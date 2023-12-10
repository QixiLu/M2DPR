import pickle
import logging
import numpy as np

from typing import Dict
from transformers.utils.logging import enable_explicit_format
from transformers.trainer_callback import PrinterCallback
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    PreTrainedTokenizerFast,
    EvalPrediction,
)

from logger_config import logger, LoggerCallback
from config import M2DPRTrainingArguments
from loaders import M2DPRPretrainDataloader
from collators import M2DPRPretrainDataCollator
from trainers import M2DPRPretrainTrainer
from models import M2DPRPretrainModel

logger = logging.getLogger(__name__)

def _common_setup(args: M2DPRTrainingArguments):
    if args.process_index > 0:
        logger.setLevel(logging.WARNING)
    enable_explicit_format()
    set_seed(args.seed)


def _compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    preds = eval_pred.predictions

    avg_enc_mlm_loss = float(np.mean(preds[0]))
    avg_recon_inputs_loss = float(np.mean(preds[2]))
    avg_recon_queries_loss = float(np.mean(preds[3]))
    avg_tfidf_loss = float(np.mean(preds[4]))
    avg_total_loss = float(np.mean(preds[5]))
    avg_replace_ratio = float(np.mean(preds[6]))

    return {'avg_enc_mlm_loss': round(avg_enc_mlm_loss, 4),
            'avg_recon_inputs_loss': round(avg_recon_inputs_loss, 4),
            'avg_recon_queries_loss': round(avg_recon_queries_loss, 4),
            'avg_tfidf_loss': round(avg_tfidf_loss, 4),
            'avg_total_loss': round(avg_total_loss, 4),
            'avg_replace_ratio': round(avg_replace_ratio, 4)}


def main():
    parser = HfArgumentParser((M2DPRTrainingArguments,))
    args: M2DPRTrainingArguments = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    _common_setup(args)
    logger.info('Args={}'.format(str(args)))

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    model: M2DPRPretrainModel = M2DPRPretrainModel.build(tokenizer.cls_token_id, args)
    
    logger.info(model)
    logger.info('Vocab size: {}'.format(len(tokenizer)))

    dataloader = M2DPRPretrainDataloader(args=args, tokenizer=tokenizer)
    train_dataset, eval_dataset = dataloader.train_dataset, dataloader.eval_dataset

    with open(args.idf_path, 'rb') as f:
        idfs = pickle.load(f)
        
    data_collator = M2DPRPretrainDataCollator(tokenizer, pad_to_multiple_of=8 if args.fp16 else None, args=args, idfs=idfs)
    trainer: M2DPRPretrainTrainer = M2DPRPretrainTrainer(model=model,
                                                         args=args,
                                                         train_dataset=train_dataset if args.do_train else None,
                                                         eval_dataset=eval_dataset if args.do_eval else None,
                                                         data_collator=data_collator,
                                                         compute_metrics=_compute_metrics,
                                                         tokenizer=tokenizer,
                                                        )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)

    if args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return


if __name__ == "__main__":
    main()

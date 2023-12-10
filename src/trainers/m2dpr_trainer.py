import os

from typing import Optional
from transformers.trainer import Trainer

from models import M2DPRConfig, M2DPROutput, M2DPRModel, M2DPRPretrainModel
from utils import AverageMeter

from logger_config import logger
import logging

from config import M2DPRTrainingArguments

logger = logging.getLogger(__name__)


class M2DPRPretrainTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(M2DPRPretrainTrainer, self).__init__(*pargs, **kwargs)
        self.model: M2DPRPretrainModel

        self.enc_mlm_loss = AverageMeter('enc_mlm_loss', round_digits=3) if self.args.encoder_mlm_task else None
        self.recon_inputs_loss = AverageMeter('recon_inputs_loss', round_digits=3) if self.args.reconstruction_inputs_task else None
        self.recon_queries_loss = AverageMeter('recon_queries_loss', round_digits=3) if self.args.reconstruction_queries_task else None
        self.tf_idf_loss = AverageMeter('tf_idf_loss', round_digits=3) if self.args.prediction_tf_idf_task else None
        self.token_replace_loss = AverageMeter('token_replace_loss', round_digits=3) if self.args.token_replace_task else None
        self.total_loss = AverageMeter('total_loss', round_digits=3)
        self.last_epoch = 0

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))
        self.model.save_model(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs: M2DPRPretrainModel = model(inputs)
        loss = outputs.total_loss
        
        if self.model.training:
            result = []
            if self.enc_mlm_loss:
                self.enc_mlm_loss.update(outputs.encoder_mlm_loss.item())
                result.append(self.enc_mlm_loss)
            if self.recon_inputs_loss:
                self.recon_inputs_loss.update(outputs.recon_inputs_loss.item())
                result.append(self.recon_inputs_loss)
            if self.recon_queries_loss:
                self.recon_queries_loss.update(outputs.recon_queries_loss.item())
                result.append(self.recon_queries_loss)
            if self.tf_idf_loss:
                self.tf_idf_loss.update(outputs.tf_idf_loss.item())
                result.append(self.tf_idf_loss)
            if self.token_replace_loss:
                self.token_replace_loss.update(outputs.token_replace_loss.item())
                result.append(self.token_replace_loss)
            self.total_loss.update(outputs.total_loss.item())
            result.append(self.total_loss)
            
            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                log_info = ', '.join(map(str, result))
                logger.info('step: {}, {}'.format(self.state.global_step, log_info))

            self._reset_meters_if_needed()

        return (loss, outputs) if return_outputs else loss

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            if self.enc_mlm_loss:
                self.enc_mlm_loss.reset()
            if self.recon_inputs_loss:
                self.recon_inputs_loss.reset()
            if self.recon_queries_loss:
                self.recon_queries_loss.reset()
            if self.tf_idf_loss:
                self.tf_idf_loss.reset()
            if self.token_replace_loss:
                self.token_replace_loss.reset()
            self.total_loss.reset()

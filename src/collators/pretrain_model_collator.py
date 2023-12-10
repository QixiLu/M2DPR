import copy
import torch
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from transformers import BatchEncoding, BertTokenizerFast
from transformers.data.data_collator import _torch_collate_batch
from transformers.file_utils import PaddingStrategy

from config import M2DPRTrainingArguments
from .collator_utils import whole_word_mask, torch_mask_tokens


@dataclass
class M2DPRPretrainDataCollator:
    tokenizer: BertTokenizerFast
    pad_to_multiple_of: Optional[int] = None
    args: M2DPRTrainingArguments = None        
    idfs: Dict = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
            
    def __call__(self, features: List[Dict]):
        return self.torch_call(features)

    def torch_call(self, examples: List[Dict[str, Any]]) -> BatchEncoding:
        if 'title' in examples[0]:
            text, text_pair = [ex['title'] for ex in examples], [ex['contents'] for ex in examples]
        else:
            text, text_pair = [ex['contents'] for ex in examples], None
        
        batch_dict = self.tokenizer(text,
                                    text_pair=text_pair,
                                    max_length=self.args.pretrain_max_length,
                                    padding=PaddingStrategy.DO_NOT_PAD,
                                    truncation=True)

        if self.args.prediction_tf_idf_task or self.args.token_replace_task:
            padded_batch_dict = self.tokenizer.pad(batch_dict,
                                                pad_to_multiple_of=self.pad_to_multiple_of,
                                                return_tensors="pt")
            
            merged_batch_dict = {}

            if self.args.prediction_tf_idf_task:
                seq_length = padded_batch_dict['input_ids'].size(1)
                tf_idfs = torch.tensor([ex['tf_idf'][:seq_length] for ex in examples])

                assert tf_idfs.size() == padded_batch_dict['input_ids'].size()
                padded_batch_dict['labels'] = tf_idfs
                merged_batch_dict['tf_idf_inputs'] = padded_batch_dict

            if self.args.token_replace_task:
                rand_num = random.randint(0, 39)
                token_replace = [ex['token_replace_candidates'][rand_num] for ex in examples]
                token_replace_batch_dict = {'input_ids': token_replace}
                token_replace_batch_dict = self.tokenizer.pad(token_replace_batch_dict,
                                                            pad_to_multiple_of=self.pad_to_multiple_of,
                                                            return_tensors="pt")

                token_replace_batch_labels = torch.zeros(token_replace_batch_dict['input_ids'].size(),dtype=torch.long)
                is_replace = (padded_batch_dict['input_ids'] != token_replace_batch_dict['input_ids'])
                token_replace_batch_labels[is_replace] = 1
                is_pad = (padded_batch_dict['input_ids'] == self.tokenizer.pad_token_id)
                token_replace_batch_labels[is_pad] = -100
                token_replace_batch_dict['labels'] = token_replace_batch_labels
                merged_batch_dict['token_replace_inputs'] = token_replace_batch_dict  # 
        
        if self.args.reconstruction_queries_task:
            queries = [ex['queries'] for ex in examples]
            queries_batch_dict = self.tokenizer(queries,
                                                max_length=self.args.pretrain_max_length,
                                                padding=PaddingStrategy.DO_NOT_PAD,
                                                truncation=True)

            queries_mask_labels = []
            for query_input_ids in queries_batch_dict['input_ids']:
                ref_tokens = []
                for token_id in query_input_ids:
                    token = self.tokenizer._convert_id_to_token(token_id)
                    ref_tokens.append(token)

                queries_mask_labels.append(whole_word_mask(self.tokenizer, ref_tokens, mlm_prob=self.args.queries_mask_prob))

            queries_batch_mask = _torch_collate_batch(queries_mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            queries_batch_dict = self.tokenizer.pad(queries_batch_dict, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors='pt')
            queries_inputs, queries_labels = torch_mask_tokens(self.tokenizer, queries_batch_dict['input_ids'], queries_batch_mask,
                                                            all_use_mask_token=self.args.all_use_mask_token)
            
            queries_batch_dict['input_ids'] = queries_inputs
            queries_batch_dict['labels'] = queries_labels
            merged_batch_dict['reconstruction_queries_inputs'] = queries_batch_dict  
        
        encoder_mask_labels = []
        decoder_mask_labels = []
        extra_mlm_prob = self.args.decoder_mask_prob - self.args.encoder_mask_prob
        extra_mlm_prob = extra_mlm_prob / (1 - self.args.encoder_mask_prob)

        for input_ids in batch_dict['input_ids']:
            ref_tokens = []
            for token_id in input_ids:
                token = self.tokenizer._convert_id_to_token(token_id)
                ref_tokens.append(token)
                
            encoder_mask_labels.append(whole_word_mask(self.tokenizer, ref_tokens, mlm_prob=self.args.encoder_mask_prob))

            decoder_mask = encoder_mask_labels[-1][:]
            if extra_mlm_prob > 1e-4:
                decoder_mask = [max(m1, m2) for m1, m2 in zip(decoder_mask, 
                                                              whole_word_mask(self.tokenizer, ref_tokens, mlm_prob=extra_mlm_prob))]

            assert len(decoder_mask) == len(encoder_mask_labels[-1])
            decoder_mask_labels.append(decoder_mask)

        encoder_batch_mask = _torch_collate_batch(encoder_mask_labels, self.tokenizer,
                                                  pad_to_multiple_of=self.pad_to_multiple_of)

        encoder_batch_dict = self.tokenizer.pad(batch_dict,
                                                pad_to_multiple_of=self.pad_to_multiple_of,
                                                return_tensors="pt")
        
        encoder_inputs, encoder_labels = torch_mask_tokens(self.tokenizer, encoder_batch_dict['input_ids'], encoder_batch_mask,
                                                           all_use_mask_token=self.args.all_use_mask_token)

        clean_input_ids = encoder_batch_dict['input_ids'].clone()
        encoder_batch_dict['input_ids'] = encoder_inputs
        encoder_batch_dict['labels'] = encoder_labels
        
        batch_size = encoder_batch_dict['input_ids'].size(0)
        merged_batch_dict['model_inputs'] = encoder_batch_dict
        num_tasks = self.args.reconstruction_inputs_task + self.args.reconstruction_queries_task + \
            self.args.prediction_tf_idf_task + self.args.token_replace_task
        merged_batch_dict['model_inputs']['task_input_ids'] = torch.tensor(list(range(num_tasks))).unsqueeze(0).repeat(batch_size,1)

        if self.args.reconstruction_inputs_task:
            decoder_batch_dict = {k: v for k, v in encoder_batch_dict.items()}
            if extra_mlm_prob > 1e-4:
                decoder_batch_mask = _torch_collate_batch(decoder_mask_labels, self.tokenizer,
                                                        pad_to_multiple_of=self.pad_to_multiple_of)
                decoder_inputs, decoder_labels = torch_mask_tokens(self.tokenizer, clean_input_ids, decoder_batch_mask,
                                                                all_use_mask_token=self.args.all_use_mask_token)
                decoder_batch_dict['input_ids'] = decoder_inputs
                decoder_batch_dict['labels'] = decoder_labels

            merged_batch_dict['reconstruction_inputs'] = decoder_batch_dict 
            
        return merged_batch_dict
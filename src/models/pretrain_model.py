import os
import copy
import torch
import torch.nn as nn
from typing import Dict, Tuple
from torch import Tensor
from config import M2DPRTrainingArguments
from dataclasses import dataclass
from torch.distributions import Categorical
from transformers import AutoModelForMaskedLM, AutoModel, AutoModelForMaskedLM
from .m2dpr_model import M2DPRModel, M2DPROutput, M2DPRConfig
from transformers.models.bert.modeling_bert import BertSelfOutput, BertLayer, BertIntermediate, BertOutput, BertEmbeddings, BertPreTrainedModel, BertOnlyMLMHead
from functools import reduce

    
class M2DPRPretrainModel(nn.Module):
    def __init__(self, args: M2DPRTrainingArguments, model: M2DPRModel):
        super().__init__()
        self.args = args
        self.model = model

        if args.is_pretrain:
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)
            self.reg = nn.MSELoss()
            
            self.cls_encoder_inputs = BertOnlyMLMHead(self.model.config)
            bert_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

            if args.reconstruction_inputs_task:
                self.recon_inputs_layer = copy.deepcopy(bert_model.bert.encoder.layer[-args.pretrain_num_layers:])
                self.cls_recon_inputs = copy.deepcopy(bert_model.cls)

            if args.reconstruction_queries_task:
                self.recon_queries_layer = copy.deepcopy(bert_model.bert.encoder.layer[-args.pretrain_num_layers:])
                self.cls_recon_queries = copy.deepcopy(bert_model.cls)

            if args.token_replace_task:
                self.token_replace_layer = copy.deepcopy(bert_model.bert.encoder.layer[-args.pretrain_num_layers:])
                self.cls_token_replace = nn.Sequential(copy.deepcopy(bert_model.cls.predictions.transform),
                                                       nn.Linear(self.args.hidden_size,2))
                self.cls_token_replace[1].weight.data.normal_(mean=0.0, std=0.02)

            if args.prediction_tf_idf_task:
                self.prediction_tf_idf_layer = copy.deepcopy(bert_model.bert.encoder.layer[-args.pretrain_num_layers:])
                self.cls_tf_idf = nn.Sequential(copy.deepcopy(bert_model.cls.predictions.transform),
                                                nn.Linear(self.args.hidden_size, 1))
                self.cls_tf_idf[1].weight.data.normal_(mean=0.0, std=0.02)
                
            del bert_model

    def forward(self, inputs):
        model_inputs = inputs['model_inputs']

        outputs = self.model(**{k: v for k, v in model_inputs.items() if k != 'labels'})  
        encoder_prediction_scores = self.cls_encoder_inputs(outputs.hidden_states)

        encoder_mlm_loss = None
        if self.args.encoder_mlm_task:
            encoder_mlm_loss = self.mlm_loss(encoder_prediction_scores, model_inputs['labels'])

        # recon_inputs_task
        recon_inputs_mlm_loss = None
        if self.args.reconstruction_inputs_task:
            reconstruction_inputs = inputs['reconstruction_inputs']
            model_recon_outputs = outputs.recon_inputs_repr
            assert len(model_recon_outputs.size()) == 2

            skip_hiddens = self.model.embeddings(input_ids=reconstruction_inputs['input_ids'])
            recon_inputs_hiddens = torch.cat([model_recon_outputs.unsqueeze(1), skip_hiddens[:, 1:]], dim=1)
            attention_mask = self.model.get_extended_attention_mask(reconstruction_inputs['attention_mask'],
                                                                    reconstruction_inputs['attention_mask'].shape,
                                                                    reconstruction_inputs['attention_mask'].device,
                                                                    )
            for layer in self.recon_inputs_layer:
                layer_out = layer(
                    recon_inputs_hiddens,
                    attention_mask,
                )
                recon_inputs_hiddens = layer_out[0]
            recon_inputs_prediction_scores = self.cls_recon_inputs(recon_inputs_hiddens)
            
            recon_inputs_mlm_loss = self.mlm_loss(recon_inputs_prediction_scores, reconstruction_inputs['labels'])

        # recon_queries_task
        recon_queries_mlm_loss = None
        if self.args.reconstruction_queries_task:
            reconstruction_queries_inputs = inputs['reconstruction_queries_inputs']
            model_recon_queries_outputs = outputs.recon_queries_repr
            assert len(model_recon_queries_outputs.size()) == 2

            skip_hiddens = self.model.embeddings(input_ids=reconstruction_queries_inputs['input_ids'])
            recon_queries_hiddens = torch.cat([model_recon_queries_outputs.unsqueeze(1), skip_hiddens[:, 1:]], dim=1)
            attention_mask = self.model.get_extended_attention_mask(reconstruction_queries_inputs['attention_mask'],
                                                                    reconstruction_queries_inputs['attention_mask'].shape,
                                                                    reconstruction_queries_inputs['attention_mask'].device,
                                                                    )
            for layer in self.recon_queries_layer:
                layer_out = layer(
                    recon_queries_hiddens,
                    attention_mask,
                )
                recon_queries_hiddens = layer_out[0]
            recon_queries_prediction_scores = self.cls_recon_queries(recon_queries_hiddens)
            recon_queries_mlm_loss = self.mlm_loss(recon_queries_prediction_scores, reconstruction_queries_inputs['labels'])
        
        # token_replace_task
        token_replace_loss = None
        if self.args.token_replace_task:
            token_replace_inputs = inputs['token_replace_inputs']
            model_token_replace_outputs = outputs.token_replace_repr

            skip_hiddens = self.model.embeddings(token_replace_inputs['input_ids'])
            token_replace_hiddens = torch.cat([model_token_replace_outputs.unsqueeze(1), skip_hiddens[:, 1:]], dim=1)
            attention_mask = self.model.get_extended_attention_mask(token_replace_inputs['attention_mask'],
                                                                    token_replace_inputs['attention_mask'].shape,
                                                                    token_replace_inputs['attention_mask'].device,
                                                                    )
            for layer in self.token_replace_layer:
                layer_out = layer(
                    token_replace_hiddens,
                    attention_mask,
                )
                token_replace_hiddens = layer_out[0]
            token_replace_prediction_scores = self.cls_token_replace(token_replace_hiddens)

            token_replace_loss = self.cross_entropy(token_replace_prediction_scores.view(-1, 2), token_replace_inputs['labels'].view(-1))
            
        # tf_idf_task
        tf_idf_loss = None
        if self.args.prediction_tf_idf_task:
            tf_idf_inputs = inputs['tf_idf_inputs']
            model_tfidf_outputs = outputs.tf_idf_repr
            assert len(model_tfidf_outputs.size()) == 2

            skip_hiddens = self.model.embeddings(input_ids=tf_idf_inputs['input_ids'])
            tf_idf_hiddens = torch.cat([model_tfidf_outputs.unsqueeze(1), skip_hiddens[:, 1:]], dim=1)
            attention_mask = self.model.get_extended_attention_mask(tf_idf_inputs['attention_mask'],
                                                                    tf_idf_inputs['attention_mask'].shape,
                                                                    tf_idf_inputs['attention_mask'].device,
                                                                    )
            for layer in self.prediction_tf_idf_layer:
                layer_out = layer(
                    tf_idf_hiddens,
                    attention_mask,
                )
                tf_idf_hiddens = layer_out[0]
            tf_idf_prediction_scores = self.cls_tf_idf(tf_idf_hiddens)
            tf_idf_loss = self.tf_idf_loss(tf_idf_prediction_scores, tf_idf_inputs['labels'])

        loss = 0.0
        if encoder_mlm_loss:
            loss += encoder_mlm_loss
        if recon_inputs_mlm_loss:
            loss += recon_inputs_mlm_loss
        if recon_queries_mlm_loss:
            loss += recon_queries_mlm_loss
        if token_replace_loss:
            loss += token_replace_loss
        if tf_idf_loss:
            loss += tf_idf_loss
    
        return M2DPROutput(encoder_mlm_loss=encoder_mlm_loss,
                           recon_inputs_loss=recon_inputs_mlm_loss,
                           recon_queries_loss=recon_queries_mlm_loss,
                           token_replace_loss=token_replace_loss,
                           tf_idf_loss=tf_idf_loss,
                           total_loss=loss,
                          )

    def mlm_loss(self, pred_scores, labels):
        masked_lm_loss = self.cross_entropy(pred_scores.view(-1, self.model.config.vocab_size), labels.view(-1))
        return masked_lm_loss

    def tf_idf_loss(self, pred_scores, labels):
        new_labels = labels == 0.0
        pred_scores[new_labels] == 0.0
        tf_idf_loss = self.reg(pred_scores.view(-1), labels.view(-1))
        return tf_idf_loss
           
    @classmethod
    def build(cls, cls_token_id, args: M2DPRTrainingArguments):
        model_config = M2DPRConfig(reconstruction_inputs_task=args.reconstruction_inputs_task,
                                   reconstruction_queries_task=args.reconstruction_queries_task,
                                   prediction_tf_idf_task=args.prediction_tf_idf_task,
                                   token_replace_task=args.token_replace_task,
                                   vocab_size=args.vocab_size,
                                   hidden_size=args.hidden_size,
                                   num_attention_heads=args.num_attention_heads,
                                   intermediate_size=args.intermediate_size,
                                   hidden_act=args.hidden_act,
                                   hidden_dropout_prob=args.hidden_dropout_prob,
                                   attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                                   max_position_embeddings=args.max_position_embeddings,
                                   type_vocab_size=args.type_vocab_size,
                                   initializer_range=args.initializer_range,
                                   layer_norm_eps=args.layer_norm_eps,
                                   position_embedding_type=args.position_embedding_type,
                                   use_cache=args.use_cache,
                                   cross_attention_num_hidden_layers=args.cross_attention_num_hidden_layers,
                                   extra_text_query=args.extra_text_query)

        encoder_model = M2DPRModel.build(model_config, cls_token_id, load_bert_checkpoint=args.load_bert_checkpoint, load_m2dpr_checkpoint=args.load_m2dpr_checkpoint)
        model = M2DPRPretrainModel(args, encoder_model)
        
        if args.is_pretrain and args.load_pretrain_checkpoint:
            state_dict = torch.load(os.path.join(args.load_pretrain_checkpoint, 'pretrain_model.pt'), map_location="cpu")
            model.load_state_dict(state_dict, strict=True)

        return model

    def save_model(self, output_dir):
        model_dict = self.state_dict()
        torch.save(model_dict, os.path.join(output_dir, 'pretrain_model.pt'))
        encoder_dict = self.model.state_dict()
        torch.save(encoder_dict, os.path.join(output_dir, 'encoder_model.pt'))
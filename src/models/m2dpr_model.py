import copy
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import BertModel, BertConfig
from typing import List, Optional, Tuple, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.modeling_outputs import MaskedLMOutput, ModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert.modeling_bert import BertSelfOutput, BertLayer, BertIntermediate, BertOutput, BertEmbeddings, BertPreTrainedModel


class M2DPRConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        reconstruction_inputs_task=True,
        reconstruction_queries_task = True,
        prediction_tf_idf_task = True,
        token_replace_task = True,
        vocab_size=30522,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3092,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
        cross_attention_num_hidden_layers=12,
        extra_text_query=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.cross_attention_num_hidden_layers = cross_attention_num_hidden_layers
        self.pretrain_cross_attention_tasks_num = \
            reconstruction_inputs_task + reconstruction_queries_task + \
            prediction_tf_idf_task + token_replace_task
        self.reconstruction_inputs_task = reconstruction_inputs_task
        self.reconstruction_queries_task = reconstruction_queries_task
        self.prediction_tf_idf_task = prediction_tf_idf_task
        self.token_replace_task = token_replace_task
        self.extra_text_query = extra_text_query

    
@dataclass
class M2DPROutput(ModelOutput):
    hidden_states: Optional[torch.Tensor] = None
    task_reprs: Optional[torch.Tensor] = None
    recon_inputs_repr: Optional[torch.Tensor] = None
    recon_queries_repr: Optional[torch.Tensor] = None
    token_replace_repr: Optional[torch.Tensor] = None
    tf_idf_repr: Optional[torch.Tensor] = None

    replace_ratio: Optional[torch.Tensor] = None
    encoder_mlm_loss: Optional[torch.Tensor] = None
    recon_inputs_loss: Optional[torch.Tensor] = None
    recon_queries_loss: Optional[torch.Tensor] = None
    token_replace_loss: Optional[torch.Tensor] = None
    tf_idf_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None


class TaskEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config: M2DPRConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.pretrain_cross_attention_tasks_num, config.hidden_size)  # 
        self.position_embeddings = nn.Embedding(config.pretrain_cross_attention_tasks_num, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.pretrain_cross_attention_tasks_num, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(input_ids)
        
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(input_ids)
            embeddings += position_embeddings
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

        
class M2DPRCrossAttention(nn.Module):
    def __init__(self, config: M2DPRConfig, position_embedding_type=None):
        super().__init__()

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if config.extra_text_query:
            self.text_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        # batch_size, num_attention_heads, seq_length, attention_head_size
        return x.permute(0, 2, 1, 3)

    def qkv(self, query_layer, key_layer, value_layer, attention_mask, output_attentions):
        # (batch_size, num_head_attention, task_num, seq_length)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # attention_mask -> (batch_size, num_attention_heads, task_num, seq_length)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        # outputs[0] -> batch_size, task_num, hidden_size
        return outputs

    def forward(
        self,
        task_reprs: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        task_query_layer = self.query(task_reprs)
        # (batch_size, num_head_attentions, task_num, attention_size)
        query_layer = self.transpose_for_scores(task_query_layer)
        task_outputs = self.qkv(query_layer, key_layer, value_layer, attention_mask, output_attentions)
        text_outputs = None, None
        if self.config.extra_text_query:
            text_query_layer = self.text_query(hidden_states)
            query_layer = self.transpose_for_scores(text_query_layer)
            text_outputs = self.qkv(query_layer, key_layer, value_layer, attention_mask, output_attentions)
        return task_outputs[0], text_outputs[0]


class M2DPRCrossAttentionBlock(nn.Module):
    def __init__(self, config: M2DPRConfig, position_embedding_type=None):
        super().__init__()
        self.config = config
        self.self = M2DPRCrossAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        task_reprs: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        task_outputs, text_outputs = self.self(task_reprs, hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        task_attention_output = self.output(task_outputs, task_reprs)
        text_attention_output = None
        if self.config.extra_text_query:
            text_attention_output = self.output(text_outputs, hidden_states)
        return task_attention_output, text_attention_output


class M2DPRLayer(nn.Module):
    def __init__(self, config: M2DPRConfig):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = M2DPRCrossAttentionBlock(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, task_reprs, hidden_states, attention_mask=None, output_attentions=None):
        task_attention_outputs, text_attention_outputs = self.attention(task_reprs,
                                                                        hidden_states,
                                                                        attention_mask,
                                                                        output_attentions=output_attentions)
        

        task_layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, task_attention_outputs
        )
        text_layer_output = None
        if self.config.extra_text_query:
            text_layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, text_attention_outputs
            )
        return task_layer_output, text_layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class M2DPRBlock(nn.Module):
    def __init__(self, config: M2DPRConfig):
        super().__init__()
        self.config = config
        self.cross_attention_layer = nn.ModuleList([M2DPRLayer(config) for _ in range(config.cross_attention_num_hidden_layers)])

    def forward(
        self,
        task_reprs: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        for i in range(self.config.cross_attention_num_hidden_layers):

            if self.config.extra_text_query:
                task_reprs, hidden_states = self.cross_attention_layer[i](task_reprs, hidden_states, attention_mask=attention_mask)
            else:
                final_query_reprs = torch.cat([task_reprs, hidden_states], dim=1).to(hidden_states.device)
                task_reprs = final_query_reprs[0][:,:self.config.pretrain_cross_attention_tasks_num,:]
                hidden_states = final_query_reprs[0][:,self.config.pretrain_cross_attention_tasks_num:,:]
        return hidden_states, task_reprs
        

class M2DPRModel(BertPreTrainedModel):
    def __init__(self, config: M2DPRConfig, add_pooling_layer=False):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.task_embeddings = TaskEmbeddings(config)
        self.encoder = M2DPRBlock(config)

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        task_input_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.embeddings(input_ids=input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids)

        task_reprs = self.task_embeddings(input_ids=task_input_ids)
        
        encoder_outputs, task_reprs = self.encoder(task_reprs,
                                                   embedding_output,
                                                   attention_mask=extended_attention_mask,
                                                   output_attentions=output_attentions,
                                                   output_hidden_states=output_hidden_states,
                                                   return_dict=return_dict,
                                                   )
        assert len(encoder_outputs.size())==3, len(task_reprs.size())==3

        cnt=0
        recon_inputs_repr = None
        recon_queries_repr = None
        token_replace_repr = None
        tf_idf_repr = None
        if self.config.reconstruction_inputs_task:
            recon_inputs_repr = task_reprs[:,cnt,:]
            recon_inputs_repr = nn.functional.normalize(recon_inputs_repr)
            cnt+=1
        if self.config.reconstruction_queries_task:
            recon_queries_repr = task_reprs[:,cnt,:]
            recon_queries_repr = nn.functional.normalize(recon_queries_repr)
            cnt+=1
        if self.config.token_replace_task:
            token_replace_repr = task_reprs[:,cnt,:]
            token_replace_repr = nn.functional.normalize(token_replace_repr)
            cnt+=1
        if self.config.prediction_tf_idf_task:
            tf_idf_repr = task_reprs[:,cnt,:]
            tf_idf_repr = nn.functional.normalize(tf_idf_repr)
            cnt+=1
        assert cnt == self.config.pretrain_cross_attention_tasks_num

        return M2DPROutput(hidden_states=encoder_outputs,
                           task_reprs=task_reprs,
                           recon_inputs_repr=recon_inputs_repr,
                           recon_queries_repr=recon_queries_repr,
                           token_replace_repr=token_replace_repr,
                           tf_idf_repr=tf_idf_repr,
                          )

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[...,None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1)/mask.sum(dim=1)[...,None]
        sentence_embeddings = nn.functional.normalize(sentence_embeddings)
        return sentence_embeddings
                           
    @classmethod
    def build(cls, config: M2DPRConfig, cls_token_id, load_bert_checkpoint=True, load_m2dpr_checkpoint=False):
        model = cls(config)
        
        if load_bert_checkpoint:
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            with torch.no_grad():
                model.embeddings = copy.deepcopy(bert_model.embeddings)
                cls_token_embedding = bert_model.embeddings.word_embeddings(
                    torch.tensor([cls_token_id])).repeat(config.pretrain_cross_attention_tasks_num, 1)
                model.task_embeddings.word_embeddings.weight.data = copy.deepcopy(cls_token_embedding)
                pos_token_embedding = bert_model.embeddings.position_embeddings(
                    torch.tensor(list(range(config.pretrain_cross_attention_tasks_num))))
                model.task_embeddings.position_embeddings.weight.data = copy.deepcopy(pos_token_embedding)
                token_type_embedding = bert_model.embeddings.token_type_embeddings(
                    torch.tensor([0] * config.pretrain_cross_attention_tasks_num))
                model.task_embeddings.token_type_embeddings.weight.data = copy.deepcopy(token_type_embedding)
            

                for i in range(min(bert_model.config.num_hidden_layers, config.cross_attention_num_hidden_layers)):
                    for param_k, param_q in zip(model.encoder.cross_attention_layer[i].attention.self.query.parameters(),
                                                bert_model.encoder.layer[i].attention.self.query.parameters()):
                        param_k.data.copy_(param_q.data)
                    for param_k, param_q in zip(model.encoder.cross_attention_layer[i].attention.self.text_query.parameters(),
                                                bert_model.encoder.layer[i].attention.self.query.parameters()):
                        param_k.data.copy_(param_q.data)
                    for param_k, param_q in zip(model.encoder.cross_attention_layer[i].attention.self.key.parameters(),
                                                bert_model.encoder.layer[i].attention.self.key.parameters()):
                        param_k.data.copy_(param_q.data)
                    for param_k, param_q in zip(model.encoder.cross_attention_layer[i].attention.self.value.parameters(),
                                                bert_model.encoder.layer[i].attention.self.value.parameters()):
                        param_k.data.copy_(param_q.data)
                    for param_k, param_q in zip(model.encoder.cross_attention_layer[i].attention.output.parameters(),
                                                bert_model.encoder.layer[i].attention.output.parameters()):
                        param_k.data.copy_(param_q.data)
                    for param_k, param_q in zip(model.encoder.cross_attention_layer[i].intermediate.parameters(),
                                                bert_model.encoder.layer[i].intermediate.parameters()):
                        param_k.data.copy_(param_q.data)
                    for param_k, param_q in zip(model.encoder.cross_attention_layer[i].output.parameters(),
                                                bert_model.encoder.layer[i].output.parameters()):
                        param_k.data.copy_(param_q.data)
            
            del bert_model
            
        if load_m2dpr_checkpoint:
            state_dict = torch.load(os.path.join(load_m2dpr_checkpoint,'m2dpr_model.pt'), map_location="cpu")
            model.load_state_dict(state_dict, stric=True)
            
        return model

    def save_model(self, output_dir):
        model_dict = self.state_dict()
        torch.save(model_dict, os.path.join(output_dir, 'm2dpr_model.pt'))

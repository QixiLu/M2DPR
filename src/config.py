import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class M2DPRTrainingArguments(TrainingArguments):
    tokenizer_name_or_path: Optional[str] = field(default="bert-base-uncased")
    load_bert_checkpoint: Optional[str] = field(default='bert-base-uncased', metadata={'help': "Path to bert model"})
    load_m2dpr_checkpoint: Optional[str] = field(default=None, metadata={'help': "Path to m2dpr model checkpoint"})
    load_pretrain_checkpoint: Optional[str] = field(default=None)
    
    reconstruction_inputs_task: Optional[bool] = field(default=True, metadata={'help': "whether to use the reconstruction inputs task"})
    reconstruction_queries_task: Optional[bool] = field(default=True, metadata={'help': "whether to use the reconstruction queries task"})
    token_replace_task: Optional[bool] = field(default=True)
    prediction_tf_idf_task: Optional[bool] = field(default=True, metadata={'help': "whether to use the prediction tfidf task"})
    encoder_mlm_task: Optional[bool] = field(default=True)
    
    pretrain_num_layers: Optional[int] = field(default=2)
    is_pretrain: Optional[bool] = field(default=True)
    pretrain_max_length: Optional[int] = field(default=150)
    
    output_dir: Optional[str] = field(default="./pretrain_checkpoint/")

    vocab_size: Optional[int] = field(default=30522)
    hidden_size: Optional[int] = field(default=768)
    num_attention_heads: Optional[int] = field(default=12)
    intermediate_size: Optional[int] = field(default=3072)
    hidden_act: Optional[str] = field(default="gelu")
    hidden_dropout_prob: Optional[float] = field(default=0.1)
    attention_probs_dropout_prob: Optional[float] = field(default=0.1)
    max_position_embeddings: Optional[int] = field(default=512)
    type_vocab_size: Optional[int] = field(default=2)
    initializer_range: Optional[float] = field(default=0.02)
    layer_norm_eps: Optional[float] = field(default=1e-12)
    position_embedding_type: Optional[str] = field(default="absolute")
    use_cache: Optional[bool] = field(default=True)
    cross_attention_num_hidden_layers: Optional[int] = field(default=12)
    extra_text_query: Optional[bool] = field(default=True)

    idf_path: Optional[str] = field(default="./data/msmarco_bm25_official/idf.p")
    train_file: Optional[str] = field(default="./data/")
    num_eval_samples: Optional[int] = field(default=0)
    max_train_samples: Optional[int] = field(default=None)
    decoder_mask_prob: Optional[float] = field(default=0.5)
    encoder_mask_prob: Optional[float] = field(default=0.3)
    queries_mask_prob: Optional[float] = field(default=0.5)
    all_use_mask_token: Optional[bool] = field(default=False)

    
    
    


    
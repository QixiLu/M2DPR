# M2DPR
This is the implementation of the paper "M2DPR: A Multi-task Multi-view Representation Learning Framework for Dense Passage Retrieval".  

This project is based on [simlm](https://github.com/microsoft/unilm/tree/master/simlm)

## Pre-training and Fine-tuning
To pretrain M2DPR, run `bash scripts/pretrain_m2dpr.sh`;  
To fine-tune M2DPR, run `bash scripts/train_biencoder_marco.sh`;  
To encode passages, run `bash scripts/encode_marco.sh /path/to/checkpoint`;  
To do evaluation, run `bash scripts/search_marco.sh /path/to/checkpoint`.

## Available Models
Pretrained M2DPR Encoder: https://drive.google.com/file/d/1bQS-r8Bbq5CllQfRL_XKtQVoQCoH9b-m/view?usp=sharing  
Fine-Tuned M2DPR with distillation on MS-MARCO: https://drive.google.com/file/d/1P01FPZSnfKX941e26uYxdM8wEGMweSaq/view?usp=sharing

## Training Data
Data used during pretraining: https://12c830b6-d90c-4b17-9c65-314b0a802de9.s3.us-west-1.amazonaws.com/pretrain_data.tar

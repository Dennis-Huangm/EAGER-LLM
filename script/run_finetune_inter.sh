#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,3

# Dataset configuration
INTER_FILE=/home/hongminjie/EAGER-LLM/sequential-multievent-500m.inter
INDEX_FILE=/home/hongminjie/EAGER-LLM/sequential-multievent-500m.index.json  # Optional: path to index.json for SID mapping
BASE_MODEL=/home/hongminjie/MiniOneRec/yambda/model
VERSION=v0
OUTPUT_DIR=/home/hongminjie/EAGER-LLM/checkpoints/multievent/$VERSION

# Training with .inter file
torchrun --nproc_per_node=3 --master_port=23325 src/finetune.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --inter_file $INTER_FILE \
    --index_file $INDEX_FILE \
    --per_device_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --epochs 3 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed /home/hongminjie/EAGER-LLM/ds_config/ds_z3_bf16.json \
    --bf16 \
    --only_train_response \
    --tasks interseqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --max_his_len 20 \
    --his_sep ","

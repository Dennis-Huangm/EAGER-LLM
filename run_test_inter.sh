#!/bin/bash

# Example script to run evaluation on sequential-multievent-500m.inter dataset

# Set paths
INTER_FILE="/root/autodl-tmp/huangminrui/EAGER-LLM/sequential-multievent-demo.inter"
INDEX_FILE="/root/autodl-tmp/huangminrui/EAGER-LLM/sequential-multievent-500m.index.json"
CKPT_PATH="/root/autodl-tmp/huangminrui/LLaMA-Factory/saves/Qwen3-1.7B/full/train_2025-12-11-23-29-46/checkpoint-2400"  # Update this to your model checkpoint path
RESULTS_FILE="./results/sequential-multievent-500m-results.json"

# Run distributed test
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 \
    src/test_ddp.py \
    --inter_file ${INTER_FILE} \
    --index_file ${INDEX_FILE} \
    --ckpt_path ${CKPT_PATH} \
    --results_file ${RESULTS_FILE} \
    --test_task SeqRec \
    --test_prompt_ids "0" \
    --test_batch_size 32 \
    --num_beams 20 \
    --sample_num -1 \
    --max_his_len 20 \
    --his_sep "," \
    --metrics "recall@5,recall@10,recall@20,ndcg@5,ndcg@10,ndcg@20" \
    --seed 42

# Notes:
# - Update CKPT_PATH to your model checkpoint
# - Adjust nproc_per_node based on your available GPUs
# - Use --sample_num to test on a subset (e.g., --sample_num 1000)
# - Use --add_prefix to add sequential prefixes to history items
# - Use --test_prompt_ids "all" to test all prompts
# - Adjust --test_batch_size based on your GPU memory

#!/bin/bash

# Example script to run evaluation on sequential-multievent-500m.inter dataset

# Set paths
INTER_FILE="/root/autodl-tmp/huangminrui/EAGER-LLM/sequential-multievent-500m.test.inter"
INDEX_FILE="/root/autodl-tmp/huangminrui/EAGER-LLM/sequential-multievent-500m.index.json"
CKPT_BASE="/root/autodl-tmp/huangminrui/EAGER-LLM/checkpoints/multievent/v0"
RESULTS_DIR="./results"

# Checkpoints to test
CHECKPOINTS=("checkpoint-236-fp32")

# Create results directory if not exists    
mkdir -p ${RESULTS_DIR}

# Loop through each checkpoint
for CKPT in "${CHECKPOINTS[@]}"; do
    echo "=============================================="
    echo "Testing ${CKPT}..."
    echo "=============================================="
    
    CKPT_PATH="${CKPT_BASE}/${CKPT}"
    RESULTS_FILE="${RESULTS_DIR}/sequential-multievent-500m-${CKPT}-results.json"
    
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
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
        --seed 2020
    
    echo "Results saved to ${RESULTS_FILE}"
    echo ""
done

echo "All checkpoints tested!"

# Notes:
# - Update CKPT_BASE to your model checkpoint base directory
# - Adjust nproc_per_node based on your available GPUs
# - Use --sample_num to test on a subset (e.g., --sample_num 1000)
# - Use --add_prefix to add sequential prefixes to history items
# - Use --test_prompt_ids "all" to test all prompts
# - Adjust --test_batch_size based on your GPU memory

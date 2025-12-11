#!/bin/bash

# vLLM-based testing script for high-throughput inference
# Usage: bash run_test_vllm.sh
# Usage with specific GPU: CUDA_VISIBLE_DEVICES=0 bash run_test_vllm.sh

# GPU selection (default: GPU 0)
export CUDA_VISIBLE_DEVICES=3

# Model and data paths
INTER_FILE="/root/autodl-tmp/huangminrui/EAGER-LLM/sequential-multievent-demo.inter"
INDEX_FILE="/root/autodl-tmp/huangminrui/EAGER-LLM/sequential-multievent-500m.index.json"
CKPT_PATH="/root/autodl-tmp/huangminrui/LLaMA-Factory/saves/Qwen3-1.7B/full/train_2025-12-11-23-29-46/checkpoint-2400"
RESULTS_FILE="./results/sequential-multievent-500m-results.json"

# Test parameters
TEST_BATCH_SIZE=32  # vLLM can handle larger batches efficiently
NUM_BEAMS=20
SAMPLE_NUM=-1  # -1 for all data, or set a number for testing
MAX_HIS_LEN=20
SEED=42

# vLLM specific parameters
TENSOR_PARALLEL_SIZE=1  # Number of GPUs for tensor parallelism
GPU_MEMORY_UTILIZATION=0.5
MAX_MODEL_LEN=2048

python src/test_vllm.py \
    --ckpt_path ${CKPT_PATH} \
    --inter_file ${INTER_FILE} \
    --index_file ${INDEX_FILE} \
    --results_file ${RESULTS_FILE} \
    --test_batch_size ${TEST_BATCH_SIZE} \
    --num_beams ${NUM_BEAMS} \
    --sample_num ${SAMPLE_NUM} \
    --max_his_len ${MAX_HIS_LEN} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --max_model_len ${MAX_MODEL_LEN} \
    --metrics "recall@5,recall@10,recall@20,ndcg@5,ndcg@10,ndcg@20" \
    --seed ${SEED}

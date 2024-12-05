#bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=Instruments # 'Instruments', 'Beauty', 'Sports'
ENCODER_TYPE='concat' # 'din' or 'text'
BASE_MODEL=meta-llama/llama-7b
VERSION=v0
DATA_PATH=./data
OUTPUT_DIR=./checkpoints/$DATASET/$ENCODER_TYPE/$VERSION
SENMANTIC_EMB=./data/${DATASET}/${ENCODER_TYPE}/${VERSION}/${DATASET}.emb-semantic.npy
BEHAVOR_EMB=./data/${DATASET}/${ENCODER_TYPE}/${VERSION}/${DATASET}.emb-behavior.npy

torchrun --nproc_per_node=4 --master_port=23325 src/finetune.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --epochs 3 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z3_bf16.json \
    --bf16 \
    --only_train_response \
    --tasks seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain \
    --train_prompt_sample_num 1,1,1,1,1,1 \
    --train_data_sample_num 0,0,0,100000,0,0 \
    --index_file .${ENCODER_TYPE}.index.json \
    --encoder_type $ENCODER_TYPE \
    --version $VERSION \
    --semantic_emb $SENMANTIC_EMB \
    --behavior_emb $BEHAVOR_EMB \

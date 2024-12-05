#bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=Beauty # 'Instruments', 'Beauty', 'Sports'
ENCODER_TYPE='concat' # 'din' or 'text'
VERSION=v0
CKPT_STEP=3
DATA_PATH=./data
OUTPUT_DIR=./checkpoints/$DATASET/$ENCODER_TYPE/$VERSION
SENMANTIC_EMB=./data/${DATASET}/${ENCODER_TYPE}/${VERSION}/${DATASET}.emb-semantic.npy
BEHAVOR_EMB=./data/${DATASET}/${ENCODER_TYPE}/${VERSION}/${DATASET}.emb-behavior.npy
CKPT_PATH=$OUTPUT_DIR/final-checkpoint-$CKPT_STEP/

if [ ! -e "$CKPT_PATH" ]; then
    cd convert
    ./convert.sh ../$OUTPUT_DIR $CKPT_STEP
    cd ..
else
    echo "Checkpoint exists: $CKPT_PATH"
fi

torchrun --nproc_per_node=4 --master_port=23322 src/anneal_finetune.py \
    --base_model $CKPT_PATH \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --epochs 1 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z3_bf16.json \
    --bf16 \
    --only_train_response \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .${ENCODER_TYPE}.index.json \
    --encoder_type $ENCODER_TYPE \
    --version $VERSION \
    --semantic_emb $SENMANTIC_EMB \
    --behavior_emb $BEHAVOR_EMB \
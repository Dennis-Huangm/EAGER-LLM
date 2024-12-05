#bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=Instruments # 'Instruments', 'Beauty', 'Sports'
ENCODER_TYPE='concat' # 'din' or 'text'
VERSION=v0
DATA_PATH=./data
CKPT_STEP=3
OUTPUT_DIR=./checkpoints/$DATASET/$ENCODER_TYPE/$VERSION/
CKPT_PATH=$OUTPUT_DIR/final-checkpoint-$CKPT_STEP/
TEST_TASK=seqrec # 
RESULTS_FILE=./results/$DATASET/$ENCODER_TYPE/$VERSION-test-result.json

if [ ! -e "$CKPT_PATH" ]; then
    cd convert
    ./convert.sh ../$OUTPUT_DIR $CKPT_STEP
    cd ..
else
    echo "Checkpoint exists: $CKPT_PATH"
fi

torchrun --nproc_per_node=4 --master_port=23325 src/test_ddp.py \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 4 \
    --num_beams 20 \
    --index_file .$ENCODER_TYPE.index.json \
    --version $VERSION \
    --encoder_type $ENCODER_TYPE \
    --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \

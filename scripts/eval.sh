#!/bin/sh

export TORCH_DISTRIBUTED_DEBUG=DETAIL
DATASET=simplescaling/s1 #gsm8k, math_500, aime
MODEL=Qwen/Qwen2.5-3B #deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, Qwen/Qwen2.5-32B
EFFICIENT=none #none, parameter_efficient
FLASH=True #True, False
export HF_TOKEN=your_token
TMP_TIME=$(date +%Y%m%d%H%M%S)
OUTPUT_DIR=eval_result/simplescaling/s1/$MODEL/${TMP_TIME}
SAFE=${DATASET//\//_}
CSV_PATH="$OUTPUT_DIR/${SAFE}_test_cal=False_output.csv"
mkdir -p "$(dirname "$CSV_PATH")" && echo ok > "$CSV_PATH" && echo "wrote $CSV_PATH"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 46381 \
    python src/eval.py \
    --base_model_name_or_path $MODEL \
    --model_name_or_path $MODEL \
    --parameter_efficient_mode $EFFICIENT \
    --dataset $DATASET \
    --batch_size 1 \
    --max_length 8192 \
    --seed 100 \
    --load_in_8bit True \
    --flash_attention $FLASH \
    --num_test 1000 \
    --output_dir $OUTPUT_DIR \
    --num_shards 2

#!/bin/sh

export TORCH_DISTRIBUTED_DEBUG=DETAIL
DATASET=simplescaling/s1 #gsm8k, math_500, aime
MODEL=Qwen/Qwen2.5-3B #deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, Qwen/Qwen2.5-32B
EFFICIENT=none #none, parameter_efficient
FLASH=True #True, False
export HF_TOKEN=your_token
TMP_TIME=$(date +%Y%m%d%H%M%S)
OUTPUT_DIR=eval_result/simplescaling/s1/$MODEL/${TMP_TIME}
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false


CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node ${gpu_count} --master_port 46381 \
    python src/eval.py \
    --block_size=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --base_model_name_or_path $MODEL \
    --model_name_or_path $MODEL \
    --parameter_efficient_mode $EFFICIENT \
    --dataset $DATASET \
    --warmup_ratio=0.05 \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --batch_size 1 \
    --max_length 8192 \
    --seed 100 \
    --load_in_8bit True \
    --flash_attention $FLASH \
    --num_test 1000 \
    --output_dir $OUTPUT_DIR \
    --num_shards 2

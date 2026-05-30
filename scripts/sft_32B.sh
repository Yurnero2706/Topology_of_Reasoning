#!/bin/bash
#PBS -A UTSUROLB
#PBS -b 2
#PBS -q gpu
#PBS -l elapstim_req=01:00:00
VENV_PREFIX=/work/UTSUROLB/utlb_ngy/work/.venv
source ${VENV_PREFIX}/bin/activate

# =============================================================================
# sft_14B.sh — SFT for Qwen2.5-14B  (Figure 9 reproduction)
# =============================================================================
# Hyperparameters are identical to cluster_s1K.sh except:
#   - Base model : Qwen/Qwen2.5-14B   (was 3B)
#   - Dataset    : switchable via DATASET env var
#   - save_strategy : steps / save_steps=200   (captures step-200 & step-400
#                     checkpoints needed for Figure 9)
#
# Usage
# -----
#   # Train on original s1K dataset  →  ckpts/s1-v1.0/
#   DATASET=s1K     bash scripts/sft_14B.sh
#
#   # Train on improved s1K-1.1 dataset  →  ckpts/s1-v1.1/
#   DATASET=s1K-1.1 bash scripts/sft_14B.sh
#
#   # Override output dir or stop early (e.g. just 400 steps for a quick test)
#   DATASET=s1K MAX_STEPS=400 CKPT_DIR=ckpts/my-run bash scripts/sft_14B.sh
#
# After training, run scripts/eval_14B.sh on each checkpoint you want to
# evaluate, then scripts/cluster_figure9.sh to produce Figure 9.
# =============================================================================
set -euo pipefail
module load cuda/11.8 2>/dev/null || true

# ---------------------------------------------------------------------------
# 1.  Dataset selection
# ---------------------------------------------------------------------------
DATASET="${DATASET:-s1K}"          # "s1K"  →  v1.0 / "s1K-1.1"  →  v1.1

case "${DATASET}" in
    s1K)
        TRAIN_FILE_PATH="simplescaling/s1K"
        CKPT_DIR="${CKPT_DIR:-ckpts/s1-v1.0}"
        ;;
    s1K-1.1)
        TRAIN_FILE_PATH="simplescaling/s1K-1.1"
        CKPT_DIR="${CKPT_DIR:-ckpts/s1-v1.1}"
        ;;
    *)
        echo "ERROR: DATASET must be 's1K' or 's1K-1.1', got: ${DATASET}"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# 2.  Hyperparameters
# ---------------------------------------------------------------------------
BASE_MODEL="Qwen/Qwen2.5-14B"

LR=1e-5
MIN_LR=0              # documented here for parity; not passed to sft.py
EPOCHS=5
WEIGHT_DECAY=1e-4
MICRO_BATCH=1
GRAD_ACCUM=1
MAX_STEPS="${MAX_STEPS:--1}"
BLOCK_SIZE=32768
WARMUP_RATIO=0.05
ADAM_B1=0.9
ADAM_B2=0.95

# ---------------------------------------------------------------------------
# 3.  Multi-node setup from PBS_NODEFILE
# ---------------------------------------------------------------------------
NODES=($(sort -u $PBS_NODEFILE))
MASTER_ADDR="${NODES[0]}"
NNODES="${#NODES[@]}"
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
WORK_DIR=/work/UTSUROLB/utlb_ngy/work/Topology_of_Reasoning

echo ""
echo "======================================================"
echo "  SFT: ${BASE_MODEL}"
echo "  Dataset    : ${DATASET} (${TRAIN_FILE_PATH})"
echo "  Output dir : ${CKPT_DIR}"
echo "  Nodes      : ${NNODES}  (${NODES[*]})"
echo "  Master     : ${MASTER_ADDR}"
echo "  GPUs/node  : ${GPUS_PER_NODE}"
echo "  Block size : ${BLOCK_SIZE}"
echo "  Epochs     : ${EPOCHS}   max_steps=${MAX_STEPS}"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# 4.  Launch torchrun on worker nodes via SSH (background)
# ---------------------------------------------------------------------------
for i in $(seq 1 $((NNODES - 1))); do
    ssh -o StrictHostKeyChecking=no "${NODES[$i]}" bash << REMOTE &
source ${VENV_PREFIX}/bin/activate
module load cuda/11.8 2>/dev/null || true
torchrun \
    --nnodes=${NNODES} \
    --nproc-per-node=${GPUS_PER_NODE} \
    --node_rank=${i} \
    --master_addr=${MASTER_ADDR} \
    --master_port=12345 \
    ${WORK_DIR}/src/sft.py \
    --block_size=${BLOCK_SIZE} \
    --per_device_train_batch_size=${MICRO_BATCH} \
    --per_device_eval_batch_size=${MICRO_BATCH} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --num_train_epochs=${EPOCHS} \
    --max_steps=${MAX_STEPS} \
    --train_file_path=${TRAIN_FILE_PATH} \
    --model_name=${BASE_MODEL} \
    --warmup_ratio=${WARMUP_RATIO} \
    --bf16=True \
    --eval_strategy=no \
    --logging_steps=1 \
    --save_strategy=steps \
    --save_steps=200 \
    --save_total_limit=20 \
    --lr_scheduler_type=cosine \
    --learning_rate=${LR} \
    --weight_decay=${WEIGHT_DECAY} \
    --adam_beta1=${ADAM_B1} \
    --adam_beta2=${ADAM_B2} \
    --output_dir=${CKPT_DIR} \
    --push_to_hub=False \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --optim=adamw_bnb_8bit \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config=${WORK_DIR}/train/fsdp_config_qwen_cpu.json \
    --report_to=none
REMOTE
done

# ---------------------------------------------------------------------------
# 5.  Launch torchrun on master node (rank 0) — foreground
# ---------------------------------------------------------------------------
torchrun \
    --nnodes=${NNODES} \
    --nproc-per-node=${GPUS_PER_NODE} \
    --node_rank=0 \
    --master_addr=${MASTER_ADDR} \
    --master_port=12345 \
    ${WORK_DIR}/src/sft.py \
    --block_size=${BLOCK_SIZE} \
    --per_device_train_batch_size=${MICRO_BATCH} \
    --per_device_eval_batch_size=${MICRO_BATCH} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --num_train_epochs=${EPOCHS} \
    --max_steps=${MAX_STEPS} \
    --train_file_path=${TRAIN_FILE_PATH} \
    --model_name=${BASE_MODEL} \
    --warmup_ratio=${WARMUP_RATIO} \
    --bf16=True \
    --eval_strategy=no \
    --logging_steps=1 \
    --save_strategy=steps \
    --save_steps=200 \
    --save_total_limit=20 \
    --lr_scheduler_type=cosine \
    --learning_rate=${LR} \
    --weight_decay=${WEIGHT_DECAY} \
    --adam_beta1=${ADAM_B1} \
    --adam_beta2=${ADAM_B2} \
    --output_dir=${CKPT_DIR} \
    --push_to_hub=False \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --optim=adamw_bnb_8bit \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config=${WORK_DIR}/train/fsdp_config_qwen_cpu.json \
    --report_to=none

wait

echo ""
echo "======================================================"
echo "  Training complete."
echo "  Checkpoints: ${CKPT_DIR}/checkpoint-200"
echo "======================================================"

#!/bin/bash
#PBS -A UTSUROLB
#PBS -b 1
#PBS -q gpu
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

# Expandable segments lets the CUDA allocator grow/shrink segments on demand,
# avoiding the fragmentation that causes spurious OOMs at the end of a step.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
# 2.  Hyperparameters — kept identical to cluster_s1K.sh
# ---------------------------------------------------------------------------
BASE_MODEL="Qwen/Qwen2.5-14B"

LR=1e-5
MIN_LR=0              # documented here for parity; not passed to sft.py
EPOCHS=5
WEIGHT_DECAY=1e-4
MICRO_BATCH=1         # effective batch = GPU_COUNT * MICRO_BATCH * GRAD_ACCUM
GRAD_ACCUM=1
MAX_STEPS="${MAX_STEPS:--1}"   # -1 = run full epochs; set 400 to stop early

# block_size=10000 matches cluster_s1K.sh
# (paper Table 3 lists 32768; change here if you want the paper-exact setting)
BLOCK_SIZE=32768

WARMUP_RATIO=0.05
ADAM_B1=0.9
ADAM_B2=0.95

# ---------------------------------------------------------------------------
# 3.  Auto-detect GPU count (same logic as cluster_s1K.sh)
# ---------------------------------------------------------------------------
GPU_COUNT=1  # $(nvidia-smi -L | wc -l) for multiple nodes

echo ""
echo "======================================================"
echo "  SFT: ${BASE_MODEL}"
echo "  Dataset    : ${DATASET} (${TRAIN_FILE_PATH})"
echo "  Output dir : ${CKPT_DIR}"
echo "  GPUs       : ${GPU_COUNT}"
echo "  Block size : ${BLOCK_SIZE}"
echo "  Epochs     : ${EPOCHS}   max_steps=${MAX_STEPS}"
echo "  Saves at   : every 200 steps  (→ checkpoint-200, checkpoint-400, …)"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# 4.  Launch training
# ---------------------------------------------------------------------------
torchrun \
    --nproc-per-node "${GPU_COUNT}" \
    --master_port 12345 \
    /work/UTSUROLB/utlb_ngy/work/Topology_of_Reasoning/src/sft.py \
    --block_size="${BLOCK_SIZE}" \
    --per_device_train_batch_size="${MICRO_BATCH}" \
    --per_device_eval_batch_size="${MICRO_BATCH}" \
    --gradient_accumulation_steps="${GRAD_ACCUM}" \
    --num_train_epochs="${EPOCHS}" \
    --max_steps="${MAX_STEPS}" \
    --train_file_path="${TRAIN_FILE_PATH}" \
    --model_name="${BASE_MODEL}" \
    --warmup_ratio="${WARMUP_RATIO}" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="steps" \
    --save_steps=200 \
    --save_total_limit=20 \
    --lr_scheduler_type="cosine" \
    --learning_rate="${LR}" \
    --weight_decay="${WEIGHT_DECAY}" \
    --adam_beta1="${ADAM_B1}" \
    --adam_beta2="${ADAM_B2}" \
    --output_dir="${CKPT_DIR}" \
    --push_to_hub=False \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --optim=paged_adamw_8bit \
    --report_to="none"

echo ""
echo "======================================================"
echo "  Training complete."
echo "  Checkpoints: ${CKPT_DIR}/checkpoint-200"
echo ""
echo "  Next steps:"
echo "    MODEL=${CKPT_DIR}/checkpoint-200 bash scripts/eval_14B.sh"
echo "======================================================"

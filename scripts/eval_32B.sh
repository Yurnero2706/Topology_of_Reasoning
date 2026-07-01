#!/bin/bash
# =============================================================================
# eval_32B.sh — Inference for a trained Qwen2.5-32B checkpoint  (Pegasus / NQSV)
# =============================================================================
# Cluster (NQSV) version of eval_14B.sh. Runs vLLM inference for ONE checkpoint
# on ONE H100. A 32B model in bf16 (~64 GB) fits on a single 80 GB H100, so this
# is a single-node job — NOT 16 nodes (vLLM here loads one model on one GPU and
# does not split work across nodes).
#
# IMPORTANT — the checkpoint must be CONSOLIDATED first.
#   sft_32B.sh saves SHARDED FSDP checkpoints (a pytorch_model_fsdp_0/ dir of
#   .distcp files). vLLM cannot read those. Merge each checkpoint once with:
#       qsub -v CKPT=ckpts/s1-v1.0/checkpoint-200 scripts/consolidate_ckpt.sh
#   (produces model-*.safetensors + index.json + config.json + tokenizer).
#
# Usage
# -----
#   qsub -v MODEL=ckpts/s1-v1.0/checkpoint-200,DATASET=aime scripts/eval_32B.sh
#   qsub -v MODEL=ckpts/s1-v1.0/checkpoint-400,DATASET=aime scripts/eval_32B.sh
#   qsub -v MODEL=ckpts/s1-v1.1/checkpoint-200,DATASET=aime scripts/eval_32B.sh
#
# Output CSV:
#   eval_result/<model_safe>/<dataset>/<dataset_safe>_test_cal=False_output.csv
# =============================================================================
#PBS -A UTSUROLB
#PBS -b 1
#PBS -q gpu
#PBS -l elapstim_req=04:00:00
#PBS -N eval_32B
# Single-node, single-GPU job: no "-T openmpi" / mpirun needed (one model on
# one H100). PyTorch/vLLM ship their own CUDA runtime, so no cuda module either.

set -euo pipefail

WORK_DIR=/work/UTSUROLB/utlb_ngy/work/Topology_of_Reasoning
# Eval uses the SEPARATE vLLM venv (torch 2.5.1). It must NOT share the training
# .venv (torch 2.1.1) — vLLM needs torch>=2.4 and would break torchrun/FSDP there.
VENV_PREFIX="${VENV_PREFIX:-/work/UTSUROLB/utlb_ngy/work/.venv-eval}"
source ${VENV_PREFIX}/bin/activate
cd "${WORK_DIR}"

# ---------------------------------------------------------------------------
# Offline HuggingFace cache (compute nodes have no internet)
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/work/UTSUROLB/utlb_ngy/work}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------------------
# Configuration  (override via:  qsub -v MODEL=...,DATASET=... )
# ---------------------------------------------------------------------------
MODEL="${MODEL:-ckpts/s1-v1.0/checkpoint-200}"
DATASET="${DATASET:-aime}"        # aime | math_500 | gsm8k | simplescaling/s1K
FLASH=True
MAX_LENGTH="${MAX_LENGTH:-8192}"
BATCH_SIZE=1
SEED=100
NUM_TEST="${NUM_TEST:-1000}"

# ---------------------------------------------------------------------------
# Derive output path (mirrors eval.py's own path logic)
# ---------------------------------------------------------------------------
MODEL_SAFE="${MODEL//\//_}"
OUTPUT_DIR="eval_result/${MODEL_SAFE}/${DATASET}"
SAFE_DATASET="${DATASET//\//_}"
CSV_PATH="${OUTPUT_DIR}/${SAFE_DATASET}_test_cal=False_output.csv"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Live log on shared /work (NQSV only returns its own stdout at job END)
# ---------------------------------------------------------------------------
mkdir -p "${WORK_DIR}/logs"
LIVE_LOG="${WORK_DIR}/logs/eval_${MODEL_SAFE}_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

{
echo ""
echo "======================================================"
echo "  Eval (32B, single H100): ${MODEL}"
echo "  Dataset    : ${DATASET}"
echo "  Max length : ${MAX_LENGTH}"
echo "  Num test   : ${NUM_TEST}"
echo "  Output CSV : ${CSV_PATH}"
echo "  Live log   : ${LIVE_LOG}"
echo "======================================================"
echo ""

python src/eval.py \
    --base_model_name_or_path "${MODEL}" \
    --model_name_or_path      "${MODEL}" \
    --parameter_efficient_mode none \
    --dataset                 "${DATASET}" \
    --batch_size              "${BATCH_SIZE}" \
    --max_length              "${MAX_LENGTH}" \
    --seed                    "${SEED}" \
    --load_in_8bit            True \
    --flash_attention         "${FLASH}" \
    --num_test                "${NUM_TEST}" \
    --output_dir              "${OUTPUT_DIR}"

echo ""
echo "======================================================"
echo "  Done.  Generated CSV:"
echo "    ${CSV_PATH}"
echo "======================================================"
} 2>&1 | tee "${LIVE_LOG}"

# Watch live from a login node with:
#   tail -f $(ls -t /work/UTSUROLB/utlb_ngy/work/Topology_of_Reasoning/logs/eval_*.log | head -1)

#!/usr/bin/env bash
# =============================================================================
# eval_14B.sh — Inference for a trained Qwen2.5-14B checkpoint
# =============================================================================
# Generates model outputs on an evaluation dataset and writes a CSV that
# cluster_figure9.sh will later feed into cluster_steps_generated.py.
#
# Parameters inherit from eval.sh (batch_size=1, max_length=8192, seed=100,
# load_in_8bit=True, flash_attention=True, num_test=1000).
#
# Usage
# -----
#   # Evaluate a specific checkpoint
#   MODEL=ckpts/s1-v1.0/checkpoint-200  DATASET=aime  bash scripts/eval_14B.sh
#   MODEL=ckpts/s1-v1.0/checkpoint-400  DATASET=aime  bash scripts/eval_14B.sh
#   MODEL=ckpts/s1-v1.1/checkpoint-200  DATASET=aime  bash scripts/eval_14B.sh
#   MODEL=ckpts/s1-v1.1/checkpoint-400  DATASET=aime  bash scripts/eval_14B.sh
#
#   # Use a different eval set
#   MODEL=ckpts/s1-v1.0/checkpoint-200  DATASET=math_500  bash scripts/eval_14B.sh
#
# Output CSV location:
#   eval_result/<model_safe>/<dataset>/<dataset_safe>_test_cal=False_output.csv
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 1.  Configuration
# ---------------------------------------------------------------------------
MODEL="${MODEL:-ckpts/s1-v1.0}"
DATASET="${DATASET:-aime}"    # aime | math_500 | gsm8k | simplescaling/s1K

# Match eval.sh defaults exactly
FLASH=True
MAX_LENGTH=8192
BATCH_SIZE=1
SEED=100
NUM_TEST=1000

export HF_TOKEN="${HF_TOKEN:-}"

# ---------------------------------------------------------------------------
# 2.  Derive output path  (must mirror eval.py's own path logic)
# ---------------------------------------------------------------------------
# Replace '/' with '_' so the model path becomes a single directory component.
MODEL_SAFE="${MODEL//\//_}"
OUTPUT_DIR="eval_result/${MODEL_SAFE}/${DATASET}"

SAFE_DATASET="${DATASET//\//_}"
CSV_PATH="${OUTPUT_DIR}/${SAFE_DATASET}_test_cal=False_output.csv"

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "======================================================"
echo "  Eval: ${MODEL}"
echo "  Dataset    : ${DATASET}"
echo "  Max length : ${MAX_LENGTH}"
echo "  Num test   : ${NUM_TEST}"
echo "  Output CSV : ${CSV_PATH}"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# 3.  Run inference
# ---------------------------------------------------------------------------
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
echo ""
echo "  Next: run scripts/cluster_figure9.sh (after all 4 checkpoints are evaluated)"
echo "======================================================"

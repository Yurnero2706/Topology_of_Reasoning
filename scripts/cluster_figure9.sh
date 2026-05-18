#!/usr/bin/env bash
# =============================================================================
# cluster_figure9.sh — Full pipeline to reproduce Figure 9
# =============================================================================
# Runs cluster_steps_generated.py for every (checkpoint × layer-ratio)
# combination needed for Figure 9, then calls plot_figure9.py to render the
# two-panel diameter-distribution figure.
#
# Figure 9 structure
# ------------------
#   Panel (a): diameter at 200 training steps — s1-v1.0 vs s1-v1.1
#   Panel (b): diameter at 400 training steps — s1-v1.0 vs s1-v1.1
#
# Prerequisites (run in order)
# ----------------------------
#   1. Train s1-v1.0:
#        DATASET=s1K     bash scripts/sft_14B.sh
#   2. Train s1-v1.1:
#        DATASET=s1K-1.1 bash scripts/sft_14B.sh
#   3. Evaluate all 4 checkpoints:
#        MODEL=ckpts/s1-v1.0/checkpoint-200 DATASET=aime bash scripts/eval_14B.sh
#        MODEL=ckpts/s1-v1.0/checkpoint-400 DATASET=aime bash scripts/eval_14B.sh
#        MODEL=ckpts/s1-v1.1/checkpoint-200 DATASET=aime bash scripts/eval_14B.sh
#        MODEL=ckpts/s1-v1.1/checkpoint-400 DATASET=aime bash scripts/eval_14B.sh
#   4. Run this script:
#        bash scripts/cluster_figure9.sh
#
# Overrides (env vars)
# --------------------
#   EVAL_DATASET      eval dataset used in step 3 (default: aime)
#   NUM_TYPES         k-means k                   (default: 200)
#   BATCH_SIZE        embedding batch size         (default: 8)
#   MAX_LENGTH        token budget                 (default: 8192)
#   RESULTS_DIR       root output directory        (default: results_figure9)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 1.  Configuration
# ---------------------------------------------------------------------------
EVAL_DATASET="${EVAL_DATASET:-aime}"      # must match what was used in eval_14B.sh
NUM_TYPES="${NUM_TYPES:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
RESULTS_DIR="${RESULTS_DIR:-results_figure9}"
TARGET_LAYER_RATIOS="0.1 0.3 0.5 0.7 0.9"

# Checkpoint paths produced by sft_14B.sh
V10_STEP200="${V10_STEP200:-ckpts/s1-v1.0/checkpoint-200}"
V10_STEP400="${V10_STEP400:-ckpts/s1-v1.0/checkpoint-400}"
V11_STEP200="${V11_STEP200:-ckpts/s1-v1.1/checkpoint-200}"
V11_STEP400="${V11_STEP400:-ckpts/s1-v1.1/checkpoint-400}"

SAFE_DATASET="${EVAL_DATASET//\//_}"

export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM=false

echo ""
echo "============================================================"
echo "  Figure 9 — Clustering Pipeline"
echo "  Eval dataset : ${EVAL_DATASET}"
echo "  Layer ratios : ${TARGET_LAYER_RATIOS}"
echo "  K-means k    : ${NUM_TYPES}"
echo "  Batch size   : ${BATCH_SIZE}"
echo "  Max length   : ${MAX_LENGTH}"
echo "  Results dir  : ${RESULTS_DIR}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 2.  Helper: cluster one checkpoint across all layer ratios
# ---------------------------------------------------------------------------
# cluster_steps_generated.py puts results at:
#   {RESULTS_DIR}/{model_name_or_path}/{dataset_name}/
#     target_layer_ratio={ratio}/k-means-k={num_types}/results.json
#
# The CSV path must match eval_14B.sh's output convention:
#   eval_result/{MODEL_SAFE}/{EVAL_DATASET}/{SAFE_DATASET}_test_cal=False_output.csv
# ---------------------------------------------------------------------------
run_cluster() {
    local ckpt="$1"
    local ckpt_safe="${ckpt//\//_}"
    local csv_path="eval_result/${ckpt_safe}/${EVAL_DATASET}/${SAFE_DATASET}_test_cal=False_output.csv"

    if [[ ! -f "${csv_path}" ]]; then
        echo "[ERROR] CSV not found: ${csv_path}"
        echo "        Please run first:"
        echo "          MODEL=${ckpt} DATASET=${EVAL_DATASET} bash scripts/eval_14B.sh"
        exit 1
    fi

    echo "── Checkpoint: ${ckpt} ──"
    for ratio in ${TARGET_LAYER_RATIOS}; do
        echo "   layer ratio = ${ratio}"
        python src/cluster_steps_generated.py \
            --model_name_or_path      "${ckpt}" \
            --tokenizer_name_or_path  "${ckpt}" \
            --batch_size              "${BATCH_SIZE}" \
            --dataset                 "${EVAL_DATASET}" \
            --num_types               "${NUM_TYPES}" \
            --df_path                 "${csv_path}" \
            --target_layer_ratio      "${ratio}" \
            --output_dir              "${RESULTS_DIR}" \
            --model_max_length        "${MAX_LENGTH}"
    done
    echo "   done: ${ckpt}"
    echo ""
}

# ---------------------------------------------------------------------------
# 3.  Run clustering for all 4 checkpoints
# ---------------------------------------------------------------------------
run_cluster "${V10_STEP200}"
run_cluster "${V10_STEP400}"
run_cluster "${V11_STEP200}"
run_cluster "${V11_STEP400}"

# ---------------------------------------------------------------------------
# 4.  Generate Figure 9
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  All clustering complete. Generating Figure 9 …"
echo "============================================================"
echo ""

python src/plot_figure9.py \
    --results_dir        "${RESULTS_DIR}" \
    --v10_ckpt_200       "${V10_STEP200}" \
    --v10_ckpt_400       "${V10_STEP400}" \
    --v11_ckpt_200       "${V11_STEP200}" \
    --v11_ckpt_400       "${V11_STEP400}" \
    --dataset            "${EVAL_DATASET}" \
    --num_types          "${NUM_TYPES}" \
    --target_layer_ratios ${TARGET_LAYER_RATIOS} \
    --output_path        "figure9.pdf"

echo ""
echo "============================================================"
echo "  Figure 9 saved → figure9.pdf"
echo "  JSON summaries in ${RESULTS_DIR}/"
echo "============================================================"

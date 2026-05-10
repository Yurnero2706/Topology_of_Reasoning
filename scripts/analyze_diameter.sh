#!/usr/bin/env bash
# =============================================================================
# analyze_diameter.sh
# =============================================================================
# Reasoning-graph diameter analysis — Section 5 / Appendix J of
#   "Topology of Reasoning" (arXiv:2506.05744)
#
# This script is the inference-side companion to sft.sh.
# Training hyper-parameters (model, block_size, etc.) are kept identical to
# sft.sh so that the SFT model and the analysis share a consistent setup.
#
# Workflow
# --------
#   Step 0: (optional) train models with sft.sh first
#   Step 1: run this script to compare base vs fine-tuned model(s)
#
# Usage
# -----
#   # Full comparison — base vs s1-v1.0 vs s1-v1.1 SFT checkpoints
#   bash analyze_diameter.sh
#
#   # Override paths via env vars:
#   CKPT_S1_V10=/my/ckpt1  CKPT_S1_V11=/my/ckpt2  bash analyze_diameter.sh
#
#   # Quick smoke-test (5 samples, ratio 0.9 only)
#   SMOKE_TEST=1 bash analyze_diameter.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0.  Shared hyper-parameters — kept in sync with sft.sh
# ---------------------------------------------------------------------------
BASE_MODEL="Qwen/Qwen2.5-14B"  # Table 3: Base Model = Qwen2.5-14B
BLOCK_SIZE=32768                         # Table 3: Block Size = 32768 tokens
BATCH_SIZE=8                             # Table 3: Batch Size = 8 (8 GPUs × micro-batch 1)
TORCH_DTYPE="bfloat16"                   # Table 3: Precision = bf16
NUM_TYPES=200                            # paper default: K-means k=200

# Which training dataset was used for each checkpoint.
# Matches sft.sh: --train_file_path="simplescaling/s1K"
DATASET_S1K="simplescaling/s1K-1.1"

# ---------------------------------------------------------------------------
# 1.  Checkpoint paths — override via env var or edit here after training
# ---------------------------------------------------------------------------
# These should be the --output_dir values from sft.sh runs.
# If a checkpoint doesn't exist yet the corresponding label/path is omitted
# automatically (see MODEL_PATHS / MODEL_LABELS construction below).
CKPT_S1_V10="${CKPT_S1_V10:-ckpts/s1-v1.0}"   # trained on s1K  (v1.0 data)
CKPT_S1_V11="${CKPT_S1_V11:-ckpts/s1-v1.1}"   # trained on s1K-v1.1 (v1.1 data)

# ---------------------------------------------------------------------------
# 2.  Analysis settings
# ---------------------------------------------------------------------------
TARGET_LAYER_RATIOS="0.1 0.3 0.5 0.7 0.9"     # paper: 5 depths
OUTPUT_DIR="results_diameter"
CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}"

# ---------------------------------------------------------------------------
# 3.  Smoke-test mode — fewer samples / ratios / clusters
# ---------------------------------------------------------------------------
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    echo "[SMOKE-TEST MODE]  max_samples=5, ratio=0.9, num_types=50, max_len=2048"
    MAX_SAMPLES="--max_samples 5"
    TARGET_LAYER_RATIOS="0.9"
    NUM_TYPES=50
    BLOCK_SIZE=2048           # reduced from 32768 just for the smoke-test
    OUTPUT_DIR="results_diameter_smoke"
else
    MAX_SAMPLES=""
fi

# ---------------------------------------------------------------------------
# 4.  Build model-path and label lists dynamically
#     (skip checkpoint entries whose directory doesn't exist yet)
# ---------------------------------------------------------------------------
MODEL_PATHS="${BASE_MODEL}"
MODEL_LABELS="Base(Qwen2.5-14B)"

if [[ -d "${CKPT_S1_V10}" ]]; then
    MODEL_PATHS="${MODEL_PATHS} ${CKPT_S1_V10}"
    MODEL_LABELS="${MODEL_LABELS} s1-v1.0"
    echo "[INFO]  Found s1-v1.0 checkpoint: ${CKPT_S1_V10}"
else
    echo "[WARN]  s1-v1.0 checkpoint not found at '${CKPT_S1_V10}'."
    echo "        Train it first with sft.sh  (output: ${CKPT_S1_V10})"
    echo "        or set CKPT_S1_V10=/path/to/ckpt"
fi

if [[ -d "${CKPT_S1_V11}" ]]; then
    MODEL_PATHS="${MODEL_PATHS} ${CKPT_S1_V11}"
    MODEL_LABELS="${MODEL_LABELS} s1-v1.1"
    echo "[INFO]  Found s1-v1.1 checkpoint: ${CKPT_S1_V11}"
else
    echo "[WARN]  s1-v1.1 checkpoint not found at '${CKPT_S1_V11}'."
    echo "        Train it first with sft.sh on the v1.1 dataset"
    echo "        or set CKPT_S1_V11=/path/to/ckpt"
fi

# ---------------------------------------------------------------------------
# 5.  Environment
# ---------------------------------------------------------------------------
export HF_TOKEN="${HF_TOKEN:-}"          # set in env if private datasets needed
export TOKENIZERS_PARALLELISM=false      # suppress HF tokenizer fork warning
export PYTHONPATH="${PYTHONPATH:-.}"     # ensure utils.py is importable

echo ""
echo "============================================================"
echo "  Reasoning-Graph Diameter Analysis"
echo "  Model(s) : ${MODEL_LABELS}"
echo "  Dataset  : ${DATASET_S1K}"
echo "  Layers   : ${TARGET_LAYER_RATIOS}"
echo "  k-means k: ${NUM_TYPES}"
echo "  Max-len  : ${BLOCK_SIZE}"
echo "  Out dir  : ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 6.  Run the analysis
# ---------------------------------------------------------------------------
python src/analyze_diameter.py \
    --model_paths  ${MODEL_PATHS} \
    --model_labels ${MODEL_LABELS} \
    --dataset      "${DATASET_S1K}" \
    --dataset_split "train" \
    --target_layer_ratios ${TARGET_LAYER_RATIOS} \
    --num_types    "${NUM_TYPES}" \
    --model_max_length "${BLOCK_SIZE}" \
    --batch_size   "${BATCH_SIZE}" \
    --torch_dtype  "${TORCH_DTYPE}" \
    --output_dir   "${OUTPUT_DIR}" \
    --cache_dir    "${CACHE_DIR}" \
    ${MAX_SAMPLES}

echo ""
echo "============================================================"
echo "  Done.  Plots and JSON in: ${OUTPUT_DIR}/"
echo ""
echo "  Key output files:"
echo "    ${OUTPUT_DIR}/summary.json"
echo "    ${OUTPUT_DIR}/diameter_comparison.png"
echo "    ${OUTPUT_DIR}/cycle_detection_ratio.png"
echo "    ${OUTPUT_DIR}/small_world_index_comparison.png"
echo "    ${OUTPUT_DIR}/loop_count_comparison.png"
echo "============================================================"


# ---------------------------------------------------------------------------
# 7.  (Optional) how to train the checkpoints this script needs
# ---------------------------------------------------------------------------
# To produce the checkpoints analysed above, run sft.sh with appropriate
# --train_file_path and --output_dir.  Example:
#
#   # s1-v1.0  (original s1K dataset)
#   CKPT_DIR="ckpts/s1-v1.0"
#   bash sft.sh  # uses simplescaling/s1K by default; edit --output_dir in sft.sh
#
#   # s1-v1.1  (improved dataset — change --train_file_path in sft.sh to point
#   #            to the v1.1 dataset, e.g. "simplescaling/s1K-v1.1")
#   CKPT_DIR="ckpts/s1-v1.1"
#   bash sft.sh
#
# Then re-run:
#   CKPT_S1_V10=ckpts/s1-v1.0  CKPT_S1_V11=ckpts/s1-v1.1  bash analyze_diameter.sh

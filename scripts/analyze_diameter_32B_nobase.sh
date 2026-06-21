#!/bin/bash
# =============================================================================
# analyze_diameter_32B_nobase.sh — diameter analysis, TRAINED MODELS ONLY
# =============================================================================
# Same as analyze_diameter_32B.sh but WITHOUT the base model — it compares only
# the two fine-tuned checkpoints (s1-v1.0 vs s1-v1.1). Use this when you don't
# need the Base(Qwen2.5-32B-Instruct) reference curve (e.g. it's slow to load,
# or you only care about the v1.0-vs-v1.1 contrast).
#
# Checkpoints must be CONSOLIDATED first (scripts/consolidate_ckpt.sh).
#
# Usage
# -----
#   # default: both checkpoint-200 dirs -> results_diameter/.../steps-200/
#   qsub scripts/analyze_diameter_32B_nobase.sh
#
#   # checkpoint-400 (override paths AND output dir so steps-200 isn't clobbered)
#   qsub -v CKPT_S1_V10=ckpts/s1-v1.0/checkpoint-400,CKPT_S1_V11=ckpts/s1-v1.1/checkpoint-400,OUTPUT_DIR=results_diameter/simplescaling/s1K/steps-400/ \
#        scripts/analyze_diameter_32B_nobase.sh
# =============================================================================
#PBS -A UTSUROLB
#PBS -b 1
#PBS -q gpu
#PBS -l elapstim_req=06:00:00
#PBS -N analyze_32B_nb
# Single-node, single-GPU job: no "-T openmpi" / mpirun needed.

set -euo pipefail

WORK_DIR=/work/UTSUROLB/utlb_ngy/work/Topology_of_Reasoning
VENV_PREFIX=/work/UTSUROLB/utlb_ngy/work/.venv
source ${VENV_PREFIX}/bin/activate
cd "${WORK_DIR}"

# ---------------------------------------------------------------------------
# Offline HuggingFace cache
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/work/UTSUROLB/utlb_ngy/work}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-.}"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BLOCK_SIZE="${BLOCK_SIZE:-32768}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TORCH_DTYPE="bfloat16"
NUM_TYPES="${NUM_TYPES:-200}"             # K-means k
DATASET_S1K="simplescaling/s1K"

CKPT_S1_V10="${CKPT_S1_V10:-ckpts/s1-v1.0/checkpoint-200}"
CKPT_S1_V11="${CKPT_S1_V11:-ckpts/s1-v1.1/checkpoint-200}"

TARGET_LAYER_RATIOS="${TARGET_LAYER_RATIOS:-0.1 0.3 0.5 0.7 0.9}"
OUTPUT_DIR="${OUTPUT_DIR:-results_diameter/${DATASET_S1K}/steps-${NUM_TYPES}/}"
# (We do NOT pass --cache_dir; HF_HOME resolves the offline cache via its
#  standard sub-dirs — see analyze_diameter_32B.sh for the full explanation.)

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    echo "[SMOKE-TEST MODE]  max_samples=5, ratio=0.9, num_types=50"
    MAX_SAMPLES="--max_samples 5"
    TARGET_LAYER_RATIOS="0.9"
    NUM_TYPES=50
    OUTPUT_DIR="results_diameter_smoke"
else
    MAX_SAMPLES=""
fi

# ---------------------------------------------------------------------------
# Build model-path / label lists — TRAINED MODELS ONLY (no base model)
# ---------------------------------------------------------------------------
MODEL_PATHS=""
MODEL_LABELS=""

if [[ -d "${CKPT_S1_V10}" ]]; then
    MODEL_PATHS="${MODEL_PATHS} ${CKPT_S1_V10}"
    MODEL_LABELS="${MODEL_LABELS} s1-v1.0"
    echo "[INFO]  Found s1-v1.0 checkpoint: ${CKPT_S1_V10}"
else
    echo "[WARN]  s1-v1.0 checkpoint not found at '${CKPT_S1_V10}' — skipping."
fi

if [[ -d "${CKPT_S1_V11}" ]]; then
    MODEL_PATHS="${MODEL_PATHS} ${CKPT_S1_V11}"
    MODEL_LABELS="${MODEL_LABELS} s1-v1.1"
    echo "[INFO]  Found s1-v1.1 checkpoint: ${CKPT_S1_V11}"
else
    echo "[WARN]  s1-v1.1 checkpoint not found at '${CKPT_S1_V11}' — skipping."
fi

# Trim leading whitespace and bail out if nothing to analyze.
MODEL_PATHS="$(echo "${MODEL_PATHS}" | sed 's/^ *//')"
MODEL_LABELS="$(echo "${MODEL_LABELS}" | sed 's/^ *//')"
if [[ -z "${MODEL_PATHS}" ]]; then
    echo "ERROR: no consolidated checkpoints found. Run scripts/consolidate_ckpt.sh first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Live log + run
# ---------------------------------------------------------------------------
mkdir -p "${WORK_DIR}/logs"
LIVE_LOG="${WORK_DIR}/logs/analyze_nobase_$(date +%Y%m%d_%H%M%S).log"

{
echo "============================================================"
echo "  Reasoning-Graph Diameter Analysis (32B, no base model)"
echo "  Model(s) : ${MODEL_LABELS}"
echo "  Dataset  : ${DATASET_S1K}"
echo "  Layers   : ${TARGET_LAYER_RATIOS}"
echo "  k-means k: ${NUM_TYPES}"
echo "  Out dir  : ${OUTPUT_DIR}"
echo "  Live log : ${LIVE_LOG}"
echo "============================================================"

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
    ${MAX_SAMPLES}

echo "Done. Violin plots + JSON in: ${OUTPUT_DIR}/"
} 2>&1 | tee "${LIVE_LOG}"

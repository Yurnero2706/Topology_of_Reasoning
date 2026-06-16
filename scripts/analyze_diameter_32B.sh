#!/bin/bash
# =============================================================================
# analyze_diameter_32B.sh — Reasoning-graph diameter analysis  (Pegasus / NQSV)
# =============================================================================
# Cluster (NQSV) version of analyze_diameter.sh for the Qwen2.5-32B run.
# Loads ONE model at a time on ONE H100 and extracts hidden states / clusters
# reasoning steps. A 32B model fits on a single 80 GB H100, so this is a
# single-node job — NOT 16 nodes (the script processes the dataset locally and
# does not split work across nodes).
#
# It produces the VIOLIN figures (Figure 9 style): diameter_comparison.png,
# small_world_index_comparison.png, loop_count_comparison.png are violin plots;
# cycle_detection_ratio.png is a line chart.
#
# IMPORTANT — checkpoints must be CONSOLIDATED first (see eval_32B.sh header):
#   accelerate merge-weights <ckpt>/pytorch_model_fsdp_0 <ckpt>
#
# Usage
# -----
#   # Analyse the step-200 checkpoints (base vs s1-v1.0 vs s1-v1.1):
#   qsub -v CKPT_S1_V10=ckpts/s1-v1.0/checkpoint-200,CKPT_S1_V11=ckpts/s1-v1.1/checkpoint-200,OUTPUT_DIR=results_diameter/simplescaling/s1K/steps-200/ \
#        scripts/analyze_diameter_32B.sh
#
#   # Smoke-test (5 samples, ratio 0.9 only):
#   qsub -v SMOKE_TEST=1 scripts/analyze_diameter_32B.sh
# =============================================================================
#PBS -A UTSUROLB
#PBS -b 1
#PBS -q gpu
#PBS -l elapstim_req=06:00:00
#PBS -N analyze_32B
# Single-node, single-GPU job: no "-T openmpi" / mpirun needed (one model on
# one H100). PyTorch ships its own CUDA runtime, so no cuda module either.

set -euo pipefail

WORK_DIR=/work/UTSUROLB/utlb_ngy/work/Topology_of_Reasoning
VENV_PREFIX=/work/UTSUROLB/utlb_ngy/work/.venv
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
export PYTHONPATH="${PYTHONPATH:-.}"

# ---------------------------------------------------------------------------
# 0.  Hyper-parameters — kept in sync with sft_32B.sh
# ---------------------------------------------------------------------------
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
BLOCK_SIZE="${BLOCK_SIZE:-32768}"
# 32B on one 80 GB H100: keep the forward-pass batch small to avoid OOM during
# hidden-state extraction at 32k context. Raise only if memory allows.
BATCH_SIZE="${BATCH_SIZE:-1}"
TORCH_DTYPE="bfloat16"
NUM_TYPES="${NUM_TYPES:-200}"             # K-means k
DATASET_S1K="simplescaling/s1K"

# ---------------------------------------------------------------------------
# 1.  Checkpoint paths — point these at CONSOLIDATED checkpoints
# ---------------------------------------------------------------------------
CKPT_S1_V10="${CKPT_S1_V10:-ckpts/s1-v1.0/checkpoint-200}"
CKPT_S1_V11="${CKPT_S1_V11:-ckpts/s1-v1.1/checkpoint-200}"

# ---------------------------------------------------------------------------
# 2.  Analysis settings
# ---------------------------------------------------------------------------
TARGET_LAYER_RATIOS="${TARGET_LAYER_RATIOS:-0.1 0.3 0.5 0.7 0.9}"
OUTPUT_DIR="${OUTPUT_DIR:-results_diameter/${DATASET_S1K}/steps-${NUM_TYPES}/}"
CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}"

# ---------------------------------------------------------------------------
# 3.  Smoke-test mode
# ---------------------------------------------------------------------------
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
# 4.  Build model-path / label lists dynamically (skip missing checkpoints)
# ---------------------------------------------------------------------------
MODEL_PATHS="${BASE_MODEL}"
MODEL_LABELS="Base(${BASE_MODEL})"

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

# ---------------------------------------------------------------------------
# 5.  Live log on shared /work (NQSV only returns its own stdout at job END)
# ---------------------------------------------------------------------------
mkdir -p "${WORK_DIR}/logs"
LIVE_LOG="${WORK_DIR}/logs/analyze_$(date +%Y%m%d_%H%M%S).log"

{
echo ""
echo "============================================================"
echo "  Reasoning-Graph Diameter Analysis (32B, single H100)"
echo "  Model(s) : ${MODEL_LABELS}"
echo "  Dataset  : ${DATASET_S1K}"
echo "  Layers   : ${TARGET_LAYER_RATIOS}"
echo "  k-means k: ${NUM_TYPES}"
echo "  Max-len  : ${BLOCK_SIZE}"
echo "  Out dir  : ${OUTPUT_DIR}"
echo "  Live log : ${LIVE_LOG}"
echo "============================================================"
echo ""

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
echo "  Done.  Violin plots + JSON in: ${OUTPUT_DIR}/"
echo "    ${OUTPUT_DIR}/diameter_comparison.png          (violin)"
echo "    ${OUTPUT_DIR}/small_world_index_comparison.png (violin)"
echo "    ${OUTPUT_DIR}/loop_count_comparison.png        (violin)"
echo "    ${OUTPUT_DIR}/cycle_detection_ratio.png        (line chart)"
echo "    ${OUTPUT_DIR}/summary.json"
echo "============================================================"
} 2>&1 | tee "${LIVE_LOG}"

# Watch live from a login node with:
#   tail -f $(ls -t /work/UTSUROLB/utlb_ngy/work/Topology_of_Reasoning/logs/analyze_*.log | head -1)

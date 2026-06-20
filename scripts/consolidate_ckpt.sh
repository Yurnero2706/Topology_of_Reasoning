#!/bin/bash
#PBS -A UTSUROLB
#PBS -b 1
#PBS -q gpu
#PBS -l elapstim_req=00:30:00
#PBS -N consolidate
# =============================================================================
# consolidate_ckpt.sh — turn a SHARDED FSDP checkpoint into a loadable HF model
# =============================================================================
# sft_32B.sh saves checkpoints with FSDP SHARDED_STATE_DICT, so each
# checkpoint-N/ holds pytorch_model_fsdp_0/*.distcp shards — NOT a single
# weights file, and NO config.json. from_pretrained() (used by eval_32B.sh and
# analyze_diameter_32B.sh) cannot load that, hence:
#   "ValueError: Unrecognized model ... Should have a model_type key in config.json"
#
# This job fixes ONE checkpoint by:
#   1. merging the .distcp shards into sharded model-*.safetensors + index
#      (src/consolidate_fsdp.py — torch 2.1 compatible, CPU, offline, low-RAM:
#       streams ~5 GB at a time so it does NOT need 64 GB to merge a 32B model)
#   2. copying config.json + tokenizer from the base model (architecture is
#      unchanged by fine-tuning, so the base config is correct)
#
# Merging a 32B model holds the full ~64 GB in RAM, which OOM-kills a login
# node — that's why this runs as a 1-node gpu-queue job (115 GiB DRAM). No GPU
# math is used; the node is just for its memory.
#
# Submit (checkpoint dir comes from -v CKPT=..., NOT a positional arg, because
# NQSV does not pass argv to the job script):
#   qsub -v CKPT=ckpts/s1-v1.0/checkpoint-200 scripts/consolidate_ckpt.sh
#   qsub -v CKPT=ckpts/s1-v1.0/checkpoint-400 scripts/consolidate_ckpt.sh
#   qsub -v CKPT=ckpts/s1-v1.1/checkpoint-200 scripts/consolidate_ckpt.sh
#
# Still runnable directly on a big-RAM node:  bash scripts/consolidate_ckpt.sh <dir>
# =============================================================================
set -euo pipefail

# Checkpoint dir: prefer -v CKPT=..., fall back to positional $1 for direct runs.
CKPT="${CKPT:-${1:-}}"
if [[ -z "${CKPT}" ]]; then
    echo "ERROR: no checkpoint given. Use:  qsub -v CKPT=ckpts/s1-v1.0/checkpoint-200 scripts/consolidate_ckpt.sh"
    exit 1
fi
BASE_MODEL="${BASE_MODEL:-${2:-Qwen/Qwen2.5-32B-Instruct}}"

# Repo root on shared /work. HARDCODED (not derived from BASH_SOURCE) because
# NQSV runs a SPOOLED COPY of this script from /var/opt/nec/nqsv/jsv/jobfile/,
# so BASH_SOURCE would point there (read-only) instead of /work. cd in so a
# RELATIVE CKPT path (e.g. ckpts/...) resolves regardless of the job's cwd.
REPO_DIR="${REPO_DIR:-/work/UTSUROLB/utlb_ngy/work/Topology_of_Reasoning}"
cd "${REPO_DIR}"

# Training venv (torch 2.1.1). src/consolidate_fsdp.py does the merge with the
# torch-2.1 DCP API (accelerate.merge_fsdp_weights needs torch>=2.3).
VENV_PREFIX="${VENV_PREFIX:-/work/UTSUROLB/utlb_ngy/work/.venv}"
source ${VENV_PREFIX}/bin/activate
export HF_HOME="${HF_HOME:-/work/UTSUROLB/utlb_ngy/work}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "${REPO_DIR}/logs"
LIVE_LOG="${REPO_DIR}/logs/consolidate_$(date +%Y%m%d_%H%M%S).log"

{
echo "============================================================"
echo "  Consolidating: ${CKPT}"
echo "  Base model   : ${BASE_MODEL}"
echo "  Live log     : ${LIVE_LOG}"
echo "============================================================"

# 1. Merge the sharded distcp weights → ${CKPT}/model-*.safetensors + index.json
if [[ -d "${CKPT}/pytorch_model_fsdp_0" ]]; then
    echo "[1/2] Merging sharded weights in ${CKPT}/pytorch_model_fsdp_0 ..."
    python "${REPO_DIR}/src/consolidate_fsdp.py" "${CKPT}"
else
    echo "[1/2] No pytorch_model_fsdp_0/ in ${CKPT} — assuming already merged."
fi

# 2. Copy config + tokenizer from the base-model snapshot (unchanged architecture)
SNAP=$(ls -d ${HF_HOME}/hub/models--${BASE_MODEL//\//--}/snapshots/*/ 2>/dev/null | head -1 || true)
if [[ -z "${SNAP}" ]]; then
    echo "ERROR: base model snapshot not found under ${HF_HOME}/hub/models--${BASE_MODEL//\//--}/snapshots/"
    echo "       Pre-download it on a login node:  huggingface-cli download ${BASE_MODEL}"
    exit 1
fi
echo "[2/2] Copying config/tokenizer from ${SNAP}"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt special_tokens_map.json added_tokens.json; do
    [[ -f "${SNAP}${f}" ]] && cp -L "${SNAP}${f}" "${CKPT}/" && echo "   + ${f}"
done

echo ""
echo "Done. ${CKPT} is now loadable via from_pretrained:"
ls -la "${CKPT}"
} 2>&1 | tee "${LIVE_LOG}"

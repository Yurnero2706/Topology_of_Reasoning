#!/bin/bash
# =============================================================================
# consolidate_ckpt.sh — turn a SHARDED FSDP checkpoint into a loadable HF model
# =============================================================================
# sft_32B.sh saves checkpoints with FSDP SHARDED_STATE_DICT, so each
# checkpoint-N/ holds pytorch_model_fsdp_0/*.distcp shards — NOT a single
# weights file, and NO config.json. from_pretrained() (used by eval_32B.sh and
# analyze_diameter_32B.sh) cannot load that, hence:
#   "ValueError: Unrecognized model ... Should have a model_type key in config.json"
#
# This script fixes ONE checkpoint by:
#   1. merging the .distcp shards into a single model.safetensors
#   2. copying config.json + tokenizer from the base model (architecture is
#      unchanged by fine-tuning, so the base config is correct)
#
# Usage (run on a node with ~64 GB+ free RAM — login node is usually fine, or
# wrap in a small qsub job; this is CPU/RAM-bound, no GPU needed):
#   bash scripts/consolidate_ckpt.sh ckpts/s1-v1.0/checkpoint-200
#   bash scripts/consolidate_ckpt.sh ckpts/s1-v1.0/checkpoint-400
#   bash scripts/consolidate_ckpt.sh ckpts/s1-v1.1/checkpoint-200 Qwen/Qwen2.5-32B-Instruct
# =============================================================================
set -euo pipefail

CKPT="${1:?usage: consolidate_ckpt.sh <checkpoint_dir> [base_model]}"
BASE_MODEL="${2:-Qwen/Qwen2.5-32B-Instruct}"

# Resolve the repo root from this script's own location so src/ is found
# regardless of the current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

# Uses the TRAINING venv (torch 2.1.1). We do NOT use accelerate.merge_fsdp_weights
# (it requires torch>=2.3) — src/consolidate_fsdp.py does the merge with the
# torch-2.1 DCP API, fully offline.
VENV_PREFIX="${VENV_PREFIX:-/work/UTSUROLB/utlb_ngy/work/.venv}"
source ${VENV_PREFIX}/bin/activate
export HF_HOME="${HF_HOME:-/work/UTSUROLB/utlb_ngy/work}"

# 1. Merge the sharded distcp weights → ${CKPT}/pytorch_model.bin
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

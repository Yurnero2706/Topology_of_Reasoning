"""
consolidate_fsdp.py — merge a SHARDED FSDP (DCP) checkpoint into a single
weights file, WITHOUT requiring torch >= 2.3 (works on torch 2.1.1).

Why this exists
---------------
sft_32B.sh saves checkpoints with FSDP SHARDED_STATE_DICT, so each checkpoint
holds `pytorch_model_fsdp_0/*.distcp` shards instead of a single weights file.
`accelerate.merge_fsdp_weights` would merge them, but it hard-requires
torch>=2.3 (it calls torch.distributed.checkpoint.format_utils, added in 2.3).
The training venv has torch 2.1.1, so we replicate the same logic with the
lower-level DCP API that DOES exist in 2.1: load the sharded checkpoint into a
plain state_dict with no_dist=True + _EmptyStateDictLoadPlanner, then torch.save.

Reading the shards with the same torch that wrote them (2.1.1) also avoids any
cross-version DCP format mismatch. Runs on CPU, fully offline.

Usage
-----
    python src/consolidate_fsdp.py ckpts/s1-v1.0/checkpoint-200
        reads  ckpts/s1-v1.0/checkpoint-200/pytorch_model_fsdp_0/
        writes ckpts/s1-v1.0/checkpoint-200/pytorch_model.bin
"""
import argparse
import os
import sys

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge a sharded FSDP/DCP checkpoint (torch 2.1 compatible).")
    ap.add_argument("checkpoint_dir", help="e.g. ckpts/s1-v1.0/checkpoint-200")
    ap.add_argument("--shard_subdir", default="pytorch_model_fsdp_0",
                    help="sub-dir holding the .distcp shards (default: pytorch_model_fsdp_0)")
    ap.add_argument("--out_name", default="pytorch_model.bin",
                    help="output weights filename written into checkpoint_dir")
    args = ap.parse_args()

    shard_dir = os.path.join(args.checkpoint_dir, args.shard_subdir)
    if not os.path.isdir(shard_dir):
        sys.exit(f"ERROR: no sharded dir found at '{shard_dir}'")

    print(f"Loading DCP shards (no_dist, CPU) from: {shard_dir}")
    state_dict = {}
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(shard_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    # accelerate's save_fsdp_model stores {"model": <real weights>}. Depending on
    # how DCP reconstructs it on this torch version, that wrapper shows up either
    # as a nested dict under "model", or flattened as a "model." prefix on every
    # key. Handle both so the result matches the HF param FQNs (model.*, lm_head.*).
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        weights = state_dict["model"]
    elif state_dict and all(k.startswith("model.") for k in state_dict):
        # flattened wrapper: strip exactly ONE leading "model." (real Qwen keys
        # include "lm_head.weight", which would NOT survive if this were already
        # unwrapped — so an all-"model." prefix means the wrapper is present).
        weights = {k[len("model."):]: v for k, v in state_dict.items()}
    else:
        weights = state_dict

    sample = list(weights.keys())[:3]
    print(f"Recovered {len(weights)} tensors. Sample keys: {sample}")

    out_path = os.path.join(args.checkpoint_dir, args.out_name)
    print(f"Saving consolidated weights -> {out_path}")
    torch.save(weights, out_path)
    print("Done.")


if __name__ == "__main__":
    main()

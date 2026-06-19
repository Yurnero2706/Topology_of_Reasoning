"""
consolidate_fsdp.py — merge a SHARDED FSDP (DCP) checkpoint into a single
weights file, WITHOUT requiring torch >= 2.3 (works on torch 2.1.1).

Why this exists
---------------
sft_32B.sh saves checkpoints with FSDP SHARDED_STATE_DICT, so each checkpoint
holds `pytorch_model_fsdp_0/*.distcp` shards instead of a single weights file.
`accelerate.merge_fsdp_weights` would merge them but hard-requires torch>=2.3,
and torch 2.1's `_EmptyStateDictLoadPlanner` (which the >=2.3 path relies on)
does not exist. So we do it with the building blocks that DO exist in 2.1:

  1. Read the checkpoint's metadata to learn every tensor's key/shape/dtype.
  2. Allocate a matching empty CPU state_dict (keys therefore match exactly).
  3. dist_cp.load_state_dict(no_dist=True) fills those tensors in place,
     gathering the shards into full tensors on one process.
  4. torch.save the result.

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
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


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

    reader = FileSystemReader(shard_dir)
    metadata = reader.read_metadata()

    # Build an empty CPU state_dict straight from the checkpoint's metadata, so
    # keys/shapes/dtypes match the stored tensors exactly. These are FULL-size
    # tensors (metadata.size is the unsharded shape); DCP gathers shards into them.
    state_dict = {}
    skipped = []
    for key, smd in metadata.state_dict_metadata.items():
        if isinstance(smd, TensorStorageMetadata):
            state_dict[key] = torch.empty(tuple(smd.size), dtype=smd.properties.dtype)
        else:
            skipped.append(key)  # non-tensor (BytesStorageMetadata) — not part of weights
    if skipped:
        print(f"Note: skipping {len(skipped)} non-tensor entries (e.g. {skipped[:3]})")

    print(f"Loading {len(state_dict)} tensors from {shard_dir} (no_dist, CPU) ...")
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=reader,
        no_dist=True,
    )

    # accelerate's save_fsdp_model stores {"model": <weights>}; in the flattened
    # metadata that wrapper shows up as a leading "model." on EVERY key. Real Qwen
    # keys include "lm_head.weight" (no "model." prefix), so if every key starts
    # with "model." the wrapper is present and we strip exactly one level.
    if state_dict and all(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

    sample = list(state_dict.keys())[:3]
    print(f"Recovered {len(state_dict)} tensors. Sample keys: {sample}")

    out_path = os.path.join(args.checkpoint_dir, args.out_name)
    print(f"Saving consolidated weights -> {out_path}")
    torch.save(state_dict, out_path)
    print("Done.")


if __name__ == "__main__":
    main()

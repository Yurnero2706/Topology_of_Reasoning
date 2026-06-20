"""
consolidate_fsdp.py — merge a SHARDED FSDP (DCP) checkpoint into a loadable HF
checkpoint, on torch 2.1.1, in LOW MEMORY (streaming, ~one shard at a time).

Why this exists
---------------
sft_32B.sh saves checkpoints with FSDP SHARDED_STATE_DICT, so each checkpoint
holds `pytorch_model_fsdp_0/*.distcp` shards instead of a loadable weights file.
`accelerate.merge_fsdp_weights` requires torch>=2.3, and the naive "load the
whole model into one dict then save" approach holds the full ~64 GB (32B in
bf16) in RAM at once — which OOM-kills even a 115 GiB node once DCP's read
buffers are added on top.

This version never holds the whole model. It:
  1. reads the checkpoint metadata (every tensor's key/shape/dtype),
  2. bin-packs the tensors into ~MAX_SHARD_GB groups,
  3. for each group: loads ONLY those tensors (DCP loads the subset given in the
     state_dict), writes them to model-XXXXX-of-NNNNN.safetensors, frees them,
  4. writes model.safetensors.index.json mapping every weight to its shard.

from_pretrained() loads this sharded layout natively. Peak RAM ≈ one shard.
Runs on CPU, fully offline, reading the shards with the same torch (2.1.1) that
wrote them.

Usage
-----
    python src/consolidate_fsdp.py ckpts/s1-v1.0/checkpoint-200
        reads  ckpts/s1-v1.0/checkpoint-200/pytorch_model_fsdp_0/
        writes ckpts/s1-v1.0/checkpoint-200/model-*.safetensors + index.json
"""
import argparse
import json
import os
import sys

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from safetensors.torch import save_file


def _nbytes(smd: TensorStorageMetadata) -> int:
    n = 1
    for d in smd.size:
        n *= d
    return n * torch.empty(0, dtype=smd.properties.dtype).element_size()


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream-merge a sharded FSDP/DCP checkpoint (torch 2.1, low RAM).")
    ap.add_argument("checkpoint_dir", help="e.g. ckpts/s1-v1.0/checkpoint-200")
    ap.add_argument("--shard_subdir", default="pytorch_model_fsdp_0")
    ap.add_argument("--max_shard_gb", type=float, default=5.0,
                    help="approx size of each output safetensors shard (GiB)")
    args = ap.parse_args()

    shard_dir = os.path.join(args.checkpoint_dir, args.shard_subdir)
    if not os.path.isdir(shard_dir):
        sys.exit(f"ERROR: no sharded dir found at '{shard_dir}'")

    reader = FileSystemReader(shard_dir)
    metadata = reader.read_metadata()

    items = [(k, smd) for k, smd in metadata.state_dict_metadata.items()
             if isinstance(smd, TensorStorageMetadata)]
    if not items:
        sys.exit("ERROR: no tensors found in checkpoint metadata.")

    # accelerate's save_fsdp_model stores {"model": <weights>}; in the flattened
    # metadata that wrapper is a leading "model." on EVERY key. Real Qwen keys
    # include "lm_head.weight" (no prefix), so an all-"model." set means wrapped.
    wrapped = all(k.startswith("model.") for k, _ in items)
    clean = (lambda k: k[len("model."):]) if wrapped else (lambda k: k)
    print(f"Wrapper prefix detected: {wrapped}. {len(items)} tensors total.")

    # Bin-pack tensors into ~max_shard_gb groups.
    max_bytes = int(args.max_shard_gb * (1024 ** 3))
    groups, cur, cur_sz = [], [], 0
    for k, smd in items:
        b = _nbytes(smd)
        if cur and cur_sz + b > max_bytes:
            groups.append(cur)
            cur, cur_sz = [], 0
        cur.append(k)
        cur_sz += b
    if cur:
        groups.append(cur)
    n_shards = len(groups)
    print(f"Writing {n_shards} shard(s) of up to {args.max_shard_gb} GiB each.")

    weight_map, total_size = {}, 0
    for i, keys in enumerate(groups, 1):
        shard_name = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        # Load ONLY this group's tensors (DCP loads exactly the keys we provide).
        sd = {k: torch.empty(tuple(metadata.state_dict_metadata[k].size),
                             dtype=metadata.state_dict_metadata[k].properties.dtype)
              for k in keys}
        dist_cp.load_state_dict(state_dict=sd, storage_reader=reader, no_dist=True)

        out_sd = {clean(k): v.contiguous() for k, v in sd.items()}
        # metadata={"format": "pt"} is REQUIRED: transformers' safetensors loader
        # does metadata.get("format"), which crashes with AttributeError if the
        # file has no metadata header (NoneType).
        save_file(out_sd, os.path.join(args.checkpoint_dir, shard_name),
                  metadata={"format": "pt"})
        for ck, v in out_sd.items():
            weight_map[ck] = shard_name
            total_size += v.numel() * v.element_size()
        print(f"  [{i}/{n_shards}] wrote {shard_name}  ({len(out_sd)} tensors)")
        del sd, out_sd  # free before the next group

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(args.checkpoint_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    sample = list(weight_map.keys())[:3]
    print(f"Recovered {len(weight_map)} tensors. Sample keys: {sample}")
    print(f"Wrote model.safetensors.index.json (total_size={total_size} bytes). Done.")


if __name__ == "__main__":
    main()

"""
analyze_diameter.py
====================
Reproduce Section 5 / Appendix J of:
  "Topology of Reasoning: Understanding Large Reasoning Models through
   Reasoning Graph Properties"  (arXiv:2506.05744)

For each model checkpoint and each target-layer ratio we:
  1. Extract per-step hidden-state representations from dataset solutions
     (splitting sequences at newline tokens, identical to cluster_steps_generated.py).
  2. Pool all step embeddings across the whole dataset and fit K-means (k=num_types).
  3. Build a reasoning graph per sample: cluster-ID path + consecutive L2 distances.
  4. Compute graph properties via analyze_graph_v2() from utils.py
     (diameter, cycles, small-world index, …).
  5. Cache per-(model, layer_ratio) JSON results so reruns are free.
  6. Plot diameter (and other metric) distributions across layer ratios,
     one box+mean-line per model — exactly the style of Figure in Appendix J.

Training hyper-parameters are inherited from sft.sh:
  model  = Qwen/Qwen2.5-3B
  block_size (model_max_length) = 10 000

Typical usage — compare base model vs two SFT checkpoints
----------------------------------------------------------
python analyze_diameter.py \
    --model_paths  Qwen/Qwen2.5-3B  ckpts/s1-v1.0  ckpts/s1-v1.1 \
    --model_labels "Base"           "s1-v1.0"       "s1-v1.1" \
    --dataset simplescaling/s1K \
    --target_layer_ratios 0.1 0.3 0.5 0.7 0.9 \
    --num_types 200 \
    --model_max_length 10000 \
    --output_dir results_diameter

Quick smoke-test (5 samples, CPU):
python analyze_diameter.py \
    --model_paths Qwen/Qwen2.5-3B \
    --model_labels "Base" \
    --dataset simplescaling/s1K \
    --max_samples 5 \
    --target_layer_ratios 0.9 \
    --num_types 50 \
    --model_max_length 2048 \
    --output_dir results_diameter_test
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import load_dataset
from sklearn.cluster import KMeans
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#  Local import — utils.py must be on the Python path (same directory).       #
# --------------------------------------------------------------------------- #
from utils import analyze_graph_v2

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── prompt template — identical to cluster_steps_generated.py ─────────────── #
QUERY_TEMPLATE = (
    "Solve the following math problem efficiently and clearly.\n"
    "The last line of your response should be of the following format: "
    "'The answer is: ANSWER.' (without quotes) where ANSWER is just the "
    "final number or expression that solves the problem.\n"
    "{Question}"
)


# =========================================================================== #
#  1.  Embedding extraction helpers                                            #
# =========================================================================== #

def _newline_token_ids(tokenizer: transformers.PreTrainedTokenizer) -> torch.Tensor:
    """
    Return all vocabulary token ids whose decoded form ends with '\\n'.
    This replicates the split_ids logic in cluster_steps_generated.py.
    """
    vocab = tokenizer.get_vocab()
    ids = [
        idx
        for tok, idx in vocab.items()
        if tokenizer.decode([idx], clean_up_tokenization_spaces=False).endswith("\n")
    ]
    return torch.tensor(ids, dtype=torch.long)


def _step_reps_from_hidden(
    hidden: torch.Tensor,        # [seq_len, hidden_size]  — single sample
    step_mask: torch.Tensor,     # [seq_len]                — cumsum of newline positions
    question_lines: List[str],
    steps: List[str],
) -> Optional[np.ndarray]:
    """
    Average hidden states within each reasoning-step segment.

    step_mask assigns every token a monotonically increasing integer that
    increments whenever a newline token is encountered (identical to the
    cumsum trick in cluster_steps_generated.py).

    Returns shape [num_answer_steps, hidden_size]  or  None if too short.
    """
    num_groups = int(step_mask.max().item()) + 1
    # skip groups that belong to the question preamble
    start = min(len(question_lines), num_groups - 1)

    rep_sums:   List[np.ndarray] = []
    rep_counts: List[int]        = []

    for j in range(start, num_groups):
        tok_mask = (step_mask == j).float()
        rep_sum  = (hidden * tok_mask.unsqueeze(-1)).sum(0)
        n_toks   = tok_mask.sum()
        if n_toks <= 0:
            continue
        rep = (rep_sum / n_toks).cpu().float().numpy()
        if not np.isfinite(rep).all():
            continue

        idx = j - start
        tgt = idx if idx < len(steps) else len(steps) - 1   # clamp overflow

        while len(rep_sums) <= tgt:
            rep_sums.append(np.zeros_like(rep))
            rep_counts.append(0)

        rep_sums[tgt]   += rep
        rep_counts[tgt] += 1

    if not rep_sums:
        return None

    averaged = []
    for s, c in zip(rep_sums, rep_counts):
        if c > 0:
            averaged.append(s / c)

    if len(averaged) < 2:          # need at least two steps to form an edge
        return None

    arr = np.stack(averaged, axis=0).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    arr = np.clip(arr, -1e5, 1e5)
    return np.require(arr, dtype=np.float32, requirements=["C"])


def collect_step_embeddings(
    model:       transformers.PreTrainedModel,
    tokenizer:   transformers.PreTrainedTokenizer,
    df,                                          # pandas DataFrame
    target_layer: int,
    max_length:   int,
) -> Tuple[List[np.ndarray], List[List[str]]]:
    """
    Iterate once over *df* and return:
      all_reps  — list of float32 arrays [T_i, H], one per valid example
      all_texts — corresponding list of solution-step string lists
    """
    device   = next(model.parameters()).device
    nl_ids   = _newline_token_ids(tokenizer).to(device)

    all_reps:  List[np.ndarray] = []
    all_texts: List[List[str]]  = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="    extracting"):
        # ── read question ────────────────────────────────────────────────── #
        question = row.get("Question") or row.get("question") or ""

        # ── read solution text (handles s1K column names) ─────────────────- #
        solution = (
            row.get("generated_text")
            or row.get("text")
            or row.get("solution")
            or ""
        )
        steps = [s.strip() for s in str(solution).strip().split("\n")
                 if len(s.strip()) > 5]
        if len(steps) < 2:
            continue

        question_lines = question.strip().split("\n")
        # Feed question + all-but-last step (mirror cluster_steps_generated.py)
        prompt = QUERY_TEMPLATE.format(Question=question) + "\n".join(steps[:-1])

        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states: tuple of [1, seq_len, H],  index 0 = embedding layer
        hidden = outputs.hidden_states[target_layer][0]   # [seq_len, H]

        # build step-index mask (same cumsum trick as cluster_steps_generated.py)
        is_newline = torch.isin(inputs["input_ids"], nl_ids)   # [1, seq_len]
        step_mask  = (
            torch.cumsum(is_newline, dim=-1) * inputs["attention_mask"]
        )[0]                                                    # [seq_len]

        rep = _step_reps_from_hidden(hidden, step_mask, question_lines, steps)
        if rep is not None:
            all_reps.append(rep)
            all_texts.append(steps)

    return all_reps, all_texts


# =========================================================================== #
#  2.  Clustering + graph-metric computation                                   #
# =========================================================================== #

def compute_graph_metrics(
    all_reps: List[np.ndarray],
    num_types: int,
) -> List[Dict]:
    """
    1. Concatenate all step embeddings → fit a single global K-means model.
    2. For each example, predict cluster labels and compute consecutive
       L2 distances between raw embeddings.
    3. Call analyze_graph_v2() (from utils.py) and return all metric dicts.
    """
    pool = np.concatenate(all_reps, axis=0).astype(np.float32)
    logger.info(f"    K-means: {pool.shape[0]} steps → k={num_types} …")
    kmeans = KMeans(n_clusters=num_types, n_init=10, random_state=0).fit(pool)
    logger.info("    K-means done")

    metrics_list = []
    for rep in all_reps:
        clusters = kmeans.predict(rep.astype(np.float32))      # [T]
        distances = np.array([
            float(np.linalg.norm(rep[i] - rep[i + 1]))
            for i in range(len(rep) - 1)
        ])                                                      # [T-1]

        (
            has_loop,       loop_count,       diameter,
            avg_clustering, avg_path_length,  clustering_norm,
            path_length_norm, avg_hop_length, hop_length_norm,
            small_world_index,
        ) = analyze_graph_v2(clusters.tolist(), distances.tolist())

        metrics_list.append(dict(
            has_loop         = bool(has_loop),
            loop_count       = int(loop_count),
            diameter         = float(diameter),
            avg_clustering   = float(avg_clustering),
            avg_path_length  = float(avg_path_length),
            clustering_norm  = float(clustering_norm),
            path_length_norm = float(path_length_norm),
            avg_hop_length   = float(avg_hop_length),
            hop_length_norm  = float(hop_length_norm),
            small_world_index= float(small_world_index),
        ))

    return metrics_list


# =========================================================================== #
#  3.  Plotting                                                                #
# =========================================================================== #

METRICS_TO_PLOT = [
    ("diameter",          "Diameter"),
    ("loop_count",        "Cycle Count"),
    ("small_world_index", "Small-World Index"),
    ("avg_path_length",   "Avg Path Length"),
    ("avg_clustering",    "Avg Clustering Coefficient"),
]


def plot_metric_comparison(
    results_per_model: Dict[str, Dict[float, List[Dict]]],
    output_dir: Path,
    metric_key:   str  = "diameter",
    metric_label: str  = "Diameter",
) -> None:
    """
    Box-plot of <metric_key> distributions across layer ratios, with a
    mean-value line overlay — mirrors the style of Appendix J in the paper.
    """
    model_labels = list(results_per_model.keys())
    ratios       = sorted(next(iter(results_per_model.values())).keys())
    n_models     = len(model_labels)

    cmap    = plt.cm.tab10
    colors  = [cmap(i / max(n_models - 1, 1)) for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(10, 5))

    x_pos   = np.arange(len(ratios))
    width   = 0.75 / n_models
    offsets = np.linspace(
        -(n_models - 1) / 2,
         (n_models - 1) / 2,
         n_models,
    ) * width

    for m_idx, label in enumerate(model_labels):
        color = colors[m_idx]
        layer_data = results_per_model[label]
        # list-of-lists: one inner list per layer ratio
        values = [
            [r[metric_key] for r in layer_data.get(ratio, [])]
            for ratio in ratios
        ]
        means = [np.mean(v) if v else 0.0 for v in values]

        bp = ax.boxplot(
            values,
            positions   = x_pos + offsets[m_idx],
            widths      = width * 0.85,
            patch_artist= True,
            boxprops    = dict(facecolor=(*color[:3], 0.25), color=color, linewidth=1.2),
            medianprops = dict(color=color, linewidth=2),
            whiskerprops= dict(color=color, linewidth=1.2),
            capprops    = dict(color=color, linewidth=1.2),
            flierprops  = dict(
                marker="o", markerfacecolor=color,
                markeredgecolor=color, alpha=0.3, markersize=3,
            ),
            manage_ticks= False,
        )
        ax.plot(
            x_pos + offsets[m_idx], means,
            "o-", color=color, label=label,
            linewidth=2, markersize=6, zorder=5,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in ratios], fontsize=11)
    ax.set_xlabel("Target Layer Ratio", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(
        f"{metric_label} Distribution across Hidden Layers",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    out_path = output_dir / f"{metric_key}_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved plot → {out_path}")


def plot_cycle_detection_ratio(
    results_per_model: Dict[str, Dict[float, List[Dict]]],
    output_dir: Path,
) -> None:
    """
    Line chart of cycle-detection ratio per layer ratio per model.
    Mirrors Figure 4 style from the paper.
    """
    model_labels = list(results_per_model.keys())
    ratios       = sorted(next(iter(results_per_model.values())).keys())
    n_models     = len(model_labels)
    colors       = [plt.cm.tab10(i / max(n_models - 1, 1)) for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, color in zip(model_labels, colors):
        layer_data = results_per_model[label]
        cdr = [
            np.mean([r["has_loop"] for r in layer_data.get(ratio, [False])])
            for ratio in ratios
        ]
        ax.plot(
            [str(r) for r in ratios], cdr,
            "o-", color=color, label=label, linewidth=2, markersize=7,
        )
        for x, y in zip(ratios, cdr):
            ax.annotate(
                f"{y:.2f}",
                xy=(str(x), y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center", fontsize=9, color=color,
            )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Target Layer Ratio", fontsize=12)
    ax.set_ylabel("Cycle Detection Ratio", fontsize=12)
    ax.set_title("Cycle Detection Ratio across Hidden Layers", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    out_path = output_dir / "cycle_detection_ratio.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved plot → {out_path}")


# =========================================================================== #
#  4.  Main                                                                    #
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diameter / graph-property analysis for SFT checkpoints "
            "(Section 5 / Appendix J of arXiv:2506.05744)."
        )
    )
    # ── model arguments ──────────────────────────────────────────────────── #
    parser.add_argument(
        "--model_paths", nargs="+", required=True,
        help="HuggingFace model IDs or local checkpoint directories.",
    )
    parser.add_argument(
        "--model_labels", nargs="+", default=None,
        help="Display label for each model (defaults to its path).",
    )
    # ── data arguments ───────────────────────────────────────────────────── #
    parser.add_argument(
        "--dataset", type=str, default="simplescaling/s1K",
        help="HuggingFace dataset id or local path.",
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train",
        help="Dataset split (default: train).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit the number of samples (useful for smoke-tests).",
    )
    # ── analysis hyper-parameters ─────────────────────────────────────────- #
    parser.add_argument(
        "--target_layer_ratios", nargs="+", type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Relative hidden-layer depths to analyse (paper: 0.1–0.9).",
    )
    parser.add_argument(
        "--num_types", type=int, default=200,
        help="K-means clusters (paper default: 200).",
    )
    # matches sft.sh block_size=10000
    parser.add_argument(
        "--model_max_length", type=int, default=10000,
        help="Max token length — matches sft.sh block_size=10000.",
    )
    # ── infra ────────────────────────────────────────────────────────────── #
    parser.add_argument(
        "--output_dir", type=str, default="results_diameter",
        help="Root directory for JSON caches and plots.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="HuggingFace model/data cache directory.",
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Ignore any cached JSON results and recompute from scratch.",
    )
    args = parser.parse_args()

    # ── validate ─────────────────────────────────────────────────────────── #
    if args.model_labels is None:
        args.model_labels = list(args.model_paths)
    if len(args.model_labels) != len(args.model_paths):
        parser.error("--model_labels must have the same length as --model_paths")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── load dataset once ────────────────────────────────────────────────── #
    logger.info(f"Loading dataset: {args.dataset} / split={args.dataset_split}")
    ds = load_dataset(args.dataset, cache_dir=args.cache_dir)
    df = ds[args.dataset_split].to_pandas()
    if args.max_samples is not None:
        df = df.head(args.max_samples)
    logger.info(f"Dataset rows: {len(df)}")

    # ── main loop ──────────────────────────────────────────────────────────
    # results_per_model[label][ratio] = list[dict]
    results_per_model: Dict[str, Dict[float, List[Dict]]] = {}

    for model_path, model_label in zip(args.model_paths, args.model_labels):
        logger.info("\n" + "=" * 65)
        logger.info(f"Model : {model_label}")
        logger.info(f"  Path: {model_path}")

        # ── load model + tokenizer ───────────────────────────────────────── #
        logger.info("  Loading model …")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = False

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            legacy=False,
            cache_dir=args.cache_dir,
        )
        # Mirror sft.py pad-token convention for Qwen / Llama
        if tokenizer.pad_token is None:
            if "Qwen" in model_path:
                tokenizer.pad_token = "<|fim_pad|>"
            elif "Llama" in model_path:
                tokenizer.pad_token = "<|reserved_special_token_5|>"
            else:
                tokenizer.pad_token_id = 0

        num_layers = model.config.num_hidden_layers
        results_per_model[model_label] = {}

        for ratio in args.target_layer_ratios:
            target_layer = int(num_layers * ratio)
            logger.info(
                f"\n  Layer ratio {ratio}  →  layer index {target_layer}"
                f" / {num_layers}"
            )

            # ── check disk cache ─────────────────────────────────────────── #
            safe_label = model_label.replace("/", "_").replace(" ", "_")
            cache_path = out_root / f"{safe_label}__ratio{ratio}.json"

            if cache_path.exists() and not args.force_recompute:
                logger.info(f"  Loading cached results: {cache_path}")
                with open(cache_path) as fh:
                    results_per_model[model_label][ratio] = json.load(fh)
                continue

            # ── extract embeddings ───────────────────────────────────────── #
            all_reps, _ = collect_step_embeddings(
                model, tokenizer, df, target_layer, args.model_max_length
            )
            logger.info(f"  Valid examples: {len(all_reps)}")
            if len(all_reps) < args.num_types:
                logger.warning(
                    f"  Only {len(all_reps)} examples — fewer than "
                    f"num_types={args.num_types}. Reducing k."
                )

            # ── cluster + graph metrics ──────────────────────────────────── #
            k = min(args.num_types, len(all_reps))
            metrics = compute_graph_metrics(all_reps, k)
            results_per_model[model_label][ratio] = metrics

            with open(cache_path, "w") as fh:
                json.dump(metrics, fh, indent=2)
            logger.info(f"  Saved {len(metrics)} metrics → {cache_path}")

        # ── free GPU memory before loading the next model ────────────────── #
        del model
        torch.cuda.empty_cache()

    # ── aggregate summary ─────────────────────────────────────────────────- #
    summary: Dict = {}
    for label, ratio_dict in results_per_model.items():
        summary[label] = {}
        for ratio, metrics in ratio_dict.items():
            diams = [m["diameter"] for m in metrics]
            summary[label][ratio] = {
                "n_samples":         len(metrics),
                "mean_diameter":     float(np.mean(diams)),
                "median_diameter":   float(np.median(diams)),
                "std_diameter":      float(np.std(diams)),
                "cycle_ratio":       float(np.mean([m["has_loop"] for m in metrics])),
                "mean_loop_count":   float(np.mean([m["loop_count"] for m in metrics])),
                "mean_small_world":  float(np.mean([m["small_world_index"] for m in metrics])),
                "mean_avg_path_len": float(np.mean([m["avg_path_length"] for m in metrics])),
            }

    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"\nSummary → {summary_path}")

    # ── print table ────────────────────────────────────────────────────────- #
    header = (
        f"{'Model':<28} {'Ratio':>6}  {'#Samp':>6}  "
        f"{'Mean Diam':>10}  {'Cycle%':>7}  {'SW-idx':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for label, ratio_dict in summary.items():
        for ratio in sorted(ratio_dict.keys()):
            s = ratio_dict[ratio]
            print(
                f"{label:<28} {ratio:>6.1f}  {s['n_samples']:>6}  "
                f"{s['mean_diameter']:>10.2f}  "
                f"{s['cycle_ratio']*100:>6.1f}%  "
                f"{s['mean_small_world']:>8.4f}"
            )
    print("=" * len(header))

    # ── generate all plots ─────────────────────────────────────────────────- #
    logger.info("\nGenerating plots …")
    for metric_key, metric_label in METRICS_TO_PLOT:
        plot_metric_comparison(
            results_per_model, out_root, metric_key, metric_label
        )
    plot_cycle_detection_ratio(results_per_model, out_root)

    logger.info(f"\n✓  All done.  Results in: {out_root.resolve()}")


if __name__ == "__main__":
    main()

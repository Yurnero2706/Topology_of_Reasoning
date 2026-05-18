"""
plot_figure9.py
================
Reproduce Figure 9 from:
  "Topology of Reasoning: Understanding Large Reasoning Models through
   Reasoning Graph Properties"  (arXiv:2506.05744)

Figure 9 — two-panel diameter-distribution comparison
  Panel (a): 200 training steps — s1-v1.0 (s1K) vs s1-v1.1 (s1K-1.1)
  Panel (b): 400 training steps — same comparison

Reads the results.json files produced by cluster_steps_generated.py.
Each results.json contains per-sample graph metrics indexed by row index.

Output path conventions (must match cluster_figure9.sh)
--------------------------------------------------------
  {results_dir}/{checkpoint_path}/{dataset}/
    target_layer_ratio={ratio}/k-means-k={num_types}/results.json

Usage (called automatically by cluster_figure9.sh, but can also run standalone)
------
  python src/plot_figure9.py \\
      --results_dir   results_figure9 \\
      --v10_ckpt_200  ckpts/s1-v1.0/checkpoint-200 \\
      --v10_ckpt_400  ckpts/s1-v1.0/checkpoint-400 \\
      --v11_ckpt_200  ckpts/s1-v1.1/checkpoint-200 \\
      --v11_ckpt_400  ckpts/s1-v1.1/checkpoint-400 \\
      --dataset       aime \\
      --num_types     200 \\
      --target_layer_ratios 0.1 0.3 0.5 0.7 0.9 \\
      --output_path   figure9.pdf
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
#  Path helpers                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def results_json_path(
    results_dir: str,
    checkpoint: str,
    dataset: str,
    ratio: float,
    num_types: int,
    method: str = "k-means",
) -> str:
    """
    Reconstruct the path that cluster_steps_generated.py writes to.

    cluster_steps_generated.py uses model_name_mapping() before building the
    path, but for Qwen / local checkpoints that function is a no-op.

    Uses the same '/' string-join as cluster_steps_generated.py to stay
    consistent on Linux GPU servers.
    """
    return (
        f"{results_dir}/{checkpoint}/{dataset}"
        f"/target_layer_ratio={ratio}/{method}-k={num_types}/results.json"
    )


def load_diameters(
    results_dir: str,
    checkpoint: str,
    dataset: str,
    ratio: float,
    num_types: int,
) -> list:
    """Return a list of per-sample diameter values (floats)."""
    path = results_json_path(results_dir, checkpoint, dataset, ratio, num_types)
    if not os.path.exists(path):
        print(f"  [WARN] Missing results file: {path}")
        return []
    with open(path) as fh:
        data = json.load(fh)
    return [v["diameter"] for v in data.values() if "diameter" in v]


# ─────────────────────────────────────────────────────────────────────────── #
#  Panel drawing                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

_COLORS = {
    "v10": "tab:blue",
    "v11": "tab:orange",
}
_LABELS = {
    "v10": "s1-v1.0  (s1K)",
    "v11": "s1-v1.1  (s1K-1.1)",
}


def _boxplot_kwargs(color: str, side: str, width: float) -> dict:
    offset = -width / 2 if side == "left" else width / 2
    rgb = mcolors.to_rgb(color)
    return dict(
        widths=width * 0.85,
        patch_artist=True,
        boxprops=dict(facecolor=(*rgb, 0.25), color=color, linewidth=1.2),
        medianprops=dict(color=color, linewidth=2),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
        flierprops=dict(
            marker="o",
            markerfacecolor=color,
            markeredgecolor=color,
            alpha=0.3,
            markersize=3,
        ),
        manage_ticks=False,
    )


def draw_panel(
    ax: plt.Axes,
    ratios: list,
    v10_data: list,
    v11_data: list,
    panel_letter: str,
    step: int,
) -> None:
    """Draw one panel of Figure 9 (box plots + mean lines for two models)."""
    x = np.arange(len(ratios))
    width = 0.35

    for key, data, side in [("v10", v10_data, "left"), ("v11", v11_data, "right")]:
        color = _COLORS[key]
        label = _LABELS[key]
        positions = x - width / 2 if side == "left" else x + width / 2

        ax.boxplot(
            data,
            positions=positions,
            **_boxplot_kwargs(color, side, width),
        )
        means = [float(np.mean(d)) if d else 0.0 for d in data]
        ax.plot(
            positions,
            means,
            "o-",
            color=color,
            label=label,
            linewidth=2,
            markersize=6,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ratios], fontsize=11)
    ax.set_xlabel("Target Layer Ratio", fontsize=12)
    ax.set_ylabel("Graph Diameter", fontsize=12)
    ax.set_title(
        f"({panel_letter}) Diameter Distribution  [{step} Training Steps]",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.35)


# ─────────────────────────────────────────────────────────────────────────── #
#  Main                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 9 of arXiv:2506.05744 from cluster results."
    )
    parser.add_argument("--results_dir",   default="results_figure9",
                        help="Root directory written by cluster_figure9.sh.")
    parser.add_argument("--v10_ckpt_200",  default="ckpts/s1-v1.0/checkpoint-200",
                        help="s1-v1.0 checkpoint at 200 steps.")
    parser.add_argument("--v10_ckpt_400",  default="ckpts/s1-v1.0/checkpoint-400",
                        help="s1-v1.0 checkpoint at 400 steps.")
    parser.add_argument("--v11_ckpt_200",  default="ckpts/s1-v1.1/checkpoint-200",
                        help="s1-v1.1 checkpoint at 200 steps.")
    parser.add_argument("--v11_ckpt_400",  default="ckpts/s1-v1.1/checkpoint-400",
                        help="s1-v1.1 checkpoint at 400 steps.")
    parser.add_argument("--dataset",       default="aime",
                        help="Evaluation dataset name (must match eval_14B.sh / cluster_figure9.sh).")
    parser.add_argument("--num_types",     type=int, default=200,
                        help="K-means k used in cluster_figure9.sh.")
    parser.add_argument("--target_layer_ratios", nargs="+", type=float,
                        default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--output_path",   default="figure9.pdf",
                        help="Where to save the figure (PDF or PNG).")
    args = parser.parse_args()

    ratios = sorted(args.target_layer_ratios)

    # ── Load all diameter lists ───────────────────────────────────────────── #
    checkpoints = {
        "v10_200": args.v10_ckpt_200,
        "v10_400": args.v10_ckpt_400,
        "v11_200": args.v11_ckpt_200,
        "v11_400": args.v11_ckpt_400,
    }

    all_data: dict = {}
    for key, ckpt in checkpoints.items():
        all_data[key] = []
        for ratio in ratios:
            diams = load_diameters(
                args.results_dir, ckpt, args.dataset, ratio, args.num_types
            )
            all_data[key].append(diams)
            mean_str = f"{np.mean(diams):.3f}" if diams else "N/A"
            print(f"  {key}  ratio={ratio:3.1f}  n={len(diams):4d}  mean_diam={mean_str}")

    # ── Plot ─────────────────────────────────────────────────────────────── #
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    draw_panel(axes[0], ratios, all_data["v10_200"], all_data["v11_200"],
               panel_letter="a", step=200)
    draw_panel(axes[1], ratios, all_data["v10_400"], all_data["v11_400"],
               panel_letter="b", step=400)

    fig.suptitle(
        "Figure 9 — Reasoning Graph Diameter: s1K vs s1K-1.1 (Qwen2.5-14B)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure 9 saved → {args.output_path}")

    # ── Print summary table ───────────────────────────────────────────────── #
    header = f"{'Key':<12} {'Ratio':>6}  {'N':>5}  {'Mean Diam':>10}  {'Median':>8}  {'Std':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for key in ("v10_200", "v10_400", "v11_200", "v11_400"):
        for i, ratio in enumerate(ratios):
            d = all_data[key][i]
            if d:
                print(
                    f"{key:<12} {ratio:>6.1f}  {len(d):>5}  "
                    f"{np.mean(d):>10.3f}  {np.median(d):>8.3f}  {np.std(d):>8.3f}"
                )
            else:
                print(f"{key:<12} {ratio:>6.1f}   {'—':>5}  {'no data':>10}")
    print("=" * len(header))


if __name__ == "__main__":
    main()

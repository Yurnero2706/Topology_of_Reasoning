"""
plot_diameter_violin.py
=======================
Re-plot the cached diameter / graph-metric results as **violin plots** in the
style of Figure 9 of "Topology of Reasoning" (arXiv:2506.05744).

This reads the per-(model, layer-ratio) JSON caches that analyze_diameter.py
already wrote — so it needs NO GPU and does NOT reload any model. It just
re-renders the existing numbers in the paper's violin aesthetic.

The cache files are named "<label>__ratio<r>.json" and each one is a list of
per-trace metric dicts (the same dicts analyze_diameter.py produces).

Usage
-----
  # Single panel from one results directory (all models found in it):
  python src/plot_diameter_violin.py \
      --results_dir results_diameter/simplescaling/s1K/steps-200 \
      --title "Diameter Distribution at 200 steps"

  # Only show specific series, in this order (e.g. match the paper: drop Base):
  python src/plot_diameter_violin.py \
      --results_dir results_diameter/simplescaling/s1K/steps-200 \
      --labels "s1-v1.0" "s1-v1.1"

  # Two-panel figure (paper layout), e.g. k=200 vs k=400 dirs:
  python src/plot_diameter_violin.py \
      --results_dir       results_diameter/simplescaling/s1K/steps-200 \
      --results_dir_right results_diameter/simplescaling/s1K/steps-400 \
      --title       "Diameter Distribution at 200 steps" \
      --title_right "Diameter Distribution at 400 steps" \
      --output results_diameter/figure9_violin.png
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Seaborn "muted"-style palette so the figure matches the paper's look even
# without seaborn installed.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

_CACHE_RE = re.compile(r"^(?P<label>.+)__ratio(?P<ratio>[0-9.]+)\.json$")


def _pretty(label: str) -> str:
    """Turn a cache label into a legend-friendly name (s1-v1.0 -> s1 v1.0)."""
    if label.lower().startswith("base"):
        return "Base"
    return label.replace("s1-v", "s1 v").replace("-", " ")


def discover(results_dir: Path) -> Dict[str, Dict[float, List[dict]]]:
    """Return {label: {ratio: [metric_dict, ...]}} for every cache in dir."""
    data: Dict[str, Dict[float, List[dict]]] = {}
    for fp in sorted(results_dir.glob("*__ratio*.json")):
        m = _CACHE_RE.match(fp.name)
        if not m:
            continue
        label = m.group("label")
        ratio = float(m.group("ratio"))
        with open(fp) as fh:
            data.setdefault(label, {})[ratio] = json.load(fh)
    return data


def _draw_panel(
    ax,
    data: Dict[str, Dict[float, List[dict]]],
    labels: List[str],
    metric_key: str,
    metric_label: str,
    title: str,
) -> List[mpatches.Patch]:
    ratios = sorted({r for lab in labels for r in data.get(lab, {})})
    x_pos = np.arange(len(ratios))
    n = len(labels)
    width = 0.8 / max(n, 1)
    offsets = (np.arange(n) - (n - 1) / 2) * width

    legend_handles: List[mpatches.Patch] = []

    for m_idx, label in enumerate(labels):
        color = PALETTE[m_idx % len(PALETTE)]
        per_ratio = data.get(label, {})

        positions, datasets, means = [], [], []
        for r_idx, ratio in enumerate(ratios):
            vals = [d[metric_key] for d in per_ratio.get(ratio, [])]
            if not vals:
                continue
            positions.append(x_pos[r_idx] + offsets[m_idx])
            datasets.append(np.asarray(vals, dtype=float))
            means.append(float(np.mean(vals)))

        if not datasets:
            continue

        parts = ax.violinplot(
            datasets,
            positions=positions,
            widths=width * 0.9,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
            body.set_linewidth(1.0)

        # Inner "stick": thin line over the central 90% range + red mean dash.
        for pos, vals, mean in zip(positions, datasets, means):
            lo, hi = np.percentile(vals, [5, 95])
            ax.plot([pos, pos], [lo, hi], color="0.25", linewidth=1.0, zorder=3)
            ax.plot(
                [pos - width * 0.28, pos + width * 0.28], [mean, mean],
                color="#B22222", linewidth=2.0, zorder=4,
            )

        legend_handles.append(
            mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.55,
                           label=_pretty(label))
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in ratios], fontsize=11)
    ax.set_xlabel("Target Layer Ratio", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    return legend_handles


def main() -> None:
    ap = argparse.ArgumentParser(description="Violin re-plot of cached diameter results.")
    ap.add_argument("--results_dir", required=True, type=Path)
    ap.add_argument("--results_dir_right", type=Path, default=None,
                    help="Optional second directory → two-panel figure.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Series to plot, in order. Default: all found (Base last).")
    ap.add_argument("--metric_key", default="diameter")
    ap.add_argument("--metric_label", default="Diameter")
    ap.add_argument("--title", default="Diameter Distribution across Hidden Layers")
    ap.add_argument("--title_right", default="Diameter Distribution (k=400)")
    ap.add_argument("--legend_title", default="Dataset")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output PNG path (default: <results_dir>/<metric>_violin.png).")
    args = ap.parse_args()

    left = discover(args.results_dir)
    if not left:
        raise SystemExit(f"No '*__ratio*.json' caches found in {args.results_dir}")

    def order_labels(found: Dict) -> List[str]:
        if args.labels:
            return [l for l in args.labels if l in found]
        # default: non-base first (alphabetical), Base series last
        labs = sorted(found.keys())
        return [l for l in labs if not l.lower().startswith("base")] + \
               [l for l in labs if l.lower().startswith("base")]

    two_panel = args.results_dir_right is not None
    if two_panel:
        right = discover(args.results_dir_right)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        h = _draw_panel(axes[0], left, order_labels(left),
                        args.metric_key, args.metric_label, args.title)
        _draw_panel(axes[1], right, order_labels(right),
                    args.metric_key, args.metric_label, args.title_right)
        axes[0].legend(handles=h, title=args.legend_title, fontsize=11,
                       title_fontsize=12, loc="upper left")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        h = _draw_panel(ax, left, order_labels(left),
                        args.metric_key, args.metric_label, args.title)
        ax.legend(handles=h, title=args.legend_title, fontsize=11,
                  title_fontsize=12, loc="upper left")

    fig.tight_layout()
    out = args.output or (args.results_dir / f"{args.metric_key}_violin.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved violin plot -> {out}")


if __name__ == "__main__":
    main()

"""Paper figure generation using matplotlib/seaborn.

Generates publication-quality figures for the paper including:
- RQ1: Performance comparison bar charts
- RQ2: Tool usage heatmaps and sequence diagrams
- RQ3: 4D score radar charts
- RQ4: Structure-aware tool impact charts
- RQ5: DSPy optimization comparison

Usage:
    from experiments.analysis.visualize import PaperFigures

    figures = PaperFigures(results_dir="data/results/")
    figures.plot_rq1_comparison("figures/rq1_comparison.pdf")
    figures.plot_all("figures/")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

# Lazy imports — matplotlib/seaborn only needed when actually plotting
_MPL_AVAILABLE = False
try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Non-interactive backend for paper figures
    _MPL_AVAILABLE = True
except ImportError:
    pass


# Paper-quality defaults
FIGSIZE_SINGLE = (6, 4)
FIGSIZE_WIDE = (10, 4)
FONT_SIZE = 11
DPI = 300


def _check_matplotlib():
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: uv pip install matplotlib seaborn"
        )


def plot_metric_comparison(
    variant_metrics: dict[str, dict[str, float]],
    metrics: list[str] = ("exact_match", "f1", "rouge_l"),
    title: str = "Performance Comparison",
    output_path: str | Path | None = None,
) -> None:
    """Bar chart comparing metrics across variants.

    Args:
        variant_metrics: {variant_name: {metric_name: value}}.
        metrics: Which metrics to plot.
        title: Figure title.
        output_path: Save path (PDF/PNG). None = plt.show().
    """
    _check_matplotlib()

    names = list(variant_metrics.keys())
    x = np.arange(len(names))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for i, metric in enumerate(metrics):
        values = [variant_metrics[n].get(metric, 0) for n in names]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xlabel("Variant")
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=FONT_SIZE + 1)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=FONT_SIZE - 1)
    ax.legend(fontsize=FONT_SIZE - 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_score_progression(
    progression: dict[str, list[float]],
    title: str = "Quality Score Progression",
    output_path: str | Path | None = None,
) -> None:
    """Line chart showing score progression across iterations.

    Args:
        progression: {dimension: [scores_per_iteration]}.
        title: Figure title.
        output_path: Save path.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for dim, values in progression.items():
        ax.plot(range(len(values)), values, marker="o", label=dim, markersize=4)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Score")
    ax.set_title(title, fontsize=FONT_SIZE + 1)
    ax.legend(fontsize=FONT_SIZE - 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_tool_frequency(
    tool_counts: dict[str, int],
    title: str = "Tool Call Frequency",
    output_path: str | Path | None = None,
) -> None:
    """Horizontal bar chart of tool call frequencies.

    Args:
        tool_counts: {tool_name: count}.
        title: Figure title.
        output_path: Save path.
    """
    _check_matplotlib()

    sorted_items = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars = ax.barh(names, counts, color="steelblue")
    ax.bar_label(bars, padding=3, fontsize=FONT_SIZE - 1)
    ax.set_xlabel("Number of Calls")
    ax.set_title(title, fontsize=FONT_SIZE + 1)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_ablation_impact(
    impact: dict[str, dict[str, float]],
    metric: str = "f1",
    title: str = "Ablation Study — F1 Impact",
    output_path: str | Path | None = None,
) -> None:
    """Bar chart showing quality delta when each component is removed.

    Args:
        impact: {variant_name: {metric: delta}}.
        metric: Which metric to plot.
        title: Figure title.
        output_path: Save path.
    """
    _check_matplotlib()

    sorted_items = sorted(impact.items(), key=lambda x: x[1].get(metric, 0))
    names = [item[0] for item in sorted_items]
    deltas = [item[1].get(metric, 0) for item in sorted_items]
    colors = ["crimson" if d < 0 else "forestgreen" for d in deltas]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.barh(names, deltas, color=colors)
    ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=FONT_SIZE - 1)
    ax.set_xlabel(f"Δ {metric.upper()} (vs Full System)")
    ax.set_title(title, fontsize=FONT_SIZE + 1)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")
        plt.close(fig)
    else:
        plt.show()


class PaperFigures:
    """Convenience class to generate all paper figures from results."""

    def __init__(self, results_dir: str | Path = "data/results") -> None:
        self.results_dir = Path(results_dir)

    def plot_all(self, output_dir: str | Path = "figures") -> None:
        """Generate all paper figures."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating paper figures in {output_dir}")
        # Individual figure methods would be called here once
        # experiment results are available
        logger.info("Figure generation requires experiment results. Run experiments first.")

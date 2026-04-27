"""
analyze_results.py — Generate tables, plots, and significance tests from experiment results.

Usage:
    cd E:/csc5051_final_proj
    python scripts/analyze_results.py --run results/run_XXXXXXXX_XXXXXX
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_metrics(path_or_dir: str) -> dict:
    """Load metrics from a JSON file. Accepts full filepath or run directory."""
    p = Path(path_or_dir)
    if p.is_file():
        target = p
    else:
        target = p / "metrics.json"
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_condition_name(condition: str) -> tuple[str, str, str]:
    """Parse a condition name into (signal, method, backbone).

    Handles multi-word signals like 'dual_enhanced' by parsing known
    methods and backbones from the right.
    """
    KNOWN_METHODS = {"bm25", "dense", "hybrid", "none"}
    KNOWN_BACKBONES = {"zero_shot", "fine_tuned"}

    tokens = condition.split("_")

    # Strip trailing k{N} or n{N} suffixes (from ablation conditions)
    if tokens and re.match(r'^[kn]\d+$', tokens[-1]):
        tokens = tokens[:-1]

    # Try to find backbone (last 1-2 tokens)
    backbone = "?"
    for width in (2, 1):
        if len(tokens) >= width:
            candidate = "_".join(tokens[-width:])
            if candidate in KNOWN_BACKBONES:
                backbone = candidate
                tokens = tokens[:-width]
                break

    # Try to find method (now last 1 token)
    method = "?"
    if tokens and tokens[-1] in KNOWN_METHODS:
        method = tokens[-1]
        tokens = tokens[:-1]

    signal = "_".join(tokens) if tokens else "?"
    return signal, method, backbone


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    """Convert metrics dict to a tidy DataFrame."""
    rows = []
    for condition, values in metrics.items():
        # Strip known prefixes before parsing
        core = condition
        for prefix in ("k_ablation_", "scale_"):
            if core.startswith(prefix):
                core = core[len(prefix):]
                break

        signal, method, backbone = _parse_condition_name(core)
        row = {"condition": condition, "signal": signal, "method": method, "backbone": backbone}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    metric_cols = [c for c in df.columns if c.startswith(("hit@", "avoidance@"))]
    display_cols = ["signal", "method", "backbone"] + metric_cols
    print(df[display_cols].to_string(index=False, float_format="%.4f"))


def plot_signal_comparison(df: pd.DataFrame, output_dir: str):
    """Bar chart: signal condition × metric."""
    metric_cols = [c for c in df.columns if c.startswith(("hit@", "avoidance@"))]
    # Exclude baseline for cleaner comparison
    df_main = df[df["signal"].isin(["positive", "dual", "negative"])]

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 5))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        grouped = df_main.groupby("signal")[metric].mean()
        colors = {"positive": "#4CAF50", "dual": "#2196F3", "negative": "#FF5722"}
        bars = ax.bar(grouped.index, grouped.values, color=[colors.get(s, "#999") for s in grouped.index])
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = Path(output_dir) / "signal_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"[plot] Saved → {out_path}")
    plt.close()


def plot_heatmap(df: pd.DataFrame, output_dir: str):
    """Heatmap: method × signal for key metrics."""
    for metric in ["hit@5", "avoidance@5"]:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(values=metric, index="method", columns="signal", aggfunc="mean")
        plt.figure(figsize=(8, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
        plt.title(f"{metric.upper()} by Method × Signal")
        plt.tight_layout()
        out_path = Path(output_dir) / f"heatmap_{metric.replace('@', '_')}.png"
        plt.savefig(out_path, dpi=150)
        print(f"[plot] Saved → {out_path}")
        plt.close()


def significance_test(run_dir: str, cond_a: str, cond_b: str, metric: str = "hit"):
    """
    Paired bootstrap significance test between two conditions.
    Loads per-user results from JSONL files.
    """
    path_a = Path(run_dir) / f"{cond_a}.jsonl"
    path_b = Path(run_dir) / f"{cond_b}.jsonl"

    if not path_a.exists() or not path_b.exists():
        print(f"⚠ Missing result files for {cond_a} or {cond_b}")
        return

    def load_scores(path, metric_name):
        # Import normalize from metrics
        from src.evaluation.metrics import _normalize_name
        scores = []
        with open(path, "r") as f:
            for line in f:
                r = json.loads(line)
                recs = [_normalize_name(rec["musical"]) for rec in r.get("recommendations", [])[:5]]
                held = _normalize_name(r.get("held_out", ""))
                if metric_name == "hit":
                    from src.evaluation.metrics import _fuzzy_match
                    scores.append(1 if any(_fuzzy_match(held, r) for r in recs) else 0)
        return np.array(scores)

    scores_a = load_scores(path_a, metric)
    scores_b = load_scores(path_b, metric)

    if len(scores_a) != len(scores_b):
        print(f"⚠ Different user counts: {len(scores_a)} vs {len(scores_b)}")
        return

    # McNemar's test for paired binary outcomes
    if metric == "hit":
        n_ab = np.sum((scores_a == 1) & (scores_b == 0))  # A right, B wrong
        n_ba = np.sum((scores_a == 0) & (scores_b == 1))  # B right, A wrong
        if n_ab + n_ba > 0:
            stat = (abs(n_ab - n_ba) - 1) ** 2 / (n_ab + n_ba)
            p_value = 1 - stats.chi2.cdf(stat, df=1)
        else:
            stat, p_value = 0, 1.0
        print(f"\nMcNemar's test: {cond_a} vs {cond_b}")
        print(f"  A wins: {n_ab}, B wins: {n_ba}")
        print(f"  χ² = {stat:.4f}, p = {p_value:.4f}")
        print(f"  {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")


def main():
    parser = argparse.ArgumentParser(description="Analyze MeloMatch results")
    parser.add_argument("--run", required=True, help="Path to run directory")
    args = parser.parse_args()

    metrics = load_metrics(args.run)
    df = metrics_to_dataframe(metrics)
    print_summary_table(df)

    # Plots
    plot_signal_comparison(df, args.run)
    plot_heatmap(df, args.run)

    # Key significance tests
    print("\n" + "=" * 60)
    print("SIGNIFICANCE TESTS")
    print("=" * 60)
    # Test: dual vs positive (main hypothesis)
    for method in ["bm25", "dense", "hybrid"]:
        for backbone in ["zero_shot", "fine_tuned"]:
            cond_a = f"positive_{method}_{backbone}"
            cond_b = f"dual_{method}_{backbone}"
            if cond_a in metrics and cond_b in metrics:
                significance_test(args.run, cond_a, cond_b)

    # Ablation: Retrieval depth
    k_abl_path = Path(args.run) / "k_ablation_metrics.json"
    if k_abl_path.exists():
        print("\n" + "=" * 60)
        print("RETRIEVAL DEPTH ABLATION")
        print("=" * 60)
        k_metrics = load_metrics(str(k_abl_path))
        k_df = metrics_to_dataframe(k_metrics)
        print(k_df.to_string(index=False, float_format="%.4f"))

    # Ablation: Data scaling
    scale_abl_path = Path(args.run) / "scale_ablation_metrics.json"
    if scale_abl_path.exists():
        print("\n" + "=" * 60)
        print("DATA SCALING ABLATION")
        print("=" * 60)
        scale_metrics = load_metrics(str(scale_abl_path))
        scale_df = metrics_to_dataframe(scale_metrics)
        print(scale_df.to_string(index=False, float_format="%.4f"))

    print("\n✓ Analysis complete.")


if __name__ == "__main__":
    main()

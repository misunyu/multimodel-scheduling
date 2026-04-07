#!/usr/bin/env python3
"""
Generate NPU (Antara) experiment result graphs from per-second CSV metrics.

Usage:
    python plot_antara_results.py <TIMESTAMP>
    python plot_antara_results.py           # auto-detect latest timestamp

Expects CSV files in results/:
    adaptive_metrics_mode{0,1,2}_antara_<TIMESTAMP>.csv

Produces in results/:
    droprate_over_time_antara.pdf
    latency_over_time_antara.pdf
    vscore_over_time_antara.pdf
    adaptive_comparison_antara_<TIMESTAMP>.pdf  (3 graphs combined)
    comparison_summary_antara.pdf               (summary table)
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCHEDULE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gen_schedules_antara", "model_schedules_vscore_test.yaml",
)

MODE_LABELS = {
    0: "Stop-and-restart",
    1: "Adaptive hot-swap",
    2: "Reactive (rollback)",
}
MODE_STYLES = {
    0: dict(color="blue",      linestyle="--",  linewidth=2.2),
    1: dict(color="red",       linestyle="-",   linewidth=2.5),
    2: dict(color="darkgreen", linestyle="-.",   linewidth=2.8),
}

ROLLBACK_OFFSET_S = 6  # approximate seconds after deploy change when reactive rollback fires


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_timestamp():
    """Auto-detect the latest antara timestamp from CSV files in results/."""
    pattern = os.path.join(RESULTS_DIR, "adaptive_metrics_mode0_antara_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    # Extract timestamp from filename
    basename = os.path.basename(files[-1])
    # adaptive_metrics_mode0_antara_YYYYMMDD_HHMMSS.csv
    parts = basename.replace("adaptive_metrics_mode0_antara_", "").replace(".csv", "")
    return parts


def load_csv(mode: int, timestamp: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, f"adaptive_metrics_mode{mode}_antara_{timestamp}.csv")
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def align_to_deploy_change(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'relative_time' column: seconds relative to the first combination change."""
    if df.empty:
        return df
    combos = df["combination"].unique()
    if len(combos) < 2:
        # No deploy change observed — align to start
        t0 = df["timestamp"].iloc[0]
        df = df.copy()
        df["relative_time"] = (df["timestamp"] - t0).dt.total_seconds()
        return df
    # Find the first row where combination changes
    first_combo = df["combination"].iloc[0]
    change_mask = df["combination"] != first_combo
    if not change_mask.any():
        t0 = df["timestamp"].iloc[0]
        df = df.copy()
        df["relative_time"] = (df["timestamp"] - t0).dt.total_seconds()
        return df
    t_change = df.loc[change_mask, "timestamp"].iloc[0]
    df = df.copy()
    df["relative_time"] = (df["timestamp"] - t_change).dt.total_seconds()
    return df


def load_combination_info() -> dict:
    """Load combination descriptions from the schedule YAML."""
    if not os.path.exists(SCHEDULE_FILE):
        return {}
    with open(SCHEDULE_FILE, "r") as f:
        data = yaml.safe_load(f) or {}
    info = {}
    for combo_name, models in data.items():
        parts = []
        for entry in models.values():
            model = entry.get("model", "?")
            exe = entry.get("execution", "?").upper()
            parts.append(f"{model} ({exe})")
        info[combo_name] = ", ".join(parts)
    return info


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_event_lines(ax, has_rollback=True):
    """Add vertical lines for deploy change and rollback."""
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.text(0, ax.get_ylim()[1] * 0.97, "Deploy\nChange",
            ha="center", va="top", fontsize=9, color="gray")
    if has_rollback:
        ax.axvline(x=ROLLBACK_OFFSET_S, color="darkgreen", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(ROLLBACK_OFFSET_S, ax.get_ylim()[1] * 0.97, "Rollback",
                ha="center", va="top", fontsize=9, color="darkgreen", style="italic")


def _smooth(series, window=3):
    """Simple rolling mean to smooth noisy metrics."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_single_metric(dfs: dict, metric_col: str, ylabel: str, title: str,
                       output_path: str, smooth_window: int = 3,
                       threshold_line: float = None, threshold_label: str = None):
    """Plot one metric across 3 modes, save as PDF."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for mode in sorted(dfs.keys()):
        df = dfs[mode]
        if df.empty:
            continue
        style = MODE_STYLES[mode]
        y = _smooth(df[metric_col], smooth_window)
        ax.plot(df["relative_time"], y, label=MODE_LABELS[mode], **style)

    # Draw threshold line (e.g. epsilon for rollback)
    if threshold_line is not None:
        ax.axhline(y=threshold_line, color="orange", linestyle="--", linewidth=1.8, alpha=0.8)
        lbl = threshold_label or f"Threshold = {threshold_line}"
        ax.text(ax.get_xlim()[1] * 0.98, threshold_line + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                lbl, ha="right", va="bottom", fontsize=10, color="orange", fontweight="bold")

    ax.set_xlabel("Time relative to deploy change (s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    # Add event lines after setting data so ylim is correct
    _add_event_lines(ax, has_rollback=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_combined_pdf(dfs: dict, output_path: str):
    """Create a multi-page PDF with all 3 graphs (like adaptive_comparison)."""
    with PdfPages(output_path) as pdf:
        # Page 1: V-Score
        fig, ax = plt.subplots(figsize=(12, 6))
        for mode in sorted(dfs.keys()):
            df = dfs[mode]
            if df.empty:
                continue
            y = _smooth(df["v_score"], 3)
            ax.plot(df["relative_time"], y, label=MODE_LABELS[mode], **MODE_STYLES[mode])
        ax.axhline(y=5.0, color="orange", linestyle="--", linewidth=1.8, alpha=0.8)
        ax.text(ax.get_xlim()[1] * 0.98, 5.15,
                r"$\epsilon$ = 5.0  (rollback when V(t) $>$ 5, strictly exceeding $\epsilon$)",
                ha="right", va="bottom", fontsize=10, color="orange", fontweight="bold")
        ax.set_xlabel("Time relative to deploy change (s)", fontsize=12)
        ax.set_ylabel("Violation Score  V(t)", fontsize=12)
        ax.set_title("QoS Violation Score Under Deploy Change", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        _add_event_lines(ax)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Latency
        fig, ax = plt.subplots(figsize=(12, 6))
        for mode in sorted(dfs.keys()):
            df = dfs[mode]
            if df.empty:
                continue
            total_latency = df[["view1_infer_ms", "view2_infer_ms",
                                "view3_infer_ms", "view4_infer_ms"]].sum(axis=1)
            y = _smooth(total_latency, 3)
            ax.plot(df["relative_time"], y, label=MODE_LABELS[mode], **MODE_STYLES[mode])
        ax.set_xlabel("Time relative to deploy change (s)", fontsize=12)
        ax.set_ylabel("Total Latency (ms)", fontsize=12)
        ax.set_title("Total Inference Latency Under Deploy Change", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        _add_event_lines(ax)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: Drop Rate
        fig, ax = plt.subplots(figsize=(12, 6))
        for mode in sorted(dfs.keys()):
            df = dfs[mode]
            if df.empty:
                continue
            y = _smooth(df["drop_rate_fps"], 3)
            ax.plot(df["relative_time"], y, label=MODE_LABELS[mode], **MODE_STYLES[mode])
        ax.set_xlabel("Time relative to deploy change (s)", fontsize=12)
        ax.set_ylabel("Drop Rate (FPS)", fontsize=12)
        ax.set_title("Drop Rate Under Deploy Change", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        _add_event_lines(ax)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: Comparison Summary Table
        combo_info = load_combination_info()
        fig = _make_summary_table(dfs, combo_info)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {output_path}")


def _make_summary_table(dfs: dict, combo_info: dict):
    """Create a summary table figure (comparison_summary style)."""
    # Compute statistics for each mode
    stats = {}
    all_combos = set()
    for mode in sorted(dfs.keys()):
        df = dfs[mode]
        if df.empty:
            stats[mode] = {}
            continue
        all_combos.update(df["combination"].unique())
        total_latency = df[["view1_infer_ms", "view2_infer_ms",
                            "view3_infer_ms", "view4_infer_ms"]].sum(axis=1)
        stats[mode] = {
            "Samples": len(df),
            "Avg Throughput (FPS)": f"{df['total_fps'].mean():.2f}",
            "Avg V-Score": f"{df['v_score'].mean():.4f}",
            "Max V-Score": f"{df['v_score'].max():.4f}",
            "Avg Drop Rate (FPS)": f"{df['drop_rate_fps'].mean():.4f}",
            "Avg Latency (ms)": f"{total_latency.mean():.2f}",
        }

    # Build combination descriptions
    sorted_combos = sorted(all_combos)
    combo_descs = []
    for c in sorted_combos:
        desc = combo_info.get(c, "")
        combo_descs.append(f"{c}: {desc}" if desc else c)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    # Title
    title_parts = " → ".join(sorted_combos) if len(sorted_combos) <= 4 else f"{len(sorted_combos)} combinations"
    ax.set_title(f"Comparison Summary (NPU/Antara)\n{title_parts}",
                 fontsize=14, fontweight="bold", pad=20)

    # Combination descriptions
    y_start = 0.85
    for i, desc in enumerate(combo_descs[:4]):  # Show at most 4
        ax.text(0.5, y_start - i * 0.04, desc, transform=ax.transAxes,
                fontsize=8, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc"))

    # Rollback policy note
    ax.text(0.5, y_start - len(combo_descs[:4]) * 0.04 - 0.03,
            r"Reactive rollback policy:  rollback triggered when V(t) $>$ 5  ($\epsilon$ = 5.0, strictly exceeding threshold)",
            transform=ax.transAxes, fontsize=9, ha="center", va="top",
            color="#CC6600", fontweight="bold")

    # Table data
    metrics = list(next(iter(stats.values()), {}).keys())
    col_labels = [MODE_LABELS[m] for m in sorted(dfs.keys())]
    cell_data = []
    for metric in metrics:
        row = []
        for mode in sorted(dfs.keys()):
            row.append(str(stats.get(mode, {}).get(metric, "N/A")))
        cell_data.append(row)

    if cell_data:
        table = ax.table(
            cellText=cell_data,
            rowLabels=metrics,
            colLabels=col_labels,
            cellLoc="center",
            rowLoc="center",
            loc="center",
            bbox=[0.05, 0.05, 0.9, 0.55],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Style header row
        for j, label in enumerate(col_labels):
            cell = table[0, j]
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        # Style row labels
        for i in range(len(metrics)):
            cell = table[i + 1, -1]
            cell.set_facecolor("#D6E4F0")
            cell.set_text_props(fontweight="bold")
        # Alternating row colors
        for i in range(len(metrics)):
            bg = "#FFFFFF" if i % 2 == 0 else "#F2F2F2"
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(bg)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        timestamp = sys.argv[1]
    else:
        timestamp = find_timestamp()
        if timestamp is None:
            print("Error: No antara CSV files found in results/. Run the experiment first.")
            print("Usage: python plot_antara_results.py <TIMESTAMP>")
            sys.exit(1)
        print(f"Auto-detected timestamp: {timestamp}")

    print(f"\nLoading CSV files for timestamp: {timestamp}")
    dfs_raw = {}
    for mode in [0, 1, 2]:
        df = load_csv(mode, timestamp)
        if df.empty:
            print(f"  Mode {mode}: NO DATA")
        else:
            print(f"  Mode {mode}: {len(df)} rows, combos={df['combination'].unique().tolist()}")
            df = align_to_deploy_change(df)
        dfs_raw[mode] = df

    if all(df.empty for df in dfs_raw.values()):
        print("\nNo data to plot. Exiting.")
        sys.exit(1)

    print("\nGenerating graphs...")

    # 1) Drop Rate over time
    plot_single_metric(
        dfs_raw,
        metric_col="drop_rate_fps",
        ylabel="Drop Rate (FPS)",
        title="Drop Rate Under Deploy Change",
        output_path=os.path.join(RESULTS_DIR, "droprate_over_time_antara.pdf"),
    )

    # 2) Latency over time (sum of 4 views)
    # Create temporary dfs with total_latency column
    dfs_lat = {}
    for mode, df in dfs_raw.items():
        if df.empty:
            dfs_lat[mode] = df
            continue
        df2 = df.copy()
        df2["total_latency"] = df2[["view1_infer_ms", "view2_infer_ms",
                                     "view3_infer_ms", "view4_infer_ms"]].sum(axis=1)
        dfs_lat[mode] = df2
    plot_single_metric(
        dfs_lat,
        metric_col="total_latency",
        ylabel="Total Latency (ms)",
        title="Total Inference Latency Under Deploy Change",
        output_path=os.path.join(RESULTS_DIR, "latency_over_time_antara.pdf"),
    )

    # 3) V-Score over time (with epsilon threshold line)
    plot_single_metric(
        dfs_raw,
        metric_col="v_score",
        ylabel="Violation Score  V(t)",
        title="QoS Violation Score Under Deploy Change",
        output_path=os.path.join(RESULTS_DIR, "vscore_over_time_antara.pdf"),
        threshold_line=5.0,
        threshold_label=r"$\epsilon$ = 5.0  (rollback when V(t) $>$ 5, strictly exceeding $\epsilon$)",
    )

    # 4) Combined multi-page PDF
    plot_combined_pdf(
        dfs_raw,
        output_path=os.path.join(RESULTS_DIR, f"adaptive_comparison_antara_{timestamp}.pdf"),
    )

    # 5) Standalone comparison summary
    combo_info = load_combination_info()
    fig = _make_summary_table(dfs_raw, combo_info)
    fig.savefig(os.path.join(RESULTS_DIR, "comparison_summary_antara.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(RESULTS_DIR, 'comparison_summary_antara.pdf')}")

    print("\nDone! All graphs saved to results/")


if __name__ == "__main__":
    main()

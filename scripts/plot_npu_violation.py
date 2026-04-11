#!/usr/bin/env python3
"""
Generate the CPU-NPU violation-score figure for the paper
(``vscore_over_time_antara.pdf``, fig:npu_violation).

The paper text compares two strategies on the constrained-accelerator
(CPU + 2x NPU) platform:

  * Adaptive hot-swap (mode 1): immediately applies the new placement
    without rollback. On NPU it cannot recover, V(t) stays above the
    threshold for the rest of the observation window.

  * BoundGuard (mode 2): same hot-swap, plus QoS-driven detection and
    rollback to the previous stable configuration when V(t) > epsilon.
    V(t) returns below the threshold within seconds.

The figure is plotted from the per-second CSVs produced by
schedule_executor_main.py --metrics-csv during the antara experiment
(see ``run_antara_experiment.sh``). The default CSVs are the
2026-04-03 17:38:20 run committed in results/, but the timestamp can
be overridden with --timestamp.

Usage:
    python scripts/plot_npu_violation.py
    python scripts/plot_npu_violation.py --timestamp 20260403_173820
    python scripts/plot_npu_violation.py --epsilon 5.0 --output results/vscore_over_time_antara.pdf
"""
import argparse
import datetime
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

DEFAULT_OUTPUT = os.path.join(RESULTS_DIR, "vscore_over_time_antara.pdf")
DEFAULT_REPRO = os.path.join(RESULTS_DIR, "vscore_over_time_antara.repro.txt")
DEFAULT_EPSILON = 5.0

# Modes shown in the paper figure: only Adaptive hot-swap and BoundGuard.
# (Mode 0 stop-and-restart and mode 3 static are NOT in this figure; the
# 3-line debug PDF is produced by plot_antara_results.py instead.)
MODES = (1, 2)
MODE_LABELS = {
    1: "Adaptive hot-swap",
    2: "BoundGuard",
}
MODE_STYLES = {
    1: dict(color="#d62728", linestyle="-",  linewidth=2.6),   # red
    2: dict(color="#2ca02c", linestyle="-.", linewidth=2.8),   # green
}


def find_default_timestamp() -> str:
    """Pick the most recent antara CSV timestamp present in results/."""
    pattern = os.path.join(RESULTS_DIR, "adaptive_metrics_mode2_antara_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return ""
    base = os.path.basename(files[-1])
    return base.replace("adaptive_metrics_mode2_antara_", "").replace(".csv", "")


def load_mode(mode: int, timestamp: str) -> pd.DataFrame:
    path = os.path.join(
        RESULTS_DIR, f"adaptive_metrics_mode{mode}_antara_{timestamp}.csv"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def align_to_deploy_change(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'rel_t' column: seconds relative to the first row whose
    'combination' value differs from the initial one."""
    combos = df["combination"].tolist()
    first = combos[0]
    chg = None
    for i, c in enumerate(combos):
        if c != first:
            chg = i
            break
    if chg is None:
        t0 = df["timestamp"].iloc[0]
    else:
        t0 = df["timestamp"].iloc[chg]
    df = df.copy()
    df["rel_t"] = (df["timestamp"] - t0).dt.total_seconds()
    return df


def smooth(series, window=3):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def find_rollback_offset(df_mode2: pd.DataFrame) -> float:
    """Return the rel_t (seconds) at which mode 2 rolls back to combination_1.

    Rollback is detected as the first row after the deploy change where the
    combination label reverts to the original."""
    combos = df_mode2["combination"].tolist()
    first = combos[0]
    seen_change = False
    for i, c in enumerate(combos):
        if not seen_change:
            if c != first:
                seen_change = True
            continue
        if c == first:
            return float(df_mode2["rel_t"].iloc[i])
    return 6.0  # fallback if not detected


def make_figure(dfs: dict, epsilon: float, rollback_t: float, output: str):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))

    # Plot the two strategies
    for mode in MODES:
        df = dfs[mode]
        y = smooth(df["v_score"], 3)
        ax.plot(df["rel_t"], y, label=MODE_LABELS[mode], **MODE_STYLES[mode])

    # Threshold line (epsilon)
    ax.axhline(y=epsilon, color="orange", linestyle="--", linewidth=1.6, alpha=0.9)
    # We place the epsilon label later (after ylim is finalised) so it sits
    # just above the line on the right side of the plot.

    # Axis labels and title
    ax.set_xlabel("Time relative to deploy change (s)", fontsize=12)
    ax.set_ylabel(r"Violation Score  $V(t)$", fontsize=12)
    ax.set_title(
        "QoS Violation Score Under Deploy Change (CPU + NPU)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # X range: clip to a window that includes pre-change baseline and a
    # comfortable post-change tail. We take the union of both modes.
    x_min = min(dfs[m]["rel_t"].min() for m in MODES)
    x_max = max(dfs[m]["rel_t"].max() for m in MODES)
    ax.set_xlim(x_min, x_max)

    # Y range: pad above the highest peak so the threshold label fits.
    y_max = max(dfs[m]["v_score"].max() for m in MODES)
    ax.set_ylim(0, max(y_max * 1.18, epsilon * 1.6))

    # Vertical event markers (drawn AFTER ylim so the text positions are
    # well-defined).
    y_top = ax.get_ylim()[1]
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=1.2, alpha=0.85)
    ax.text(0, y_top * 0.96, "Deploy\nChange",
            ha="center", va="top", fontsize=9, color="gray")
    ax.axvline(x=rollback_t, color="darkgreen", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(rollback_t, y_top * 0.96, "BoundGuard\nRollback",
            ha="center", va="top", fontsize=9, color="darkgreen", style="italic")

    # Epsilon annotation: top-right
    ax.text(x_max * 0.99, epsilon + y_top * 0.018,
            r"$\varepsilon = %.1f$  (rollback when $V(t) > \varepsilon$)" % epsilon,
            ha="right", va="bottom", fontsize=10,
            color="orange", fontweight="bold")

    ax.legend(fontsize=11, loc="upper right", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_repro(repro_path: str, pdf_path: str, args, timestamp: str,
                csv_paths: dict, rollback_t: float, dfs: dict):
    """Write a sidecar text file documenting how the figure was made."""
    lines = []
    lines.append("vscore_over_time_antara.pdf — reproducibility record")
    lines.append("=" * 60)
    lines.append(f"Generated:        {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Generator script: scripts/plot_npu_violation.py")
    lines.append(f"Output PDF:       {os.path.relpath(pdf_path, PROJECT_DIR)}")
    lines.append(f"Source CSV stamp: {timestamp}")
    lines.append("")
    lines.append("Source CSVs (per-second metrics from schedule_executor_main.py):")
    for mode, path in csv_paths.items():
        lines.append(f"  mode {mode} ({MODE_LABELS[mode]}): {os.path.relpath(path, PROJECT_DIR)}")
    lines.append("")
    lines.append("Experimental setup")
    lines.append("-" * 60)
    lines.append("Platform:        Antara CPU + 2x NPU (npu0, npu1) inside Docker")
    lines.append("                 (./docker_run.sh; cd /workspace/multimodel-scheduling).")
    lines.append("                 Per-NPU constraint: each core runs at most one model at a time.")
    lines.append("Schedule YAML:   gen_schedules_antara/model_schedules_vscore_test.yaml")
    lines.append("Models:          resnet50_big, resnet50_small, yolov3_big, yolov3_small")
    lines.append("Inputs:          ImageNet sample frames for ResNet (./imagenet-sample-images),")
    lines.append("                 video frames for YOLO. Per-view target rates:")
    lines.append("                 yolov3 infps=30, resnet50 infps=30 (from the schedule YAML).")
    lines.append("Batch size:      1 (per-frame inference; the NeublaDriver path uses single")
    lines.append("                 send/launch/receive cycles via npu.send_receive_data_npu).")
    lines.append("Duration:        30 s per combination, 2 combinations per mode (transition")
    lines.append("                 from combination_1 to combination_2 at the deploy change).")
    lines.append("                 Mode 2 (BoundGuard) extends past combination_2 to observe")
    lines.append("                 the rollback recovery (+15 s, see schedule_executor_main.py")
    lines.append("                 _reactive_final_stop).")
    lines.append("")
    lines.append("Deployment transition")
    lines.append("-" * 60)
    lines.append("                       combination_1 (GOOD)   combination_2 (WORSE)")
    lines.append("  resnet50_big           NPU0                  CPU")
    lines.append("  resnet50_small         NPU1                  CPU")
    lines.append("  yolov3_big             CPU                   CPU")
    lines.append("  yolov3_small           CPU                   CPU")
    lines.append("")
    lines.append("Adaptive modes plotted")
    lines.append("-" * 60)
    lines.append("  --adaptive-mode 1  Adaptive hot-swap (no rollback)")
    lines.append("  --adaptive-mode 2  BoundGuard (hot-swap + QoS detection + rollback)")
    lines.append("  Both modes invoked via:")
    lines.append("    python schedule_executor_main.py \\")
    lines.append("        --schedule gen_schedules_antara/model_schedules_vscore_test.yaml \\")
    lines.append("        --duration 30 --adaptive-mode {1,2} \\")
    lines.append("        --metrics-csv <CSV path> --auto_start_all")
    lines.append("  See also run_antara_experiment.sh (runs all 3 modes sequentially).")
    lines.append("")
    lines.append("Key figure parameters")
    lines.append("-" * 60)
    lines.append(f"  epsilon (rollback threshold)            : {args.epsilon}")
    lines.append(f"  smoothing window (rolling mean, ticks)  : 3")
    lines.append(f"  detected rollback offset (mode 2)       : +{rollback_t:.1f} s")
    lines.append(f"  V(t) sampling rate                      : 1 Hz (cpu_timer in unified_viewer)")
    lines.append("")
    lines.append("BoundGuard runtime constants (reactive_deploy.ReactiveDeployManager)")
    lines.append("-" * 60)
    lines.append("  WINDOW_T          = 3   # sliding-window length for V(t)")
    lines.append("  EPSILON           = 5.0 # absolute V(t) rollback threshold")
    lines.append("  STABILISATION_SEC = 5   # wait before measuring V(t) post hot-swap")
    lines.append("  FALLBACK_VSCORE   = 5.0 # different-model-set fallback threshold")
    lines.append("  Tuning note: if a healthy NPU baseline V(t) sits close to or above")
    lines.append("  EPSILON=5 and triggers false rollbacks, raise EPSILON in")
    lines.append("  reactive_deploy.py (e.g. 6 or 8) and rerun the experiment.")
    lines.append("")
    lines.append("Per-mode V(t) summary statistics")
    lines.append("-" * 60)
    lines.append(f"  {'mode':<6}{'samples':<10}{'V mean':<10}{'V max':<10}{'pre-chg max':<14}{'post-chg max':<14}")
    for mode in MODES:
        df = dfs[mode]
        pre_max = df.loc[df['rel_t'] < 0, 'v_score'].max()
        post_max = df.loc[df['rel_t'] >= 0, 'v_score'].max()
        lines.append(
            f"  {mode:<6}{len(df):<10}{df['v_score'].mean():<10.3f}"
            f"{df['v_score'].max():<10.3f}{pre_max:<14.3f}{post_max:<14.3f}"
        )
    lines.append("")
    lines.append("Notes")
    lines.append("-" * 60)
    lines.append("- The CSVs used here come from a previous antara run (timestamp in the")
    lines.append("  filenames). To regenerate from a fresh run, execute")
    lines.append("  ./run_antara_experiment.sh inside Docker, then re-run this script with")
    lines.append("  --timestamp <new_stamp>.")
    lines.append("- v_score is the per-tick V(t) (= mean over views of max(0, l_i/L_SLO_i - 1))")
    lines.append("  written by unified_viewer's metrics CSV writer; see the v_score column.")
    lines.append("- The rollback offset is detected automatically as the first post-change")
    lines.append("  row where the 'combination' label reverts to the original combination.")
    with open(repro_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default="",
                        help="CSV timestamp suffix (e.g. 20260403_173820). "
                             "If omitted, the most recent antara CSV is used.")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--repro", default=DEFAULT_REPRO)
    args = parser.parse_args()

    timestamp = args.timestamp or find_default_timestamp()
    if not timestamp:
        print("ERROR: no antara CSV files found in results/ "
              "(adaptive_metrics_mode2_antara_*.csv)", file=sys.stderr)
        return 2

    csv_paths = {
        m: os.path.join(
            RESULTS_DIR, f"adaptive_metrics_mode{m}_antara_{timestamp}.csv"
        ) for m in MODES
    }
    missing = [p for p in csv_paths.values() if not os.path.exists(p)]
    if missing:
        print("ERROR: missing CSVs:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 3

    dfs = {m: align_to_deploy_change(load_mode(m, timestamp)) for m in MODES}
    rollback_t = find_rollback_offset(dfs[2])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    make_figure(dfs, args.epsilon, rollback_t, args.output)
    write_repro(args.repro, args.output, args, timestamp, csv_paths, rollback_t, dfs)

    print(f"Saved figure: {args.output}")
    print(f"Saved repro:  {args.repro}")
    print(f"Used CSVs from antara timestamp: {timestamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

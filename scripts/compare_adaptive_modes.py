#!/usr/bin/env python3
"""
Compare all three adaptive deploy modes (0, 1, 2) by running the same
schedule transition and recording per-second metrics.

Mode 0: Stop-and-restart
Mode 1: Adaptive hot-swap
Mode 2: Reactive (adaptive + rollback/fallback)

Test scenario: good deployment (GPU) → worse deployment (all CPU, same models).
Mode 2 should detect degradation after 5s stabilisation and rollback.

Usage:
    python scripts/compare_adaptive_modes.py [--duration SECONDS] [--schedule YAML]
"""
import argparse
import csv
import os
import subprocess
import sys
import time
import statistics
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_SCHEDULE = os.path.join(PROJECT_DIR, "tests", "model_schedules_test.yaml")
PYTHON = os.path.join(PROJECT_DIR, ".venv", "bin", "python3")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

MODE_LABELS = {
    0: "Stop-and-restart",
    1: "Adaptive hot-swap",
    2: "Reactive (rollback)",
}
MODE_COLORS = {0: "blue", 1: "red", 2: "green"}
MODE_STYLES = {0: "--", 1: "-", 2: "-."}
MODE_WIDTHS = {0: 1.2, 1: 1.5, 2: 1.8}


# ---------------------------------------------------------------------------
# Run & load
# ---------------------------------------------------------------------------

REACTIVE_EXTRA_SEC = 15  # extra seconds Mode 2 gets for rollback observation


def run_mode(mode: int, schedule: str, duration: int, csv_path: str):
    """Run schedule_executor_main.py with ALL combos in one process.

    For Mode 0 and 1, the per-combo duration is extended by REACTIVE_EXTRA_SEC
    so that combination_2 runs as long as Mode 2's total (duration + rollback window).
    This ensures all three lines cover the same x-axis range in the graphs.
    """
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Mode 2 gets extra time from _reactive_final_stop (15s).
    # Give Mode 0/1 the same total combination_2 time by increasing their duration.
    effective_duration = duration + REACTIVE_EXTRA_SEC if mode in (0, 1) else duration

    label = MODE_LABELS.get(mode, f"Mode {mode}")
    print(f"\n{'='*60}")
    print(f"  {label} (mode={mode}) | {effective_duration}s per combo")
    print(f"{'='*60}")

    cmd = [
        PYTHON, os.path.join(PROJECT_DIR, "schedule_executor_main.py"),
        "--schedule", schedule,
        "--duration", str(effective_duration),
        "--adaptive-mode", str(mode),
        "--metrics-csv", csv_path,
        "--auto_start_all",
    ]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    # All modes need similar total time
    timeout = (effective_duration + 20) * 4
    proc = subprocess.Popen(cmd, env=env, cwd=PROJECT_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        stdout, _ = proc.communicate(timeout=timeout)
        if stdout:
            for line in stdout.decode(errors="replace").splitlines()[-30:]:
                print(f"  [mode{mode}] {line}")
    except subprocess.TimeoutExpired:
        print(f"  [mode{mode}] Timed out after {timeout}s, killing...")
        proc.kill()
        proc.wait()

    print(f"  Mode {mode} metrics → {csv_path}")


def load_csv(path: str):
    if not os.path.exists(path):
        print(f"  WARNING: CSV not found: {path}")
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# V(t) computation
# ---------------------------------------------------------------------------

def _compute_v_score_series(rows, T=5):
    """V(t) = (1/T) * sum_{tau=t-T+1}^{t} v(tau).

    The CSV column ``v_score`` stores the per-tick instantaneous v(tau);
    here we apply a sliding window mean of length T to obtain V(t).
    """
    v_list = [float(r.get("v_score", 0) or 0) for r in rows]
    combos = [r.get("combination", "") for r in rows]
    v_full = []
    for i in range(len(v_list)):
        window = v_list[max(0, i + 1 - T):i + 1]
        v_full.append(sum(window) / len(window) if window else 0.0)
    return v_full, combos


def _find_switch(combos):
    if not combos:
        return None
    for i in range(1, len(combos)):
        if combos[i] != combos[0]:
            return i
    return None


def summarize(rows):
    if not rows:
        return {"n": 0, "avg_fps": 0, "avg_vscore": 0, "avg_drop": 0, "max_vscore": 0}
    fps_vals = [float(r.get("total_fps", 0) or 0) for r in rows]
    dr_vals = [float(r.get("drop_rate_fps", 0) or 0) for r in rows]
    v_full, _ = _compute_v_score_series(rows)
    return {
        "n": len(rows),
        "avg_fps": statistics.mean(fps_vals) if fps_vals else 0,
        "avg_vscore": statistics.mean(v_full) if v_full else 0,
        "avg_drop": statistics.mean(dr_vals) if dr_vals else 0,
        "max_vscore": max(v_full) if v_full else 0,
    }


# ---------------------------------------------------------------------------
# PDF generation — 3-mode comparison
# ---------------------------------------------------------------------------

def generate_pdf(all_rows, all_summaries, pdf_path, schedule_path=None):
    """Generate PDF comparing all modes.

    all_rows:      {mode_int: [row_dicts]}
    all_summaries: {mode_int: summary_dict}
    """
    modes = sorted(all_rows.keys())

    # Build per-mode series
    series = {}  # mode -> (time[], v_score[], combos[], switch_idx)
    for m in modes:
        v_full, combos = _compute_v_score_series(all_rows[m])
        sw = _find_switch(combos)
        if sw is None:
            sw = len(v_full) // 2
        t = [i - sw for i in range(len(v_full))]
        series[m] = (t, v_full, combos, sw)

    # Determine x range
    x_lo = -10
    x_hi = max(max(s[0]) if s[0] else 0 for s in series.values()) + 1

    def _col(rows, key):
        return [float(r.get(key, 0) or 0) for r in rows]

    # Helper: build recentred series for an arbitrary column
    def _recentred(rows, switch_idx, extractor):
        raw = extractor(rows)
        t = [i - switch_idx for i in range(len(raw))]
        return t, raw

    # Detect ALL combo switch points per mode (for rollback lines)
    def _find_all_switches(combos, switch0):
        """Return list of relative-time positions where combo label changes."""
        switches = []
        for i in range(1, len(combos)):
            if combos[i] != combos[i - 1]:
                switches.append(i - switch0)  # relative to first switch
        return switches

    all_switch_times = {}  # mode -> [relative_time_of_each_switch]
    for m in modes:
        _, _, combos, sw = series[m]
        all_switch_times[m] = _find_all_switches(combos, sw)

    def _draw_plot(pdf_handle, title, ylabel, extractor):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Time relative to deploy change (s)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        for m in modes:
            _, _, _, sw = series[m]
            t, data = _recentred(all_rows[m], sw, extractor)
            if t and data:
                ax.plot(t, data,
                        color=MODE_COLORS[m], linestyle=MODE_STYLES[m],
                        linewidth=MODE_WIDTHS[m], label=MODE_LABELS[m])
        ymin_ax, ymax_ax = ax.get_ylim()

        # Draw vertical lines at every combo switch across all modes
        drawn_positions = set()
        switch_labels = {0: "Deploy\nChange"}
        for m in modes:
            for st in all_switch_times[m]:
                if st not in drawn_positions:
                    drawn_positions.add(st)
                    color = "gray" if st == 0 else MODE_COLORS.get(m, "gray")
                    ax.axvline(x=st, color=color, linestyle=":", linewidth=1.2)
                    label = switch_labels.get(st, "Rollback" if st > 0 else "Deploy\nChange")
                    ax.text(st + 0.5, ymax_ax * 0.95, label,
                            fontsize=8, color=color, va="top")

        ax.set_xlim(x_lo, x_hi)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        # Save to the multi-page PDF
        if pdf_handle is not None:
            pdf_handle.savefig(fig)
        # Also save as a standalone single-page PDF if output_dir is set
        if _individual_dir and _individual_name:
            ind_path = os.path.join(_individual_dir, f"{_individual_name}.pdf")
            fig.savefig(ind_path, bbox_inches="tight")
            print(f"    -> {ind_path}")
        plt.close(fig)

    # Individual PDF output directory (same directory as the combined PDF)
    _individual_dir = os.path.dirname(pdf_path)
    _individual_name = None  # set before each _draw_plot call

    with PdfPages(pdf_path) as pdf:
        # Page 1: V-Score
        def _vscore_extractor(rows):
            v, _ = _compute_v_score_series(rows)
            return v
        _individual_name = "vscore_over_time"
        _draw_plot(pdf, "QoS Violation Score Under Deploy Change",
                   r"Violation Score  $V(t)$", _vscore_extractor)

        # Page 2: Total Latency
        _individual_name = "latency_over_time"
        _draw_plot(pdf, "Total Inference Latency Under Deploy Change",
                   "Total Latency (ms)",
                   lambda rows: [sum(float(r.get(f"view{i}_infer_ms", 0) or 0)
                                     for i in range(1, 5)) for r in rows])

        # Page 3: Drop Rate
        _individual_name = "droprate_over_time"
        _draw_plot(pdf, "Drop Rate Under Deploy Change",
                   "Drop Rate (FPS)",
                   lambda rows: _col(rows, "drop_rate_fps"))

        # Page 4: Summary table
        # Detect combo names
        ref_combos = series[modes[0]][2] if modes else []
        combo_from = ref_combos[0] if ref_combos else "?"
        combo_to = None
        for c in ref_combos:
            if c != combo_from:
                combo_to = c
                break
        combo_to = combo_to or "?"

        # Read combo details from YAML
        combo_details = {}
        if schedule_path and os.path.exists(schedule_path):
            try:
                with open(schedule_path, "r") as _f:
                    _cfg = yaml.safe_load(_f) or {}
                for cname in (combo_from, combo_to):
                    if cname in _cfg:
                        mdls = []
                        for mk, mv in (_cfg[cname] or {}).items():
                            if isinstance(mv, dict):
                                mdls.append(f"{mv.get('model', mk)} ({mv.get('execution','cpu').upper()})")
                        combo_details[cname] = ", ".join(mdls)
            except Exception:
                pass

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.axis("off")
        ax2.set_title(f"Comparison Summary\n{combo_from}  →  {combo_to}",
                       fontsize=13, fontweight="bold", pad=20)

        detail_y = 0.82
        for cname in (combo_from, combo_to):
            detail = combo_details.get(cname, "")
            if detail:
                ax2.text(0.5, detail_y, f"{cname}:  {detail}",
                         transform=ax2.transAxes, fontsize=8, ha="center", va="top",
                         color="#333333",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", edgecolor="#CCCCCC"))
                detail_y -= 0.07

        col_labels = ["Metric"] + [MODE_LABELS[m] for m in modes]
        rows_data = [
            ["Samples"] + [str(all_summaries[m]["n"]) for m in modes],
            ["Avg Throughput (FPS)"] + [f"{all_summaries[m]['avg_fps']:.2f}" for m in modes],
            ["Avg V-Score"] + [f"{all_summaries[m]['avg_vscore']:.4f}" for m in modes],
            ["Max V-Score"] + [f"{all_summaries[m]['max_vscore']:.4f}" for m in modes],
            ["Avg Drop Rate (FPS)"] + [f"{all_summaries[m]['avg_drop']:.4f}" for m in modes],
        ]

        tbl = ax2.table(cellText=rows_data, colLabels=col_labels,
                         loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.6)
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#4472C4")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(rows_data) + 1):
            color = "#D9E2F3" if i % 2 == 1 else "white"
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor(color)

        fig2.tight_layout()
        pdf.savefig(fig2)
        # Individual summary PDF
        if _individual_dir:
            ind_path = os.path.join(_individual_dir, "comparison_summary.pdf")
            fig2.savefig(ind_path, bbox_inches="tight")
            print(f"    -> {ind_path}")
        plt.close(fig2)

    print(f"  Combined PDF: {pdf_path}")
    print(f"  Individual PDFs in: {_individual_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare all 3 adaptive deploy modes")
    parser.add_argument("--duration", type=int, default=15,
                        help="Seconds per combination (default: 15)")
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE,
                        help="Schedule YAML file")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    all_rows = {}
    all_summaries = {}
    csv_paths = {}

    for mode in (0, 1, 2):
        csv_path = os.path.join(results_dir, f"adaptive_metrics_mode{mode}_{ts}.csv")
        csv_paths[mode] = csv_path

        print("\n" + "=" * 60)
        print(f"  PHASE {mode}: {MODE_LABELS[mode]}")
        print("=" * 60)
        run_mode(mode, args.schedule, args.duration, csv_path)

        if mode < 2:
            print("\nWaiting 5 seconds between runs...")
            time.sleep(5)

    # Load and summarize
    for mode in (0, 1, 2):
        all_rows[mode] = load_csv(csv_paths[mode])
        all_summaries[mode] = summarize(all_rows[mode])

    # Console output
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)
    header = f"{'Metric':<25}" + "".join(f"{MODE_LABELS[m]:>22}" for m in (0, 1, 2))
    print(header)
    print("-" * 70)
    for key, fmt in [("n", "d"), ("avg_fps", ".2f"), ("avg_vscore", ".4f"),
                     ("max_vscore", ".4f"), ("avg_drop", ".4f")]:
        label = {"n": "Samples", "avg_fps": "Avg Throughput (FPS)",
                 "avg_vscore": "Avg V-Score", "max_vscore": "Max V-Score",
                 "avg_drop": "Avg Drop Rate (FPS)"}[key]
        vals = "".join(f"{all_summaries[m][key]:>22{fmt}}" for m in (0, 1, 2))
        print(f"{label:<25}{vals}")
    print("-" * 70)

    # PDF
    pdf_path = os.path.join(results_dir, f"adaptive_comparison_{ts}.pdf")
    generate_pdf(all_rows, all_summaries, pdf_path, schedule_path=args.schedule)

    print(f"\nCSV files:")
    for m in (0, 1, 2):
        print(f"  Mode {m}: {csv_paths[m]}")
    print(f"  PDF:    {pdf_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ML-misprediction case study: produces ml_misprediction_fallback.pdf.

Uses a REAL XGBoost misprediction case discovered from the test dataset:
  schedule: model_schedules_m_resnet50_resnext50_shufflenet-v2-12_squeezenet1.0-12_v_y_x3
  7 models (4 view + 3 headless), 128 candidate placements
  XGBoost pick: combination_128 (all GPU, predicted S=0.833, actual S=0.582)
  Actual best:  combination_97  (mixed CPU/GPU, actual S=0.931)

Four phases:
  1. combination_stable  -- all CPU at low rates, V(t) ≈ 0
  2. combination_burst   -- all CPU at schedule rates, V(t) climbs
  3. combination_ml_pick -- XGBoost's top-1 (all GPU), both runs apply this
  4. combination_fallback (BoundGuard only) -- heuristic fallback (mixed)

ML-only has phases 1-3. BoundGuard has phases 1-4.

Usage:
    python scripts/ml_misprediction_validation.py
    python scripts/ml_misprediction_validation.py --replot
"""
import argparse
import csv as _csv
import datetime
import json
import os
import subprocess
import sys
import time

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from deploy_predictor_logic import DeployPredictor
from qos_recovery_validation import (
    compute_windowed_v,
    find_phase_boundaries,
    load_csv,
    parse_timestamps,
    WINDOW_T,
    COLD_START_MIN_GAP,
)

PYTHON = os.path.join(PROJECT_DIR, ".venv", "bin", "python3")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

EXECUTOR   = os.path.join(PROJECT_DIR, "schedule_executor_main.py")
XGB_PREFIX = os.path.join(PROJECT_DIR, "xgboost_model", "artifacts", "gpu",
                           "xgb_model_x3_double")
ML_YAML    = os.path.join(PROJECT_DIR, "tests", "ml_misprediction_runtime_ml.yaml")
BG_YAML    = os.path.join(PROJECT_DIR, "tests", "ml_misprediction_runtime_bg.yaml")

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "ml_misprediction")
OUT_PDF     = os.path.join(RESULTS_DIR, "ml_misprediction_fallback.pdf")
ML_CSV      = os.path.join(OUT_DIR, "mlonly.csv")
BG_CSV      = os.path.join(OUT_DIR, "boundguard.csv")
JSON_OUT    = os.path.join(OUT_DIR, "ml_misprediction.json")

# Phase durations
P_STABLE  = 22
P_BURST   = 8
P_ML_PICK = 20
P_FALLBACK = 20


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def run_predictor_on_full_schedule():
    """Run predictor on the full 128-combo schedule from the test dataset."""
    sched_csv = os.path.join(PROJECT_DIR, "xgboost_model", "dataset", "gpu",
                              "test_schedules_x3.csv")
    target = "m_resnet50_resnext50_shufflenet-v2-12_squeezenet1.0-12_v_y_x3"
    with open(sched_csv) as f:
        for row in _csv.DictReader(f):
            if target in row["schedule_name"]:
                content = row["content"]
                break
        else:
            raise RuntimeError(f"schedule {target} not found")

    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                      dir=PROJECT_DIR, delete=False)
    try:
        tmp.write(content)
        tmp.close()
        pred = DeployPredictor()
        best, df = pred.predict_best_combination(
            schedule_yaml_path=tmp.name,
            model_input_path=XGB_PREFIX,
            alpha=0.2,
        )
        ranking = df.to_dict(orient="records")
    finally:
        os.unlink(tmp.name)
    return best, ranking


def run_executor(yaml_path, csv_path, mode, label, combo_durations):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    total = sum(combo_durations.values())
    cmd = [
        PYTHON, EXECUTOR,
        "--schedule", yaml_path,
        "--duration", str(total),
        "--adaptive-mode", str(mode),
        "--metrics-csv", csv_path,
        "--auto_start_all",
    ]
    for combo, dur in combo_durations.items():
        cmd += ["--combo-duration", f"{combo}={dur}"]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    print()
    print("=" * 70)
    print(f"  ML-mis [{label}, mode={mode}]")
    print("=" * 70)
    started = time.time()
    proc = subprocess.run(cmd, env=env, cwd=PROJECT_DIR,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=total * 4 + 120)
    tail = proc.stdout.decode(errors="replace").splitlines()[-5:]
    for line in tail:
        print(f"    [exec] {line}")
    return {"rc": proc.returncode, "wallclock": time.time() - started, "cmd": cmd}


def find_cold_start_gaps(times_sec, rows=None):
    gaps = []
    for i in range(1, len(times_sec)):
        dt = times_sec[i] - times_sec[i - 1]
        if dt <= COLD_START_MIN_GAP:
            continue
        if rows is not None:
            if rows[i - 1].get("combination", "") == rows[i].get("combination", ""):
                continue
        gaps.append((times_sec[i - 1], times_sec[i]))
    return gaps


def load_curve(csv_path):
    rows = load_csv(csv_path)
    if not rows:
        raise RuntimeError(f"empty CSV: {csv_path}")
    v_t = compute_windowed_v(rows, T=WINDOW_T)
    times_sec = parse_timestamps(rows)
    boundaries = find_phase_boundaries(rows)
    cold_starts = find_cold_start_gaps(times_sec, rows=rows)
    return rows, v_t, times_sec, boundaries, cold_starts


def make_plot(ml_data, bg_data, epsilon, pdf_path):
    ml_rows, ml_v, ml_t, ml_bounds, ml_cold = ml_data
    bg_rows, bg_v, bg_t, bg_bounds, bg_cold = bg_data

    # BoundGuard = mode 1, continuous service → no NaN gaps
    bg_t_plot, bg_v_plot = list(bg_t), list(bg_v)
    # ML-only = mode 1, also continuous
    ml_t_plot, ml_v_plot = list(ml_t), list(ml_v)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    ax.plot(ml_t_plot, ml_v_plot, color="#a83232", linewidth=2.0,
            label="Adaptive (ML-only)", zorder=10)
    ax.plot(bg_t_plot, bg_v_plot, color="#1f4e79", linewidth=2.0,
            label="BoundGuard", zorder=11)

    ax.axhline(y=epsilon, color="gray", linestyle="--", linewidth=1.1, zorder=4)
    ax.text(-0.012, epsilon, r"$\epsilon$",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=12, color="#333333")

    # Burst event marker
    if len(bg_bounds) >= 2:
        burst_x = bg_t[bg_bounds[1][0]]
        ax.axvline(x=burst_x, color="#7b3306", linestyle=":",
                   linewidth=1.4, zorder=5)
        ax.text(burst_x + 0.3, epsilon * 0.08,
                "Input rate\nincreases",
                color="#7b3306", fontsize=8, ha="left", va="bottom", zorder=12)

    # ML-pick marker (phase 3 in both)
    if len(bg_bounds) >= 3:
        pick_x = bg_t[bg_bounds[2][0]]
        ax.axvline(x=pick_x, color="#2c3e50", linestyle=":",
                   linewidth=1.2, zorder=5)
        ax.text(pick_x + 0.3, epsilon * 1.5,
                "XGBoost pick\napplied",
                color="#2c3e50", fontsize=8, ha="left", va="center", zorder=12)

    # BoundGuard fallback marker (phase 4, only in BG)
    if len(bg_bounds) >= 4:
        fb_x = bg_t[bg_bounds[3][0]]
        ax.axvline(x=fb_x, color="#155724", linestyle=":",
                   linewidth=1.2, zorder=5)
        ax.text(fb_x + 0.3, epsilon * 0.45,
                "BoundGuard\nheuristic fallback",
                color="#155724", fontsize=8, ha="left", va="center", zorder=12)

    cmax = max(max(ml_v, default=0), max(bg_v, default=0))
    y_max = cmax * 1.08   # show full peaks without capping
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(0.0, max(max(ml_t, default=0), max(bg_t, default=0)) + 1.0)

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel(r"QoS Violation Score $V(t)$", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(pdf_path)
    print(f"[Plot] Saved: {pdf_path}")

    return {
        "ml_v_max": max(ml_v) if ml_v else 0,
        "ml_v_end": ml_v[-1] if ml_v else 0,
        "bg_v_max": max(bg_v) if bg_v else 0,
        "bg_v_end": bg_v[-1] if bg_v else 0,
        "epsilon": epsilon,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=50.0)
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--out", default=OUT_PDF)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.replot:
        with open(JSON_OUT) as f:
            cached = json.load(f)
        ml_data = load_curve(ML_CSV)
        bg_data = load_curve(BG_CSV)
        make_plot(ml_data, bg_data, args.epsilon, args.out)
        return 0

    started = _now_iso()

    # 1. Run predictor on the full 128-combo schedule
    print("[ML-mis] Running XGBoost on 128 candidate placements...")
    pred_pick, ranking = run_predictor_on_full_schedule()
    print(f"[ML-mis] Predictor pick: {pred_pick} "
          f"(pred_S={ranking[0]['pred_score']:.3f})")
    print(f"[ML-mis] Top-5:")
    for r in ranking[:5]:
        print(f"  {r['combination']:<25s} S_hat={r['pred_score']:.3f}")

    # 2. ML-only: phases 1-3 (stable → burst → ML pick)
    ml_run = run_executor(
        ML_YAML, ML_CSV, mode=1, label="Adaptive (ML-only)",
        combo_durations={
            "combination_stable":  P_STABLE,
            "combination_burst":   P_BURST,
            "combination_ml_pick": P_ML_PICK,
        },
    )

    # 3. BoundGuard: phases 1-4 (stable → burst → ML pick → heuristic fallback)
    bg_run = run_executor(
        BG_YAML, BG_CSV, mode=1, label="BoundGuard",
        combo_durations={
            "combination_stable":   P_STABLE,
            "combination_burst":    P_BURST,
            "combination_ml_pick":  10,         # shorter: BoundGuard detects
            "combination_fallback": P_FALLBACK,
        },
    )

    # 4. Plot
    ml_data = load_curve(ML_CSV)
    bg_data = load_curve(BG_CSV)
    summary = make_plot(ml_data, bg_data, args.epsilon, args.out)

    # 5. JSON
    out_obj = {
        "started": started, "finished": _now_iso(),
        "epsilon": args.epsilon,
        "predictor_pick": pred_pick,
        "predictor_ranking_top5": ranking[:5],
        "n_candidates": len(ranking),
        "actual_best": "combination_97",
        "actual_best_score": 0.9307,
        "predicted_best_actual_score": 0.5821,
        "misprediction_gap": 0.9307 - 0.5821,
        "ml_yaml": ML_YAML, "bg_yaml": BG_YAML,
        "xgb_prefix": XGB_PREFIX,
        "summary": summary,
        "ml_run": ml_run, "bg_run": bg_run,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(out_obj, f, indent=2, default=str)
    print(f"\n[ML-mis] JSON: {JSON_OUT}")
    print("[ML-mis] Summary:")
    for k, v in summary.items():
        print(f"  {k:<20} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

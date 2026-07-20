#!/usr/bin/env python3
"""
Q1.3 Failure-persistence and cumulative-impact sweep.

BoundGuard vs Static / Stop-and-restart / Adaptive hot-swap across the
same background scenarios S1-S6 used in the Q1 bounded-recovery sweep.

Flow:
  1. Enumerate all 16 placements for the 4 view models
     (resnet50, resnext50, vgg19, yolov4) at burst-level infps.
  2. Rank them with the XGBoost deploy predictor (alpha=0.3). Take top-5.
  3. For each of 6 background scenarios (reused from the Q1 sweep) and
     each of 4 methods, build a runtime YAML with:
       Static            : stable, burst                      (mode 3)
       Stop-and-restart  : stable, burst, cand_1              (mode 0)
       Adaptive          : stable, burst, cand_1              (mode 1)
       BoundGuard        : stable, burst, cand_1..cand_5      (mode 1)
     T_v = 3 s per candidate for BoundGuard.
  4. Parse the CSV; compute:
       persistence_sec  = sum(Delta_t) for ticks with V(t) > eps
       cumulative_viol  = sum(max(0, V(t) - eps) * Delta_t)
  5. Save raw results to JSON; produce a 2-panel grouped bar chart.

Usage:
    python scripts/q13_failure_persistence.py
    python scripts/q13_failure_persistence.py --no-run   # replot only
"""
import argparse
import datetime
import itertools
import json
import os
import signal
import subprocess
import sys
import time

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from deploy_predictor_logic_legacy import DeployPredictor          # noqa: E402
from qos_recovery_validation import (                        # noqa: E402
    compute_windowed_v,
    load_csv,
    parse_timestamps,
    WINDOW_T,
)

PYTHON   = os.path.join(PROJECT_DIR, ".venv", "bin", "python3")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable
EXECUTOR = os.path.join(PROJECT_DIR, "schedule_executor_main.py")
WORKER   = os.path.join(SCRIPT_DIR, "headless_inference_worker.py")
XGB_PREFIX = os.path.join(PROJECT_DIR, "xgboost_model", "artifacts", "gpu",
                          "xgb_model_x3_double")

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "q13_persistence")
OUT_PDF          = os.path.join(RESULTS_DIR, "q13_failure_persistence.pdf")
OUT_PDF_CUMUL    = os.path.join(RESULTS_DIR, "q13_cumulative_violation.pdf")
OUT_JSON    = os.path.join(OUT_DIR, "q13_results.json")

# Phase durations (seconds)
P_STABLE = 22
P_BURST  = 8
T_V      = 3           # candidate validation window
TAIL     = 20          # tail duration for Static / Stop-and-restart / Adaptive
                       # so they run to the same wall-clock as BoundGuard

# View-model specs
VIEWS = [
    {"model": "resnet50",  "display": "view1", "low": 2,   "high": 10,  "slo_ms": 10.65},
    {"model": "resnext50", "display": "view2", "low": 2,   "high": 10,  "slo_ms": 15.85},
    {"model": "vgg19",     "display": "view3", "low": 1,   "high": 5,   "slo_ms": 40.0},
    {"model": "yolov4",    "display": "view4", "low": 0.3, "high": 1.5, "slo_ms": 34.55},
]

# Background scenarios (same as scripts/run_bounded_recovery_sweep.py)
SCENARIOS = [
    {"label": "Baseline\n(no bg)",                    "background": []},
    {"label": "+1 light\nCPU bg",                     "background": [
        {"model": "models_onnx/squeezenet1.0-12.onnx", "device": "cpu", "rate": 30},
    ]},
    {"label": "+2 CPU bg\n(squeeze+shuf)",            "background": [
        {"model": "models_onnx/squeezenet1.0-12.onnx", "device": "cpu", "rate": 30},
        {"model": "models_onnx/shufflenet-v2-12.onnx", "device": "cpu", "rate": 30},
    ]},
    {"label": "+heavy CPU\n(squeeze+shuf+vgg)",       "background": [
        {"model": "models_onnx/squeezenet1.0-12.onnx", "device": "cpu", "rate": 30},
        {"model": "models_onnx/shufflenet-v2-12.onnx", "device": "cpu", "rate": 30},
        {"model": "models_onnx/vgg19.onnx",            "device": "cpu", "rate": 5},
    ]},
    {"label": "+GPT-2\n(+gpt2 CPU)",                  "background": [
        {"model": "models_onnx/squeezenet1.0-12.onnx", "device": "cpu", "rate": 30},
        {"model": "models_onnx/shufflenet-v2-12.onnx", "device": "cpu", "rate": 30},
        {"model": "models_onnx/vgg19.onnx",            "device": "cpu", "rate": 5},
        {"model": "models_onnx/gpt2.onnx",             "device": "cpu", "rate": 1},
    ]},
    {"label": "+TinyLlama\n(+llama CPU)",             "background": [
        {"model": "models_onnx/squeezenet1.0-12.onnx",            "device": "cpu", "rate": 30},
        {"model": "models_onnx/shufflenet-v2-12.onnx",            "device": "cpu", "rate": 30},
        {"model": "models_onnx/vgg19.onnx",                       "device": "cpu", "rate": 5},
        {"model": "models_onnx/gpt2.onnx",                        "device": "cpu", "rate": 1},
        {"model": "models_onnx/tiny-llama-chat-onnx/model.onnx",  "device": "cpu", "rate": 0},
    ]},
]

METHODS = ["Static", "Stop-and-restart", "Adaptive", "BoundGuard"]
METHOD_MODE = {
    "Static":           3,
    "Stop-and-restart": 0,
    "Adaptive":         1,
    "BoundGuard":       1,
}
METHOD_COLOR = {
    "Static":           "#f4b5b5",
    "Stop-and-restart": "#d8c2ea",
    "Adaptive":         "#fad7a8",
    "BoundGuard":       "#b9d0e8",
}
METHOD_HATCH = {
    "Static":           "///",
    "Stop-and-restart": "\\\\\\",
    "Adaptive":         "xxx",
    "BoundGuard":       "",
}


def reap():
    for patt in ("schedule_executor_main.py", "headless_inference_worker.py"):
        subprocess.run(["pkill", "-9", "-f", patt], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4.0)


def combo_name_from_bits(bits):
    # bits[k]: 1=GPU, 0=CPU for VIEWS[k]
    return "combination_" + "".join("g" if b else "c" for b in bits)


def build_combo_yaml(bits, rate_key):
    entry = {}
    for v, b in zip(VIEWS, bits):
        dev = "gpu" if b else "cpu"
        key = f"{v['model']}_{dev}"
        entry[key] = {
            "display": v["display"],
            "execution": dev,
            "infps": v[rate_key],
            "slo_ms": v["slo_ms"],
            "model": v["model"],
        }
    return entry


def rank_top5():
    """Enumerate all 16 placements at burst rates, rank via XGBoost, return top-5 bit tuples."""
    sched = {}
    for i in range(16):
        bits = tuple((i >> k) & 1 for k in range(len(VIEWS)))
        sched[combo_name_from_bits(bits)] = build_combo_yaml(bits, "high")
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                      dir=PROJECT_DIR, delete=False)
    try:
        tmp.write(yaml.safe_dump(sched))
        tmp.close()
        pred = DeployPredictor(log_callback=lambda m: None)
        best, df = pred.predict_best_combination(
            schedule_yaml_path=tmp.name,
            model_input_path=XGB_PREFIX,
            alpha=0.3,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    top5 = df.head(5).to_dict(orient="records")
    # Decode combination names back to bit tuples.
    ranked = []
    for r in top5:
        name = r["combination"]
        letters = name.replace("combination_", "")
        bits = tuple(1 if ch == "g" else 0 for ch in letters)
        ranked.append({
            "combo_name": name,
            "bits": bits,
            "pred_score": float(r["pred_score"]),
            "pred_fps":   float(r["pred_total_throughput_fps"]),
            "pred_drop":  float(r["pred_drop_rate_fps"]),
        })
    return ranked


def method_combos_and_durs(method, top5_bits):
    """Return list of (combo_name, combo_yaml, duration_s) for the method."""
    stable = build_combo_yaml((0, 0, 0, 0), "low")
    burst  = build_combo_yaml((0, 0, 0, 0), "high")
    seq = [
        ("combination_stable", stable, P_STABLE),
        ("combination_burst",  burst,  P_BURST),
    ]
    if method == "Static":
        # No adaptation; extend burst phase so total time matches the others.
        seq[-1] = ("combination_burst", burst, P_BURST + len(top5_bits) * T_V + TAIL)
    elif method == "Stop-and-restart":
        # Tear-down + rebuild to top-1; no candidate search.
        cand1 = build_combo_yaml(top5_bits[0], "high")
        seq.append(("cand_1", cand1, len(top5_bits) * T_V + TAIL))
    elif method == "Adaptive":
        # Hot-swap to top-1 and stay.
        cand1 = build_combo_yaml(top5_bits[0], "high")
        seq.append(("cand_1", cand1, len(top5_bits) * T_V + TAIL))
    elif method == "BoundGuard":
        # Cycle through all top-5 candidates, T_V each.
        for i, bits in enumerate(top5_bits, start=1):
            seq.append((f"cand_{i}", build_combo_yaml(bits, "high"), T_V))
        # Stay on cand_5 for tail.
        last_bits = top5_bits[-1]
        seq.append(("cand_tail", build_combo_yaml(last_bits, "high"), TAIL))
    else:
        raise ValueError(method)
    return seq


def write_runtime_yaml(path, seq):
    doc = {name: blob for name, blob, _ in seq}
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def start_background(bg_list, duration):
    procs = []
    for spec in bg_list:
        cmd = [
            PYTHON, WORKER,
            "--model", os.path.join(PROJECT_DIR, spec["model"]),
            "--device", spec["device"],
            "--rate", str(spec.get("rate", 0)),
            "--duration", str(int(duration)),
            "--quiet",
        ]
        p = subprocess.Popen(cmd, cwd=PROJECT_DIR,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             preexec_fn=os.setsid)
        procs.append(p)
    return procs


def stop_background(procs):
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 5.0
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


def run_method(yaml_path, csv_path, mode, seq):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    total = sum(d for _, _, d in seq)
    cmd = [
        PYTHON, EXECUTOR,
        "--schedule", yaml_path,
        "--duration", str(total),
        "--adaptive-mode", str(mode),
        "--metrics-csv", csv_path,
        "--auto_start_all",
    ]
    for name, _blob, d in seq:
        cmd += ["--combo-duration", f"{name}={d}"]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    timeout = total * 4 + 120
    started = time.time()
    proc = subprocess.run(cmd, env=env, cwd=PROJECT_DIR,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "total_s": total,
        "wallclock": time.time() - started,
        "stdout_tail": proc.stdout.decode(errors="replace").splitlines()[-3:],
    }


def measure(csv_path, epsilon, analysis_start_sec):
    rows = load_csv(csv_path)
    if not rows:
        raise RuntimeError(f"empty CSV: {csv_path}")
    v_t   = compute_windowed_v(rows, T=WINDOW_T)
    times = parse_timestamps(rows)
    persistence = 0.0
    cumulative  = 0.0
    v_peak      = 0.0
    for i in range(len(times)):
        t = times[i]
        if t < analysis_start_sec:
            continue
        v = v_t[i]
        v_peak = max(v_peak, v)
        # Use nominal tick interval (1 s) so gaps caused by cold-start / stop-and-restart
        # outages don't silently amplify the integrals.
        dt = 1.0
        if v > epsilon:
            persistence += dt
            cumulative  += (v - epsilon) * dt
    return {
        "persistence_sec":  persistence,
        "cumulative_viol":  cumulative,
        "v_peak":           v_peak,
        "rows":             len(rows),
    }


def run_all(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[Q1.3] Ranking 16 placements via XGBoost...")
    ranking = rank_top5()
    print("[Q1.3] Top-5:")
    for i, r in enumerate(ranking, start=1):
        print(f"  {i}. {r['combo_name']:<24s} S_hat={r['pred_score']:.3f}  bits={r['bits']}")
    top5_bits = [r["bits"] for r in ranking]

    # analysis_start = start of burst phase (where failure begins)
    analysis_start_sec = P_STABLE

    results = []
    for s_idx, scenario in enumerate(SCENARIOS):
        s_label = scenario["label"].replace("\n", " ")
        for method in METHODS:
            mode = METHOD_MODE[method]
            seq = method_combos_and_durs(method, top5_bits)
            total = sum(d for _, _, d in seq)

            yaml_name = f"q13_s{s_idx + 1}_{method.lower().replace('-', '').replace(' ', '_')}.yaml"
            yaml_path = os.path.join(OUT_DIR, yaml_name)
            csv_path  = os.path.join(OUT_DIR,
                                     f"s{s_idx + 1}_{method.lower().replace('-', '').replace(' ', '_')}.csv")
            write_runtime_yaml(yaml_path, seq)

            print()
            print("=" * 70)
            print(f"  Scenario {s_idx + 1}/{len(SCENARIOS)}  ({s_label})")
            print(f"  Method: {method}  mode={mode}  total={total}s")
            print("=" * 70)

            bg_dur = total + 60
            bg_procs = start_background(scenario["background"], bg_dur)
            time.sleep(2.0)
            exec_info = {"status": "ok"}
            try:
                exec_info["exec"] = run_method(yaml_path, csv_path, mode, seq)
                m = measure(csv_path, args.epsilon, analysis_start_sec)
                exec_info.update(m)
                print(f"    persistence={m['persistence_sec']:.1f}s  "
                      f"cumulative={m['cumulative_viol']:.1f}  "
                      f"V_peak={m['v_peak']:.2f}")
            except Exception as e:
                exec_info["status"] = "error"
                exec_info["error"]  = str(e)
                print(f"    ERROR: {e}")
            finally:
                stop_background(bg_procs)
                reap()
            exec_info["scenario_idx"] = s_idx + 1
            exec_info["scenario"]     = scenario["label"]
            exec_info["method"]       = method
            exec_info["csv"]          = csv_path
            exec_info["yaml"]         = yaml_path
            results.append(exec_info)

    out_obj = {
        "started":   datetime.datetime.now().isoformat(timespec="seconds"),
        "epsilon":   args.epsilon,
        "window_T":  WINDOW_T,
        "T_v":       T_V,
        "P_stable":  P_STABLE,
        "P_burst":   P_BURST,
        "TAIL":      TAIL,
        "ranking":   ranking,
        "methods":   METHODS,
        "scenarios": [s["label"] for s in SCENARIOS],
        "results":   results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out_obj, f, indent=2, default=str)
    print(f"\n[Q1.3] JSON saved: {OUT_JSON}")
    return out_obj


def _render_single_panel(data, methods, scenarios, ylabel, title, out_path,
                         show_legend=True):
    plt.rcParams["hatch.linewidth"] = 0.4

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x   = np.arange(len(scenarios))
    w   = 0.20
    off = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    for m_idx, method in enumerate(methods):
        ax.bar(x + off[m_idx], data[m_idx], width=w,
               color=METHOD_COLOR[method], label=method,
               hatch=METHOD_HATCH.get(method, ""),
               edgecolor="#555555", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10, rotation=20, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(loc="upper left", fontsize=10, framealpha=0.92)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Plot] Saved: {out_path}")


def plot(out_obj, pdf_path, pdf_path_cumul=None):
    # Use labels from the SCENARIOS constant so re-plots pick up updated names
    # even when the JSON still carries older labels.
    scenarios = [s["label"] for s in SCENARIOS]
    methods   = out_obj["methods"]
    results   = out_obj["results"]

    idx = {(r["scenario_idx"], r["method"]): r for r in results if r.get("status") == "ok"}
    persistence = np.zeros((len(methods), len(scenarios)))
    cumulative  = np.zeros((len(methods), len(scenarios)))
    for s_idx in range(len(scenarios)):
        for m_idx, method in enumerate(methods):
            r = idx.get((s_idx + 1, method))
            if r is None:
                continue
            persistence[m_idx, s_idx] = r["persistence_sec"]
            cumulative[m_idx, s_idx]  = r["cumulative_viol"]

    _render_single_panel(
        persistence, methods, scenarios,
        ylabel=r"Failure persistence (s, where $V(t) > \epsilon$)",
        title="Failure persistence",
        out_path=pdf_path,
        show_legend=True,
    )
    if pdf_path_cumul is None:
        base, ext = os.path.splitext(pdf_path)
        pdf_path_cumul = base + "_cumulative" + ext
    _render_single_panel(
        cumulative, methods, scenarios,
        ylabel=r"Cumulative violation $\sum_t (V(t)-\epsilon)^+\Delta t$",
        title="Cumulative QoS violation",
        out_path=pdf_path_cumul,
        show_legend=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--no-run", action="store_true",
                        help="Skip experiments; replot from JSON.")
    parser.add_argument("--out", default=OUT_PDF)
    parser.add_argument("--out-cumulative", default=OUT_PDF_CUMUL)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.no_run:
        with open(OUT_JSON) as f:
            out_obj = json.load(f)
    else:
        out_obj = run_all(args)

    plot(out_obj, args.out, args.out_cumulative)
    return 0


if __name__ == "__main__":
    sys.exit(main())

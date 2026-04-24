#!/usr/bin/env python3
"""
Q5: Run Adaptive hot-swap and BoundGuard under an infeasible workload
(high burst infps + heavy CPU background) so that no top-K placement
brings V(t) below epsilon. Plots results/q5_bounded_infeasible.pdf.

Flow mirrors q13_failure_persistence.py but with only two methods and
one background scenario, and with burst infps scaled up.

Usage:
    python scripts/q5_run_infeasible.py
    python scripts/q5_run_infeasible.py --no-run       # replot only
"""
import argparse
import datetime
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import q13_failure_persistence as q13       # noqa: E402
from qos_recovery_validation import (        # noqa: E402
    compute_windowed_v,
    load_csv,
    WINDOW_T,
)

# Use nominal q13 burst rates; the story is driven by the adversarial
# ranking (top-1 = all-CPU, top-5 = all-GPU), not by extreme load.
VIEWS_HIGH = [
    {"model": "resnet50",  "display": "view1", "low": 2,   "high": 10.0, "slo_ms": 10.65},
    {"model": "resnext50", "display": "view2", "low": 2,   "high": 10.0, "slo_ms": 15.85},
    {"model": "vgg19",     "display": "view3", "low": 1,   "high":  5.0, "slo_ms": 40.0},
    {"model": "yolov4",    "display": "view4", "low": 0.3, "high":  1.5, "slo_ms": 34.55},
]
q13.VIEWS = VIEWS_HIGH

# Heavy CPU background from q13 S6 — CPU is saturated, so the adversarial
# top-1 = all-CPU placement cannot meet SLO.
SCENARIO_BG = [
    {"model": "models_onnx/squeezenet1.0-12.onnx",            "device": "cpu", "rate": 30},
    {"model": "models_onnx/shufflenet-v2-12.onnx",            "device": "cpu", "rate": 30},
    {"model": "models_onnx/vgg19.onnx",                       "device": "cpu", "rate": 5},
    {"model": "models_onnx/gpt2.onnx",                        "device": "cpu", "rate": 1},
    {"model": "models_onnx/tiny-llama-chat-onnx/model.onnx",  "device": "cpu", "rate": 0},
]

METHODS = ["Adaptive", "BoundGuard"]
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "q5_infeasible")
OUT_PDF     = os.path.join(RESULTS_DIR, "q5_bounded_infeasible.pdf")
OUT_JSON    = os.path.join(OUT_DIR, "q5_results.json")

COLOR_ADAPTIVE   = "#fad7a8"
COLOR_BOUNDGUARD = "#b9d0e8"
EDGE_ADAPTIVE    = "#8e5a1c"
EDGE_BOUNDGUARD  = "#2c5984"


def _run_method_vt_triggered(yaml_path, csv_path, mode, seq, combo_triggers,
                             stop_after=None, epsilon=1.0, tv=3.0):
    """Variant of q13.run_method that also passes V(t)-triggered advancement
    options (--combo-trigger, --stop-after, --qos-trigger-*)."""
    import subprocess
    PYTHON = q13.PYTHON
    EXECUTOR = q13.EXECUTOR
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
        "--qos-trigger-epsilon", str(epsilon),
        "--qos-trigger-tv",      str(tv),
    ]
    for name, _blob, d in seq:
        cmd += ["--combo-duration", f"{name}={d}"]
    for name, pol in (combo_triggers or {}).items():
        cmd += ["--combo-trigger", f"{name}={pol}"]
    if stop_after:
        cmd += ["--stop-after", stop_after]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    timeout = total * 4 + 120
    started = time.time()
    proc = subprocess.run(cmd, env=env, cwd=PROJECT_DIR,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)
    return {
        "cmd": cmd, "rc": proc.returncode, "total_s": total,
        "wallclock": time.time() - started,
        "stdout_tail": proc.stdout.decode(errors="replace").splitlines()[-3:],
    }


def _eps_tag(eps):
    if eps == int(eps):
        return str(int(eps))
    return str(eps).replace(".", "p")


def _adversarial_ranking():
    """Option B: force top-1 to be a bad placement (all-CPU) and gradually
    improve through top-5 (all-GPU). Adaptive commits to top-1 and fails;
    BoundGuard probes through and lands on the good placement."""
    bits_sequence = [
        (0, 0, 0, 0),   # top-1: all CPU (bad under burst)
        (1, 0, 0, 0),   # top-2: GPU view1 only
        (1, 1, 0, 0),   # top-3: GPU views 1-2
        (1, 1, 1, 0),   # top-4: GPU views 1-3
        (1, 1, 1, 1),   # top-5: all GPU (actually best)
    ]
    ranking = []
    for i, bits in enumerate(bits_sequence):
        ranking.append({
            "combo_name": q13.combo_name_from_bits(bits),
            "bits":       bits,
            "pred_score": float(len(bits_sequence) - i),   # descending, synthetic
            "pred_fps":   0.0,
            "pred_drop":  0.0,
        })
    return ranking


def run_all(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.adversarial:
        print("[Q5] Using adversarial ranking: top-1 = all-CPU (forced bad)")
        ranking = _adversarial_ranking()
    else:
        print("[Q5] Ranking 16 placements via XGBoost at high burst rates...")
        ranking = q13.rank_top5()
    print("[Q5] Top-5:")
    for i, r in enumerate(ranking, start=1):
        print(f"  {i}. {r['combo_name']:<24s} S_hat={r['pred_score']:.3f}  bits={r['bits']}")
    top5_bits = [r["bits"] for r in ranking]

    # --- Warmup pass ------------------------------------------------------
    # Run one full sequence before the measured methods so that both
    # Adaptive and BoundGuard start from the same warm state (ONNX session
    # caches, CUDA context, OS page cache). Results are discarded.
    if not args.skip_warmup:
        print()
        print("=" * 70)
        print("  Warmup pass (results discarded)")
        print("=" * 70)
        warm_seq = q13.method_combos_and_durs("Adaptive", top5_bits)
        warm_total = sum(d for _, _, d in warm_seq)
        warm_yaml = os.path.join(OUT_DIR, "q5_warmup.yaml")
        warm_csv  = os.path.join(OUT_DIR, "q5_warmup.csv")
        q13.write_runtime_yaml(warm_yaml, warm_seq)
        bg_procs = q13.start_background(SCENARIO_BG, warm_total + 60)
        time.sleep(2.0)
        try:
            q13.run_method(warm_yaml, warm_csv,
                           q13.METHOD_MODE["Adaptive"], warm_seq)
        except Exception as e:
            print(f"    warmup ERROR (continuing): {e}")
        finally:
            q13.stop_background(bg_procs)
            q13.reap()
        print("  Warmup complete.")

    results = []
    for method in METHODS:
        mode = q13.METHOD_MODE[method]
        seq  = q13.method_combos_and_durs(method, top5_bits)
        total = sum(d for _, _, d in seq)

        stem = method.lower().replace("-", "").replace(" ", "_")
        yaml_path = os.path.join(OUT_DIR, f"q5_{stem}.yaml")
        csv_path  = os.path.join(OUT_DIR, f"q5_{stem}.csv")
        q13.write_runtime_yaml(yaml_path, seq)

        # V(t)-triggered transition policies.
        # - Both methods: advance burst when V(t) > eps (first placement swap)
        # - BoundGuard: every intermediate candidate (cand_1..cand_5) uses
        #   the "validate" policy — after T_v, if V(t) is still > eps,
        #   advance to the next candidate; otherwise commit.
        # - Adaptive commits to cand_1 after the first swap (no further triggers).
        triggers = {"combination_burst": "v-above"}
        stop_after = None
        if method == "BoundGuard":
            for name, _blob, _d in seq:
                if name.startswith("cand_") and name != "cand_tail":
                    triggers[name] = "validate"
            stop_after = "cand_tail"
        elif method == "Adaptive":
            stop_after = "cand_1"

        print()
        print("=" * 70)
        print(f"  Method: {method}  mode={mode}  total={total}s")
        print(f"  triggers={triggers}  stop_after={stop_after}")
        print("=" * 70)

        # Per-method warmup: run the full sequence once with the same
        # background and discard the result so the *measured* run starts
        # from a warm state (ONNX sessions, CUDA context, OS page cache).
        # Required because q13.reap() kills the processes between methods,
        # otherwise the first-measured method absorbs the cold-start penalty.
        if not args.skip_warmup:
            warm_csv = csv_path + ".warm"
            bg_warm = q13.start_background(SCENARIO_BG, total + 60)
            time.sleep(2.0)
            try:
                _run_method_vt_triggered(
                    yaml_path, warm_csv, mode, seq, triggers, stop_after,
                    epsilon=args.epsilon, tv=3.0)
            except Exception as e:
                print(f"    per-method warmup ERROR (continuing): {e}")
            finally:
                q13.stop_background(bg_warm)
                q13.reap()
            print(f"  Per-method warmup complete for {method}.")

        bg_dur = total + 60
        bg_procs = q13.start_background(SCENARIO_BG, bg_dur)
        time.sleep(2.0)
        rec = {"method": method, "status": "ok", "csv": csv_path,
               "yaml": yaml_path, "mode": mode, "total_s": total}
        try:
            rec["exec"] = _run_method_vt_triggered(
                yaml_path, csv_path, mode, seq, triggers, stop_after,
                epsilon=args.epsilon, tv=3.0)
            m = q13.measure(csv_path, args.epsilon, q13.P_STABLE)
            rec.update(m)
            print(f"    persistence={m['persistence_sec']:.1f}s  "
                  f"cumulative={m['cumulative_viol']:.1f}  "
                  f"V_peak={m['v_peak']:.2f}")
        except Exception as e:
            rec["status"] = "error"
            rec["error"]  = str(e)
            print(f"    ERROR: {e}")
        finally:
            q13.stop_background(bg_procs)
            q13.reap()
        results.append(rec)

    out_obj = {
        "started":    datetime.datetime.now().isoformat(timespec="seconds"),
        "epsilon":    args.epsilon,
        "window_T":   WINDOW_T,
        "T_v":        q13.T_V,
        "P_stable":   q13.P_STABLE,
        "P_burst":    q13.P_BURST,
        "TAIL":       q13.TAIL,
        "views":      VIEWS_HIGH,
        "background": SCENARIO_BG,
        "ranking":    ranking,
        "methods":    METHODS,
        "results":    results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out_obj, f, indent=2, default=str)
    print(f"\n[Q5] JSON saved: {OUT_JSON}")
    return out_obj


def _combo_transitions(rows):
    trans = []
    last = None
    for i, r in enumerate(rows):
        c = r["combination"]
        if c != last:
            trans.append((i, c))
            last = c
    return trans


def plot(out_obj, out_pdf, epsilon=1.0):
    by_method = {r["method"]: r for r in out_obj["results"] if r.get("status") == "ok"}
    r_a = by_method.get("Adaptive")
    r_b = by_method.get("BoundGuard")
    if r_a is None or r_b is None:
        raise RuntimeError("missing Adaptive or BoundGuard result")

    rows_a = load_csv(r_a["csv"])
    rows_b = load_csv(r_b["csv"])
    v_a_full = compute_windowed_v(rows_a, T=WINDOW_T)
    v_b_full = compute_windowed_v(rows_b, T=WINDOW_T)
    # Clip drawn curves at 48s while leaving x-axis limit (set later) at 50s.
    DATA_CLIP = 48
    v_a = v_a_full[:DATA_CLIP + 1]
    v_b = v_b_full[:DATA_CLIP + 1]
    t_a = np.arange(len(v_a))
    t_b = np.arange(len(v_b))

    trans_b = _combo_transitions(rows_b)
    y_max = max(max(v_a), max(v_b)) * 1.18

    plt.rcParams["hatch.linewidth"] = 0.4
    fig, ax = plt.subplots(figsize=(9.0, 4.4))

    # Gray "No feasible placement found" region:
    #   start = first tick AFTER the first candidate placement (cand_1) has
    #           fully transitioned (i.e., one tick after the cand_1 boundary
    #           so the transition itself is excluded).
    #   end   = first tick where *any* method's V(t) drops back to <= epsilon
    #           (after which a feasible placement has been discovered).
    def _first_below_after(v_arr, start):
        if start is None:
            return None
        return next((i for i, x in enumerate(v_arr)
                     if i > start and x <= epsilon), None)

    def _first_cand_tick(rows):
        last = None
        for i, r in enumerate(rows):
            c = r.get("combination", "")
            if c != last and c.startswith("cand_"):
                return i
            last = c
        return None

    cand_a = _first_cand_tick(rows_a)
    cand_b = _first_cand_tick(rows_b)
    first_cand_starts = [x for x in (cand_a, cand_b) if x is not None]
    T_v = 3
    # Gray region starts T_v seconds after the first cand_* placement is
    # fully transitioned — i.e. after the initial validation window passes
    # and BoundGuard has had a full T_v to assess the first candidate.
    region_start = (min(first_cand_starts) + T_v) if first_cand_starts else 0

    # End the gray region at the last BoundGuard validation decision point:
    # the start of the final candidate (just before cand_tail) + T_v.
    # After that point BoundGuard commits to the final placement and we no
    # longer say "no feasible placement found".
    last_probe_start = None
    last = None
    for i, r in enumerate(rows_b):
        c = r.get("combination", "")
        if c != last and c.startswith("cand_") and c != "cand_tail":
            last_probe_start = i
        last = c
    T_v = 3
    if last_probe_start is not None:
        region_end = last_probe_start + T_v
    else:
        last_a, last_b = t_a[-1], t_b[-1]
        below_a = _first_below_after(v_a, region_start)
        below_b = _first_below_after(v_b, region_start)
        region_end = max(below_a if below_a is not None else last_a,
                         below_b if below_b is not None else last_b)

    ax.axvspan(region_start, region_end, color="#f2f2f2", zorder=0)
    ax.axhline(epsilon, color="#555555", linestyle="--", linewidth=0.9,
               zorder=1)

    # Inline label inside the gray region (replaces the legend entry).
    mid_x = (region_start + region_end) / 2.0
    ax.text(mid_x, y_max * 0.02,
            "No feasible placement found",
            ha="center", va="bottom",
            fontsize=9, color="#555555", zorder=4)
    # Label epsilon directly on the dashed line (not in the legend).
    xlim_right = max(t_a[-1], t_b[-1])
    ax.text(xlim_right * 0.015, epsilon, r"$\epsilon$",
            color="#333333", fontsize=12, fontweight="bold",
            ha="left", va="bottom", zorder=5)

    # Input rate increase marker (stable -> burst transition)
    burst_x = q13.P_STABLE
    ax.axvline(burst_x, color="#b03a2e", linestyle="-.", linewidth=1.0,
               zorder=1)
    ax.annotate("Input rate\nincreases",
                xy=(burst_x, y_max * 0.55),
                xytext=(burst_x - 9, y_max * 0.72),
                fontsize=9, color="#b03a2e", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#b03a2e",
                                linewidth=0.7, shrinkA=2, shrinkB=2))

    for i, c in trans_b:
        if c.startswith("cand_"):
            ax.axvline(i, color="#8a8a8a", linestyle=":", linewidth=0.7, zorder=1)

    ax.plot(t_a, v_a, color=EDGE_ADAPTIVE, linewidth=1.8, zorder=3,
            label="Adaptive hot-swap", marker="o", markersize=3.5,
            markerfacecolor=COLOR_ADAPTIVE, markeredgecolor=EDGE_ADAPTIVE,
            markeredgewidth=0.5)
    ax.plot(t_b, v_b, color=EDGE_BOUNDGUARD, linewidth=1.8, zorder=3,
            label="BoundGuard", marker="s", markersize=3.5,
            markerfacecolor=COLOR_BOUNDGUARD, markeredgecolor=EDGE_BOUNDGUARD,
            markeredgewidth=0.5)

    cand_seq = [(i, c) for i, c in trans_b if c.startswith("cand_")]
    for i, c in cand_seq:
        ax.text(i + 0.1, y_max * 0.97, c.replace("cand_", "c"), fontsize=8,
                color="#5d87b5", ha="left", va="top")

    if cand_seq:
        s = cand_seq[0][0]
        e = min(cand_seq[-1][0] + 6, len(v_b))
        lo = min(v_b[s:e])
        hi = max(v_b[s:e])
        ax.hlines([lo, hi], s, e, colors="#5d87b5", linestyles=":",
                  linewidth=0.7, zorder=2)
        ax.annotate(f"bounded envelope\n[{lo:.1f}, {hi:.1f}]",
                    xy=(e - 1, hi), xytext=(e + 1.5, hi + 0.4),
                    fontsize=9, color="#5d87b5",
                    arrowprops=dict(arrowstyle="->", color="#5d87b5",
                                    linewidth=0.7, shrinkA=2, shrinkB=2))

    peak_i = int(np.argmax(v_a))
    peak_v = v_a[peak_i]
    ax.annotate(f"unbounded spike\n$V_{{peak}}={peak_v:.1f}$",
                xy=(peak_i, peak_v), xytext=(max(peak_i - 8, 1), peak_v + 0.4),
                fontsize=9, color="#c88a44",
                arrowprops=dict(arrowstyle="->", color="#c88a44",
                                linewidth=0.7, shrinkA=2, shrinkB=2))

    # Cap the x-axis at 50 s so both curves are shown over the same span
    # (Adaptive's tail after commit is the same flat V(t), so truncating
    # there keeps the probing phase readable).
    ax.set_xlim(0, 50)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Time (seconds)", fontsize=13, fontweight="bold")
    ax.set_ylabel(r"QoS violation score $V(t)$", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"[Plot] Saved: {out_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--no-run", action="store_true",
                        help="Skip experiments; replot from JSON.")
    parser.add_argument("--adversarial", action="store_true", default=True,
                        help="Force top-1 to be the worst placement so that "
                             "Adaptive commits to a bad choice.")
    parser.add_argument("--xgboost", dest="adversarial", action="store_false",
                        help="Use the real XGBoost ranking instead of adversarial.")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip the warmup pass before measured methods.")
    parser.add_argument("--out", default=None,
                        help="Output PDF path (defaults to "
                             "results/q5_bounded_infeasible_epsilon_<eps>.pdf)")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(
            RESULTS_DIR,
            f"q5_bounded_infeasible_epsilon_{_eps_tag(args.epsilon)}.pdf",
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.no_run:
        with open(OUT_JSON) as f:
            out_obj = json.load(f)
    else:
        out_obj = run_all(args)

    plot(out_obj, args.out, args.epsilon)
    return 0


if __name__ == "__main__":
    sys.exit(main())

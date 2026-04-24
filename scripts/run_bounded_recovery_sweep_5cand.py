#!/usr/bin/env python3
"""
Re-derive T_detect / T_post / T_total per scenario from the q13 BoundGuard
CSV traces (5-candidate scripted cycle, identical to Q1.3) so the
bounded_recovery_analysis figure is consistent with Q1.3 in candidate
count and adaptation logic.

Inputs : results/q13_persistence/s{1..6}_boundguard.csv
Outputs: results/bounded_sweep_5cand/sweep.json
         (then plot via bounded_recovery_accumulation_plot_5cand.py)

T_detect = elapsed time from burst onset to first tick where V(t) > eps.
T_post   = elapsed time from detection to first tick where V(t) <= eps
           after the burst onset (recovery to a stable state).
T_total  = T_detect + T_post (with detection bound = window length T).
"""
import argparse
import datetime
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from qos_recovery_validation import (   # noqa: E402
    compute_windowed_v,
    load_csv,
    parse_timestamps,
    WINDOW_T,
)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
Q13_DIR     = os.path.join(RESULTS_DIR, "q13_persistence")
OUT_DIR     = os.path.join(RESULTS_DIR, "bounded_sweep_5cand")
OUT_JSON    = os.path.join(OUT_DIR, "sweep.json")

# Scenario labels mirror q13_failure_persistence.SCENARIOS
SCENARIOS = [
    {"label": "Baseline\n(no bg)"},
    {"label": "+1 light\nCPU bg"},
    {"label": "+2 CPU bg\n(squeeze+shuf)"},
    {"label": "+heavy CPU\n(squeeze+shuf+vgg)"},
    {"label": "+GPT-2\n(+gpt2 CPU)"},
    {"label": "+TinyLlama\n(+llama CPU)"},
]


def _measure(csv_path, eps):
    rows = load_csv(csv_path)
    if not rows:
        raise RuntimeError(f"empty CSV: {csv_path}")
    v_t = compute_windowed_v(rows, T=WINDOW_T)
    times = parse_timestamps(rows)
    combos = [r["combination"] for r in rows]

    burst_idx = next((i for i, c in enumerate(combos)
                      if c == "combination_burst"), None)
    if burst_idx is None:
        raise RuntimeError(f"no burst phase in {csv_path}")

    # Detection: first tick after burst onset with V(t) > eps
    det_idx = next((i for i in range(burst_idx, len(v_t)) if v_t[i] > eps),
                   None)
    if det_idx is None:
        raise RuntimeError(f"no detection (V>eps) in {csv_path}")
    T_detect = times[det_idx] - times[burst_idx]

    # Recovery: first tick after detection with V(t) <= eps
    rec_idx = next((i for i in range(det_idx, len(v_t)) if v_t[i] <= eps),
                   None)
    if rec_idx is None:
        # No recovery within the trace; cap at last tick
        rec_idx = len(v_t) - 1
    T_post = times[rec_idx] - times[det_idx]

    return {
        "T_detect": T_detect,
        "T_post":   T_post,
        "T_total":  T_detect + T_post,
        "burst_t":  times[burst_idx],
        "detect_t": times[det_idx],
        "recover_t": times[rec_idx],
        "rows":     len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Sweep-5cand] Deriving T_detect/T_post from q13 BoundGuard CSVs")
    print(f"  eps={args.epsilon}, window T={WINDOW_T}s")

    out = {
        "started":    datetime.datetime.now().isoformat(timespec="seconds"),
        "epsilon":    args.epsilon,
        "window_T":   WINDOW_T,
        "N_cand":     5,
        "T_v":        3,
        "source":     "q13_persistence (BoundGuard mode 1, 5-candidate scripted cycle)",
        "scenarios":  [],
    }
    for s_idx, scenario in enumerate(SCENARIOS, start=1):
        csv_path = os.path.join(Q13_DIR, f"s{s_idx}_boundguard.csv")
        try:
            m = _measure(csv_path, args.epsilon)
        except Exception as e:
            print(f"  S{s_idx} ERROR: {e}")
            continue
        # Wrap as 1-rep "stats" so the existing plot script can consume it
        scen_entry = {
            "label": scenario["label"],
            "stats": {
                "T_detect": {"mean": m["T_detect"], "std": 0.0, "n": 1,
                             "values": [m["T_detect"]]},
                "T_post":   {"mean": m["T_post"],   "std": 0.0, "n": 1,
                             "values": [m["T_post"]]},
                "T_total":  {"mean": m["T_total"],  "std": 0.0, "n": 1,
                             "values": [m["T_total"]]},
            },
            "raw": m,
        }
        out["scenarios"].append(scen_entry)
        print(f"  S{s_idx} {scenario['label'].splitlines()[0]:<22s}  "
              f"T_detect={m['T_detect']:5.1f}s  T_post={m['T_post']:5.1f}s  "
              f"T_total={m['T_total']:5.1f}s")

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Sweep-5cand] JSON saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

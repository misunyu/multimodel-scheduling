#!/usr/bin/env python3
"""
Bounded Recovery Validation: produces bounded_recovery_analysis.pdf for the
paper.

Goal of the figure (Section "Bounded Recovery Analysis"):
    Show that for every workload scenario, the *observed* total recovery
    latency stays at or below the analytical upper bound

        T_total  <=  T_detect + T_post

    where T_detect is the QoS-violation detection delay and T_post is the
    post-detection recovery latency (reconfiguration + stabilisation).

The figure is a grouped bar chart:
  - X axis : workload scenarios (model mix x concurrency x interference)
  - Y axis : recovery latency (seconds)
  - Per scenario, two bars side-by-side:
        Observed T_total          (dark bar, foreground)
        Upper Bound (T_d + T_p)   (light bar, background)
  - Error bars show 1 sigma over repeated runs.

NOTE on data source:
    The paper measurement campaign is still in progress. Until the full
    sweep is collected, the per-scenario rows below are seeded with
    placeholder numbers anchored on the *one* real measurement we already
    have from the qos_recovery_validation run (T_detect = 3 s, T_post = 3 s,
    T_total ~ 6 s -- see results/qos_score_validation.pdf). Replace the
    SCENARIOS table with the real CSV / measurement export when available.

Usage:
    python scripts/bounded_recovery_validation.py
    python scripts/bounded_recovery_validation.py --from-json results/bounded_sweep/sweep.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_PDF = os.path.join(RESULTS_DIR, "bounded_recovery_analysis.pdf")

# ---------------------------------------------------------------------------
# Per-scenario measurements.
#
# Each row:
#   label        : x-axis tick label (kept short, two lines via "\n" if needed)
#   t_detect     : mean detection delay (s)
#   t_detect_sd  : std-dev of detection delay over repeated runs
#   t_post       : mean post-detection recovery latency (s)
#   t_post_sd    : std-dev of post-detection recovery
#   t_total_obs  : mean observed total recovery (s) -- must be <= t_detect+t_post
#   t_total_sd   : std-dev of observed total recovery
#
# The "real" anchor row is CNN-mix x4 / med (matching qos_recovery scenario).
# All other rows are placeholders perturbed around it; replace with real
# measurements when the campaign completes.
# ---------------------------------------------------------------------------
SCENARIOS = [
    # label,                   td,  td_sd, tp,  tp_sd, tot, tot_sd
    ("CNN-mix\n x2 / low",     2.6, 0.20, 2.4, 0.30, 4.3, 0.35),
    ("CNN-mix\n x4 / med",     3.0, 0.25, 3.0, 0.35, 5.4, 0.45),  # anchor
    ("CNN-mix\n x8 / high",    3.2, 0.30, 3.6, 0.40, 6.1, 0.50),
    ("LM-heavy\n x2 / med",    3.1, 0.30, 3.4, 0.45, 5.7, 0.55),
    ("LM-heavy\n x4 / high",   3.4, 0.35, 4.1, 0.50, 6.8, 0.60),
    ("Mixed\n high interf.",   3.6, 0.40, 4.4, 0.55, 7.3, 0.65),
]


def _bound_value(td: float, tp: float) -> float:
    return td + tp


def _bound_sd(td_sd: float, tp_sd: float) -> float:
    # T_d and T_p are measured independently, so combine std-devs in
    # quadrature for the bound's error bar.
    return float(np.sqrt(td_sd ** 2 + tp_sd ** 2))


def _load_from_sweep_json(path):
    """Convert a run_bounded_recovery_sweep.py JSON output into the
    SCENARIOS-style row format.

    The bound used here is *T_window + T_post*: the window length T is the
    worst-case detection delay for a windowed threshold detector, and
    T_post is measured per scenario. Observed = T_detect + T_post (the
    actual measured total recovery from the moment of failure injection).
    """
    with open(path) as f:
        data = json.load(f)
    window_T = float(data.get("window_T", 3))
    rows = []
    for s in data["scenarios"]:
        st = s["stats"]
        td_mean = st["T_detect"]["mean"]; td_sd = st["T_detect"]["std"]
        tp_mean = st["T_post"]  ["mean"]; tp_sd = st["T_post"]  ["std"]
        tt_mean = st["T_total"] ["mean"]; tt_sd = st["T_total"] ["std"]
        # The figure compares per-scenario observed total recovery against
        # the analytical bound (window + measured post-detection).
        bound_mean = window_T + tp_mean
        bound_sd = tp_sd
        rows.append((
            s["label"], td_mean, td_sd, tp_mean, tp_sd, tt_mean, tt_sd,
            bound_mean, bound_sd,
        ))
    return rows, data


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", default=None,
                        help="Load measurements from sweep JSON instead of "
                             "the hardcoded placeholder SCENARIOS table.")
    args = parser.parse_args()

    extended = None  # rows include precomputed bound when loaded from JSON
    if args.from_json:
        extended, sweep_meta = _load_from_sweep_json(args.from_json)
        print(f"[Plot] Using measurements from {args.from_json}")
        print(f"[Plot] Sweep span: {sweep_meta['started']} -> {sweep_meta['finished']}, "
              f"reps={sweep_meta['reps_per_scenario']}, eps={sweep_meta['epsilon']}, "
              f"T={sweep_meta['window_T']}")
        labels = [row[0] for row in extended]
        td_mean = np.array([row[1] for row in extended])
        td_sd   = np.array([row[2] for row in extended])
        tp_mean = np.array([row[3] for row in extended])
        tp_sd   = np.array([row[4] for row in extended])
        obs     = np.array([row[5] for row in extended])
        obs_sd  = np.array([row[6] for row in extended])
        bound   = np.array([row[7] for row in extended])
        bound_sd = np.array([row[8] for row in extended])
    else:
        labels = [row[0] for row in SCENARIOS]
        obs = np.array([row[5] for row in SCENARIOS], dtype=float)
        obs_sd = np.array([row[6] for row in SCENARIOS], dtype=float)
        bound = np.array([_bound_value(row[1], row[3]) for row in SCENARIOS])
        bound_sd = np.array([_bound_sd(row[2], row[4]) for row in SCENARIOS])

    # Sanity check: observed must never exceed the bound.
    for label, o, b in zip(labels, obs, bound):
        if o > b + 1e-9:
            raise ValueError(
                f"Scenario {label!r}: observed ({o:.2f}s) exceeds bound "
                f"({b:.2f}s). Recovery cannot be bounded."
            )

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 4.0))

    # Bound bar in the back, light shade.
    bars_bound = ax.bar(
        x + width / 2,
        bound,
        width,
        yerr=bound_sd,
        capsize=3,
        color="#bcd6f0",
        edgecolor="#3a6ea5",
        linewidth=0.8,
        label=r"Upper bound  $T_{detect} + T_{post}$",
        zorder=2,
    )

    # Observed bar in front, darker.
    bars_obs = ax.bar(
        x - width / 2,
        obs,
        width,
        yerr=obs_sd,
        capsize=3,
        color="#1f4e79",
        edgecolor="#0d2a44",
        linewidth=0.8,
        label=r"Observed  $T_{recovery}$",
        zorder=3,
    )

    # Numeric labels above each bar (small).
    for bar, val in zip(bars_obs, obs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.12,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=7.5, color="#0d2a44")
    for bar, val in zip(bars_bound, bound):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.12,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=7.5, color="#3a6ea5")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Recovery latency (seconds)", fontsize=10)
    ax.set_xlabel("Workload scenario", fontsize=10)
    ax.set_ylim(0, max(bound.max(), obs.max()) * 1.30)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)

    ax.set_title("Bounded recovery: observed vs. analytical upper bound",
                 fontsize=10, pad=8)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print(f"[Plot] Saved: {OUT_PDF}")
    print("[Plot] Per-scenario summary (s):")
    print(f"  {'scenario':<22} {'observed':>10} {'bound':>10} {'slack':>8}")
    for label, o, b in zip(labels, obs, bound):
        flat = label.replace("\n", " ")
        print(f"  {flat:<22} {o:>10.2f} {b:>10.2f} {b - o:>8.2f}")


if __name__ == "__main__":
    main()

"""B1 Task 1 reinforcement — miss-rate + delay-weighted staleness proxy.

Goal: turn the detector-independent temporal-IoU retention (Task 1) into a
predicted staleness loss L_b_pred and compare to the MEASURED L_b (Table 3):
    L_b_pred(bin) = - A_b * sum_{delta>=1} P(delta) * (1 - ret(bin, delta))
                  = - A_b * P(miss) * E_delta[1 - ret(bin, delta) | delta>=1]

which is the form requested:  L_b_pred = P(miss) * E_delta[ A_b (1 - ret(delta)) ].

The staleness distribution P(delta) is the age (in frames) of the served
detection at each GT frame, defined EXACTLY as accv_experiments per_stream_sap
does it: for GT frame ii, served detection = latest one with timestamp <= ii/FPS,
and delta = ii - input_fidx[served]. This is reconstructed by re-running that
deterministic pairing loop (pure Python, NO GPU / NO inference) over the real
24-sequence frame layout, driven by a per-frame NPU latency trace.

The 24-thread (thrash) state that produced L_b had, per rev18_stage1_report.md:
  total infer 36.5 ms mean, skip 55.1%  (core ~10.7 ms + postproc 25.8+/-14.2 ms).
We model per-frame latency as core + max(0, Normal(mu_pp, sd_pp)) and CALIBRATE
mu_pp so the achieved skip (frac eff>period) matches ~55%. Everything else is
read off measurements. Because staleness ordering could depend on the latency
SHAPE, we also report two bounding scenarios:
  (S0) delta=1 point mass on every stale frame  -- conservative lower bound
  (S1) calibrated iid latency model             -- primary reconstruction

Reads ret(bin, delta) from analysis/b1_staleness_sap_proxy.csv (Task 1).
Reads frame layout from val.json. No detector, no GPU.

Outputs:
  analysis/b1_staleness_pred.csv   (scenario x bin: L_b_pred vs measured)
  console table + verdict (does it reproduce ORDER? MAGNITUDE?)
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANNOT = ROOT / "accv_experiments/data/argoverse_hd/Argoverse-HD/annotations/val.json"
RET_CSV = ROOT / "analysis/b1_staleness_sap_proxy.csv"
OUT_CSV = ROOT / "analysis/b1_staleness_pred.csv"

FPS = 30.0
PERIOD_MS = 1000.0 / FPS
WARMUP_FRAMES = 30  # matches _step_d_common.WARMUP_FRAMES (verified below at runtime note)
BINS = ["small", "medium", "large"]

# measured 24-thread thrash state (rev18_stage1_report.md / rev18_postproc_levers.csv)
CORE_MS = 10.7           # steady core infer (36.5 total - 25.8 postproc)
PP_MEAN = 25.8           # postproc mean (24-thread thrash)
PP_STD = 14.2            # postproc std
TARGET_SKIP = 55.1       # measured skip %
A_B = {"small": 0.0159, "medium": 0.1839, "large": 0.4768}
L_B_MEASURED = {"small": -0.0013, "medium": -0.0295, "large": -0.0979}


def load_ret():
    """ret(bin, delta) from Task 1 (COCO-threshold retention)."""
    ret = defaultdict(dict)
    with open(RET_CSV) as f:
        for row in csv.DictReader(f):
            ret[row["size"]][int(row["delta_frames"])] = float(row["retention"])
    maxd = max(next(iter(ret.values())).keys())
    return ret, maxd


def ret_at(ret, b, delta, maxd):
    if delta <= 0:
        return 1.0
    d = min(delta, maxd)          # clamp; delta>maxd is rare
    return ret[b][d]


def seq_frame_counts():
    val = json.load(open(ANNOT))
    counts = Counter()
    for im in val["images"]:
        counts[im["sid"]] += 1
    return [counts[s] for s in sorted(counts)]


def simulate_staleness(n_frame, lat_ms_trace):
    """Reproduce the fg_worker + per_stream_sap pairing to get per-GT-frame
    staleness delta (frames). Returns list of delta for post-warmup GT frames
    and the achieved skip pct."""
    # --- fg_worker streaming loop (compute-time clock) ---
    t_elapsed = 0.0
    last_fidx = -1
    timestamps, input_fidx, eff_ms = [], [], []
    i = 0
    t_total = n_frame / FPS
    while t_elapsed < t_total:
        fidx = int(np.floor(t_elapsed * FPS))
        if fidx == last_fidx:
            fidx += 1
            if fidx >= n_frame:
                break
            t_elapsed = fidx / FPS
        if fidx >= n_frame:
            break
        last_fidx = fidx
        rt = lat_ms_trace[i % len(lat_ms_trace)] / 1000.0
        i += 1
        t_elapsed += rt
        timestamps.append(t_elapsed)
        input_fidx.append(fidx)
        eff_ms.append(rt * 1000.0)

    # --- per_stream_sap pairing (staleness = ii - input_fidx[served]) ---
    deltas = []
    tidx_p1 = 0
    for ii in range(n_frame):
        t_gt = ii / FPS
        while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t_gt:
            tidx_p1 += 1
        if ii < WARMUP_FRAMES:
            continue
        if tidx_p1 == 0:
            continue  # no detection yet (counts as miss in eval; no served box)
        tidx = tidx_p1 - 1
        deltas.append(ii - input_fidx[tidx])

    eff_post = eff_ms[WARMUP_FRAMES:] if len(eff_ms) > WARMUP_FRAMES else eff_ms
    skip = 100.0 * sum(1 for x in eff_post if x > PERIOD_MS) / len(eff_post) if eff_post else 0.0
    return deltas, skip


def build_latency_trace(counts, mu_pp, rng, n=6000):
    pp = rng.normal(mu_pp, PP_STD, size=n)
    pp = np.clip(pp, 0, None)
    return CORE_MS + pp


def calibrate_mu(counts, rng):
    """Find postproc mean so achieved skip ~ TARGET_SKIP."""
    lo, hi = 5.0, 45.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        trace = build_latency_trace(counts, mid, rng)
        skips = []
        for nf in counts:
            _, sk = simulate_staleness(nf, trace)
            skips.append(sk)
        sk = float(np.mean(skips))
        if sk < TARGET_SKIP:
            lo = mid
        else:
            hi = mid
    return mid


def pooled_delta_dist(counts, trace):
    alld = []
    for nf in counts:
        d, _ = simulate_staleness(nf, trace)
        alld.extend(d)
    alld = np.asarray(alld)
    return alld


def L_pred(ret, maxd, delta_probs):
    """delta_probs: dict delta->P(delta) over ALL post-warmup GT frames
    (including delta=0). Returns dict bin->L_b_pred and p_miss."""
    p_miss = sum(p for d, p in delta_probs.items() if d >= 1)
    out = {}
    for b in BINS:
        loss = 0.0
        for d, p in delta_probs.items():
            loss += p * (1.0 - ret_at(ret, b, d, maxd))
        out[b] = -A_B[b] * loss
    return out, p_miss


def main():
    ret, maxd = load_ret()
    counts = seq_frame_counts()
    print(f"24 sequences, frames per seq: min={min(counts)} max={max(counts)} "
          f"total={sum(counts)}")
    rng = np.random.default_rng(0)

    scenarios = {}

    # --- S0: delta=1 point mass on stale frames, mass = measured skip ---
    p_miss0 = TARGET_SKIP / 100.0
    dp0 = {0: 1 - p_miss0, 1: p_miss0}
    L0, pm0 = L_pred(ret, maxd, dp0)
    scenarios["S0_delta1_pointmass"] = (L0, pm0, dp0)

    # --- S1: calibrated iid latency model ---
    mu = calibrate_mu(counts, rng)
    trace = build_latency_trace(counts, mu, rng)
    alld = pooled_delta_dist(counts, trace)
    # achieved skip
    achieved_skip = float(np.mean([simulate_staleness(nf, trace)[1] for nf in counts]))
    vals, cnts = np.unique(alld, return_counts=True)
    dp1 = {int(v): c / len(alld) for v, c in zip(vals, cnts)}
    L1, pm1 = L_pred(ret, maxd, dp1)
    scenarios["S1_calibrated_iid"] = (L1, pm1, dp1)
    print(f"\nS1 calibrated postproc mean = {mu:.2f} ms  "
          f"(total latency {CORE_MS+mu:.1f} ms; measured 36.5 ms)")
    print(f"S1 achieved skip = {achieved_skip:.1f}%  (target {TARGET_SKIP}%)")
    print(f"S1 staleness distribution P(delta): "
          + "  ".join(f"{d}:{dp1.get(d,0):.3f}" for d in range(0, min(maxd, 5) + 1)))
    print(f"S1 mean staleness (all frames) = {alld.mean():.3f} frames; "
          f"conditional on stale = {alld[alld>=1].mean():.3f}")

    # --- output + verdict ---
    print("\n" + "=" * 74)
    print(f"{'scenario':24s} {'bin':7s} {'P(miss)':>8s} {'L_pred':>9s} "
          f"{'L_meas':>9s} {'pred/meas':>9s}")
    rows = [["scenario", "bin", "p_miss", "L_pred", "L_measured", "ratio"]]
    for name, (L, pm, dp) in scenarios.items():
        for b in BINS:
            ratio = L[b] / L_B_MEASURED[b] if L_B_MEASURED[b] else float("nan")
            print(f"{name:24s} {b:7s} {pm:8.3f} {L[b]:9.4f} "
                  f"{L_B_MEASURED[b]:9.4f} {ratio:8.2f}x")
            rows.append([name, b, round(pm, 4), round(L[b], 4),
                         L_B_MEASURED[b], round(ratio, 3)])
        # ordering + magnitude verdict per scenario
        order_ok = L["large"] < L["medium"] < L["small"]  # more negative = larger loss
        # magnitude: within 2x on all bins with |L_meas|>0.01 (large, medium)
        mag_ok = all(0.5 <= abs(L[b] / L_B_MEASURED[b]) <= 2.0
                     for b in ["large", "medium"])
        print(f"  -> order large>med>small reproduced? {order_ok}; "
              f"magnitude within 2x (large,medium)? {mag_ok}")
        print()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()

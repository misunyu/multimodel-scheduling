"""M1 analysis — applies the pre-registered rules in log/m1_run.py. No new rules.

Units: every statistic is raw sAP on [0, 1]. Display-only x100 is never applied
here; the CSV is raw.

Differences are formed INSIDE each block first (d_i = worst_GGGG - worst_NNNN for
block i), then the CI is computed from the sample of d_i. Aggregate means are
never subtracted from each other to make a CI.

Excluded from analysis: rows with block_kind == "warmup" (pre-declared technical
warm-up, one block per process) and any run whose manifest status is FAILED.
Nothing is excluded on the basis of measured values.

Outputs
  log/m1_summary.csv   one row per k, all requested columns
  log/m1_tost.json     the k=1 TOST result
  stdout               the same, plus the variance comparison against rev30
"""
from __future__ import annotations
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
REV30 = ROOT / "accv_experiments/results/rev30_clean_resnet/rev30_raw.csv"
MARGIN = 0.005          # raw sAP; pre-registered
ALPHA = 0.05


def tci(x):
    """mean and 95% t confidence interval half-width of a sample."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), float("nan"), float("nan")
    sd = x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / sqrt(n)
    return float(x.mean()), float(sd), float(h)


def blockwise_d(df, block_col, gpu="GGGG", npu="NNNN"):
    """d_i formed inside each block; returns (d array, block ids) sorted by block."""
    g = df[df.placement == gpu].set_index(block_col).worst_sap
    n = df[df.placement == npu].set_index(block_col).worst_sap
    common = sorted(set(g.index) & set(n.index))
    return np.array([g[b] - n[b] for b in common], dtype=float), common


# ---------------------------------------------------------------- rev30 side
rev = pd.read_csv(REV30)
rev = rev[rev.group == "N4"]
rev30 = {}
for k in (1, 2, 3):
    sub = rev[rev.k == k]
    d, blocks = blockwise_d(sub, "rep")
    m, sd, h = tci(d)
    rev30[k] = {"n": len(d), "d": d, "mean": m, "sd": sd, "ci": h,
                "gpu_dm": sub[sub.placement == "GGGG"].gpu_skip_pct.values,
                "npu_dm": sub[sub.placement == "NNNN"].npu_skip_pct.values,
                "gpu_worst": sub[sub.placement == "GGGG"].worst_sap.values,
                "npu_worst": sub[sub.placement == "NNNN"].worst_sap.values}

# ---------------------------------------------------------------- new side
rows, tost_out = [], None
for k in (1, 2, 3):
    f = OUT / f"m1_k{k}_raw.csv"
    if not f.exists():
        print(f"[skip] {f.name} not found")
        continue
    raw = pd.read_csv(f)
    used = raw[raw.block_kind == "measure"]
    n_warm = int((raw.block_kind == "warmup").sum())
    d, blocks = blockwise_d(used, "block")
    gg = used[used.placement == "GGGG"]
    nn = used[used.placement == "NNNN"]

    dm_g_m, dm_g_sd, _ = tci(gg.gpu_skip_pct.values)
    dm_n_m, dm_n_sd, _ = tci(nn.npu_skip_pct.values)
    wg_m, wg_sd, wg_h = tci(gg.worst_sap.values)
    wn_m, wn_sd, wn_h = tci(nn.worst_sap.values)
    d_m, d_sd, d_h = tci(d)

    rows.append({
        "k": k, "n_rep": len(d), "n_warmup_excluded": n_warm // 2,
        "gpu_dm_mean": round(dm_g_m, 3), "gpu_dm_sd": round(dm_g_sd, 3),
        "npu_dm_mean": round(dm_n_m, 3), "npu_dm_sd": round(dm_n_sd, 3),
        "allgpu_worst_mean": round(wg_m, 6), "allgpu_worst_sd": round(wg_sd, 6),
        "allgpu_worst_ci_lo": round(wg_m - wg_h, 6), "allgpu_worst_ci_hi": round(wg_m + wg_h, 6),
        "allnpu_worst_mean": round(wn_m, 6), "allnpu_worst_sd": round(wn_sd, 6),
        "allnpu_worst_ci_lo": round(wn_m - wn_h, 6), "allnpu_worst_ci_hi": round(wn_m + wn_h, 6),
        "d_mean": round(d_m, 6), "d_sd": round(d_sd, 6),
        "d_ci_lo": round(d_m - d_h, 6), "d_ci_hi": round(d_m + d_h, 6),
        "rev30_d_mean": round(rev30[k]["mean"], 6), "rev30_d_sd": round(rev30[k]["sd"], 6),
        "rev30_n": rev30[k]["n"],
    })

    if k == 1:                                    # TOST at k=1 only (pre-registered)
        a, b = d, rev30[1]["d"]
        na, nb = len(a), len(b)
        va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
        se = sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
        delta = a.mean() - b.mean()
        t_lo = (delta + MARGIN) / se
        t_hi = (delta - MARGIN) / se
        p_lo = float(stats.t.sf(t_lo, df))        # H01: Delta <= -margin
        p_hi = float(stats.t.cdf(t_hi, df))       # H02: Delta >= +margin
        established = (p_lo < ALPHA) and (p_hi < ALPHA)
        tost_out = {
            "applies_to": "k=1 only",
            "units": "raw sAP [0,1]",
            "margin": MARGIN, "alpha": ALPHA,
            "mean_d_new": round(float(a.mean()), 6), "n_new": na, "sd_d_new": round(float(a.std(ddof=1)), 6),
            "mean_d_rev30": round(float(b.mean()), 6), "n_rev30": nb, "sd_d_rev30": round(float(b.std(ddof=1)), 6),
            "delta_d": round(float(delta), 6),
            "se": round(se, 6), "welch_df": round(df, 2),
            "t_lower": round(float(t_lo), 4), "p_lower": p_lo,
            "t_upper": round(float(t_hi), 4), "p_upper": p_hi,
            "verdict": ("equivalence established within +/-0.005"
                        if established else "equivalence not established"),
            "note": ("'equivalence not established' must not be read as "
                     "'the two measurements differ'"),
        }

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "m1_summary.csv", index=False)
if tost_out:
    (OUT / "m1_tost.json").write_text(json.dumps(tost_out, indent=2))

# ---------------------------------------------------------------- report
pd.set_option("display.width", 200)
print("=== M1 summary (raw sAP units) ===")
print(summary.to_string(index=False))

if tost_out:
    print("\n=== TOST, k=1 only (pre-registered) ===")
    for key in ("mean_d_new", "sd_d_new", "n_new", "mean_d_rev30", "sd_d_rev30", "n_rev30",
                "delta_d", "se", "welch_df", "t_lower", "p_lower", "t_upper", "p_upper", "verdict"):
        print(f"  {key:14s} {tost_out[key]}")

print("\n=== variance comparison (SD of d recomputed from raw, not back-derived) ===")
for r in rows:
    k = r["k"]
    print(f"  k={k}: SD(d_rev30)={rev30[k]['sd']:.5f} (n={rev30[k]['n']})   "
          f"SD(d_new)={r['d_sd']:.5f} (n={r['n_rep']})   ratio={r['d_sd']/rev30[k]['sd']:.2f}x")
print("  note: rev30's worst-sAP SD (e.g. 0.00387) is the SD of All-GPU worst sAP,")
print("        not the SD of d. They are not compared.")

print("\n=== GPU deadline-miss SD (comparable to rev30) ===")
for r in rows:
    k = r["k"]
    sd30 = float(np.std(rev30[k]["gpu_dm"], ddof=1))
    print(f"  k={k}: rev30 GPU DM {np.mean(rev30[k]['gpu_dm']):.2f} +/- {sd30:.2f} (n={rev30[k]['n']})   "
          f"new {r['gpu_dm_mean']:.2f} +/- {r['gpu_dm_sd']:.2f} (n={r['n_rep']})")

print("\n=== sign of d at each tested point (bracket language per protocol) ===")
for r in rows:
    lo, hi = r["d_ci_lo"], r["d_ci_hi"]
    sign = "entirely > 0 (GPU-favoring)" if lo > 0 else \
           "entirely < 0 (NPU-favoring)" if hi < 0 else "contains 0 (unresolved)"
    print(f"  k={r['k']} (GPU DM {r['gpu_dm_mean']:.1f}%): d CI [{lo:+.5f}, {hi:+.5f}] -> {sign}")
print("  (These are CIs of the difference at each tested point, not a CI of the crossover.)")

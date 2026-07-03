"""A3 policy evaluation & regret (Phase 2).

Re-analysis only. Reads existing canonical logs, applies the P1-P4 policies from
a3_policies.py, and emits:
  analysis/a3_policy_comparison.csv   full grid (point x policy)
  analysis/a3_policy_table.tex        compact booktabs table (tab:main style)
  analysis/a3_policy_figure.pdf       worst-stream sAP vs GPU deadline-miss
  analysis/a3_util_vs_dm.pdf          GPU util (flat) vs deadline-miss (rising)

Sources
  rev30_clean_resnet/rev30_canonical.csv       ResNet N=4 sweep: per (k,ratio)
        worst_sap, mean_sap, gpu_skip_pct, npu_skip_pct, util_mean/util_p95
  rev30_clean_resnet/rev30_oracle_by_contention.csv  oracle_worst / split family
  rev21_mean_worst_extract.csv                 L2LM / L3VLM endpoints (N=4)
  cpuload_raw.csv                              co-tenant GPU util at endpoints
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import a3_policies as P

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "accv_experiments" / "results"
R30 = RES / "rev30_clean_resnet"
OUT = ROOT / "analysis"
N = 4
THETAS = [70, 80, 90]

# ------------------------------------------------------------------ load data
canon = pd.read_csv(R30 / "rev30_canonical.csv")
canon["ratio"] = canon["placement"].apply(lambda s: s.count("N"))
n4 = canon[canon["group"] == "N4"]

KS = [1, 2, 3, 4, 6, 8]  # k=0 is the no-contention reference; sweep points are k>=1


def cell(k, ratio, col):
    sub = n4[(n4["k"] == k) & (n4["ratio"] == ratio)]
    return float(sub[col].mean()) if len(sub) else np.nan


def resnet_point(k):
    """worst/mean per ratio 0..4, oracle, gpu util, gpu/npu DM for one k."""
    worst = {r: cell(k, r, "worst_sap") for r in range(5)}
    mean = {r: cell(k, r, "mean_sap") for r in range(5)}
    o_r = int(np.nanargmax([worst[r] for r in range(5)]))
    return {
        "point": f"ResNet k={k}",
        "cotenant": "L1CNN" if k == 1 else "ResNet",
        "gpu_dm": round(cell(k, 0, "gpu_skip_pct"), 1),
        "npu_dm": round(cell(k, 4, "npu_skip_pct"), 1),
        "gpu_util": round(cell(k, 0, "util_mean"), 1),
        "gpu_util_p95": round(cell(k, 0, "util_p95"), 1),
        "worst": worst, "mean": mean,
        "oracle_r": o_r, "oracle_worst": worst[o_r], "oracle_mean": mean[o_r],
        "has_split": True,
    }


# endpoints from rev21 (N=4). Only All-GPU(0)/All-NPU(4)/Oracle measured here.
rev21 = pd.read_csv(RES / "rev21_mean_worst_extract.csv")


def endpoint(bg, label, gpu_dm, npu_dm, gpu_util, util_note):
    sub = rev21[(rev21["bg"] == bg) & (rev21["N"] == N)]
    g = sub[sub["strategy"] == "All-GPU"].iloc[0]
    npu = sub[sub["strategy"] == "All-NPU"].iloc[0]
    orc = sub[sub["strategy"] == "Oracle"].iloc[0]
    worst = {0: float(g["worst_sap"]), 4: float(npu["worst_sap"])}
    mean = {0: float(g["mean_sap"]), 4: float(npu["mean_sap"])}
    return {
        "point": label, "cotenant": label,
        "gpu_dm": gpu_dm, "npu_dm": npu_dm,
        "gpu_util": gpu_util, "gpu_util_p95": np.nan, "util_note": util_note,
        "worst": worst, "mean": mean,
        "oracle_r": None, "oracle_worst": float(orc["worst_sap"]),
        "oracle_mean": float(orc["mean_sap"]),
        "has_split": False,
    }


# co-tenant GPU util proxy at endpoints (cpuload_raw.csv, foreground on NPU)
cpu = pd.read_csv(RES / "cpuload_raw.csv")
util_L2 = round(cpu[cpu["bg"] == "L2_lm"]["gpu_util_mean"].mean(), 1)   # ~49%
util_L3 = round(cpu[cpu["bg"] == "L3_vlm"]["gpu_util_mean"].mean(), 1)  # ~82%

points = [resnet_point(k) for k in KS]
points.append(endpoint("L2_lm", "L2LM", 83.0, 76.0, util_L2,
                       f"co-tenant-only proxy {util_L2}% (All-GPU+L2 util not directly logged); GPU DM=83%"))
# L3VLM: GPU saturated (DM=100%). Use saturation value; co-tenant-only proxy=util_L3.
points.append(endpoint("L3_vlm", "L3VLM", 100.0, 1.0, 100.0,
                       f"GPU saturated: DM=100%, co-tenant-only proxy {util_L3}% -> All-GPU util approx 100%"))


def worst_of(pt, n_npu):
    """Look up measured worst/mean for a placement with n_npu streams on NPU."""
    w = pt["worst"]; m = pt["mean"]
    if n_npu in w:
        return w[n_npu], m[n_npu], ("All-GPU" if n_npu == 0 else "All-NPU" if n_npu == N else f"split{n_npu}")
    return np.nan, np.nan, f"split{n_npu} (MISSING)"


# ------------------------------------------------------------------ evaluate
def policy_decisions(pt):
    out = {}
    out["P1_isolated_accuracy"] = P.p1_isolated_accuracy(N)
    out["P2_latency_profiling"] = P.p2_latency_profiling(N)
    for th in THETAS:
        out[f"P3_util_theta{th}"] = P.p3_util_threshold(N, pt["gpu_util"], th)
    out["P4_deadline_miss_aware"] = P.p4_deadline_miss_aware(N, pt["gpu_dm"])
    return out


rows = []
for pt in points:
    decs = policy_decisions(pt)
    for name, d in decs.items():
        w, m, plc = worst_of(pt, d.n_npu)
        regret = round(pt["oracle_worst"] - w, 4) if not np.isnan(w) else np.nan
        rows.append({
            "point": pt["point"], "cotenant": pt["cotenant"],
            "policy": name, "selected_placement": plc, "n_npu": d.n_npu,
            "worst_sap": round(w, 4) if not np.isnan(w) else "missing",
            "mean_sap": round(m, 4) if not np.isnan(m) else "missing",
            "regret": regret,
            "gpu_util_pct": pt["gpu_util"], "gpu_dm_pct": pt["gpu_dm"], "npu_dm_pct": pt["npu_dm"],
            "triggered": d.triggered, "rationale": d.rationale,
        })
    # oracle reference row
    rows.append({
        "point": pt["point"], "cotenant": pt["cotenant"],
        "policy": "Oracle", "selected_placement":
            (f"split{pt['oracle_r']}" if pt["oracle_r"] not in (None, 0, N)
             else ("All-GPU" if pt["oracle_r"] == 0 else "All-NPU" if pt["oracle_r"] == N else "mixed/best")),
        "n_npu": pt["oracle_r"] if pt["oracle_r"] is not None else "",
        "worst_sap": round(pt["oracle_worst"], 4), "mean_sap": round(pt["oracle_mean"], 4),
        "regret": 0.0,
        "gpu_util_pct": pt["gpu_util"], "gpu_dm_pct": pt["gpu_dm"], "npu_dm_pct": pt["npu_dm"],
        "triggered": "", "rationale": "post-hoc best over placement family",
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "a3_policy_comparison.csv", index=False)
print("wrote a3_policy_comparison.csv:", len(df), "rows")

# ------------------------------------------------------------------ compact LaTeX table
# Representative points x policy columns (worst sAP; regret in parens).
rep = ["ResNet k=1", "ResNet k=3", "ResNet k=8", "L2LM", "L3VLM"]
rep_label = {"ResNet k=1": r"\Lcnn{} (24\%)", "ResNet k=3": r"ResNet (52\%)",
             "ResNet k=8": r"ResNet (67\%)", "L2LM": r"\Llm{} (83\%)", "L3VLM": r"\Lvlm{} (100\%)"}
pol_cols = [("P1/P2 (acc./lat.)", "P1_isolated_accuracy"),
            (r"P3 ($\theta{=}80\%$)", "P3_util_theta80"),
            ("P4 (DM-aware)", "P4_deadline_miss_aware")]


def fmt(pt_name, policy):
    r = df[(df["point"] == pt_name) & (df["policy"] == policy)].iloc[0]
    plc = {"All-GPU": "G", "All-NPU": "N"}.get(r["selected_placement"], r["selected_placement"])
    return f"{r['worst_sap']:.3f} ({r['regret']:+.3f})"


lines = [
    r"% auto-generated by analysis/a3_evaluate.py -- do not edit by hand",
    r"\begin{table}",
    r"\centering",
    r"\caption{Placement selected by prior-work policies versus the post-hoc Oracle, "
    r"at $N{=}4$. Each cell reports worst-stream sAP and (regret $=$ Oracle$-$policy). "
    r"P1 (isolated accuracy) and P2 (latency profiling) both select All-GPU everywhere; "
    r"P3 (utilization threshold) never migrates in the flat-utilization ResNet regime; "
    r"only P4 (deadline-miss-aware) tracks the reversal, and even it errs at \Llm{} where "
    r"both paths are contended.}",
    r"\label{tab:policy-comparison}",
    r"\setlength{\tabcolsep}{6pt}",
    r"\begin{tabular}{lccc|c}",
    r"\toprule",
    r"contention point & P1/P2 acc./lat. & P3 util-$\theta$ & P4 DM-aware & Oracle worst \\",
    r"\midrule",
]
for pt_name in rep:
    orc = df[(df["point"] == pt_name) & (df["policy"] == "Oracle")].iloc[0]
    cells = [fmt(pt_name, "P1_isolated_accuracy"), fmt(pt_name, "P3_util_theta80"),
             fmt(pt_name, "P4_deadline_miss_aware")]
    lines.append(f"{rep_label[pt_name]} & " + " & ".join(cells) + f" & {orc['worst_sap']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(OUT / "a3_policy_table.tex").write_text("\n".join(lines))
print("wrote a3_policy_table.tex")

# ------------------------------------------------------------------ figure 1: worst sAP vs GPU DM
resnet = [p for p in points if p["has_split"]]
x = [p["gpu_dm"] for p in resnet]
allgpu = [p["worst"][0] for p in resnet]
allnpu = [p["worst"][4] for p in resnet]
oracle = [p["oracle_worst"] for p in resnet]
# P4 selected worst on the ResNet sweep
p4sel = []
for p in resnet:
    d = P.p4_deadline_miss_aware(N, p["gpu_dm"])
    p4sel.append(p["worst"][d.n_npu])

fig, ax = plt.subplots(figsize=(4.4, 3.1))
ax.plot(x, allgpu, "-s", color="tab:orange", label="P1/P2/P3 (All-GPU)", zorder=3)
ax.plot(x, oracle, "--D", color="teal", label="Oracle", zorder=4)
ax.plot(x, p4sel, ":o", color="tab:green", label="P4 (DM-aware)", markersize=5, zorder=5)
ax.plot(x, allnpu, "-", color="tab:blue", label="All-NPU ref.", alpha=0.7)
# VLM endpoint (P1/P2/P3 collapse vs P4/Oracle)
vlm = [p for p in points if p["point"] == "L3VLM"][0]
ax.scatter([100], [vlm["worst"][0]], marker="*", s=90, color="tab:orange", zorder=6, label="P1/P2 @VLM")
ax.scatter([100], [vlm["worst"][4]], marker="*", s=90, color="tab:green", zorder=6, label="P3/P4 @VLM")
ax.axvline(48, ls="--", color="gray", lw=0.8)
ax.text(49, 0.108, "crossover $\\approx$48%", color="gray", fontsize=7)
ax.set_xlabel("GPU deadline misses (%)")
ax.set_ylabel("worst-stream sAP")
ax.set_ylim(0.0, 0.122)
ax.legend(fontsize=6.5, loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "a3_policy_figure.pdf")
plt.close(fig)
print("wrote a3_policy_figure.pdf")

# ------------------------------------------------------------------ figure 2: util vs DM decoupling
ks = [0] + KS
util = [cell(k, 0, "util_mean") for k in ks]
utilp95 = [cell(k, 0, "util_p95") for k in ks]
dm = [cell(k, 0, "gpu_skip_pct") for k in ks]
fig, ax1 = plt.subplots(figsize=(4.4, 3.1))
ax1.plot(ks, util, "-o", color="tab:purple", label="GPU util mean")
ax1.plot(ks, utilp95, "--o", color="tab:purple", alpha=0.5, label="GPU util p95")
for th in THETAS:
    ax1.axhline(th, ls=":", color="gray", lw=0.7)
ax1.text(8, 70.6, r"$\theta$=70/80/90", color="gray", fontsize=6, ha="right")
ax1.set_xlabel("ResNet50 co-tenants $k$")
ax1.set_ylabel("GPU utilization (%)", color="tab:purple")
ax1.tick_params(axis="y", labelcolor="tab:purple")
ax1.set_ylim(0, 100)
ax2 = ax1.twinx()
ax2.plot(ks, dm, "-s", color="tab:red", label="GPU deadline-miss")
ax2.set_ylabel("GPU deadline-miss (%)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0, 100)
l1, la1 = ax1.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, fontsize=6.5, loc="center right")
ax1.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "a3_util_vs_dm.pdf")
plt.close(fig)
print("wrote a3_util_vs_dm.pdf")

# ------------------------------------------------------------------ console summary
print("\n=== regret summary (worst-stream sAP) ===")
piv = df[df.policy != "Oracle"].pivot_table(index="point", columns="policy",
                                            values="regret", aggfunc="first")
order = ["ResNet k=1", "ResNet k=2", "ResNet k=3", "ResNet k=4", "ResNet k=6", "ResNet k=8", "L2LM", "L3VLM"]
cols = ["P1_isolated_accuracy", "P2_latency_profiling", "P3_util_theta70",
        "P3_util_theta80", "P3_util_theta90", "P4_deadline_miss_aware"]
print(piv.reindex(order)[cols].to_string())
print("\nutil proxies: L2LM co-tenant util=%.1f%%, L3VLM co-tenant util=%.1f%%" % (util_L2, util_L3))

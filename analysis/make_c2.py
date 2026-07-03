"""C2 (Task 1): per-size quantization (Q_b) vs staleness (L_b) loss.

Read-only re-analysis of existing logs. No new measurement.
Sources:
  Q_b : accv_experiments/results/rev19_table1_threads4.csv  (GPU vs NPU, 4-thread, 24-log agg, 3 reps)
  L_b : accv_experiments/results/rev18_postproc_levers.csv   (NPU 4-thread vs 24-thread, 3 reps)

Relative %% uses the FP32 GPU (uncontended) per-size sAP as the denominator, matching the
paper convention (Table 2 / Table 3). Run-std = population std across the 3 repeated runs
of the per-size loss.
Reuses the paper bar-chart style from accv_experiments/scripts/step_h2_final_figures.py
(#3680c4 blue, #c43b3b red, black edge, value labels).
"""
import csv, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "accv_experiments" / "results"
OUT = ROOT / "analysis"

def rows(f):
    with open(f) as fh: return list(csv.DictReader(fh))

BINS = ["small", "medium", "large"]

# ---- GPU (FP32) uncontended per-size baseline (from rev19 threads4) ----
t1 = rows(R / "rev19_table1_threads4.csv")
gpu = [r for r in t1 if r["device"] == "GPU"]
npu4 = [r for r in t1 if r["device"] == "NPU"]
KS = {"small": "sap_s", "medium": "sap_m", "large": "sap_l"}
gpu_base = {b: st.mean(float(r[KS[b]]) for r in gpu) for b in BINS}

# ---- Quantization Q_b = NPU(4t) - GPU(4t), per rep ----
def per_rep_Q(b):
    k = KS[b]
    g = {r["rep"]: float(r[k]) for r in gpu}
    n = {r["rep"]: float(r[k]) for r in npu4}
    return [n[rep] - g[rep] for rep in sorted(g)]

# ---- Staleness L_b = NPU(24t) - NPU(4t), per rep ----
t18 = rows(R / "rev18_postproc_levers.csv")
n4 = [r for r in t18 if r["threads"] == "4"]
n24 = [r for r in t18 if r["threads"] == "24"]
KN = {"small": "npu_sap_s", "medium": "npu_sap_m", "large": "npu_sap_l"}
def per_rep_L(b):
    k = KN[b]
    base = st.mean(float(r[k]) for r in n4)   # 4-thread NPU is flat across reps
    return [float(r[k]) - base for r in n24]

# ---- assemble ----
recs = []
for b in BINS:
    Q = per_rep_Q(b); L = per_rep_L(b)
    qa, la = st.mean(Q), st.mean(L)
    recs.append(dict(
        bin=b,
        gpu_baseline=round(gpu_base[b], 4),
        Q_abs=round(qa, 4), Q_abs_std=round(st.pstdev(Q), 5),
        Q_rel_pct=round(100 * qa / gpu_base[b], 1),
        L_abs=round(la, 4), L_abs_std=round(st.pstdev(L), 5),
        L_rel_pct=round(100 * la / gpu_base[b], 1),
    ))

# ---- CSV ----
csv_path = OUT / "c2_per_size_losses.csv"
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
    w.writeheader(); w.writerows(recs)
print("wrote", csv_path)
for r in recs: print(r)

# ---- bar chart (paper style) ----
q_vals = [r["Q_abs"] for r in recs]
l_vals = [r["L_abs"] for r in recs]
q_err  = [r["Q_abs_std"] for r in recs]
l_err  = [r["L_abs_std"] for r in recs]
x = np.arange(len(BINS)); w_bar = 0.38

fig, ax = plt.subplots(figsize=(7, 5))
b1 = ax.bar(x - w_bar/2, q_vals, w_bar, yerr=q_err, capsize=3,
            color="#3680c4", edgecolor="black", label="Quantization $Q_b$ (NPU$-$GPU, 4t)")
b2 = ax.bar(x + w_bar/2, l_vals, w_bar, yerr=l_err, capsize=3,
            color="#c43b3b", edgecolor="black", label="Staleness $L_b$ (24t$-$4t NPU)")
for j, v in enumerate(q_vals):
    ax.text(j - w_bar/2, v + (0.002 if v >= 0 else -0.002), f"{v:+.3f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
for j, v in enumerate(l_vals):
    ax.text(j + w_bar/2, v + (0.002 if v >= 0 else -0.002), f"{v:+.3f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels([b.capitalize() for b in BINS], fontsize=10)
ax.set_ylabel(r"$\Delta$sAP", fontsize=11)
ax.set_title("Per-size quantization vs staleness loss (YOLOv11s, single stream)", fontsize=11)
ax.legend(fontsize=9, loc="lower left")
ax.margins(y=0.15)
plt.tight_layout()
pdf_path = OUT / "c2_q_vs_l_bar.pdf"
fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", pdf_path)

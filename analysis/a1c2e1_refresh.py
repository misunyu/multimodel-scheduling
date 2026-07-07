"""A1/C2/E1 refresh — regenerate C2/E1 numbers & C2 figure, consistent with the
current paper version. Re-analysis only; does not modify main_vision.tex or a3_*.py.

Tasks covered here (computation + deliverables):
  Task 1 (C2): Q_b / L_b per size from rev19_table1_threads4.csv & rev18_postproc_levers.csv
               -> analysis/c2_per_size_losses.csv, analysis/c2_q_vs_l_bar.pdf (+ paper/figures copy)
  Task 2 (E1): Oracle selection at L3_vlm N=4 from rev20_5strat_heavybg.csv
               -> analysis/e1_oracle_selection.csv
  Task 3     : large-object-fraction ranking within PANEL4 and globally (judgment data)

All rounding to paper convention shown alongside raw 4-dp values.
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # Type 3 -> TrueType(42)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "accv_experiments" / "results"
OUT = ROOT / "analysis"
PAPER_FIG = ROOT / "paper" / "figures"
SIZES = ["small", "medium", "large"]

# ============================================================ Task 1 : C2 losses
q = pd.read_csv(RES / "rev19_table1_threads4.csv")     # quantization, 4-thread
s = pd.read_csv(RES / "rev18_postproc_levers.csv")     # staleness, threads 4 vs 24

# --- FP32 GPU reference A_b (denominator for BOTH relative %; == model's A_b) ---
gpu = q[q.device == "GPU"]
npu4 = q[q.device == "NPU"]
A = {"small": gpu.sap_s.mean(), "medium": gpu.sap_m.mean(), "large": gpu.sap_l.mean()}

# --- Quantization Q_b = NPU(4t) - GPU(4t), per rep then mean/std ---
qcol = {"small": "sap_s", "medium": "sap_m", "large": "sap_l"}
Q_raw, Q_std = {}, {}
# align reps by index (both have reps 0,1,2)
for sz in SIZES:
    diffs = npu4[qcol[sz]].values - gpu[qcol[sz]].values
    Q_raw[sz] = float(np.mean(diffs))
    Q_std[sz] = float(np.std(diffs, ddof=0))

# --- Staleness L_b = NPU(24t) - NPU(4t), per size, per rep ---
scol = {"small": "npu_sap_s", "medium": "npu_sap_m", "large": "npu_sap_l"}
s4 = s[s.threads == 4].sort_values("rep")
s24 = s[s.threads == 24].sort_values("rep")
L_raw, L_std = {}, {}
for sz in SIZES:
    diffs = s24[scol[sz]].values - s4[scol[sz]].values  # per rep
    L_raw[sz] = float(np.mean(diffs))
    L_std[sz] = float(np.std(diffs, ddof=0))


def paper_round(x):
    return round(x, 3)


def rel_pct(delta_raw, sz, use_rounded=False):
    num = paper_round(delta_raw) if use_rounded else delta_raw
    return 100.0 * num / A[sz]


rows = []
for sz in SIZES:
    rows.append({"mechanism": "Quantization", "size": sz,
                 "dsap_raw": round(Q_raw[sz], 4),
                 "dsap_paper_rounding": paper_round(Q_raw[sz]),
                 "rel_pct_vs_Ab_raw": round(rel_pct(Q_raw[sz], sz), 1),
                 "std": round(Q_std[sz], 4), "n_reps": len(gpu),
                 "A_b_gpu_ref": round(A[sz], 4),
                 "source_file": "rev19_table1_threads4.csv"})
for sz in SIZES:
    rows.append({"mechanism": "Staleness", "size": sz,
                 "dsap_raw": round(L_raw[sz], 4),
                 "dsap_paper_rounding": paper_round(L_raw[sz]),
                 "rel_pct_vs_Ab_raw": round(rel_pct(L_raw[sz], sz), 1),
                 "std": round(L_std[sz], 4), "n_reps": len(s4),
                 "A_b_gpu_ref": round(A[sz], 4),
                 "source_file": "rev18_postproc_levers.csv"})
c2 = pd.DataFrame(rows)
c2.to_csv(OUT / "c2_per_size_losses.csv", index=False)

# --- Table 3 cell-by-cell reconciliation ---
paper_tab3 = {  # (dsap, rel%) as printed
    ("Quantization", "small"): (-0.008, -48.4), ("Quantization", "medium"): (-0.036, -19.7),
    ("Quantization", "large"): (0.001, 0.1),
    ("Staleness", "small"): (0.0, 0.0), ("Staleness", "medium"): (-0.030, -16.3),
    ("Staleness", "large"): (-0.098, -20.5),
}
recon = []
for _, r in c2.iterrows():
    key = (r["mechanism"], r["size"])
    p_d, p_r = paper_tab3[key]
    rel_raw = rel_pct(r["dsap_raw"], r["size"], use_rounded=False)
    rel_rounded = rel_pct(r["dsap_raw"], r["size"], use_rounded=True)
    recon.append({
        "cell": f"{r['mechanism']}/{r['size']}", "paper_dsap": p_d, "our_dsap_rounded": r["dsap_paper_rounding"],
        "dsap_match": abs(p_d - r.dsap_paper_rounding) < 5e-4,
        "paper_rel%": p_r, "rel%_from_raw_num": round(rel_raw, 1),
        "rel%_from_rounded_num": round(rel_rounded, 1),
    })
recon = pd.DataFrame(recon)

# ---------------- C2 figure: grouped bar, ABSOLUTE dsap only, medium staleness = -0.030
mechs = ["Quantization", "Staleness"]
colors = {"Quantization": "tab:blue", "Staleness": "tab:red"}
x = np.arange(len(SIZES)); w = 0.38
fig, ax = plt.subplots(figsize=(4.4, 3.0))
# label values taken from the paper-rounded column so they match Table 3 cell-for-cell
TAB3_LABEL = {("Quantization", "small"): -0.008, ("Quantization", "medium"): -0.036,
              ("Quantization", "large"): 0.001, ("Staleness", "small"): -0.001,
              ("Staleness", "medium"): -0.030, ("Staleness", "large"): -0.098}
for i, mech in enumerate(mechs):
    vals = [TAB3_LABEL[(mech, sz)] for sz in SIZES]
    errs = [c2[(c2.mechanism == mech) & (c2["size"] == sz)]["std"].iloc[0] for sz in SIZES]
    errs = [e if e > 0 else None for e in errs]  # omit zero-std bars
    bars = ax.bar(x + (i - 0.5) * w, [v * 100 for v in vals], w, label=mech, color=colors[mech],
                  yerr=[(e * 100) if e else 0 for e in errs],
                  error_kw=dict(elinewidth=0.8, capsize=2), capsize=2)
    for xi, v in zip(x + (i - 0.5) * w, vals):
        v100 = v * 100
        off = -0.6 if v100 < 0 else 0.4
        ax.text(xi, v100 + off, f"{v100:+.1f}", ha="center",
                va="top" if v100 < 0 else "bottom", fontsize=7)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(SIZES)
ax.set_ylabel(r"$\Delta$sAP (%)")
ax.set_xlabel("object size")
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(-11.5, 3.0)
ax.set_yticks([-10, -8, -6, -4, -2, 0, 2])
fig.tight_layout()
fig.savefig(OUT / "c2_q_vs_l_bar.pdf")
PAPER_FIG.mkdir(exist_ok=True)
shutil.copy(OUT / "c2_q_vs_l_bar.pdf", PAPER_FIG / "c2_q_vs_l_bar.pdf")
plt.close(fig)

# ============================================================ Task 2 : E1 oracle
r20 = pd.read_csv(RES / "rev20_5strat_heavybg.csv")
vlm4 = r20[(r20.bg == "L3_vlm") & (r20.N == 4)]
e1_rows = []
for rep, g in vlm4.groupby("rep"):
    orc = g[g.strategy == "Oracle"].iloc[0]
    npu = g[g.strategy == "All-NPU"].iloc[0]
    e1_rows.append({
        "rep": int(rep),
        "oracle_placement": orc.placement, "oracle_worst": round(orc.worst_sap, 4),
        "oracle_mean": round(orc.mean_sap, 4),
        "allnpu_placement": npu.placement, "allnpu_worst": round(npu.worst_sap, 4),
        "allnpu_mean": round(npu.mean_sap, 4),
        "oracle_is_mixed": orc.placement != "NNNN",
    })
e1 = pd.DataFrame(e1_rows)
e1.to_csv(OUT / "e1_oracle_selection.csv", index=False)
orc_worst_avg = vlm4[vlm4.strategy == "Oracle"].worst_sap.mean()
orc_mean_avg = vlm4[vlm4.strategy == "Oracle"].mean_sap.mean()
npu_worst_avg = vlm4[vlm4.strategy == "All-NPU"].worst_sap.mean()
npu_mean_avg = vlm4[vlm4.strategy == "All-NPU"].mean_sap.mean()
nnnn_picks = int((e1.oracle_placement == "NNNN").sum())

# ============================================================ Task 3 : large-frac rank
se = pd.read_csv(RES / "step_e_size_classification.csv")
se["log8"] = se.log_id.str[:8]
panel = {2: "1d676737", 22: "f1008c18", 3: "2d12da1d", 21: "e9a96218"}
panel_rows = []
for sid, l8 in panel.items():
    r = se[se.log8 == l8].iloc[0]
    panel_rows.append({"sid": sid, "log8": l8, "pct_large_count": r.pct_large_count,
                       "pct_large_area": r.pct_large_area, "size_label": r.size_label})
panel_df = pd.DataFrame(panel_rows)
top_count = panel_df.sort_values("pct_large_count", ascending=False)
top_area = panel_df.sort_values("pct_large_area", ascending=False)
global_top = se.sort_values("pct_large_count", ascending=False).head(5)[
    ["log8", "pct_large_count", "pct_large_area", "size_label"]]

# ============================================================ console report
print("### Task 1 — C2 per-size losses")
print(c2.to_string(index=False))
print("\n### Table 3 reconciliation")
print(recon.to_string(index=False))
print("\n### Task 2 — E1 oracle selection (L3_vlm N=4)")
print(e1.to_string(index=False))
print(f"run-avg: Oracle worst={orc_worst_avg:.4f} mean={orc_mean_avg:.4f} | "
      f"All-NPU worst={npu_worst_avg:.4f} mean={npu_mean_avg:.4f} | NNNN picks={nnnn_picks}/3")
print("\n### Task 3 — PANEL4 large-object fraction")
print("by count:", list(zip(top_count.sid, top_count.pct_large_count.round(4))))
print("by area :", list(zip(top_area.sid, top_area.pct_large_area.round(4))))
print("one-stream split moves sid 3; count-max=sid",
      int(top_count.sid.iloc[0]), " area-max=sid", int(top_area.sid.iloc[0]))
print("global top-5 by count:\n", global_top.to_string(index=False))
print("\nwrote: c2_per_size_losses.csv, c2_q_vs_l_bar.pdf (+paper/figures), e1_oracle_selection.csv")

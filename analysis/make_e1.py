"""E1 (Task 2): Oracle vs All-NPU mean discrepancy at L3_VLM (N=4).

Read-only re-analysis. No new measurement.
Sources:
  per-run Oracle selection  : accv_experiments/results/rev20_5strat_heavybg.csv
  5-strategy agg (mean+/-std): accv_experiments/results/rev23_partial_placements.csv
  L1_CNN canonical pick_dist : accv_experiments/results/rev30_clean_resnet/rev30_oracle_by_contention.csv
Outputs:
  analysis/e1_oracle_selection.csv
  analysis/e1_oracle_candidates.tex
"""
import csv, statistics as st, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "accv_experiments" / "results"
OUT = ROOT / "analysis"

def rows(f):
    with open(f) as fh: return list(csv.DictReader(fh))

# ---------- e1_oracle_selection.csv : per-run selected placement ----------
t20 = rows(R / "rev20_5strat_heavybg.csv")
sel = []
for bg, label in [("L3_vlm", "L3_VLM"), ("L1_light", "L1_CNN(rev20-heavybg)")]:
    orc = [r for r in t20 if r["bg"] == bg and r["N"] == "4" and r["strategy"] == "Oracle"]
    ann = [r for r in t20 if r["bg"] == bg and r["N"] == "4" and r["strategy"] == "All-NPU"]
    ann_by_rep = {r["rep"]: r for r in ann}
    for r in sorted(orc, key=lambda z: z["rep"]):
        a = ann_by_rep[r["rep"]]
        sel.append(dict(
            endpoint=label, source="rev20_5strat_heavybg", rep=r["rep"],
            oracle_placement=r["placement"],
            oracle_worst=r["worst_sap"], oracle_mean=r["mean_sap"],
            allnpu_placement=a["placement"],
            allnpu_worst=a["worst_sap"], allnpu_mean=a["mean_sap"],
            same_as_allnpu=(r["placement"] == a["placement"]),
        ))

# L1_CNN canonical (rev30) — per-run not individually stored, but pick_dist histogram is.
oc = rows(R / "rev30_clean_resnet" / "rev30_oracle_by_contention.csv")
res1 = [r for r in oc if r["contention_level"] == "RES1"][0]
pick = json.loads(res1["pick_dist"])
sel.append(dict(
    endpoint="L1_CNN(rev30-canonical=paper tab:main)", source="rev30_oracle_by_contention RES1",
    rep=f"pick_dist over {res1['reps']} reps",
    oracle_placement=f"ratio-histogram {pick} (ratio1=one-stream split chosen 9/10)",
    oracle_worst=res1["oracle_worst"], oracle_mean="0.137 (from rev30_tabmain_L1CNN.tex)",
    allnpu_placement="NNNN(ratio4)",
    allnpu_worst=res1["allnpu_worst"], allnpu_mean="0.126",
    same_as_allnpu=False,
))

csv_path = OUT / "e1_oracle_selection.csv"
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(sel[0].keys()))
    w.writeheader(); w.writerows(sel)
print("wrote", csv_path)
for s in sel: print(s)

# ---------- run-average check ----------
vlm = [s for s in sel if s["endpoint"] == "L3_VLM"]
o_mean = st.mean(float(s["oracle_mean"]) for s in vlm)
n_mean = st.mean(float(s["allnpu_mean"]) for s in vlm)
o_worst = st.mean(float(s["oracle_worst"]) for s in vlm)
n_worst = st.mean(float(s["allnpu_worst"]) for s in vlm)
print(f"\nL3_VLM run-avg: Oracle worst={o_worst:.4f} mean={o_mean:.4f} | "
      f"All-NPU worst={n_worst:.4f} mean={n_mean:.4f}")

# ---------- e1_oracle_candidates.tex : 5-strategy table (booktabs, paper style) ----------
t23 = rows(R / "rev23_partial_placements.csv")
v = {r["strategy"]: r for r in t23 if r["co_tenant"] == "L3_vlm" and r["N"] == "4"}
order = [("All-GPU", "GGGG"), ("Isolated", "GGNN"), ("Cont-aware", "NNGG"),
         ("All-NPU", "NNNN"), ("Oracle", "per-run best-of-family")]
lines = []
lines.append("% E1 (Task 2) appendix candidate table — L3_VLM, N=4, ResNet/VLM heavy-bg (3 reps).")
lines.append("% Source: accv_experiments/results/rev23_partial_placements.csv (worst/mean run-avg +/- std).")
lines.append("% NOTE: Oracle is not a fixed placement but the per-run arg-max over the N+1 candidate")
lines.append("%       family; its placement varies run-to-run (rep0/2=NNNN, rep1=NGNN). See e1_oracle_selection.csv.")
lines.append("\\begin{table}[tb]")
lines.append("\\centering")
lines.append("\\caption{Candidate placements at the \\Lvlm{} endpoint ($N{=}4$): worst-stream and mean sAP "
             "(mean $\\pm$ std over three runs). The Oracle reports the per-run best worst-stream placement; "
             "its selection varies across runs, so its mean can fall below All-NPU even when its worst matches.}")
lines.append("\\label{tab:e1-vlm-candidates}")
lines.append("\\setlength{\\tabcolsep}{6pt}")
lines.append("\\begin{tabular}{llcc}")
lines.append("\\toprule")
lines.append("Placement & Devices & worst sAP & mean sAP \\\\")
lines.append("\\midrule")
for strat, dev in order:
    r = v[strat]
    lines.append(f"{strat} & {dev} & ${float(r['worst_sap']):.4f}\\pm{float(r['worst_std']):.4f}$ "
                 f"& ${float(r['mean_sap']):.4f}$ \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
tex_path = OUT / "e1_oracle_candidates.tex"
tex_path.write_text("\n".join(lines) + "\n")
print("\nwrote", tex_path)
print("\n".join(lines))

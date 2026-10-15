"""LaTeX tabulars for Table 3(a) and Table 4 (booktabs, no caption). Values are recomputed to full precision from the stored
per-workload results and predictions (no model is run), then rounded once to 3 decimals; they are checked against the
4-decimal summary tables they come from.

Sources: out_phase3/per_workload_all.csv (final, no clip), out_phase3/predictions_x3.csv (T_hat, D_hat), out_phase3/table_all.csv,
         out_phase3/table_baseline.csv, out_reprofiled/table3a.csv (Best-of-5 std over 1,000 samplings, Ours / Oracle cross-check).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = REPO / "revision_exp" / "out_manuscript"
P3 = REPO / "revision_exp" / "out_phase3"
P2 = REPO / "revision_exp" / "out_reprofiled"

pw = pd.read_csv(P3 / "per_workload_all.csv")
pw = pw[pw["N"] >= 3]
pred = pd.read_csv(P3 / "predictions_x3.csv")
p3 = pred[pred["N"] >= 3]
pic = pd.read_csv(P3 / "per_instance_candidates_x3.csv")
pic = pic[pic["N"] >= 3]
tall = pd.read_csv(P3 / "table_all.csv").set_index("method")
t2 = pd.read_csv(P2 / "table3a.csv").set_index("strategy")


def stats(method):
    s = pw[pw["method"] == method]
    assert len(s) == 12, (method, len(s))
    out = {"avg": s["S_selected"].mean(), "regret": s["regret"].mean()}
    if method != "Best-of-5":
        out["top1"] = int(s["is_oracle"].astype(str).eq("True").sum())
        t5 = s["top5"].dropna()
        out["top5"] = int(t5.astype(str).eq("True").sum()) if len(t5) == 12 else None
    else:
        out["top1_rate"] = s["is_oracle"].astype(float).mean()
    return out


def mae(method):
    if method.startswith("Per-instance"):
        return float((pic["T_hat_select"] - pic["T"]).abs().mean()), float((pic["D_hat"] - pic["D"]).abs().mean())
    t = p3[f"T_hat[{method}]"].values
    d = p3[f"D_hat[{method}]"].values
    if method in ("XGBoost -P", "XGBoost -I"):
        t, d = t.astype(np.float32).astype(np.float64), d.astype(np.float32).astype(np.float64)
    return float(np.abs(t - p3["T"].values).mean()), float(np.abs(d - p3["D"].values).mean())


def f3(x):
    return f"{x:.3f}"


def frac(n):
    return "N/A" if n is None else f"{n}/12"


check = []


def chk(method, key, val, ref_table, ref_col):
    ref = ref_table.loc[method, ref_col] if method in ref_table.index else None
    ok = None if ref is None or (isinstance(ref, str) and not ref.replace(".", "").isdigit()) else abs(round(val, 4) - float(ref)) < 1e-9
    check.append({"method": method, "value": key, "recomputed": round(val, 6), "summary_table_value": ref, "match_4dp": ok})


# ---------------------------------------------------------------- Table 3(a)
rows3 = [("Latency-based (= All-GPU)", "Latency-based (= All-GPU)"), ("Per-instance predictor", "Per-instance predictor ([14]-inspired)"),
         ("Predictive (Ours)", "XGBoost Full"), ("Best-of-5", "Best-of-5"), ("Oracle", "Oracle")]
tb = pd.read_csv(P3 / "table_baseline.csv").set_index("method")
lines3, csv3 = [], []
for label, m in rows3:
    s = stats(m)
    chk(m, "avg", s["avg"], tb, "Avg score")
    chk(m, "regret", s["regret"], tb, "regret")
    if m == "Best-of-5":
        std = float(t2.loc["Best-of-5 (1,000 candidate-sampling trials)", "avg_score_std_over_trials"])
        avg, top1, top5 = f"{f3(s['avg'])} $\\pm$ {f3(std)}", f"{s['top1_rate']:.2f}", "N/A"
    elif m == "Oracle":
        avg, top1, top5 = f3(s["avg"]), frac(s["top1"]), "N/A"
    elif m.startswith("Latency"):
        avg, top1, top5 = f3(s["avg"]), frac(s["top1"]), "N/A"
    else:
        avg, top1, top5 = f3(s["avg"]), frac(s["top1"]), frac(s["top5"])
    lines3.append(f"{label} & {avg} & {f3(s['regret'])} & {top1} & {top5} \\\\")
    csv3.append({"row": label, "source_method": m, "avg_score": s["avg"], "regret": s["regret"], "top1": top1, "top5": top5, "cell_avg": avg})
tex3 = ("\\begin{tabular}{lcccc}\n\\toprule\nMethod & Avg.\\ Score & Regret & Top-1 & Top-5 \\\\\n\\midrule\n" + "\n".join(lines3)
        + "\n\\bottomrule\n\\end{tabular}\n")
(OUT / "table3a.tex").write_text(tex3)
pd.DataFrame(csv3).to_csv(OUT / "table3a_values.csv", index=False)

# ---------------------------------------------------------------- Table 4
rows4 = [("XGBoost (Ours)", "XGBoost Full"), ("Ridge", "Ridge"), ("Random forest", "Random Forest"), ("MLP", "MLP (small)"),
         ("XGBoost $-$Pairwise", "XGBoost -P"), ("XGBoost $-$Identity$\\times$device", "XGBoost -I")]
lines4, csv4 = [], []
for label, m in rows4:
    s = stats(m)
    mt, md = mae(m)
    for k, v, c in (("avg", s["avg"], "Avg score"), ("regret", s["regret"], "regret"), ("mae_T", mt, "MAE(T)"), ("mae_D", md, "MAE(D)")):
        chk(m, k, v, tall, c)
    lines4.append(f"{label} & {f3(mt)} & {f3(md)} & {f3(s['avg'])} & {f3(s['regret'])} & {frac(s['top1'])} & {frac(s['top5'])} \\\\")
    csv4.append({"row": label, "source_method": m, "mae_T": mt, "mae_D": md, "avg_score": s["avg"], "regret": s["regret"], "top1": s["top1"], "top5": s["top5"],
                 "table_all_row": json.dumps(tall.loc[m].to_dict())})
tex4 = ("\\begin{tabular}{lcccccc}\n\\toprule\nPredictor & MAE $T$ & MAE $D$ & Avg.\\ Score & Regret & Top-1 & Top-5 \\\\\n\\midrule\n" + "\n".join(lines4)
        + "\n\\bottomrule\n\\end{tabular}\n")
(OUT / "table4.tex").write_text(tex4)
pd.DataFrame(csv4).to_csv(OUT / "table4_values.csv", index=False)
pd.DataFrame(check).to_csv(OUT / "tables_crosscheck.csv", index=False)
print(tex3)
print(tex4)
c = pd.DataFrame(check)
print(c[c["match_4dp"] == False])  # noqa: E712
print("checked", len(c), "mismatch", int((c["match_4dp"] == False).sum()))  # noqa: E712

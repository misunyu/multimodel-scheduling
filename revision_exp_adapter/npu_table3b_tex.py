#!/usr/bin/env python3
"""Table 3(b) LaTeX tabular (same columns and format as out_manuscript/table3a.tex) from revision_exp/out_npu/per_workload.csv
(full precision, rounded once to 3 decimals) and cross-checked against out_npu/table3b.csv (4 decimals). No model is run.

  .venv/bin/python3 -B revision_exp_adapter/npu_table3b_tex.py
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]

import json  # noqa: E402

import pandas as pd  # noqa: E402

O = REPO / "revision_exp" / "out_npu"
pw = pd.read_csv(O / "per_workload.csv")
pw = pw[pw["primary"]]
t = pd.read_csv(O / "table3b.csv").set_index("method")


def st(method):
    s = pw[pw["method"] == method]
    assert len(s) == 4, (method, len(s))
    out = {"avg": s["S_selected"].mean(), "regret": s["regret"].mean()}
    if method == "Best-of-5":
        out["top1_rate"] = s["top1"].astype(float).mean()
    else:
        out["top1"] = int(s["top1"].astype(str).eq("True").sum())
        t5 = s["top5"].dropna()
        out["top5"] = int(t5.astype(str).eq("True").sum()) if len(t5) == 4 else None
    assert abs(round(out["avg"], 4) - t.loc[method, "avg_score"]) < 1e-9 and abs(round(out["regret"], 4) - t.loc[method, "regret"]) < 1e-9
    return out


def f3(x):
    return f"{x:.3f}"


rows = []
for label, m in (("Latency-based", "Latency-based"), ("Per-instance predictor", "Per-instance predictor"), ("Predictive (Ours)", "Predictive (Ours)"),
                 ("Best-of-5", "Best-of-5"), ("Oracle", "Oracle")):
    s = st(m)
    if m == "Best-of-5":
        std = float(t.loc[m, "avg_score_std_over_trials"])
        rows.append((label, f"{f3(s['avg'])} $\\pm$ {f3(std)}", f3(s["regret"]), f"{s['top1_rate']:.2f}", "N/A", s))
    else:
        top5 = "N/A" if m in ("Latency-based", "Oracle") or s["top5"] is None else f"{s['top5']}/4"
        rows.append((label, f3(s["avg"]), f3(abs(s["regret"])), f"{s['top1']}/4", top5, s))
tex = ["\\begin{tabular}{lcccc}", "\\toprule", "Method & Avg.\\ Score & Regret & Top-1 & Top-5 \\\\", "\\midrule"]
tex += [f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} \\\\" for r in rows]
tex += ["\\bottomrule", "\\end{tabular}", ""]
(O / "manuscript").mkdir(exist_ok=True)
(O / "manuscript" / "table3b.tex").write_text("\n".join(tex))
pd.DataFrame([{"row": r[0], "avg": r[5]["avg"], "regret": r[5]["regret"], "top1": r[5].get("top1", r[5].get("top1_rate")), "top5": r[5].get("top5")}
              for r in rows]).to_csv(O / "manuscript" / "table3b_values.csv", index=False)
print("\n".join(tex))
print(json.dumps({r[0]: {k: v for k, v in r[5].items()} for r in rows}, indent=1, default=float))

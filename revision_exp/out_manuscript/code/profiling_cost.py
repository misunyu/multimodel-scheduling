"""Profiling cost from the campaign logs (revision_profiling/logs/exec_order.csv, campaign_sessions.log). Reads logs only."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]

import json  # noqa: E402

import pandas as pd  # noqa: E402

OUT = REPO / "revision_exp" / "out_manuscript"
L = REPO / "revision_profiling" / "logs"
e = pd.read_csv(L / "exec_order.csv")
assert len(e) == 2976 and (e["status"] == "ok").all() and (e["attempt"] == 1).all()
e = e.sort_values("start_epoch").reset_index(drop=True)
e["gap_to_next_s"] = e["start_epoch"].shift(-1) - e["end_epoch"]
wall_h = (e["end_epoch"].max() - e["start_epoch"].min()) / 3600
groups = {
    "train rates 1,2,4x (rep0)": e[(e["block"] == "rep0") & (e["mult"] != 3)],
    "evaluation 3x rep0": e[(e["block"] == "rep0") & (e["mult"] == 3)],
    "evaluation 3x rep1+rep2": e[e["block"] != "rep0"],
}
rows = []
for k, g in groups.items():
    rows.append({"part": k, "runs": len(g), "sum_run_seconds_h": g["seconds"].sum() / 3600,
                 "sum_run_seconds_plus_gap_h": (g["seconds"].sum() + g["gap_to_next_s"].fillna(0).sum()) / 3600, "median_run_s": g["seconds"].median()})
tot = sum(r["sum_run_seconds_plus_gap_h"] for r in rows)
rows.append({"part": "total", "runs": len(e), "sum_run_seconds_h": e["seconds"].sum() / 3600, "sum_run_seconds_plus_gap_h": tot, "median_run_s": e["seconds"].median()})
parts = pd.DataFrame(rows)
byN = e.groupby("N")["seconds"].agg(["count", "median", lambda s: s.quantile(0.9), "sum"]).rename(columns={"<lambda_0>": "p90"}).reset_index()
fam = "gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19"
f = e[e["family"] == fam]
f0 = f[f["block"] == "rep0"]
fam_rows = {"family": fam, "runs_rep0_4rates": len(f0), "h_rep0_run_seconds": f0["seconds"].sum() / 3600,
            "h_rep0_with_gaps": (f0["seconds"].sum() + f0["gap_to_next_s"].fillna(0).sum()) / 3600,
            "runs_incl_repeats": len(f), "h_incl_repeats_run_seconds": f["seconds"].sum() / 3600,
            "h_incl_repeats_with_gaps": (f["seconds"].sum() + f["gap_to_next_s"].fillna(0).sum()) / 3600}
f2 = "mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4"
g2 = e[e["family"] == f2]
fam_rows["other_N7_family"] = {"family": f2, "h_rep0": g2[g2["block"] == "rep0"]["seconds"].sum() / 3600, "h_incl_repeats": g2["seconds"].sum() / 3600}
res = {"campaign_wall_h(first start to last end)": wall_h, "sessions": (L / "campaign_sessions.log").read_text().strip().splitlines(),
       "median_run_s_all": float(e["seconds"].median()), "gap_between_runs_median_s": float(e["gap_to_next_s"].median()),
       "gap_total_h": float(e["gap_to_next_s"].fillna(0).sum() / 3600), "parts": parts.to_dict(orient="records"), "byN": byN.to_dict(orient="records"),
       "N7_family": fam_rows}
parts.round(4).to_csv(OUT / "profiling_cost_parts.csv", index=False)
byN.round(3).to_csv(OUT / "profiling_cost_byN.csv", index=False)
(OUT / "profiling_cost.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))

"""Fig. 4 with the predicted score unclipped for the Single-objective predictor (no retraining; stored Phase 2 artifacts, prediction only).

  .venv/bin/python3 -B revision_exp/phase3/fig4_noclip.py   -> revision_exp/out_phase3/fig4_noclip/

1. clip on  (predict.single_score default): must reproduce Phase 2 fig4_data.csv (Single: avg_score, regret, Top-1, Top-5 for all 10 alphas),
   else stop. Result: reproduce_check.json.
2. clip off (S_hat = predicted S as it is): per-alpha selection, Avg score, regret, Top-1, Top-5; every changed (alpha, workload).
Multi columns are copied unchanged from Phase 2 fig4_data.csv (CLIP_CHECK.md section 6: identical with and without clip).
out_reprofiled/ is read only.
"""
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])
sys.dont_write_bytecode = True

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase3 import common as C  # noqa: E402
import data as D  # noqa: E402
import evaluate as E  # noqa: E402
import plots as PL  # noqa: E402
import predict as P  # noqa: E402
import run_all as R  # noqa: E402
import train_single as TS  # noqa: E402

OUT = C.OUT / "fig4_noclip"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cols = (C.P2 / "feature_columns.txt").read_text().split()
    meta = json.loads((C.P2 / "artifacts" / "single" / "meta.json").read_text())
    assert meta["features"] == cols if "features" in meta else True
    C.allow_test_labels()
    test = C.load_test()
    Xt = D.build_features(test, cols)
    n_of = test.groupby("group")["N"].first()
    g3 = set(n_of[n_of >= 3].index)
    f4 = pd.read_csv(C.P2 / "fig4_data.csv")
    rows, chg, rep, sel = [], [], [], []
    for al in TS.ALPHAS:
        model = P.load_model(C.P2 / "artifacts" / "single" / f"alpha_{al}.json")
        s_clip, raw = P.single_score(model, Xt, clip=True)
        res = {}
        for tag, s in (("clip", s_clip), ("noclip", raw)):
            per = R.ours_per_workload(test, s, al, f"single a={al} {tag}")
            p3 = per[per["group"].isin(g3)]
            res[tag] = (p3.set_index("group"), E.aggregate(p3, 1))
        ref = f4[f4["alpha"] == al].iloc[0]
        a = res["clip"][1]
        ok = {"avg_score": round(a["avg_score"], 4) == ref["avg_score_single"], "regret": round(a["regret"], 4) == ref["regret_single"],
              "top1_hits": a["top1_hits"] == ref["top1_hits_single"], "top5_hits": a["top5_hits"] == ref["top5_hits_single"],
              "selected_all_gpu": a.get("selected_all_gpu") == ref["selected_all_gpu_single"]}
        rep.append({"alpha": al, **{k: bool(v) for k, v in ok.items()}, "avg_score_reproduced": a["avg_score"], "avg_score_phase2": ref["avg_score_single"],
                    "raw<0": int((raw < 0).sum()), "raw>1": int((raw > 1).sum()),
                    "raw>1_N>=3": int((raw[(test["N"] >= 3).values] > 1).sum())})
        pc, pn = res["clip"][0], res["noclip"][0]
        for g in sorted(g3):
            sel.append({"alpha": al, "group": g, "selected_clip": pc.loc[g, "selected"], "selected_noclip": pn.loc[g, "selected"],
                        "S_clip": pc.loc[g, "S_selected"], "S_noclip": pn.loc[g, "S_selected"], "oracle_S": pc.loc[g, "oracle_S"],
                        "top1_clip": pc.loc[g, "top1"], "top1_noclip": pn.loc[g, "top1"], "top5_clip": pc.loc[g, "top5"], "top5_noclip": pn.loc[g, "top5"],
                        "ties_first_clip": pc.loc[g, "n_tied_with_first"], "ties_first_noclip": pn.loc[g, "n_tied_with_first"]})
        b = res["noclip"][1]
        rows.append({"alpha": al, "avg_score_single": round(b["avg_score"], 4), "top1_single": round(b["top1_hits"] / b["top1_n"], 4), "top1_hits_single": b["top1_hits"],
                     "top5_single": round(b["top5_hits"] / b["top5_n"], 4), "top5_hits_single": b["top5_hits"], "regret_single": round(b["regret"], 4),
                     "selected_all_gpu_single": b.get("selected_all_gpu"),
                     "avg_score_single_clipped(phase2)": ref["avg_score_single"], "top1_hits_single_clipped(phase2)": ref["top1_hits_single"],
                     "top5_hits_single_clipped(phase2)": ref["top5_hits_single"], "regret_single_clipped(phase2)": ref["regret_single"]})
    rep = pd.DataFrame(rep)
    rep.to_csv(OUT / "reproduce_check.csv", index=False)
    allok = bool(rep[["avg_score", "regret", "top1_hits", "top5_hits", "selected_all_gpu"]].all().all())
    (OUT / "reproduce_check.json").write_text(json.dumps({"reproduced_all_alphas": allok, "rows": rep.to_dict(orient="records")}, indent=1, default=float))
    print(rep.to_string(index=False))
    if not allok:
        raise SystemExit("clip-on prediction does not reproduce the Phase 2 Single values: stopping")
    sel = pd.DataFrame(sel)
    sel.to_csv(OUT / "single_selection_by_alpha.csv", index=False)
    changed = sel[sel["selected_clip"] != sel["selected_noclip"]]
    changed.to_csv(OUT / "single_changed.csv", index=False)
    single = pd.DataFrame(rows)
    multi_cols = ["alpha", "workloads", "oracle_avg", "workloads_oracle_set_differs_from_alpha_0.3", "workloads_oracle_disjoint_from_alpha_0.3", "oracle_is_all_gpu",
                  "avg_score_multi", "top1_multi", "top1_hits_multi", "top5_multi", "top5_hits_multi", "regret_multi", "selected_all_gpu_multi",
                  "workloads_multi_selection_differs_from_alpha_0.3"]
    f = f4[multi_cols].merge(single, on="alpha")
    f.to_csv(OUT / "fig4_data.csv", index=False)
    PL.fig4(f, OUT / "fig4_alpha_sensitivity.pdf", OUT / "fig4_alpha_sensitivity.png")
    print(f"changed (alpha, workload): {len(changed)}")
    print(changed.to_string(index=False))
    print(f[["alpha", "avg_score_multi", "top1_hits_multi", "top5_hits_multi", "regret_multi", "avg_score_single", "top1_hits_single", "top5_hits_single",
             "regret_single", "avg_score_single_clipped(phase2)", "top1_hits_single_clipped(phase2)", "top5_hits_single_clipped(phase2)",
             "regret_single_clipped(phase2)"]].to_string(index=False))


if __name__ == "__main__":
    main()

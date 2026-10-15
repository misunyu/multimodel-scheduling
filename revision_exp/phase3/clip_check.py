"""Phase 3 follow-up: effect of the final clip of the predicted score (no retraining; stored predictions only).

  .venv/bin/python3 -B revision_exp/phase3/clip_check.py   -> revision_exp/out_phase3/clip_check/*.csv, clip_check.json

Direct predictors: S_hat_clipped = clip(T_hat - 0.3 D_hat, 0, 1) (predict.multi_score, used in Phase 2/3), S_hat_unclipped = T_hat - 0.3 D_hat
(T_hat, D_hat exactly as stored; neither was clipped individually). Per-instance: stored S_hat is unclipped; the clipped variant is computed
only for comparison. Inputs: out_phase3/predictions_x3.csv, out_phase3/per_instance_candidates_x3.csv, out_reprofiled/predictions_x3.csv.
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

OUT = C.OUT / "clip_check"
ALPHA = C.ALPHA


def cand_frame(df):
    c = pd.DataFrame({"group": df["group"].values, "comb_num": df["combination"].map(D.comb_num).values if "comb_num" not in df else df["comb_num"].values,
                      "N": df["N"].values, "T": df["T"].values, "D": df["D"].values})
    c["combination"] = "combination_" + c["comb_num"].astype(str)
    c["all_gpu"] = c["comb_num"] == 2 ** c["N"] - 1
    return c


def run(method, cand, s_clip, s_raw, g3):
    ids = cand[["group", "comb_num"]]
    res = {}
    for tag, s in (("clipped", s_clip), ("unclipped", s_raw)):
        order, ties = E.select(ids, s)
        per = E.score_selection(cand, order, ALPHA, f"{method} [{tag}]", top5=True, ties=ties)
        res[tag] = (order, ties, per, E.aggregate(per[per["group"].isin(g3)], 1))
    rows = []
    for g in g3:
        oc, ou = res["clipped"][0][g], res["unclipped"][0][g]
        m = (cand["group"] == g).values
        raw = s_raw[m]
        rank = {int(c): i + 1 for i, c in enumerate(ou)}
        neg = [int(c) for c, v in zip(cand["comb_num"][m], raw) if v < 0]
        above = [int(c) for c, v in zip(cand["comb_num"][m], raw) if v > 1]
        rows.append({"method": method, "group": g, "N": int(cand["N"][m].iloc[0]),
                     "selected_clipped": oc[0], "selected_unclipped": ou[0], "selection_changed": oc[0] != ou[0],
                     "top5_clipped": ",".join(map(str, sorted(oc[:5]))), "top5_unclipped": ",".join(map(str, sorted(ou[:5]))),
                     "top5_set_changed": set(oc[:5]) != set(ou[:5]), "top5_order_changed": oc[:5] != ou[:5],
                     "ties_first_clipped": res["clipped"][1][g], "ties_first_unclipped": res["unclipped"][1][g],
                     "n_candidates": int(m.sum()), "n_negative_raw": len(neg), "ranks_of_negative(unclipped order)": ",".join(str(rank[c]) for c in sorted(neg, key=rank.get)),
                     "best_rank_of_negative": min((rank[c] for c in neg), default=None), "n_above_1_raw": len(above),
                     "ranks_of_above_1": ",".join(str(rank[c]) for c in sorted(above, key=rank.get))})
    agg = []
    for tag in ("clipped", "unclipped"):
        a = res[tag][3]
        agg.append({"method": method, "score": tag, "workloads": a["workloads"], "avg_score": a["avg_score"], "regret": a["regret"],
                    "top1": f"{a['top1_hits']}/{a['top1_n']}", "top5": f"{a['top5_hits']}/{a['top5_n']}"})
    return rows, agg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p3 = pd.read_csv(C.OUT / "predictions_x3.csv")
    cand = cand_frame(p3)
    g3 = sorted(cand.loc[cand["N"] >= 3, "group"].unique())
    assert len(g3) == 12
    rows, agg, nrows = [], [], []
    methods = ["XGBoost Full", "Ridge", "Random Forest", "MLP (small)", "XGBoost -P", "XGBoost -I"]
    for m in methods:
        t, d = p3[f"T_hat[{m}]"].values, p3[f"D_hat[{m}]"].values
        if m in ("XGBoost -P", "XGBoost -I"):
            # these models returned float32; the CSV holds the shortest float32 text, which float32 parsing restores exactly
            t, d = t.astype(np.float32).astype(np.float64), d.astype(np.float32).astype(np.float64)
        raw = t - ALPHA * d
        clip = np.clip(raw, 0, 1)
        stored = p3[f"S_hat[{m}]"].values
        assert np.abs(clip - stored).max() < 1e-12, f"stored S_hat of {m} is not clip(T_hat - 0.3 D_hat, 0, 1)"
        o_st, _ = E.select(cand[["group", "comb_num"]], stored)
        o_re, _ = E.select(cand[["group", "comb_num"]], clip)
        assert o_st == o_re, f"{m}: recomputed clipped score orders candidates differently from the stored score"
        r, a = run(m, cand, clip, raw, g3)
        rows += r
        agg += a
        n3 = (cand["N"] >= 3).values
        nrows.append({"method": m, "candidates": len(raw), "raw<0": int((raw < 0).sum()), "raw>1": int((raw > 1).sum()),
                      "raw<0 (N>=3)": int((raw[n3] < 0).sum()), "raw>1 (N>=3)": int((raw[n3] > 1).sum()),
                      "T_hat<0": int((t < 0).sum()), "T_hat>1": int((t > 1).sum()), "D_hat<0": int((d < 0).sum()), "D_hat>1": int((d > 1).sum())})
    pi = pd.read_csv(C.OUT / "per_instance_candidates_x3.csv")
    ci = cand_frame(pi)
    raw = pi["S_hat"].values
    assert np.allclose(raw, pi["T_hat_select"] - ALPHA * pi["D_hat"])
    r, a = run("Per-instance predictor ([14]-inspired)", ci, np.clip(raw, 0, 1), raw, g3)
    rows += r
    agg += a
    nrows.append({"method": "Per-instance predictor ([14]-inspired)", "candidates": len(raw), "raw<0": int((raw < 0).sum()), "raw>1": int((raw > 1).sum()),
                  "raw<0 (N>=3)": int((raw[(ci['N'] >= 3).values] < 0).sum()), "raw>1 (N>=3)": int((raw[(ci['N'] >= 3).values] > 1).sum()),
                  "T_hat<0": int((pi["T_hat_select"] < 0).sum()), "T_hat>1": int((pi["T_hat_select"] > 1).sum()),
                  "D_hat<0": int((pi["D_hat"] < 0).sum()), "D_hat>1": int((pi["D_hat"] > 1).sum())})
    # Phase 2 stored predictions: XGBoost D_noweight and the alpha sweep of the multi-objective predictor (Fig. 4)
    p2 = pd.read_csv(C.P2 / "predictions_x3.csv")
    c2 = cand_frame(p2)
    assert list(c2["group"]) == list(cand["group"]) and (c2["comb_num"].values == cand["comb_num"].values).all()
    for lab, dcol, alphas in (("Phase 2 Ours D_noweight", "D_hat_noweight", [ALPHA]), ("Phase 2 Fig.4 multi", "D_hat", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])):
        for al in alphas:
            raw = p2["T_hat"].values - al * p2[dcol].values
            cc = c2.copy()
            r, a = run(f"{lab} alpha={al}", cc.assign(D=cc["D"]), np.clip(raw, 0, 1), raw, g3) if al == ALPHA else run_alpha(f"{lab} alpha={al}", cc, raw, g3, al)
            rows += r
            agg += a
            nrows.append({"method": f"{lab} alpha={al}", "candidates": len(raw), "raw<0": int((raw < 0).sum()), "raw>1": int((raw > 1).sum())})
    pr = pd.DataFrame(rows)
    pa = pd.DataFrame(agg)
    pn = pd.DataFrame(nrows)
    pr.to_csv(OUT / "per_workload.csv", index=False)
    pa.to_csv(OUT / "aggregate.csv", index=False)
    pn.to_csv(OUT / "out_of_range_counts.csv", index=False)
    ch = pr[pr["selection_changed"] | pr["top5_set_changed"]]
    ch.to_csv(OUT / "changed.csv", index=False)
    summ = {"selection_changed": int(pr["selection_changed"].sum()), "top5_set_changed": int(pr["top5_set_changed"].sum()),
            "top5_order_changed": int(pr["top5_order_changed"].sum()), "rows": len(pr),
            "aggregate_identical": bool(all((pa[pa.method == m].iloc[0][["avg_score", "regret", "top1", "top5"]].values ==
                                             pa[pa.method == m].iloc[1][["avg_score", "regret", "top1", "top5"]].values).all() for m in pa["method"].unique()))}
    (OUT / "clip_check.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(pn.to_string(index=False))
    print(ch.to_string(index=False))


def run_alpha(method, cand, raw, g3, al):
    """Fig. 4 at another alpha: the measured S of the evaluator uses the same alpha."""
    global ALPHA
    keep = ALPHA
    ALPHA = al
    try:
        return run(method, cand, np.clip(raw, 0, 1), raw, g3)
    finally:
        ALPHA = keep


if __name__ == "__main__":
    main()

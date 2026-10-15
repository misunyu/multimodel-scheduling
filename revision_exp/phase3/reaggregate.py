"""Re-aggregate the Phase 3 tables from STORED predictions (no retraining), with or without the final clip of S_hat.

  .venv/bin/python3 -B revision_exp/phase3/reaggregate.py --clip   -> clip_check/reproduce/  (must reproduce the existing tables exactly)
  .venv/bin/python3 -B revision_exp/phase3/reaggregate.py           -> rewrites the Phase 3 tables with S_hat = T_hat - 0.3 D_hat (no clip);
                                                                       the previous files are copied to clip_check/before/ first
Direct predictors: T_hat, D_hat from out_phase3/predictions_x3.csv (XGBoost -P / -I parsed back to float32, the dtype they were predicted in).
Per-instance: out_phase3/per_instance_candidates_x3.csv (its S_hat never had a clip).
"""
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase3 import common as C  # noqa: E402
from phase3 import run_phase3 as RP  # noqa: E402
import baselines as B  # noqa: E402
import evaluate as E  # noqa: E402
import run_all as R  # noqa: E402

ALPHA = C.ALPHA
DIRECT = ("full", "ridge", "rf", "mlp", "P", "I")
FILES = ["table_predictors.csv", "table_predictors.md", "table_ablation.csv", "table_ablation.md", "table_baseline.csv", "table_baseline.md",
         "table_all.csv", "table_all.md", "paired_per_workload.csv", "paired_summary.csv", "per_workload_all.csv", "predictions_x3.csv", "summary.json"]


def build(clip, out_dir):
    C.allow_test_labels()
    test = C.load_test()
    pred = pd.read_csv(C.OUT / "predictions_x3.csv")
    assert list(pred["row_uid"]) == list(test["row_uid"])
    n_of = test.groupby("group")["N"].first()
    g3 = sorted(n_of[n_of >= 3].index)
    m3 = (test["N"] >= 3).values
    cand = R.cand_of(test)
    S_true = (test["T"] - ALPHA * test["D"]).values
    per, mae, shat, th, dh = {}, {}, {}, {}, {}
    for k in DIRECT:
        t, d = pred[f"T_hat[{RP.NAME[k]}]"].values, pred[f"D_hat[{RP.NAME[k]}]"].values
        if k in ("P", "I"):
            t, d = t.astype(np.float32).astype(np.float64), d.astype(np.float32).astype(np.float64)
        raw = t - ALPHA * d
        s = np.clip(raw, 0.0, 1.0) if clip else raw
        shat[k], th[k], dh[k] = s, t, d
        per[k] = R.ours_per_workload(test, s, ALPHA, RP.NAME[k])
        mae[k] = (float(np.abs(t[m3] - test["T"].values[m3]).mean()), float(np.abs(d[m3] - test["D"].values[m3]).mean()))
    pi = pd.read_csv(C.OUT / "per_instance_candidates_x3.csv")
    s_inst = pi["S_hat"].values if not clip else pi["S_hat"].values      # the per-instance score has no clip in either case
    order, ties = E.select(pi[["group", "comb_num"]], s_inst)
    per["inst"] = E.score_selection(cand, order, ALPHA, RP.NAME["inst"], top5=True, ties=ties)
    p3 = pi[pi["N"] >= 3]
    mae["inst"] = (float((p3["T_hat_select"] - p3["T"]).abs().mean()), float((p3["D_hat"] - p3["D"]).abs().mean()))
    per["lat"] = RP.RR_latency(cand)
    per["gpu"] = R.single_per_workload(test, B.choose_all_device(cand, "all_gpu"), ALPHA, RP.NAME["gpu"])
    per["orc"] = R.ours_per_workload(test, S_true, ALPHA, RP.NAME["orc"])
    bo5 = B.best_of_k(cand, ALPHA)
    bo5r = C.RR.bo5_row(bo5, test, ALPHA, set(g3))

    paired, pw_all = {}, []
    for k in DIRECT + ("inst",):
        pw, s = C.RR.paired_difference(test, per[k], per["gpu"], ALPHA, "macro", g3, f"{RP.NAME[k]} - All-GPU")
        paired[k] = s["mean_diff"]
        pw_all.append(pw)
        if k != "full":
            pw_all.append(C.RR.paired_difference(test, per[k], per["full"], ALPHA, "macro", g3, f"{RP.NAME[k]} - XGBoost Full")[0])
    pw_all = pd.concat(pw_all, ignore_index=True)
    ps = pw_all.groupby("comparison").agg(workloads=("diff", "size"), mean_diff=("diff", "mean"), se=("diff", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                                          n_pos=("sign", lambda x: int((x == "+").sum())), n_zero=("sign", lambda x: int((x == "0").sum())),
                                          n_neg=("sign", lambda x: int((x == "-").sum())), min_diff=("diff", "min"), max_diff=("diff", "max"),
                                          label_se_median=("diff_label_se", "median")).reset_index()

    def mk(k, top5=True):
        return RP.row_of(RP.NAME[k], per[k], g3, mae.get(k), paired.get(k), top5)
    cols = (C.OUT / "feature_columns_full.txt").read_text().split()
    nP, nI = len(cols) - 56, len(cols) - 108
    t_pred = pd.DataFrame([mk(k) for k in ("full", "ridge", "rf", "mlp")])
    t_abl = pd.DataFrame([mk(k) for k in ("full", "P", "I")])
    t_abl.insert(1, "features", [len(cols), nP, nI])
    t_abl.insert(2, "removed (non-zero in train)", ["0", "56 (56)", "108 (54)"])
    lat_row = RP.row_of(RP.NAME["lat"], per["lat"], g3, None, None, top5=False)
    orc_row = RP.row_of(RP.NAME["orc"], per["orc"], g3, None, None, top5=False)
    bo5_line = {"method": RP.NAME["bo5"], "workloads": bo5r["workloads"], "MAE(T)": "N/A", "MAE(D)": "N/A", "Avg score": bo5r["avg_score"],
                "regret": bo5r["regret"], "Top-1": bo5r["top1"].replace(" (mean rate)", " (mean hit rate over 1,000 samplings)"), "Top-5": "N/A",
                "Ours-All-GPU paired mean (ref.)": ""}
    t_base = pd.DataFrame([lat_row, mk("inst"), mk("full"), bo5_line, orc_row])
    t_all = pd.DataFrame([mk(k) for k in DIRECT + ("inst",)] + [lat_row, bo5_line, orc_row])
    marg = pd.read_csv(C.OUT / "top1_margin.csv")
    rule = ("Selection score: S_hat = clip(T_hat - 0.3 D_hat, 0, 1) for the direct predictors (Phase 2 rule); per-instance S_hat = T_hat_select - 0.3 D_hat_macro.\n"
            if clip else
            "Selection score (all methods, no clip, as in the manuscript definition): S_hat = T_hat - 0.3 D_hat for the direct predictors "
            "(T_hat, D_hat not clipped individually); per-instance S_hat = T_hat_select - 0.3 D_hat_macro. See CLIP_CHECK.md.\n")
    foot = (f"\nTop-1 / Top-5: hits / evaluated workloads (3x, N>=3). Top-5 only for methods that rank every candidate. "
            f"(i) Workloads whose oracle-to-second measured S margin is <= 0.0002: {int((marg['margin_1st_2nd'] <= 0.0002).sum())}/12 "
            f"(of which exact ties: {int(marg['exact_tie'].sum())}). (ii) Workloads whose per-repeat oracle is not the same in all three repeats: "
            f"4/12 (results/INTEGRITY.md section 6). {RP.TOP1_NOTE} Best-of-5: 1,000 repetitions of candidate sampling, not of measurement; "
            f"its Top-1 is the mean hit rate. Latency-based selects all-GPU for every model on the new offline profile. {RP.NOTE_14} "
            "MAE: N>=3 candidates (480); per-instance MAE(T) uses T_hat_select = F_hat / max F_hat (predicted maximum). " + rule)
    out_dir.mkdir(parents=True, exist_ok=True)
    for nm, t in (("table_predictors", t_pred), ("table_ablation", t_abl), ("table_baseline", t_base), ("table_all", t_all)):
        t.to_csv(out_dir / f"{nm}.csv", index=False)
        (out_dir / f"{nm}.md").write_text(f"# {nm}\n\n3x hold-out, N>=3, 12 workloads, alpha = 0.3, D = D_macro.\n\n" + C.md_table(t) + foot)
    pw_all.to_csv(out_dir / "paired_per_workload.csv", index=False)
    ps.to_csv(out_dir / "paired_summary.csv", index=False)
    pwl = []
    for k in DIRECT + ("inst", "lat", "gpu", "orc"):
        for _, r in per[k][per[k]["group"].isin(g3)].iterrows():
            pwl.append({"method": RP.NAME[k], "group": r["group"], "N": r["N"], "selected": r["selected"], "S_selected": r["S_selected"],
                        "oracle_S": r["oracle_S"], "regret": r["regret"], "is_oracle": r["top1"], "top5": r.get("top5"), "selected_all_gpu": r.get("selected_all_gpu")})
    g_, S_sel, top1, _c = bo5
    orc = per["orc"].set_index("group")["oracle_S"]
    for i, g in enumerate(g_):
        if g in g3:
            pwl.append({"method": RP.NAME["bo5"], "group": g, "N": int(n_of[g]), "selected": "1,000 samplings", "S_selected": float(S_sel[:, i].mean()),
                        "oracle_S": float(orc[g]), "regret": float(orc[g] - S_sel[:, i].mean()), "is_oracle": float(top1[:, i].mean())})
    pd.DataFrame(pwl).to_csv(out_dir / "per_workload_all.csv", index=False)
    pr = pred.copy()
    for k in DIRECT:
        pr[f"S_hat[{RP.NAME[k]}]"] = shat[k]
    pr.to_csv(out_dir / "predictions_x3.csv", index=False)
    old = json.loads((C.OUT / "summary.json").read_text())
    old["tables"] = {nm: t.to_dict(orient="records") for nm, t in (("predictors", t_pred), ("ablation", t_abl), ("baseline", t_base))}
    old["selection_score"] = "clip(T_hat - 0.3 D_hat, 0, 1)" if clip else "T_hat - 0.3 D_hat (no clip; see CLIP_CHECK.md)"
    (out_dir / "summary.json").write_text(json.dumps(old, indent=1, default=str))
    return {"pred": t_pred, "abl": t_abl, "base": t_base, "all": t_all, "paired": ps}


def same_csv(a, b):
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return False
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            if not np.allclose(x[c].astype(float), y[c].astype(float), atol=1e-12, equal_nan=True):
                return False
        elif not (x[c].astype(str) == y[c].astype(str)).all():
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="store_true")
    a = ap.parse_args()
    if a.clip:
        d = C.OUT / "clip_check" / "reproduce"
        build(True, d)
        res = {f: same_csv(C.OUT / f, d / f) for f in ("table_predictors.csv", "table_ablation.csv", "table_baseline.csv", "table_all.csv",
                                                      "paired_per_workload.csv", "paired_summary.csv", "per_workload_all.csv")}
        print(json.dumps(res, indent=1))
        (C.OUT / "clip_check" / "reproduce_check.json").write_text(json.dumps(res, indent=1))
        assert all(res.values()), "stored predictions do not reproduce the existing tables"
        return
    rep = json.loads((C.OUT / "clip_check" / "reproduce_check.json").read_text())
    assert all(rep.values()), "run --clip first and make sure it reproduces the existing tables"
    before = C.OUT / "clip_check" / "before"
    before.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (before / f).exists():
            shutil.copy2(C.OUT / f, before / f)
    t = build(False, C.OUT)
    for k, v in t.items():
        print(k)
        print(v.to_string(index=False))


if __name__ == "__main__":
    main()

"""Phase 3 (v1.2): reviewer experiments on the re-profiled CPU-GPU dataset.

  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp/phase3/run_phase3.py        (run precheck.py first; it must pass)

Order: training (Ridge / RF / MLP, XGBoost -P / -I, per-instance) on 1, 2, 4x only -> logs/config_frozen.json ->
per-instance selection on 3x from schedules only -> final_eval() (the only place that reads 3x labels) -> tables.
XGBoost Full = the Phase 2 artifact (not retrained). Outputs: revision_exp/out_phase3/.
"""
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])
sys.dont_write_bytecode = True

import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase3 import common as C  # noqa: E402
from phase3 import precheck as PC  # noqa: E402
from phase3 import predictors as PR  # noqa: E402
from phase3.baselines import compose_select as CS  # noqa: E402
from phase3.baselines import fit_instance as FI  # noqa: E402
from phase3.baselines import instance_rows as IR  # noqa: E402
import baselines as B  # noqa: E402
import data as D  # noqa: E402
import evaluate as E  # noqa: E402
import folds as F  # noqa: E402
import predict as P  # noqa: E402
import run_all as R  # noqa: E402
import train_multi as TM  # noqa: E402

assert Path(B.__file__).resolve() == C.EXP / "baselines.py", B.__file__
OUT, ALPHA = C.OUT, C.ALPHA
NAME = {"full": "XGBoost Full", "ridge": "Ridge", "rf": "Random Forest", "mlp": "MLP (small)", "P": "XGBoost -P", "I": "XGBoost -I",
        "inst": "Per-instance predictor ([14]-inspired)", "lat": "Latency-based (= All-GPU)", "gpu": "All-GPU", "bo5": "Best-of-5", "orc": "Oracle"}
NOTE_14 = "Per-instance predictor: inspired by the instance-wise interference prediction of Kim et al. [14]; not a reproduction of their system."
# Final clip of the predicted score: off (manuscript section 3, S_hat = T_hat - 0.3 D_hat). See out_phase3/CLIP_CHECK.md.
CLIP_S_HAT = False
SCORE_RULE = ("Selection score (all methods, no clip, as in the manuscript definition): S_hat = T_hat - 0.3 D_hat for the direct predictors "
              "(T_hat, D_hat not clipped individually); per-instance S_hat = T_hat_select - 0.3 D_hat_macro. See CLIP_CHECK.md.\n")
TOP1_NOTE = ("Top-1 accuracy is sensitive to small differences between the highest-ranked measured placements; we therefore report placement score "
             "and regret alongside Top-1 and Top-5.")


def log(m):
    C.log(m)


def main():
    pre = json.loads((OUT / "precheck.json").read_text())
    assert pre["all_pass"], "section 0-pre has not passed"
    t_start = time.time()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / "git_status_before.txt").write_text(subprocess.run(["git", "-C", str(C.REPO), "status", "--short"], capture_output=True, text=True).stdout)
    import sklearn
    import xgboost
    env = {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    log(f"start | env {env}")

    # ---------------------------------------------------------------- training data only
    train = C.load_train()
    X = D.build_features(train)
    cols = list(X.columns)
    assert cols == (OUT / "feature_columns_full.txt").read_text().split()
    fold = C.phase2_folds(train)
    assert (fold == F.make_folds(train)).all()
    family_fold = dict(zip(train["family"], fold))
    timing, hp, info = {}, [], {}

    models = {}
    for kind in ("ridge", "rf", "mlp"):
        t0 = time.time()
        models[kind], rec, info[kind] = PR.train_direct(kind, X, train, fold, log)
        hp.append(rec)
        timing[NAME[kind]] = round(time.time() - t0, 1)
    pd.concat(hp, ignore_index=True).to_csv(OUT / "hparam_search_predictors.csv", index=False)

    ab_cols, hpa = {}, []
    for g in ("P", "I"):
        ab_cols[g] = PC.masked(cols, g)
        assert 0 < len(ab_cols[g]) < len(cols)
        t0 = time.time()
        models[g], rec, info[g] = PR.train_xgb_ablation(X[ab_cols[g]], train, fold, OUT / "artifacts" / f"xgb_minus_{g}", f"-{g}", log)
        hpa.append(rec)
        timing[NAME[g]] = round(time.time() - t0, 1)
    pd.concat(hpa, ignore_index=True).to_csv(OUT / "hparam_search_ablation.csv", index=False)

    iso = IR.iso_capacity(C.PROF / "offline_profile" / "offline_profile.json")
    rows_tr = IR.training_rows(C.DATA / "train_schedules_x3.csv", C.DATA / "train_x3.csv", C.DATA / "labels_train.csv", iso, C.WINDOW_SEC)
    fold_i = FI.instance_folds(rows_tr, family_fold)
    t0 = time.time()
    models["inst"], rec, info["inst"] = FI.fit_instance_models(rows_tr, fold_i, OUT / "artifacts" / "per_instance", log)
    timing[NAME["inst"]] = round(time.time() - t0, 1)
    rec.to_csv(OUT / "hparam_search_instance.csv", index=False)
    tr_ratio = {"rows": len(rows_tr), "t_ratio_min": float(rows_tr["t_ratio"].min()), "t_ratio_max": float(rows_tr["t_ratio"].max()),
                "t_ratio_rows_above_1": int((rows_tr["t_ratio"] > 1).sum()), "q_i_max": float(rows_tr["q_i"].max()),
                "q_i_rows_above_0.01": int((rows_tr["q_i"] > 0.01).sum())}
    log(f"instance training rows: {tr_ratio}")

    frozen = {"frozen_at": datetime.now().isoformat(sep=" ", timespec="seconds"), "alpha": ALPHA, "features_full": len(cols),
              "features_minus_P": len(ab_cols["P"]), "features_minus_I": len(ab_cols["I"]), "masks": PC.MASKS, "grids": {**PR.GRIDS, "xgb": TM.GRID},
              "d_weight": {"threshold": TM.D_WEIGHT_THRESHOLD, "value": TM.D_WEIGHT_VALUE, "applies_to": "D regressors (placement rows, D_macro)"},
              "q_weight": {"threshold": FI.Q_WEIGHT_THRESHOLD, "value": FI.Q_WEIGHT_VALUE, "applies_to": "per-instance q_i regressor (instance rows)"},
              "family_fold": {k: int(v) for k, v in sorted(family_fold.items())}, "selected": {NAME[k]: v for k, v in info.items()},
              "xgb_full_artifact": {n: C.sha(C.P2_ART / f"{n}.json") for n in ("T", "D")}, "env": env, "training_seconds": timing,
              "instance_training_target_range": tr_ratio,
              "statement": "written before any 3x label is read by this process; nothing below changes models, features, grids, rules or alpha"}
    (OUT / "logs" / "config_frozen.json").write_text(json.dumps(frozen, indent=1, default=str))
    log("configuration frozen")

    # ---------------------------------------------------------------- per-instance selection on 3x from schedules only
    rows_te = IR.rows_from_schedules(C.DATA / "test_schedules_x3.csv", iso)
    assert set(rows_te["rate"]) == {"x3"}
    Xi = rows_te[IR.FEATURES].values
    pt, pq = models["inst"]["t_ratio"].predict(Xi), models["inst"]["q_i"].predict(Xi)
    comp = CS.compose(rows_te[sorted(CS.ALLOWED_INPUT & set(rows_te.columns))], pt, pq)
    s_inst, order_inst, ties_inst = CS.select_candidates(comp, ALPHA)
    comp["S_hat"] = s_inst
    rows_te = rows_te.assign(pred_t_ratio=pt, pred_q_i=pq)
    log("per-instance selection done (no 3x label read)")

    final_eval(train, X, cols, models, ab_cols, rows_te, comp, order_inst, ties_inst, info, timing, tr_ratio)
    (OUT / "logs" / "git_status_after.txt").write_text(subprocess.run(["git", "-C", str(C.REPO), "status", "--short"], capture_output=True, text=True).stdout)
    timing["total"] = round(time.time() - t_start, 1)
    (OUT / "logs" / "timing.json").write_text(json.dumps(timing, indent=1))
    log(f"done in {timing['total']} s")


def collisions(Xc, df, tag):
    """Candidates of one workload with identical feature vectors (X rounded to 9 digits)."""
    S = (df["T"] - ALPHA * df["D"]).values
    key = pd.util.hash_pandas_object(Xc.round(9).reset_index(drop=True), index=False).values
    rows = []
    for g, idx in df.reset_index(drop=True).groupby("group").indices.items():
        k = pd.Series(key[idx])
        dup = k[k.duplicated(keep=False)]
        groups = dup.groupby(dup).groups
        rng = [float(S[idx][list(ix)].max() - S[idx][list(ix)].min()) for ix in groups.values()]
        rows.append({"features": tag, "group": g, "N": int(df["N"].values[idx[0]]), "candidates": len(idx), "distinct_vectors": int(k.nunique()),
                     "colliding_sets": len(groups), "colliding_candidates": int(len(dup)),
                     "colliding_combinations": ";".join(",".join(str(int(df["comb_num"].values[idx][i])) for i in ix) for ix in groups.values()),
                     "max_S_range_in_set": max(rng) if rng else 0.0})
    return rows


def row_of(name, per, groups, mae=None, paired=None, top5=True):
    a = E.aggregate(per[per["group"].isin(groups)], 1, name)
    return {"method": name, "workloads": a["workloads"],
            "MAE(T)": round(mae[0], 4) if mae else "N/A", "MAE(D)": round(mae[1], 4) if mae else "N/A",
            "Avg score": round(a["avg_score"], 4), "regret": round(a["regret"], 4),
            "Top-1": f"{a['top1_hits']}/{a['top1_n']}", "Top-5": f"{a['top5_hits']}/{a['top5_n']}" if top5 else "N/A",
            "Ours-All-GPU paired mean (ref.)": round(paired, 4) if paired is not None else ""}


def final_eval(train, X, cols, models, ab_cols, rows_te, comp, order_inst, ties_inst, info, timing, tr_ratio):
    C.allow_test_labels()
    test = C.load_test()
    Xt = D.build_features(test, cols)
    n_of = test.groupby("group")["N"].first()
    g3 = sorted(n_of[n_of >= 3].index)
    S_true = (test["T"] - ALPHA * test["D"]).values
    cand = R.cand_of(test)
    m3 = (test["N"] >= 3).values
    assert m3.sum() == 480 and len(g3) == 12

    preds = {"full": P.predict_multi(P.load_model(C.P2_ART / "T.json"), P.load_model(C.P2_ART / "D.json"), Xt)}
    for k in ("ridge", "rf", "mlp"):
        preds[k] = (models[k]["T"].predict(Xt.values), models[k]["D"].predict(Xt.values))
    for g in ("P", "I"):
        preds[g] = (models[g]["T"].predict(Xt[ab_cols[g]].values), models[g]["D"].predict(Xt[ab_cols[g]].values))
    per, mae, shat = {}, {}, {}
    for k, (t, d) in preds.items():
        s, raw = P.multi_score(np.asarray(t, float), np.asarray(d, float), ALPHA, clip=CLIP_S_HAT)
        shat[k] = s
        per[k] = R.ours_per_workload(test, s, ALPHA, NAME[k])
        mae[k] = (float(np.abs(t[m3] - test["T"].values[m3]).mean()), float(np.abs(d[m3] - test["D"].values[m3]).mean()))
    # per-instance baseline: selection was made before this function; here only scoring
    key_test = set(zip(test["group"], test["comb_num"]))
    assert set(zip(comp["group"], comp["comb_num"])) == key_test, "per-instance candidates differ from the measured candidates"
    per["inst"] = E.score_selection(cand, order_inst, ALPHA, NAME["inst"], top5=True, ties=ties_inst)
    cm = comp.merge(test[["group", "comb_num", "N", "T", "D", "F_mean", "F_max", "all_gpu"]], on=["group", "comb_num"], how="left", validate="one_to_one")
    c3 = cm[cm["N"] >= 3]
    mae["inst"] = (float((c3["T_hat_select"] - c3["T"]).abs().mean()), float((c3["D_hat"] - c3["D"]).abs().mean()))
    lat = RR_latency(cand)
    per["lat"] = lat
    per["gpu"] = R.single_per_workload(test, B.choose_all_device(cand, "all_gpu"), ALPHA, NAME["gpu"])
    per["orc"] = R.ours_per_workload(test, S_true, ALPHA, NAME["orc"])
    bo5 = B.best_of_k(cand, ALPHA)
    bo5r = C.RR.bo5_row(bo5, test, ALPHA, set(g3))

    paired, pw_all = {}, []
    for k in ("full", "ridge", "rf", "mlp", "P", "I", "inst"):
        pw, s = C.RR.paired_difference(test, per[k], per["gpu"], ALPHA, "macro", g3, f"{NAME[k]} - All-GPU")
        paired[k] = s["mean_diff"]
        pw_all.append(pw)
        if k != "full":
            pw2, s2 = C.RR.paired_difference(test, per[k], per["full"], ALPHA, "macro", g3, f"{NAME[k]} - XGBoost Full")
            pw_all.append(pw2)
    pw_all = pd.concat(pw_all, ignore_index=True)
    pw_all.to_csv(OUT / "paired_per_workload.csv", index=False)
    ps = pw_all.groupby("comparison").agg(workloads=("diff", "size"), mean_diff=("diff", "mean"), se=("diff", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                                          n_pos=("sign", lambda x: int((x == "+").sum())), n_zero=("sign", lambda x: int((x == "0").sum())),
                                          n_neg=("sign", lambda x: int((x == "-").sum())), min_diff=("diff", "min"), max_diff=("diff", "max"),
                                          label_se_median=("diff_label_se", "median")).reset_index()
    ps.to_csv(OUT / "paired_summary.csv", index=False)

    # ---------------------------------------------------------------- tables
    def mk(k, top5=True):
        return row_of(NAME[k], per[k], g3, mae.get(k), paired.get(k), top5)
    t_pred = pd.DataFrame([mk(k) for k in ("full", "ridge", "rf", "mlp")])
    t_abl = pd.DataFrame([mk(k) for k in ("full", "P", "I")])
    t_abl.insert(1, "features", [len(cols), len(ab_cols["P"]), len(ab_cols["I"])])
    t_abl.insert(2, "removed (non-zero in train)", ["0", f"{len(cols) - len(ab_cols['P'])} (56)", f"{len(cols) - len(ab_cols['I'])} (54)"])
    lat_row = row_of(NAME["lat"], per["lat"], g3, None, None, top5=False)
    orc_row = row_of(NAME["orc"], per["orc"], g3, None, None, top5=False)
    bo5_line = {"method": NAME["bo5"], "workloads": bo5r["workloads"], "MAE(T)": "N/A", "MAE(D)": "N/A", "Avg score": bo5r["avg_score"],
                "regret": bo5r["regret"], "Top-1": bo5r["top1"].replace(" (mean rate)", " (mean hit rate over 1,000 samplings)"), "Top-5": "N/A",
                "Ours-All-GPU paired mean (ref.)": ""}
    t_base = pd.DataFrame([lat_row, mk("inst"), mk("full"), bo5_line, orc_row])
    t_all = pd.DataFrame([mk(k) for k in ("full", "ridge", "rf", "mlp", "P", "I", "inst")] + [lat_row, bo5_line, orc_row])

    # Top-1 footnote statistics
    marg = []
    for g in g3:
        s = np.sort(S_true[(test["group"] == g).values])[::-1]
        marg.append({"group": g, "margin_1st_2nd": float(s[0] - s[1]), "exact_tie": bool(s[0] - s[1] <= E.TIE_TOL)})
    marg = pd.DataFrame(marg)
    ox = pd.read_csv(C.PROF / "results" / "integrity" / "oracle_x3.csv")
    n_rep_diff = int((ox["reps_equal_to_mean_oracle"] < 3).sum()) if "reps_equal_to_mean_oracle" in ox else None
    foot = (f"\nTop-1 / Top-5: hits / evaluated workloads (3x, N>=3). Top-5 only for methods that rank every candidate. "
            f"(i) Workloads whose oracle-to-second measured S margin is <= 0.0002: {int((marg['margin_1st_2nd'] <= 0.0002).sum())}/12 "
            f"(of which exact ties: {int(marg['exact_tie'].sum())}). (ii) Workloads whose per-repeat oracle is not the same in all three repeats: "
            f"{n_rep_diff}/12 (results/INTEGRITY.md section 6). {TOP1_NOTE} Best-of-5: 1,000 repetitions of candidate sampling, not of measurement; "
            f"its Top-1 is the mean hit rate. Latency-based selects all-GPU for every model on the new offline profile. {NOTE_14} "
            "MAE: N>=3 candidates (480); per-instance MAE(T) uses T_hat_select = F_hat / max F_hat (predicted maximum). "
            + SCORE_RULE)
    for nm, t in (("table_predictors", t_pred), ("table_ablation", t_abl), ("table_baseline", t_base), ("table_all", t_all)):
        t.to_csv(OUT / f"{nm}.csv", index=False)
        (OUT / f"{nm}.md").write_text(f"# {nm}\n\n3x hold-out, N>=3, 12 workloads, alpha = 0.3, D = D_macro.\n\n" + C.md_table(t) + foot)
    marg.to_csv(OUT / "top1_margin.csv", index=False)
    for nm, t in (("predictors", t_pred), ("ablation", t_abl), ("baseline", t_base)):
        log(f"\n{nm}\n" + t.to_string(index=False))

    # ---------------------------------------------------------------- per-workload table
    pwl = []
    for k in ("full", "ridge", "rf", "mlp", "P", "I", "inst", "lat", "gpu", "orc"):
        p = per[k][per[k]["group"].isin(g3)]
        for _, r in p.iterrows():
            pwl.append({"method": NAME[k], "group": r["group"], "N": r["N"], "selected": r["selected"], "S_selected": r["S_selected"],
                        "oracle_S": r["oracle_S"], "regret": r["regret"], "is_oracle": r["top1"], "top5": r.get("top5"),
                        "selected_all_gpu": r.get("selected_all_gpu")})
    g_, S_sel, top1, contains = bo5
    for i, g in enumerate(g_):
        if g in g3:
            pwl.append({"method": NAME["bo5"], "group": g, "N": int(n_of[g]), "selected": "1,000 samplings", "S_selected": float(S_sel[:, i].mean()),
                        "oracle_S": float(per["orc"].set_index("group").loc[g, "oracle_S"]), "regret": float(per["orc"].set_index("group").loc[g, "oracle_S"] - S_sel[:, i].mean()),
                        "is_oracle": float(top1[:, i].mean())})
    pd.DataFrame(pwl).to_csv(OUT / "per_workload_all.csv", index=False)

    # ---------------------------------------------------------------- collisions (direct feature sets)
    col_rows = []
    for tag, cc in (("Full", cols), ("-P", ab_cols["P"]), ("-I", ab_cols["I"])):
        col_rows += [dict(r, split="test 3x") for r in collisions(Xt[cc], test, tag)]
        col_rows += [dict(r, split="train 1,2,4x") for r in collisions(X[cc], train, tag)]
    # per-instance representation: a candidate = its multiset of instance feature rows
    sig = rows_te.sort_values(["group", "comb_num", "model"]).groupby(["group", "comb_num"])[IR.FEATURES].apply(
        lambda d: hash(np.round(d.values, 9).tobytes())).rename("sig").reset_index()
    sig = sig.merge(test[["group", "comb_num", "N", "T", "D"]], on=["group", "comb_num"])
    for g, sub in sig.groupby("group"):
        dup = sub[sub["sig"].duplicated(keep=False)]
        S = sub["T"] - ALPHA * sub["D"]
        col_rows.append({"features": "per-instance rows", "split": "test 3x", "group": g, "N": int(sub["N"].iloc[0]), "candidates": len(sub),
                         "distinct_vectors": int(sub["sig"].nunique()), "colliding_sets": int(dup["sig"].nunique()), "colliding_candidates": len(dup),
                         "colliding_combinations": "", "max_S_range_in_set": float(max([S[dup.index[dup["sig"] == s]].max() - S[dup.index[dup["sig"] == s]].min()
                                                                                         for s in dup["sig"].unique()], default=0.0))})
    col = pd.DataFrame(col_rows)
    col.to_csv(OUT / "ablation_collisions.csv", index=False)
    col_sum = col.groupby(["features", "split"]).agg(workloads=("group", "size"), workloads_with_collision=("colliding_sets", lambda x: int((x > 0).sum())),
                                                     colliding_sets=("colliding_sets", "sum"), colliding_candidates=("colliding_candidates", "sum"),
                                                     max_S_range=("max_S_range_in_set", "max")).reset_index()
    col_sum.to_csv(OUT / "ablation_collisions_summary.csv", index=False)
    log("\n" + col_sum.to_string(index=False))

    # ---------------------------------------------------------------- per-instance diagnostics
    diag = baseline_diagnostics(test, rows_te, cm, g3)
    diag["training_target_range"] = tr_ratio
    (OUT / "baseline_diagnostics.json").write_text(json.dumps(diag, indent=1, default=float))
    log(f"baseline diagnostics: { {k: v for k, v in diag.items() if not isinstance(v, (list, dict))} }")

    pd.DataFrame({"row_uid": test["row_uid"], "group": test["group"], "combination": test["combination"], "N": test["N"], "T": test["T"], "D": test["D"], "S": S_true,
                  **{f"T_hat[{NAME[k]}]": np.asarray(v[0]) for k, v in preds.items()}, **{f"D_hat[{NAME[k]}]": np.asarray(v[1]) for k, v in preds.items()},
                  **{f"S_hat[{NAME[k]}]": shat[k] for k in shat}}).to_csv(OUT / "predictions_x3.csv", index=False)
    cm.to_csv(OUT / "per_instance_candidates_x3.csv", index=False)
    summary = {"tables": {nm: t.to_dict(orient="records") for nm, t in (("predictors", t_pred), ("ablation", t_abl), ("baseline", t_base))},
               "training_info": info, "training_seconds": timing, "top1_margin_le_0.0002": int((marg["margin_1st_2nd"] <= 0.0002).sum()),
               "top1_exact_ties": int(marg["exact_tie"].sum()), "per_repeat_oracle_not_all_equal": n_rep_diff,
               "selection_score": "clip(T_hat - 0.3 D_hat, 0, 1)" if CLIP_S_HAT else "T_hat - 0.3 D_hat (no clip; see CLIP_CHECK.md)"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


def RR_latency(cand):
    tg = B.latency_targets(C.RR.offline_latency())
    assert set(tg.values()) == {"GPU"}, tg
    test_like = cand
    return E.score_selection(test_like, E.single_choice(cand[["group", "comb_num"]], B.choose_by_targets(cand, tg)), ALPHA, NAME["lat"], top5=False)


def baseline_diagnostics(test, rows_te, cm, g3):
    """(1) exact reconstruction of the measured labels from measured instance values, (2) instance MAE, (3) F_hat error and the effect
    of the predicted-max normalisation, (4) regret is in the tables. T_hat_eval uses the MEASURED F_max: diagnostic only, never used to select."""
    C.assert_final("baseline_diagnostics")
    inst = pd.read_csv(C.DATA / "labels_instances_all.csv")
    inst = inst[inst["rate"] == "x3"].copy()
    inst["group"] = inst["family"] + "_x3.yaml"
    inst["comb_num"] = inst["combination"].map(D.comb_num)
    lab = pd.read_csv(C.DATA / "labels_test.csv")
    lab["comb_num"] = lab["combination"].map(D.comb_num)
    lab = lab.rename(columns={"schedule_file": "group"})
    out = {}
    # (1) reconstruction, per repeat first
    rec = []
    for (g, c), sub in inst.groupby(["group", "comb_num"]):
        L = lab[(lab["group"] == g) & (lab["comb_num"] == c)].iloc[0]
        reps = [k for k in range(3) if sub[f"generated_i_rep{k}"].notna().all()]
        q_rk = {k: (sub[f"dropped_i_rep{k}"] / sub[f"generated_i_rep{k}"]).values for k in reps}
        F_rk = {k: float(sub[f"completed_i_rep{k}"].sum() / C.WINDOW_SEC) for k in reps}
        d_bar = float(np.mean([np.mean([q_rk[k][i] for k in reps]) for i in range(len(sub))]))
        alt = float(np.mean(np.mean([sub[f"dropped_i_rep{k}"] for k in reps], axis=0) / np.mean([sub[f"generated_i_rep{k}"] for k in reps], axis=0)))
        rec.append({"group": g, "comb_num": c, "reps": len(reps), "D_rebuilt": d_bar, "D_label": float(L["D_macro_mean"]), "D_alt_mean_d_over_mean_g": alt,
                    "F_rebuilt": float(np.mean([F_rk[k] for k in reps])), "F_label": float(L["F_mean"]),
                    "max_abs_F_rep": max(abs(F_rk[k] - L[f"F_rep{k}"]) for k in reps),
                    "max_abs_Drep": max(abs(float(np.mean(q_rk[k])) - L[f"D_macro_rep{k}"]) for k in reps)})
    rec = pd.DataFrame(rec)
    rec.to_csv(OUT / "baseline_reconstruction_x3.csv", index=False)
    out["reconstruction_candidates"] = len(rec)
    out["reconstruction_max_abs_D"] = float((rec["D_rebuilt"] - rec["D_label"]).abs().max())
    out["reconstruction_max_abs_F"] = float((rec["F_rebuilt"] - rec["F_label"]).abs().max())
    out["reconstruction_max_abs_per_repeat_F"] = float(rec["max_abs_F_rep"].max())
    out["reconstruction_max_abs_per_repeat_D"] = float(rec["max_abs_Drep"].max())
    out["reconstruction_exact(1e-9)"] = bool(out["reconstruction_max_abs_D"] < 1e-9 and out["reconstruction_max_abs_F"] < 1e-9)
    out["alt_rule_mean_d_over_mean_g_max_abs_diff"] = float((rec["D_alt_mean_d_over_mean_g"] - rec["D_label"]).abs().max())
    out["alt_rule_candidates_differing(>1e-9)"] = int(((rec["D_alt_mean_d_over_mean_g"] - rec["D_label"]).abs() > 1e-9).sum())
    # (2) instance-level MAE (3x, N>=3)
    inst["t_ratio_true"] = np.nanmean(np.stack([inst[f"completed_i_rep{k}"] / C.WINDOW_SEC / inst["lambda_i"] for k in range(3)]), axis=0)
    inst["q_true"] = np.nanmean(np.stack([inst[f"dropped_i_rep{k}"] / inst[f"generated_i_rep{k}"] for k in range(3)]), axis=0)
    assert np.allclose(inst["q_true"], inst["q_i_i"], atol=1e-12)
    m = rows_te.merge(inst[["group", "comb_num", "model", "t_ratio_true", "q_true", "N"]].rename(columns={"N": "N_inst"}), on=["group", "comb_num", "model"],
                      how="left", validate="one_to_one")
    assert m["t_ratio_true"].notna().all()
    m["t_hat_ratio"] = np.clip(m["pred_t_ratio"], 0, None)
    m["q_hat"] = np.clip(m["pred_q_i"], 0, 1)
    m3 = m[m["N_inst"] >= 3]
    out["instance_rows_N>=3"] = len(m3)
    out["instance_mae_t_ratio"] = float((m3["t_hat_ratio"] - m3["t_ratio_true"]).abs().mean())
    out["instance_mae_q_i"] = float((m3["q_hat"] - m3["q_true"]).abs().mean())
    out["instance_mae_q_i_where_q>0.01"] = float((m3["q_hat"] - m3["q_true"]).abs()[m3["q_true"] > 0.01].mean())
    out["x3_t_ratio_true_max"] = float(m["t_ratio_true"].max())
    out["x3_t_ratio_true_rows_above_1"] = int((m["t_ratio_true"] > 1).sum())
    out["instance_mae_by_device"] = {d: {"t_ratio": float((s["t_hat_ratio"] - s["t_ratio_true"]).abs().mean()), "q_i": float((s["q_hat"] - s["q_true"]).abs().mean()),
                                         "rows": len(s)} for d, s in m3.groupby("device")}
    m.to_csv(OUT / "per_instance_rows_x3.csv", index=False)
    # (3) F_hat error and normalisation
    c3 = cm[cm["N"] >= 3].copy()
    c3["F_rel_err"] = (c3["F_hat"] - c3["F_mean"]) / c3["F_mean"]
    out["F_hat_rel_err"] = {q: float(c3["F_rel_err"].quantile(v)) for q, v in (("p05", .05), ("p50", .5), ("p95", .95))}
    out["F_hat_abs_rel_err_mean"] = float(c3["F_rel_err"].abs().mean())
    c3["T_hat_eval"] = c3["F_hat"] / c3["F_max"]
    out["mae_T_select(primary)"] = float((c3["T_hat_select"] - c3["T"]).abs().mean())
    out["mae_T_eval(measured-normalisation diagnostic)"] = float((c3["T_hat_eval"] - c3["T"]).abs().mean())
    s_sel = c3["T_hat_select"] - ALPHA * c3["D_hat"]
    s_ev = c3["T_hat_eval"] - ALPHA * c3["D_hat"]
    ids = c3[["group", "comb_num"]].reset_index(drop=True)
    o1, _ = E.select(ids, s_sel.values)
    o2, _ = E.select(ids, s_ev.values)
    out["workloads_selection_changes_with_measured_normalisation"] = int(sum(o1[g][0] != o2[g][0] for g in g3))
    out["note_T_eval"] = "T_hat_eval uses the measured F_max of the 3x workload: post-hoc diagnostic only, not used for selection or the primary results"
    pmax = c3.groupby("group").apply(lambda s: pd.Series({"pred_argmax_F": int(s.loc[s["F_hat"].idxmax(), "comb_num"]), "meas_argmax_F": int(s.loc[s["F_mean"].idxmax(), "comb_num"]),
                                                          "F_hat_max": float(s["F_hat"].max()), "F_max": float(s["F_max"].iloc[0])}), include_groups=False)
    pmax.to_csv(OUT / "baseline_fmax_x3.csv")
    out["predicted_Fmax_over_measured_Fmax"] = {g: float(r["F_hat_max"] / r["F_max"]) for g, r in pmax.iterrows()}
    return out


if __name__ == "__main__":
    main()

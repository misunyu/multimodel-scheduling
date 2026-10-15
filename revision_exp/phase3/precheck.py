"""Phase 3 section 0-pre: checks P1-P5. Nothing is trained for the comparison here.

  .venv/bin/python3 -B revision_exp/phase3/precheck.py      -> revision_exp/out_phase3/precheck.json, PRECHECK.md,
                                                               feature_columns_full.txt, ablation_columns.txt

P1 re-evaluates the frozen Phase 2 artifact on the 3x hold-out (it reads 3x labels by definition: it reproduces the Phase 2 numbers).
P5 checks the label rule on the training rates only; the 3x reconstruction is checked inside the final evaluation.
"""
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])
sys.dont_write_bytecode = True

import json  # noqa: E402
import re  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase3 import common as C  # noqa: E402
import data as D  # noqa: E402
import evaluate as E  # noqa: E402
import folds as F  # noqa: E402
import predict as P  # noqa: E402
import run_all as R  # noqa: E402
import train_multi as TM  # noqa: E402

# Final masks (P4). The draft "^pairhash_" matches the real names (no prefix); the -I draft patterns are anchored to "views.".
MASKS = {"P": [r"^pairhash_\d+_on_"],
         "I": [r"^views\.(sum|mean|max)\.model_hash_\d+_on_", r"^views\.(sum|mean|max)\.view\.infps_on_"]}


def group_of(c):
    if re.match(r"^pairhash_", c):
        return "pairwise"
    if re.search(r"\.model_hash_\d+_on_", c):
        return "identity x device"
    if re.search(r"\.view\.infps_on_", c):
        return "demand x device"
    if re.search(r"\.model_hash_\d+$", c):
        return "identity"
    if re.search(r"\.view\.exec_", c):
        return "device"
    if re.search(r"\.view\.infps$", c):
        return "demand"
    if c == "views.count.views":
        return "N"
    return "other"


def masked(cols, drop):
    pats = [p for g in drop for p in MASKS[g]]
    return [c for c in cols if not any(re.search(p, c) for p in pats)]


def p4(cols, X):
    out = {"columns_full": len(cols)}
    grp = pd.Series({c: group_of(c) for c in cols})
    zero = {c for c in cols if not X[c].any()}
    out["groups_full"] = {g: int((grp == g).sum()) for g in sorted(grp.unique())}
    out["groups_full_nonzero_in_train"] = {g: int(((grp == g) & ~grp.index.isin(zero)).sum()) for g in sorted(grp.unique())}
    lines = []
    ok = True
    for g, expect in (("P", {"pairwise"}), ("I", {"identity x device", "demand x device"})):
        removed = [c for c in cols if c not in set(masked(cols, g))]
        rg = set(grp[removed])
        whole = all((grp == e).sum() == sum(1 for c in removed if grp[c] == e) for e in expect)
        out[f"minus_{g}"] = {"removed": len(removed), "removed_nonzero_in_train": len([c for c in removed if c not in zero]),
                             "removed_groups": sorted(rg), "remaining": len(cols) - len(removed), "whole_groups_removed": whole,
                             "other_groups_untouched": rg == expect}
        ok &= len(removed) > 0 and whole and rg == expect
        lines += [f"# -{g}: {len(removed)} columns removed (patterns: {MASKS[g]})"] + removed + [""]
    old = pd.read_csv(C.REPO / "revision_audit" / "tables" / "feature_groups.csv")
    old = old[old["artifact"] == "gpu/xgb_model_x3_double"]
    out["old_feature_groups(gpu/xgb_model_x3_double)"] = dict(zip(old["feature_group"], old["columns"].astype(int)))
    out["pass"] = bool(ok)
    return out, lines


def p3(X):
    import sklearn
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    Xs = X.values[:300]
    y = rng.random(300)
    w = np.where(rng.random(300) > 0.7, 10.0, 1.0)
    res = {"sklearn": sklearn.__version__, "X_dtype": str(X.values.dtype), "X_is_sparse": False,
           "X_zero_fraction": float((X.values == 0).mean()), "X_type": type(X).__name__}
    ok = True
    cases = {"ridge": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "ridge__sample_weight"),
             "rf": (lambda: RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1), "sample_weight"),
             "mlp": (lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), max_iter=50, random_state=42)), "mlpregressor__sample_weight")}
    for k, (mk, kw) in cases.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                a = mk().fit(Xs, y, **{kw: w}).predict(Xs)
                b = mk().fit(Xs, y).predict(Xs)
                ones = mk().fit(Xs, y, **{kw: np.ones(300)}).predict(Xs)
                res[k] = {"fit_kwarg": kw, "accepted": True, "weight_changes_fit": bool(not np.allclose(a, b)),
                          "unit_weight_equals_unweighted": bool(np.allclose(ones, b))}
                ok &= res[k]["weight_changes_fit"]
            except Exception as e:  # noqa: BLE001
                res[k] = {"fit_kwarg": kw, "accepted": False, "error": repr(e)}
                ok = False
    res["pass"] = bool(ok)
    return res


def p2(train, X):
    path = C.P2 / "folds.csv"
    out = {"file": str(path.relative_to(C.REPO)), "exists": path.exists()}
    f = pd.read_csv(path)
    stored = dict(zip(f["family"], f["fold"]))
    out["families_in_file"] = len(stored)
    out["all_22_families_have_fold"] = bool(len(stored) == 22 and set(stored) == set(train["family"]))
    fold_now = F.make_folds(train)
    remade = dict(zip(train["family"], fold_now))
    out["make_folds_reproduces_file"] = bool(remade == stored)
    from sklearn.model_selection import GroupKFold
    g = GroupKFold(n_splits=F.N_SPLITS)
    out["GroupKFold_shuffle"] = g.shuffle
    out["GroupKFold_random_state"] = g.random_state
    out["note_seed"] = "folds.make_folds uses GroupKFold(n_splits=3) with shuffle=False; no random_state is involved (seed 42 has no effect on the folds)"
    # identity with the folds that trained the Phase 2 artifact: refit the selected configurations fold by fold and compare the stored fold MAEs
    fold = C.phase2_folds(train)
    meta = json.loads((C.P2_ART / "meta.json").read_text())
    rec = []
    for name, y, w in (("T", train["T"].values, None), ("D", train["D"].values, TM.d_weights(train["D"].values))):
        best = meta["models"][name]["best_params"]
        for k in range(3):
            tr, va = fold != k, fold == k
            m = TM.fit_one(X.values[tr], y[tr], best, None if w is None else w[tr])
            mae = float(np.abs(m.predict(X.values[va]) - y[va]).mean())
            rec.append({"model": name, "fold": k, "mae_refit": mae, "mae_phase2": meta["models"][name]["cv_mae_folds"][k],
                        "abs_diff": abs(mae - meta["models"][name]["cv_mae_folds"][k])})
    out["fold_mae_refit"] = rec
    out["max_abs_diff_fold_mae"] = max(r["abs_diff"] for r in rec)
    out["pass"] = bool(out["exists"] and out["all_22_families_have_fold"] and out["make_folds_reproduces_file"] and out["max_abs_diff_fold_mae"] < 1e-6)
    return out


def p5():
    """Label rule on the training rates: q_{i,r} = d_{i,r}/g_{i,r} per repeat, D_macro_r = mean_i q_{i,r}, D_macro = mean_r D_macro_r."""
    inst = pd.read_csv(C.DATA / "labels_instances_all.csv")
    inst = inst[inst["rate"] != "x3"]
    cand = pd.read_csv(C.DATA / "labels_train.csv")
    out = {"instances_checked(rate != x3)": len(inst), "candidates_checked": len(cand)}
    q_rep = inst["dropped_i_rep0"] / inst["generated_i_rep0"]
    out["generated_zero_rows"] = int((inst["generated_i_rep0"] == 0).sum())
    out["max_abs(q_i_rep0 - dropped/generated)"] = float((q_rep - inst["q_i_i_rep0"]).abs().max())
    key = ["family", "multiplier", "combination"]
    g = inst.groupby(key).agg(Dm=("q_i_i_rep0", "mean"), F=("completed_i_rep0", "sum"), n=("instance", "size")).reset_index()
    cand["multiplier"] = cand["multiplier"].astype(float)
    g["multiplier"] = g["multiplier"].astype(float)
    m = cand.merge(g, on=key, how="left", validate="one_to_one")
    out["max_abs(D_macro_rep0 - mean_i q_i_rep0)"] = float((m["D_macro_rep0"] - m["Dm"]).abs().max())
    out["max_abs(D_macro_mean - D_macro_rep0)  (train has 1 repeat)"] = float((m["D_macro_mean"] - m["D_macro_rep0"]).abs().max())
    out["max_abs(F_rep0 - sum_i completed_i_rep0 / 10s)"] = float((m["F_rep0"] - m["F"] / C.WINDOW_SEC).abs().max())
    out["n_reps_in_train"] = sorted(cand["n_reps"].unique().tolist())
    src = (C.PROF / "code" / "recompute_scores.py").read_text()
    out["code_rule"] = ("recompute_scores.build_tables: D_macro_mean = mean over repeats of system.drop_macro; counters.py: "
                        "drop_macro = mean_i q_i with q_i = dropped_full / generated of the same run; instance q_i_i = mean over repeats of q_i")
    out["code_lines_found"] = bool("Dm.mean()" in src and '("q_i", "q_i")' in src)
    out["pass"] = bool(out["generated_zero_rows"] == 0 and out["max_abs(q_i_rep0 - dropped/generated)"] < 1e-9
                       and out["max_abs(D_macro_rep0 - mean_i q_i_rep0)"] < 1e-9 and out["max_abs(F_rep0 - sum_i completed_i_rep0 / 10s)"] < 1e-9
                       and out["code_lines_found"])
    return out


def p1(train, X, cols):
    C.allow_test_labels()
    test = C.load_test()
    Xt = D.build_features(test, cols)
    mT, mD = P.load_model(C.P2_ART / "T.json"), P.load_model(C.P2_ART / "D.json")
    t, d = P.predict_multi(mT, mD, Xt)
    s, _ = P.multi_score(t, d, C.ALPHA)
    per = R.ours_per_workload(test, s, C.ALPHA, "XGBoost Full")
    a = E.aggregate(per, 3)
    m = next(x for x in E.mae_table(test, t, d, "x") if x["subset"] == "N>=3")
    p2pw = pd.read_csv(C.P2 / "per_workload_x3.csv")
    p2pw = p2pw[(p2pw["strategy"] == "Ours (retrained)") & (p2pw["N"] >= 3)].set_index("group")["selected"]
    sel = per[per["N"] >= 3].set_index("group")["selected"]
    out = {"features": len(cols), "candidates_N>=3": int((test["N"] >= 3).sum()), "workloads": a["workloads"],
           "avg_score": round(a["avg_score"], 4), "regret": round(a["regret"], 4), "top1": f"{a['top1_hits']}/{a['top1_n']}",
           "top5": f"{a['top5_hits']}/{a['top5_n']}", "mae_T": round(m["mae_T"], 4), "mae_D": round(m["mae_D"], 4),
           "selections_equal_phase2": bool((sel.sort_index() == p2pw.sort_index()).all()),
           "artifact_sha256": {n: C.sha(C.P2_ART / f"{n}.json") for n in ("T", "D")}}
    frozen = json.loads((C.P2 / "logs" / "config_frozen.json").read_text())
    out["artifact_sha256_equals_config_frozen"] = all(frozen["artifact_sha256"][f"artifacts/multi/{n}.json"] == out["artifact_sha256"][n] for n in ("T", "D"))
    expect = {"features": 204, "candidates_N>=3": 480, "workloads": 12, "avg_score": 0.9930, "regret": 0.0044, "top1": "7/12", "top5": "11/12"}
    out["expected"] = expect
    out["pass"] = bool(all(out[k] == v for k, v in expect.items()) and out["selections_equal_phase2"] and out["artifact_sha256_equals_config_frozen"])
    return out


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    train = C.load_train()
    X = D.build_features(train)
    cols = list(X.columns)
    assert cols == (C.P2 / "feature_columns.txt").read_text().split()
    (C.OUT / "feature_columns_full.txt").write_text("\n".join(cols) + "\n")
    res = {}
    res["P1"] = p1(train, X, cols)
    C.log(f"P1 {res['P1']}", "precheck.log")
    res["P2"] = p2(train, X)
    C.log(f"P2 pass={res['P2']['pass']} max fold-MAE diff={res['P2']['max_abs_diff_fold_mae']:.2e}", "precheck.log")
    res["P3"] = p3(X)
    C.log(f"P3 {res['P3']}", "precheck.log")
    res["P4"], lines = p4(cols, X)
    (C.OUT / "ablation_columns.txt").write_text("\n".join(lines))
    C.log(f"P4 {json.dumps({k: v for k, v in res['P4'].items() if k.startswith('minus') or k == 'pass'})}", "precheck.log")
    res["P5"] = p5()
    C.log(f"P5 {res['P5']}", "precheck.log")
    res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    (C.OUT / "precheck.json").write_text(json.dumps(res, indent=1, default=str))
    C.log(f"ALL PASS = {res['all_pass']}", "precheck.log")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Adapter: run the UNMODIFIED revision_exp pipeline modules on the re-profiled dataset (Phase 2 v1.2 section 4).

  .venv/bin/python3 -B revision_exp_adapter/run_reprofiled.py --data_root revision_profiling/dataset --out revision_exp/out_reprofiled
  .venv/bin/python3 -B revision_exp_adapter/run_reprofiled.py --data_root revision_profiling/dataset --out revision_exp/out_reprofiled_agg --drop agg

Only paths are changed: revision_exp/{data,folds,train_multi,train_single,predict,evaluate,baselines,plots,run_all}.py are imported
as they are (D.DS, D.SETS and D.OUT are pointed at the new folders at run time). Feature builder, grid, fold scheme, sample weight,
tie rules and the evaluator are the existing ones.

Fixed before the 3x labels are read (written to <out>/logs/config_frozen.json): alpha = 0.3, features, grid, selected hyper-parameters.
test_x3.csv is read only inside final_eval(), after training has finished and the frozen configuration has been written.

--drop macro (primary): D = D_macro (the drop_rate_norm column).  --drop agg (reference): D = drop_agg_norm, models retrained on it.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "revision_exp"))
os.environ.setdefault("MPLBACKEND", "Agg")

import baselines as B  # noqa: E402
import data as D  # noqa: E402
import evaluate as E  # noqa: E402
import folds as F  # noqa: E402
import predict as P  # noqa: E402
import run_all as R  # noqa: E402  (helper functions only; its main() is not called)
import train_multi as TM  # noqa: E402
import train_single as TS  # noqa: E402

ALPHA = D.ALPHA
assert ALPHA == 0.3
PROF = REPO / "revision_profiling"
LABEL_COLS = ["schedule_file", "combination", "n_reps", "F_mean", "F_max", "R_max", "D_macro_mean", "drop_agg_norm", "drop_max",
              "F_rep0", "F_rep1", "F_rep2", "R_agg_rep0", "R_agg_rep1", "R_agg_rep2", "D_macro_rep0", "D_macro_rep1", "D_macro_rep2"]
MANUSCRIPT = {"Latency-based": "0.63 / hit 0.17", "Ours": "0.95 / Top-1 0.72 / Top-5 0.94 (MAE 0.06 / 0.09)", "Best-of-5": "0.78 +- 0.040 / hit 0.30", "Oracle": "0.99"}
_T0 = time.time()
CTX = {}


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    with open(CTX["logs"] / "run_reprofiled.log", "a") as f:
        f.write(line + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name, drop):
    """Existing loader + the label columns of the same rows. D = D_macro (loader value) or drop_agg_norm."""
    df, dropped = D.load_set(name, dedup=True)
    assert len(dropped) == 0, "the re-profiled dataset must not contain duplicated candidates"
    lab = pd.read_csv(D.DS / ("labels_train.csv" if name == "train" else "labels_test.csv"))[LABEL_COLS]
    n = len(df)
    df = df.merge(lab, on=["schedule_file", "combination"], how="left", validate="one_to_one")
    assert len(df) == n and df["F_max"].notna().all()
    assert np.allclose(df["D"], df["D_macro_mean"]), "drop_rate_norm is not D_macro"
    df["D_macro"] = df["D"]
    if drop == "agg":
        df["D"] = df["drop_agg_norm"]
    return df


def s_rep(df, alpha, drop):
    """[rows, 3] per-repeat S with the workload normalisation of the mean labels (NaN where a repeat does not exist)."""
    out = np.full((len(df), 3), np.nan)
    for k in range(3):
        pen = df[f"D_macro_rep{k}"] if drop == "macro" else np.where(df["R_max"] > 0, df[f"R_agg_rep{k}"] / df["R_max"].replace(0, np.nan), 0.0)
        out[:, k] = (df[f"F_rep{k}"] / df["F_max"] - alpha * pen).values
    return out


def se3(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if np.isfinite(x).all() and len(x) > 1 else float("nan")


def num(c):
    return int(str(c).rsplit("_", 1)[1])


def add_se(per, df, alpha, drop):
    """Per-workload repeat statistics: SE (3 repeats) of the measured S of the selected candidate and of the oracle candidate."""
    idx = {(g, c): i for i, (g, c) in enumerate(zip(df["group"], df["comb_num"]))}
    sr = s_rep(df, alpha, drop)
    per = per.copy()
    per["label_reps"] = [int(df["n_reps"].values[idx[(g, num(o))]]) for g, o in zip(per["group"], per["oracle_combination"])]
    per["S_selected_se"] = [se3(sr[idx[(g, num(s))]]) if ev else np.nan for g, s, ev in zip(per["group"], per.get("selected", [None] * len(per)), per["evaluated"])]
    per["oracle_S_se"] = [se3(sr[idx[(g, num(o))]]) for g, o in zip(per["group"], per["oracle_combination"])]
    return per


def paired_difference(df, per_a, per_b, alpha, drop, groups, label):
    """Workload-wise S(a) - S(b) on the mean labels; label SE = SE over the 3 repeats of the paired per-repeat difference."""
    idx = {(g, c): i for i, (g, c) in enumerate(zip(df["group"], df["comb_num"]))}
    s_mean = (df["T"] - alpha * df["D"]).values
    sr = s_rep(df, alpha, drop)
    a = per_a.set_index("group")
    b = per_b.set_index("group")
    rows, skipped = [], []
    for g in groups:
        if not (a.loc[g, "evaluated"] and b.loc[g, "evaluated"]):
            skipped.append(g)
            continue
        ia, ib = idx[(g, num(a.loc[g, "selected"]))], idx[(g, num(b.loc[g, "selected"]))]
        d = float(s_mean[ia] - s_mean[ib])
        dr = sr[ia] - sr[ib]
        rows.append({"comparison": label, "group": g, "N": int(a.loc[g, "N"]), "selected_a": a.loc[g, "selected"], "selected_b": b.loc[g, "selected"],
                     "same_candidate": ia == ib, "S_a": float(s_mean[ia]), "S_b": float(s_mean[ib]), "diff": d,
                     "diff_rep0": dr[0], "diff_rep1": dr[1], "diff_rep2": dr[2], "diff_label_se": 0.0 if ia == ib else se3(dr),
                     "sign": "+" if d > 1e-9 else "-" if d < -1e-9 else "0"})
    pw = pd.DataFrame(rows)
    if not len(pw):
        return pw, {"comparison": label, "workloads": 0, "skipped": ";".join(skipped)}
    n = len(pw)
    se_l = pw["diff_label_se"].values
    summ = {"comparison": label, "workloads": n, "skipped": ";".join(skipped), "mean_diff": float(pw["diff"].mean()),
            "se_across_workloads": float(pw["diff"].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "n_positive": int((pw["sign"] == "+").sum()), "n_zero": int((pw["sign"] == "0").sum()), "n_negative": int((pw["sign"] == "-").sum()),
            "median_diff": float(pw["diff"].median()), "min_diff": float(pw["diff"].min()), "max_diff": float(pw["diff"].max()),
            "label_se_of_mean(repeat noise only)": float(np.sqrt(np.nansum(se_l ** 2)) / n),
            "label_se_per_workload_median": float(np.nanmedian(se_l)), "label_se_per_workload_max": float(np.nanmax(se_l)),
            "workloads_with_|diff|<=label_se": int((np.abs(pw["diff"].values) <= se_l).sum() - int((pw["sign"] == "0").sum()))}
    return pw, summ


def agg_row(per, name, groups):
    p = per[per["group"].isin(groups)]
    a = E.aggregate(p, 1, name)
    return {"strategy": name, "workloads": a["workloads"], "not_evaluated": a["workloads_not_evaluated"], "avg_score": round(a["avg_score"], 4),
            "top1": E.frac(a["top1_hits"], a["top1_n"]), "top5": E.frac(a["top5_hits"], a["top5_n"]), "regret": round(a["regret"], 4),
            "oracle_avg_same_workloads": round(a["oracle_avg_same_workloads"], 4), "selected_all_gpu": a.get("selected_all_gpu", "")}


def bo5_row(bo5, df, alpha, groups):
    g, S_sel, top1, contains = bo5
    m = np.array([x in groups for x in g])
    orc = df.assign(S=df["T"] - alpha * df["D"]).groupby("group")["S"].max()
    orc = np.array([orc[x] for x in g])[m]
    means = S_sel[:, m].mean(axis=1)
    return {"strategy": "Best-of-5 (1,000 candidate-sampling trials)", "workloads": int(m.sum()), "not_evaluated": 0, "avg_score": round(float(means.mean()), 4),
            "avg_score_std_over_trials": round(float(means.std(ddof=0)), 4), "top1": f"{top1[:, m].mean():.3f} (mean rate)", "top5": "N/A",
            "regret": round(float(orc.mean() - means.mean()), 4), "oracle_avg_same_workloads": round(float(orc.mean()), 4),
            "sample_contains_oracle": round(float(contains[:, m].mean()), 3)}


def offline_latency():
    """Isolated (closed-loop) service latency per model and device from the new offline profile."""
    p = D.note_input(PROF / "offline_profile" / "offline_profile.json")
    prof = json.loads(Path(p).read_text())["models"]
    rows = []
    for m, v in sorted(prof.items()):
        for dev in ("cpu", "gpu"):
            rows.append({"model": m, "device": dev.upper(), "latency_ms": v[dev]["service_ms_p50"], "throughput_fps": v[dev]["throughput_fps"],
                         "source": "offline_profile.json (closed loop, isolated)"})
    return pd.DataFrame(rows)


def smoke_latency():
    """1x open-loop single-instance service latency (Phase 1b section 5.1 smoke runs). Used only for the 5.1-exception analysis."""
    p = D.note_input(PROF / "pilot" / "smoke" / "smoke_5_1.csv")
    s = pd.read_csv(p)
    return pd.DataFrame({"model": s["model"], "device": s["device"].str.upper(), "latency_ms": s["service_p50_ms"],
                         "source": "pilot/smoke (open loop, 1x, beta 0.1)"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="revision_profiling/dataset")
    ap.add_argument("--out", default="revision_exp/out_reprofiled")
    ap.add_argument("--drop", choices=("macro", "agg"), default="macro")
    ap.add_argument("--reuse", action="store_true", help="reuse <out>/artifacts instead of retraining")
    a = ap.parse_args()
    OUT = (REPO / a.out).resolve()
    assert OUT.parent == REPO / "revision_exp" and OUT.name.startswith("out_reprofiled"), "outputs must go to revision_exp/out_reprofiled*"
    LOGS, ART = OUT / "logs", OUT / "artifacts"
    for d in (OUT, LOGS, OUT / "figs", ART / "multi", ART / "single"):
        d.mkdir(parents=True, exist_ok=True)
    CTX["logs"] = LOGS
    D.DS = (REPO / a.data_root).resolve()
    D.SETS = {"train": ("train_x3.csv", "train_schedules_x3.csv", []), "x3": ("test_x3.csv", "test_schedules_x3.csv", [])}
    D.OUT = OUT
    drop = a.drop
    log(f"command: {' '.join(sys.argv)} | data_root={D.DS} | out={OUT} | drop={drop} | alpha={ALPHA}")
    (LOGS / "git_status_before.txt").write_text(subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout)
    import sklearn
    import xgboost
    env = {"python": platform.python_version(), "xgboost": xgboost.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
           "pandas": pd.__version__, "cpu_count": os.cpu_count(), "master_seed": D.MASTER_SEED, "alpha": ALPHA, "drop_definition": drop}
    (LOGS / "environment.json").write_text(json.dumps(env, indent=1))
    log(E.self_test())

    # ------------------------------------------------------------------ training data only
    train = load("train", drop)
    assert set(train["rate"]) == {"x1", "x2", "x4"}, set(train["rate"])
    X = D.build_features(train)
    cols = list(X.columns)
    (OUT / "feature_columns.txt").write_text("\n".join(cols) + "\n")
    fold = F.make_folds(train)                       # asserts that no family is split across folds
    for k in range(F.N_SPLITS):
        assert set(train["family"][fold == k]).isdisjoint(train["family"][fold != k]), "family in train and validation of the same fold"
    fam, per_fold = F.describe(train, fold)
    fam.to_csv(OUT / "folds.csv", index=False)
    per_fold.to_csv(OUT / "folds_summary.csv", index=False)
    log(f"train rows={len(train)} workloads={train['group'].nunique()} families={train['family'].nunique()} features={len(cols)}")
    log("\n" + per_fold.to_string(index=False))
    pd.DataFrame([D.summarize("train", train, train, pd.DataFrame())]).to_csv(OUT / "data_summary_train.csv", index=False)

    if a.reuse and (ART / "multi" / "meta.json").exists() and (ART / "single" / "meta.json").exists():
        meta_m = json.loads((ART / "multi" / "meta.json").read_text())
        meta_s = json.loads((ART / "single" / "meta.json").read_text())
        assert meta_m["features"] == cols and meta_m["drop_definition"] == drop
        log("reusing artifacts")
    else:
        t0 = time.time()
        meta_m, rec_m = TM.train_multi(X, train, fold, ART / "multi", log)
        meta_m.update({"env": env, "train_rows": len(train), "fold_scheme": "GroupKFold(3) by family", "drop_definition": drop})
        (ART / "multi" / "meta.json").write_text(json.dumps(meta_m, indent=1))
        rec_m.to_csv(OUT / "hparam_search_multi.csv", index=False)
        meta_s, rec_s = TS.train_single(X, train, fold, ART / "single", log)
        meta_s.update({"env": env, "train_rows": len(train), "fold_scheme": "GroupKFold(3) by family", "drop_definition": drop})
        (ART / "single" / "meta.json").write_text(json.dumps(meta_s, indent=1))
        rec_s.to_csv(OUT / "hparam_search_single.csv", index=False)
        (LOGS / "training_time.json").write_text(json.dumps({"seconds": time.time() - t0}))
        log(f"training done in {(time.time() - t0) / 60:.1f} min")
    mT, mD, mDn = (P.load_model(ART / "multi" / f"{n}.json") for n in ("T", "D", "D_noweight"))
    single = {al: P.load_model(ART / "single" / f"alpha_{al}.json") for al in TS.ALPHAS}

    # training-data performance (in-sample; never mixed with the test table)
    t_in, d_in = P.predict_multi(mT, mD, X)
    s_in, _ = P.multi_score(t_in, d_in, ALPHA)
    rows = []
    for rate, sub in train.assign(_s=s_in).groupby("rate"):
        per = R.ours_per_workload(sub, sub["_s"].values, ALPHA, "Ours")
        for mn in (3, 1):
            ag = E.aggregate(per, mn)
            rows.append({"data": "TRAINING DATA (in-sample)", "rate": rate, "min_N": mn, "workloads": ag["workloads"], "avg_score": round(ag["avg_score"], 4),
                         "oracle_avg": round(ag["oracle_avg_same_workloads"], 4), "top1": E.frac(ag["top1_hits"], ag["top1_n"]),
                         "top5": E.frac(ag["top5_hits"], ag["top5_n"]), "regret": round(ag["regret"], 4)})
    pd.DataFrame(rows).to_csv(OUT / "train_performance.csv", index=False)
    cv = [{"model": k, "cv_mae_mean": v["cv_mae_mean"], "cv_mae_folds": json.dumps(v["cv_mae_folds"]), "best_params": json.dumps(v["best_params"])}
          for k, v in meta_m["models"].items()]
    pd.DataFrame(cv).to_csv(OUT / "cv_mae_multi.csv", index=False)

    frozen = {"frozen_at": datetime.now().isoformat(sep=" ", timespec="seconds"), "alpha": ALPHA, "drop_definition": drop, "n_features": len(cols),
              "grid": TM.GRID, "base_params": TM.BASE_PARAMS, "d_weight": {"threshold": TM.D_WEIGHT_THRESHOLD, "value": TM.D_WEIGHT_VALUE},
              "fold_scheme": "GroupKFold(3) by family", "train_rates": sorted(train["rate"].unique()),
              "best_params": {k: v["best_params"] for k, v in meta_m["models"].items()},
              "single_best_params": {k: v["best_params"] for k, v in meta_s["models"].items()},
              "artifact_sha256": {str(p.relative_to(OUT)): sha(p) for p in sorted(ART.rglob("*.json")) if p.name != "meta.json"},
              "train_csv_sha256": sha(D.DS / "train_x3.csv"),
              "statement": "written before test_x3.csv is opened by this process; nothing below this point changes models, features, hyper-parameters or alpha"}
    (LOGS / "config_frozen.json").write_text(json.dumps(frozen, indent=1))
    log("configuration frozen; reading the 3x hold-out now")
    final_eval(OUT, LOGS, drop, cols, (mT, mD, mDn), single, meta_m, meta_s)
    D.write_manifest()
    (LOGS / "git_status_after.txt").write_text(subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout)
    log(f"total wall time {time.time() - _T0:.1f}s")


def final_eval(OUT, LOGS, drop, cols, models, single, meta_m, meta_s):
    mT, mD, mDn = models
    df = load("x3", drop)
    assert set(df["rate"]) == {"x3"}
    Xn = D.build_features(df, cols)
    t_hat, d_hat = P.predict_multi(mT, mD, Xn)
    _, d_nw = P.predict_multi(mT, mDn, Xn)
    S_true = (df["T"] - ALPHA * df["D"]).values
    s_new, raw_new = P.multi_score(t_hat, d_hat, ALPHA)
    s_nw, _ = P.multi_score(t_hat, d_nw, ALPHA)
    cand = R.cand_of(df)
    lat_closed, lat_open = offline_latency(), smoke_latency()
    tg, tg_open = B.latency_targets(lat_closed), B.latency_targets(lat_open)
    lt = lat_closed.pivot(index="model", columns="device", values="latency_ms").add_prefix("closed_loop_ms_")
    lt = lt.join(lat_open.pivot(index="model", columns="device", values="latency_ms").add_prefix("open_loop_1x_ms_"))
    lt["target_closed_loop(primary)"] = pd.Series(tg)
    lt["target_open_loop_1x"] = pd.Series(tg_open)
    lt["targets_differ"] = lt["target_closed_loop(primary)"] != lt["target_open_loop_1x"]
    lt["cpu_open_over_closed"] = lt["open_loop_1x_ms_CPU"] / lt["closed_loop_ms_CPU"]
    lt.reset_index().to_csv(OUT / "latency_targets.csv", index=False)
    log(f"latency targets (closed loop): {tg} | open loop 1x: {tg_open}")
    names = {"lat": "Latency-based", "ours": "Ours (retrained)", "gpu": "All-GPU", "cpu": "All-CPU", "orc": "Oracle",
             "lat_open": "Latency-based (1x open-loop latency)", "nw": "Ours (retrained, D without weight)"}
    per = {
        names["lat"]: R.single_per_workload(df, B.choose_by_targets(cand, tg), ALPHA, names["lat"]),
        names["ours"]: R.ours_per_workload(df, s_new, ALPHA, names["ours"]),
        names["gpu"]: R.single_per_workload(df, B.choose_all_device(cand, "all_gpu"), ALPHA, names["gpu"]),
        names["cpu"]: R.single_per_workload(df, B.choose_all_device(df[["group", "comb_num", "all_cpu"]], "all_cpu"), ALPHA, names["cpu"]),
        names["orc"]: R.ours_per_workload(df, S_true, ALPHA, names["orc"]),
        names["lat_open"]: R.single_per_workload(df, B.choose_by_targets(cand, tg_open), ALPHA, names["lat_open"]),
        names["nw"]: R.ours_per_workload(df, s_nw, ALPHA, names["nw"]),
    }
    bo5 = B.best_of_k(cand, ALPHA)
    mae = {names["ours"]: E.mae_table(df, t_hat, d_hat, "T + D weighted"), names["nw"]: E.mae_table(df, t_hat, d_nw, "T + D_noweight")}
    pd.DataFrame({"row_uid": df["row_uid"], "group": df["group"], "combination": df["combination"], "N": df["N"], "placement": df["placement"],
                  "T": df["T"], "D": df["D"], "D_macro": df["D_macro"], "D_agg": df["drop_agg_norm"], "D_max": df["drop_max"],
                  "T_hat": t_hat, "D_hat": d_hat, "D_hat_noweight": d_nw, "S": S_true, "S_hat_raw": raw_new, "S_hat": s_new,
                  "all_gpu": df["all_gpu"], "n_reps": df["n_reps"]}).to_csv(OUT / "predictions_x3.csv", index=False)

    n_of = df.groupby("group")["N"].first()
    g3 = set(n_of[n_of >= 3].index)
    g_all = set(n_of.index)
    gm_path = PROF / "results" / "integrity" / "gen_mismatch.csv"
    gm = pd.read_csv(D.note_input(gm_path))
    gm["group"] = gm["workload"] + ".yaml"
    flagged = set(gm.loc[gm["gen_violation"], "group"])
    g3_clean = g3 - flagged

    # ---- per-workload file with repeat statistics
    pw = pd.concat([add_se(p, df, ALPHA, drop) for p in per.values()], ignore_index=True)
    g, S_sel, top1, contains = bo5
    pw_b = pd.DataFrame({"strategy": "Best-of-5 (1,000 candidate-sampling trials)", "group": g, "S_selected": S_sel.mean(axis=0),
                         "S_selected_std_over_trials": S_sel.std(axis=0), "top1_rate": top1.mean(axis=0), "sample_contains_oracle": contains.mean(axis=0)})
    pw_b = pw_b.merge(df.groupby("group").agg(N=("N", "first"), n_candidates=("row_uid", "size")).reset_index(), on="group")
    pw = pd.concat([pw, pw_b], ignore_index=True)
    pw = pw.merge(gm[["group", "gen_mismatch_max", "gen_mismatch_abs_max", "gen_violation"]], on="group", how="left")
    pw.to_csv(OUT / "per_workload_x3.csv", index=False)

    # ---- Table 3(a) and the all-N table; previous numbers in separate columns
    prev = pd.read_csv(D.note_input(REPO / "revision_exp" / "out" / "table3a.csv")).set_index("strategy")
    prev_name = {names["lat"]: "Latency-based", names["ours"]: "Ours (retrained)", names["gpu"]: "All-GPU", names["cpu"]: "All-CPU", names["orc"]: "Oracle",
                 names["nw"]: "Ours (retrained, D without weight)", "Best-of-5 (1,000 candidate-sampling trials)": "Best-of-5 (1,000 trials)"}
    order = [names["lat"], names["ours"], "Best-of-5", names["gpu"], names["cpu"], names["orc"], names["lat_open"], names["nw"]]

    def table(groups):
        rows = []
        for nm in order:
            if nm == "Best-of-5":
                r = bo5_row(bo5, df, ALPHA, groups)
            else:
                r = agg_row(per[nm], nm, groups)
                mm = next((x for x in mae.get(nm, []) if x["subset"] == "N>=3"), None)
                if mm and groups == g3:
                    r["mae_T"], r["mae_D"] = round(mm["mae_T"], 4), round(mm["mae_D"], 4)
            rows.append(r)
        return pd.DataFrame(rows)

    t3 = table(g3)
    t3["previous_data_retrained(revision_exp/out, 18 workloads, D_agg)"] = [
        (f"{prev.loc[prev_name[s], 'avg_score']} / {prev.loc[prev_name[s], 'top1']} / {prev.loc[prev_name[s], 'top5']}" if prev_name.get(s) in prev.index else "")
        for s in t3["strategy"]]
    t3["manuscript_Table_3a(old data)"] = [MANUSCRIPT.get(s.split(" (")[0], "") if s in (names["lat"], names["ours"], names["orc"]) or s.startswith("Best-of-5") else ""
                                           for s in t3["strategy"]]
    t3.to_csv(OUT / "table3a.csv", index=False)
    note = ("\nNotes: evaluation workloads = 3x, N>=3 (denominators in the table). Oracle and all measured scores use the 3-repeat mean labels. "
            "Best-of-5: 1,000 repetitions of CANDIDATE SAMPLING (5 placements drawn without replacement, the best measured one taken); "
            "it is not a repetition of measurements. D = " + ("D_macro (mean of the per-instance drop fractions)" if drop == "macro" else "D_agg (aggregate drop rate / workload maximum)")
            + f", alpha = {ALPHA}. The two right-most columns are earlier numbers on the OLD data with the old drop definition; they are not comparable row by row.\n")
    (OUT / "table3a.md").write_text(f"# Table 3(a) - re-profiled CPU-GPU data, 3x hold-out, N>=3, alpha={ALPHA}, drop={drop}\n\n" + R.md_table(t3) + note)
    table(g_all).to_csv(OUT / "table_all_n.csv", index=False)
    log("\n" + t3[["strategy", "workloads", "avg_score", "top1", "top5", "regret"]].to_string(index=False))

    # ---- paired differences
    pds, summ = [], []
    for label, pa, pb, groups in (("Ours - All-GPU", names["ours"], names["gpu"], g3), ("Oracle - All-GPU", names["orc"], names["gpu"], g3),
                                  ("Ours - Oracle", names["ours"], names["orc"], g3), ("Ours - Latency-based", names["ours"], names["lat"], g3),
                                  ("Ours - All-GPU [clean subset]", names["ours"], names["gpu"], g3_clean),
                                  ("Ours - All-GPU [all N]", names["ours"], names["gpu"], g_all)):
        p, s = paired_difference(df, per[pa], per[pb], ALPHA, drop, sorted(groups), label)
        pds.append(p)
        summ.append(s)
    pd.concat(pds, ignore_index=True).to_csv(OUT / "paired_per_workload.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / "paired_summary.csv", index=False)
    log("\n" + pd.DataFrame(summ).to_string(index=False))

    # ---- clean subset (generated-count difference); no correction of the observed values
    cl = []
    for tag, groups in (("all (N>=3)", g3), ("clean subset (N>=3, no generated-count violation)", g3_clean)):
        t = table(groups)
        t.insert(0, "subset", tag)
        t.insert(1, "evaluation_workloads", len(groups))
        cl.append(t)
    cl = pd.concat(cl, ignore_index=True)
    cl.to_csv(OUT / "table_clean_subset.csv", index=False)
    pd.DataFrame({"group": sorted(g3), "flagged_gen_violation": [x in flagged for x in sorted(g3)]}).merge(
        gm[["group", "gen_mismatch_max", "gen_mismatch_abs_max", "violations"]], on="group", how="left").to_csv(OUT / "clean_subset_membership.csv", index=False)

    # ---- MAE and D-weight ablation
    pd.DataFrame([dict(x, strategy=k) for k, v in mae.items() for x in v]).to_csv(OUT / "mae_x3.csv", index=False)
    ab = []
    for nm, key, mk in (("T + D (weighted, primary)", names["ours"], "D"), ("T + D_noweight", names["nw"], "D_noweight")):
        ag = E.aggregate(per[key][per[key]["group"].isin(g3)], 1)
        mm = next(x for x in mae[key] if x["subset"] == "N>=3")
        ab.append({"variant": nm, "workloads": ag["workloads"], "mae_T": round(mm["mae_T"], 4), "mae_D": round(mm["mae_D"], 4),
                   "mae_D_where_D>0.01": round(mm["mae_D_where_D>0.01"], 4), "candidates_D>0.01": mm["candidates_D>0.01"], "candidates": mm["candidates"],
                   "avg_score": round(ag["avg_score"], 4), "top1": E.frac(ag["top1_hits"], ag["top1_n"]), "top5": E.frac(ag["top5_hits"], ag["top5_n"]),
                   "regret": round(ag["regret"], 4), "cv_mae_D": round(meta_m["models"][mk]["cv_mae_mean"], 4), "best_params_D": json.dumps(meta_m["models"][mk]["best_params"])})
    pd.DataFrame(ab).to_csv(OUT / "ablation_d_weight.csv", index=False)

    # ---- validation C: retrained + evaluator, previous data vs re-profiled data
    pc = pd.read_csv(D.note_input(REPO / "revision_exp" / "out" / "validation_C.csv"))
    pc = pc[(pc["step"].str.startswith("C")) & (pc["min_N"] == 3)].assign(data="previous data (revision_exp/out)")
    ag = E.aggregate(per[names["ours"]][per[names["ours"]]["group"].isin(g3)], 1)
    vc = pd.concat([pc, pd.DataFrame([{"step": "C retrained + new evaluator", "min_N": 3, "workloads": ag["workloads"],
                                       "oracle_avg": round(ag["oracle_avg_same_workloads"], 4), "avg_score": round(ag["avg_score"], 4),
                                       "top1": E.frac(ag["top1_hits"], ag["top1_n"]), "top5": E.frac(ag["top5_hits"], ag["top5_n"]),
                                       "regret": round(ag["regret"], 4), "note": f"re-profiled data, drop={drop}", "data": "re-profiled data"}])], ignore_index=True)
    vc.to_csv(OUT / "validation_C.csv", index=False)

    # ---- Fig. 3
    old3 = pd.read_csv(D.note_input(REPO / "revision_exp" / "out" / "fig3_data.csv")).set_index("N>=")
    rows = []
    for n in (3, 4, 5, 6, 7):
        groups = set(n_of[n_of >= n].index)
        b = bo5_row(bo5, df, ALPHA, groups)
        row = {"N>=": n, "workloads": len(groups)}
        for label, key in (("Oracle", names["orc"]), ("Ours", names["ours"]), ("Latency-based", names["lat"]), ("All-GPU", names["gpu"]), ("All-CPU", names["cpu"])):
            ag = agg_row(per[key], key, groups)
            row[label] = ag["avg_score"]
            row[f"{label}_workloads"] = ag["workloads"]
        row["Best-of-5"], row["Best-of-5_std"] = b["avg_score"], b["avg_score_std_over_trials"]
        for k in ("Oracle", "Ours", "Latency-based", "All-GPU", "Best-of-5"):
            row[f"previous_data_retrained_{k}"] = old3.loc[n, k] if n in old3.index else ""
        for k in ("Oracle", "Ours", "Best-of-5", "Latency-based"):
            row[f"manuscript_figure_{k}"] = old3.loc[n, f"old_figure_{k}"] if n in old3.index else ""
        rows.append(row)
    f3 = pd.DataFrame(rows)
    f3.to_csv(OUT / "fig3_data.csv", index=False)
    log("\n" + f3[["N>=", "workloads", "Oracle", "Ours", "All-GPU", "Best-of-5", "Latency-based", "All-CPU"]].to_string(index=False))

    # ---- Fig. 4: alpha sensitivity (alpha is NOT selected here; 0.3 stays the primary setting)
    d3 = df[df["group"].isin(g3)]

    def oracle_sets(alpha):
        out = {}
        for gname, sub in d3.groupby("group"):
            s = (sub["T"] - alpha * sub["D"]).values
            out[gname] = frozenset(int(c) for c, v in zip(sub["comb_num"], s) if v >= s.max() - E.TIE_TOL)
        return out
    ref_or = oracle_sets(ALPHA)
    ref_sel = None
    rows, sel_by_alpha = [], {}
    for al in TS.ALPHAS:
        S_a = (df["T"] - al * df["D"]).values
        sm, _ = P.multi_score(t_hat, d_hat, al)
        ss, _ = P.single_score(single[al], Xn)
        per_m = R.ours_per_workload(df, sm, al, "multi")
        per_s = R.ours_per_workload(df, ss, al, "single")
        per_o = R.ours_per_workload(df, S_a, al, "oracle")
        pm, ps, po = (E.aggregate(x[x["group"].isin(g3)], 1) for x in (per_m, per_s, per_o))
        osets = oracle_sets(al)
        sel_by_alpha[al] = per_m[per_m["group"].isin(g3)].set_index("group")["selected"].to_dict()
        row = {"alpha": al, "workloads": pm["workloads"], "oracle_avg": round(po["avg_score"], 4),
               "workloads_oracle_set_differs_from_alpha_0.3": sum(osets[x] != ref_or[x] for x in osets),
               "workloads_oracle_disjoint_from_alpha_0.3": sum(not (osets[x] & ref_or[x]) for x in osets),
               "oracle_is_all_gpu": int(per_o[per_o["group"].isin(g3)]["selected_all_gpu"].astype(bool).sum())}
        for tag, p in (("multi", pm), ("single", ps)):
            row.update({f"avg_score_{tag}": round(p["avg_score"], 4), f"top1_{tag}": round(p["top1_hits"] / p["top1_n"], 4), f"top1_hits_{tag}": p["top1_hits"],
                        f"top5_{tag}": round(p["top5_hits"] / p["top5_n"], 4), f"top5_hits_{tag}": p["top5_hits"], f"regret_{tag}": round(p["regret"], 4),
                        f"selected_all_gpu_{tag}": p.get("selected_all_gpu")})
        row["single_best_params"] = json.dumps(meta_s["models"][str(al)]["best_params"])
        rows.append(row)
    ref_sel = sel_by_alpha[ALPHA]
    for row in rows:
        row["workloads_multi_selection_differs_from_alpha_0.3"] = sum(sel_by_alpha[row["alpha"]][x] != ref_sel[x] for x in ref_sel)
    f4 = pd.DataFrame(rows)
    o4 = pd.read_csv(D.note_input(REPO / "revision_exp" / "out" / "fig4_data.csv"))
    keep = ["alpha", "avg_score_multi", "top1_hits_multi", "top5_hits_multi", "avg_score_single", "top1_hits_single", "top5_hits_single", "oracle_avg",
            "old_figure_avg_multi", "old_figure_top1_multi", "old_figure_top5_multi", "old_figure_avg_single", "old_figure_top1_single", "old_figure_top5_single"]
    o4 = o4[keep].rename(columns={c: (c.replace("old_figure_", "manuscript_figure_") if c.startswith("old_figure_") else f"previous_data_retrained_{c}")
                                  for c in keep if c != "alpha"})
    f4 = f4.merge(o4, on="alpha", how="left")
    f4.to_csv(OUT / "fig4_data.csv", index=False)
    log("\n" + f4[["alpha", "oracle_avg", "workloads_oracle_set_differs_from_alpha_0.3", "avg_score_multi", "top1_hits_multi", "top5_hits_multi",
                   "avg_score_single", "top1_hits_single", "top5_hits_single", "workloads_multi_selection_differs_from_alpha_0.3"]].to_string(index=False))

    import plots as PL
    PL.fig3(f3, OUT / "figs" / "fig3_model_count_vs_score.pdf", OUT / "figs" / "fig3_model_count_vs_score.png")
    PL.fig4(f4, OUT / "figs" / "fig4_alpha_comparison.pdf", OUT / "figs" / "fig4_alpha_comparison.png")

    # ---- oracle under the three drop definitions (macro run only), with the instances that make them differ
    if drop == "macro":
        inst = pd.read_csv(D.note_input(D.DS / "labels_instances_all.csv"))
        inst = inst[inst["rate"] == "x3"]
        rows, irows = [], []
        for gname, sub in d3.groupby("group"):
            sub = sub.sort_values("comb_num")
            defs = {"macro(b)": sub["T"] - ALPHA * sub["D_macro"], "agg(a)": sub["T"] - ALPHA * sub["drop_agg_norm"], "max(c)": sub["T"] - ALPHA * sub["drop_max"]}
            row = {"group": gname, "N": int(sub["N"].iloc[0])}
            pick = {}
            for dn, s in defs.items():
                s = s.values
                oset = [int(c) for c, v in zip(sub["comb_num"], s) if v >= s.max() - E.TIE_TOL]
                pick[dn] = oset
                r0 = sub[sub["comb_num"] == oset[0]].iloc[0]
                row.update({f"oracle_{dn}": f"combination_{oset[0]:03d}", f"placement_{dn}": "".join("G" if x.endswith("GPU") else "C" for x in r0["placement"].split(";")),
                            f"n_oracle_{dn}": len(oset), f"all_gpu_{dn}": bool(r0["all_gpu"]), f"S_{dn}": float(s.max())})
            row["b_vs_a_differ"] = not (set(pick["macro(b)"]) & set(pick["agg(a)"]))
            row["b_vs_c_differ"] = not (set(pick["macro(b)"]) & set(pick["max(c)"]))
            sb = dict(zip(sub["comb_num"], defs["macro(b)"]))
            sa = dict(zip(sub["comb_num"], defs["agg(a)"]))
            row["S_b_of_a_oracle"], row["S_a_of_b_oracle"] = float(sb[pick["agg(a)"][0]]), float(sa[pick["macro(b)"][0]])
            row["regret_b_if_a_oracle_used"] = row["S_macro(b)"] - row["S_b_of_a_oracle"]
            rows.append(row)
            if row["b_vs_a_differ"] or row["b_vs_c_differ"]:
                fam = gname[:-len("_x3.yaml")]
                for dn in defs:
                    ii = inst[(inst["family"] == fam) & (inst["combination"] == f"combination_{pick[dn][0]:03d}")]
                    for _, x in ii.iterrows():
                        irows.append({"group": gname, "definition": dn, "combination": x["combination"], "placement": x["placement"], "instance": x["instance"],
                                      "device": x["device"], "lambda_i": x["lambda_i"], "generated_i": x["generated_i"], "completed_i": x["completed_i"],
                                      "dropped_i": x["dropped_i"], "q_i": x["q_i_i"]})
        od = pd.DataFrame(rows)
        od.to_csv(OUT / "oracle_definitions.csv", index=False)
        pd.DataFrame(irows).to_csv(OUT / "oracle_definitions_instances.csv", index=False)
        log(f"oracle (b) vs (a) differ in {int(od['b_vs_a_differ'].sum())}/{len(od)} workloads; (b) vs (c): {int(od['b_vs_c_differ'].sum())}/{len(od)}")

    summary = {"drop": drop, "alpha": ALPHA, "evaluation_workloads_N>=3": len(g3), "evaluation_workloads_all_N": len(g_all), "candidates_x3": len(df),
               "clean_subset_workloads": len(g3_clean), "flagged_workloads": sorted(g3 & flagged), "latency_targets_closed": tg, "latency_targets_open": tg_open,
               "s_hat_clipped_rows": int((raw_new != s_new).sum()),
               "ties_at_first_place_N>=3": int((per[names["ours"]][per[names["ours"]]["group"].isin(g3)]["n_tied_with_first"] > 1).sum())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

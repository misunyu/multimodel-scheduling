#!/usr/bin/env python3
"""End-to-end reproduction: data -> folds -> timing probe -> training -> validation A/B/C -> tables -> figures.

Run from the repository root:
    .venv/bin/python3 -B revision_exp/run_all.py              # full run (retrains)
    .venv/bin/python3 -B revision_exp/run_all.py --reuse      # reuse revision_exp/artifacts, redo evaluation/figures
    .venv/bin/python3 -B revision_exp/run_all.py --probe-only # data checks + fit-time estimate, then stop
Everything is written under revision_exp/. Existing code, data, artifacts and figures are only read.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baselines as B  # noqa: E402
import data as D  # noqa: E402
import evaluate as E  # noqa: E402
import folds as F  # noqa: E402
import predict as P  # noqa: E402
import train_multi as TM  # noqa: E402
import train_single as TS  # noqa: E402

OUT, LOGS, ART = D.OUT, D.OUT / "logs", D.EXP / "artifacts"
ALPHA = D.ALPHA
PROTECTED = ["experimental_results", "xgboost_model/artifacts", "xgboost_model/prediction_result", "xgboost_model/dataset", "scripts",
             "results", "results_recompute", "results_recompute_additional", "gen_schedules", "gen_schedules_additional",
             "sample_profiling_data", "xgboost_model/deploy_selector_xgb_suite.py"]
MANUSCRIPT = {"Latency-based": "0.63 / hit 0.17", "Ours": "0.95 / Top-1 0.72 / Top-5 0.94 (MAE 0.06 / 0.09, gap 0.03)",
              "Best-of-5": "0.78 +- 0.040 / hit 0.30", "Oracle": "0.99"}
_T0 = time.time()
STAGES = []


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    with open(LOGS / "run_all.log", "a") as f:
        f.write(line + "\n")


class stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        self.start = datetime.now().isoformat(sep=" ", timespec="seconds")
        log(f"== {self.name} ==")

    def __exit__(self, *a):
        STAGES.append({"stage": self.name, "start": self.start, "end": datetime.now().isoformat(sep=" ", timespec="seconds"),
                       "seconds": round(time.time() - self.t, 2)})


def protected_hashes():
    out = {}
    for p in PROTECTED:
        root = D.REPO / p
        files = [root] if root.is_file() else sorted(x for x in root.rglob("*") if x.is_file() and x.suffix != ".pyc")
        for f in files:
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[str(f.relative_to(D.REPO))] = h.hexdigest()
    return out


def git_status():
    return subprocess.run(["git", "-C", str(D.REPO), "status", "--short"], capture_output=True, text=True).stdout


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) or v is None else str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- evaluation
def cand_of(df):
    return df[["group", "comb_num", "combination", "N", "T", "D", "all_gpu", "placement"]].copy()


def ours_per_workload(df, s_hat, alpha, name):
    cand = cand_of(df)
    ids = cand[["group", "comb_num"]]
    order, ties = E.select(ids, s_hat)
    return E.score_selection(cand, order, alpha, name, top5=True, ties=ties, legacy_top5=E.legacy_top5_groups(ids, s_hat))


def single_per_workload(df, chosen, alpha, name):
    cand = cand_of(df)
    return E.score_selection(cand, E.single_choice(cand[["group", "comb_num"]], chosen), alpha, name, top5=False)


def bo5_summary(df, alpha, min_n, bo5=None):
    groups, S_sel, top1, contains = bo5 if bo5 is not None else B.best_of_k(cand_of(df), alpha)
    n_of = df.groupby("group")["N"].first()
    m = np.array([n_of[g] >= min_n for g in groups])
    orc = (df.assign(S=df["T"] - alpha * df["D"]).groupby("group")["S"].max())
    orc = np.array([orc[g] for g in groups])[m]
    means = S_sel[:, m].mean(axis=1)
    return {"strategy": "Best-of-5 (1,000 trials)", "min_N": min_n, "workloads": int(m.sum()), "avg_score": float(means.mean()),
            "avg_score_std": float(means.std(ddof=0)), "top1_rate": float(top1[:, m].mean()), "top1_rate_std": float(top1[:, m].mean(axis=1).std(ddof=0)),
            "sample_contains_oracle": float(contains[:, m].mean()), "regret": float(orc.mean() - means.mean()),
            "oracle_avg_same_workloads": float(orc.mean())}


def table_rows(per_by_name, bo5_row, mae_by_name, min_n, manuscript=True):
    rows = []
    for name, per in per_by_name.items():
        a = E.aggregate(per, min_n, name)
        mae = mae_by_name.get(name, {})
        key = "N>=3" if min_n >= 3 else "N>=1"
        mm = next((r for r in mae.get("rows", []) if r["subset"] == key), {})
        rows.append({"strategy": name, "workloads": a["workloads"], "avg_score": round(a["avg_score"], 4), "avg_score_std": "",
                     "top1": E.frac(a["top1_hits"], a["top1_n"]), "top5": E.frac(a["top5_hits"], a["top5_n"]),
                     "regret": round(a["regret"], 4), "mae_T": round(mm["mae_T"], 4) if mm else "", "mae_D": round(mm["mae_D"], 4) if mm else "",
                     "sample_contains_oracle": "", "selected_all_gpu": a.get("selected_all_gpu", ""),
                     "not_evaluated": a["not_evaluated_list"]})
    if bo5_row is not None:
        rows.append({"strategy": bo5_row["strategy"], "workloads": bo5_row["workloads"], "avg_score": round(bo5_row["avg_score"], 4),
                     "avg_score_std": round(bo5_row["avg_score_std"], 4), "top1": f"{bo5_row['top1_rate']:.3f} (mean rate)", "top5": "N/A",
                     "regret": round(bo5_row["regret"], 4), "mae_T": "", "mae_D": "",
                     "sample_contains_oracle": round(bo5_row["sample_contains_oracle"], 3), "selected_all_gpu": "", "not_evaluated": ""})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse artifacts under revision_exp/artifacts instead of retraining")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()
    for d in (OUT, LOGS, OUT / "figs", ART / "multi", ART / "single"):
        d.mkdir(parents=True, exist_ok=True)
    log(f"command: {' '.join(sys.argv)} | cwd: {os.getcwd()} | python {platform.python_version()}")
    if not (LOGS / "git_status_before.txt").exists():
        (LOGS / "git_status_before.txt").write_text(git_status())
    hb_path = LOGS / "protected_hash_before.json"
    if not hb_path.exists():
        hb_path.write_text(json.dumps(protected_hashes(), indent=0))
    hash_before = json.loads(hb_path.read_text())

    import sklearn
    import xgboost
    env = {"python": platform.python_version(), "xgboost": xgboost.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
           "pandas": pd.__version__, "cpu_count": os.cpu_count(), "n_jobs": TM.BASE_PARAMS["n_jobs"],
           "cpu_model": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), ""),
           "master_seed": D.MASTER_SEED, "alpha": ALPHA}
    (LOGS / "environment.json").write_text(json.dumps(env, indent=1))
    log(f"env: {env}")
    log(E.self_test())

    # ------------------------------------------------------------------ 3-1 data
    with stage("data"):
        sets, raw, dropped_all, summ, checks = {}, {}, [], [], []
        for name in D.SETS:
            raw[name], _ = D.load_set(name, dedup=False)
            sets[name], dropped = D.load_set(name, dedup=True)
            if len(dropped):
                dropped_all.append(dropped)
            summ.append(D.summarize(name, raw[name], sets[name], dropped))
            checks.append(D.check_against_existing_builder(name, raw[name]))
            log(f"  {summ[-1]}")
            log(f"  builder check: {checks[-1]}")
            assert checks[-1]["features_equal"] and checks[-1]["labels_equal"] and checks[-1]["group_ids_equal"]
        pd.DataFrame(summ).merge(pd.DataFrame(checks), on="set").to_csv(OUT / "data_summary.csv", index=False)
        dd = pd.concat(dropped_all, ignore_index=True) if dropped_all else pd.DataFrame()
        dd.to_csv(OUT / "dedup_dropped.csv", index=False)
        assert not len(dd) or (dd["kept_source_file"] != "NOT FOUND").all() and (dd["dropped_source_file"] != "NOT FOUND").all()
        train = sets["train"]
        X = D.build_features(train)
        cols = list(X.columns)
        (OUT / "feature_columns.txt").write_text("\n".join(cols) + "\n")
        assert set(sets["x3"]["row_uid"]).isdisjoint(train["row_uid"]) and set(train["rate"]) == {"x1", "x2", "x4"} and set(sets["x3"]["rate"]) == {"x3"}
        log(f"  train rows={len(train)} features={len(cols)}; x3 rows={len(sets['x3'])} (csv {len(raw['x3'])}, dropped {len(dd)})")
        mix = sets["x3"][sets["x3"]["source_file"] != ""].groupby("group")["source_file"].nunique()
        log(f"  x3 groups mixing two runs after dedup: {list(mix[mix > 1].index)}")

    # ------------------------------------------------------------------ 3-2 folds
    with stage("folds"):
        fold = F.make_folds(train)
        fam, per_fold = F.describe(train, fold)
        fam.to_csv(OUT / "folds.csv", index=False)
        old_p = D.REPO / "revision_audit" / "tables" / "cv_fold_composition.csv"
        if old_p.exists():
            D.note_input(old_p)
            old = pd.read_csv(old_p)
            old = old[old["dataset"] == "gpu/train_x3"]
            per_fold = pd.concat([per_fold, pd.DataFrame({"scheme": "KFold(3) without shuffle (existing)", "fold": old["fold"], "rows": old["rows"],
                                                           "families": old["families"], "schedule_files": old["schedule_files"],
                                                           "N_distribution(rows)": old["N_distribution"], "N_distribution(families)": ""})], ignore_index=True)
        per_fold.to_csv(OUT / "folds_summary.csv", index=False)
        log("\n" + per_fold.to_string(index=False))

    # ------------------------------------------------------------------ timing probe
    with stage("timing_probe"):
        probe = {"learning_rate": 0.01, "max_depth": 6, "n_estimators": 1000, "subsample": 0.8}
        t = []
        for k in range(3):
            t0 = time.perf_counter()
            TM.fit_one(X.values[fold != k], train["T"].values[fold != k], probe)
            t.append(time.perf_counter() - t0)
        n_fits = 3 * 16 * 3 + 10 * 16 * 3 + 13
        est = {"probe_params": probe, "probe_target": "T", "fit_seconds_per_fold": t, "fit_seconds_mean": float(np.mean(t)),
               "planned_fits": {"multi_cv": 144, "single_cv": 480, "final_refits": 13, "total": n_fits},
               "estimated_total_seconds_upper_bound": float(np.mean(t) * n_fits),
               "note": "upper bound: the probe is the most expensive grid point (1000 trees, depth 6)"}
        (LOGS / "timing_probe.json").write_text(json.dumps(est, indent=1))
        log(f"  probe fit {np.mean(t):.2f}s -> upper bound for {n_fits} fits: {est['estimated_total_seconds_upper_bound'] / 60:.1f} min")
        if est["estimated_total_seconds_upper_bound"] > 7200:
            log("  estimate exceeds 2 h: stopping before training (grid unchanged). Report first.")
            return
    if args.probe_only:
        return

    # ------------------------------------------------------------------ 3-3 / 3-4 training
    with stage("train"):
        if args.reuse and (ART / "multi" / "meta.json").exists() and (ART / "single" / "meta.json").exists():
            log("  reusing existing artifacts under revision_exp/artifacts")
            meta_m = json.loads((ART / "multi" / "meta.json").read_text())
            meta_s = json.loads((ART / "single" / "meta.json").read_text())
            assert meta_m["features"] == cols
        else:
            t0 = time.time()
            meta_m, rec_m = TM.train_multi(X, train, fold, ART / "multi", log)
            meta_m.update({"env": env, "train_rows": len(train), "train_set": D.SETS["train"][0], "fold_scheme": "GroupKFold(3) by family"})
            (ART / "multi" / "meta.json").write_text(json.dumps(meta_m, indent=1))
            rec_m.to_csv(OUT / "hparam_search_multi.csv", index=False)
            t_multi = time.time() - t0
            t0 = time.time()
            meta_s, rec_s = TS.train_single(X, train, fold, ART / "single", log)
            meta_s.update({"env": env, "train_rows": len(train), "fold_scheme": "GroupKFold(3) by family"})
            (ART / "single" / "meta.json").write_text(json.dumps(meta_s, indent=1))
            rec_s.to_csv(OUT / "hparam_search_single.csv", index=False)
            t_single = time.time() - t0
            (LOGS / "training_time.json").write_text(json.dumps({"multi_seconds": t_multi, "single_seconds": t_single,
                                                                 "total_seconds": t_multi + t_single, "n_jobs": -1, "cpu_count": os.cpu_count()}, indent=1))
            log(f"  training done: multi {t_multi / 60:.1f} min, single {t_single / 60:.1f} min")
    mT, mD, mDn = (P.load_model(ART / "multi" / f"{n}.json") for n in ("T", "D", "D_noweight"))
    single = {a: P.load_model(ART / "single" / f"alpha_{a}.json") for a in TS.ALPHAS}
    oT, oD, ocols, ometa = P.load_old_artifact()

    def preds(df):
        Xn = D.build_features(df, cols)
        Xo = D.build_features(df, ocols)
        t, d = P.predict_multi(mT, mD, Xn)
        _, dn = P.predict_multi(mT, mDn, Xn)
        to, do = P.predict_multi(oT, oD, Xo)
        return Xn, {"new": (t, d), "new_noweight": (t, dn), "old": (to, do)}

    # ------------------------------------------------------------------ 3-5 validation A / B / C
    with stage("validation"):
        x3, x3raw = sets["x3"], raw["x3"].sort_values("csv_row").reset_index(drop=True)
        _, pr_raw = preds(x3raw)
        _, pr = preds(x3)
        s_old_raw, _ = P.multi_score(*pr_raw["old"], ALPHA)
        leg = E.legacy_evaluate(x3raw, s_old_raw, ALPHA)

        def leg_row(tag, lg, note):
            out = []
            for mn in (3, 1):
                g = lg[lg["N"] >= mn]
                out.append({"step": tag, "min_N": mn, "workloads": len(g), "oracle_avg": round(g["oracle_S"].mean(), 4),
                            "avg_score": round(g["S_selected"].mean(), 4), "top1": E.frac(int(g["top1"].sum()), len(g)),
                            "top5": E.frac(int(g["top5"].sum()), len(g)), "regret": round(g["regret"].mean(), 4), "note": note})
            return out
        mae_old_raw = E.mae_table(x3raw, *pr_raw["old"], "old artifact, csv rows")
        A = leg_row("A legacy replay: old artifact, no dedup, old rules", leg, "target: Oracle 0.987, Ours 0.945, 13/18, 17/18 (REPORT2)")
        for r in A:
            r["mae_T_all_rows"], r["mae_D_all_rows"] = round(mae_old_raw[0]["mae_T"], 4), round(mae_old_raw[0]["mae_D"], 4)
        pd.DataFrame(A).to_csv(OUT / "validation_A.csv", index=False)
        log("  A: " + json.dumps(A[0]))
        a3 = A[0]
        assert a3["top1"].startswith("13/18") and a3["top5"].startswith("17/18") and abs(a3["avg_score"] - 0.945) < 0.002 and abs(a3["oracle_avg"] - 0.987) < 0.001, \
            "legacy replay does not reproduce the stored result"

        s_old, _ = P.multi_score(*pr["old"], ALPHA)
        leg_d = E.legacy_evaluate(x3.sort_values("csv_row"), pd.Series(s_old, index=x3.index).loc[x3.sort_values("csv_row").index].values, ALPHA)
        per_old = ours_per_workload(x3, s_old, ALPHA, "Ours (existing artifact + new evaluator)")

        def new_row(tag, per, note):
            out = []
            for mn in (3, 1):
                a = E.aggregate(per, mn)
                out.append({"step": tag, "min_N": mn, "workloads": a["workloads"], "oracle_avg": round(a["oracle_avg_same_workloads"], 4),
                            "avg_score": round(a["avg_score"], 4), "top1": E.frac(a["top1_hits"], a["top1_n"]), "top5": E.frac(a["top5_hits"], a["top5_n"]),
                            "regret": round(a["regret"], 4), "note": note})
            return out
        Bv = (leg_row("A (reference)", leg, "old artifact, no dedup, old rules")
              + leg_row("B1 dedup only", leg_d, "old artifact, 24 duplicate rows removed (later run kept), old rules")
              + new_row("B2 dedup + new rules", per_old, "old artifact, dedup, deterministic ties, Top-5 = 5 candidates"))
        pd.DataFrame(Bv).to_csv(OUT / "validation_B.csv", index=False)
        s_new, raw_new = P.multi_score(*pr["new"], ALPHA)
        per_new = ours_per_workload(x3, s_new, ALPHA, "Ours (retrained)")
        Cv = new_row("B2 (reference)", per_old, "old artifact + new evaluator") + new_row("C retrained + new evaluator", per_new, "new artifact (GroupKFold-selected)")
        pd.DataFrame(Cv).to_csv(OUT / "validation_C.csv", index=False)
        for r in Bv + Cv:
            if r["min_N"] == 3:
                log(f"  {r['step']}: avg={r['avg_score']} top1={r['top1']} top5={r['top5']} oracle={r['oracle_avg']} regret={r['regret']}")
        # unit check on real data: s_hat = S -> every workload is a Top-1 hit and Avg = Oracle
        S_true = (x3["T"] - ALPHA * x3["D"]).values
        per_or = ours_per_workload(x3, S_true, ALPHA, "Oracle")
        ao = E.aggregate(per_or, 3)
        assert ao["top1_hits"] == ao["top1_n"] == 18 and abs(ao["avg_score"] - ao["oracle_avg_same_workloads"]) < 1e-12
        log(f"  unit check (s_hat = S): Top-1 {ao['top1_hits']}/{ao['top1_n']}, Avg == Oracle == {ao['avg_score']:.4f}")

    # ------------------------------------------------------------------ 3-6 / 3-7 / 3-8 strategies on every evaluation set
    with stage("evaluate"):
        results, lat_src, inter_rows, all_tables = {}, [], [], {}
        for name in ("x3", "x1-5", "x2-5", "x3-5"):
            df = sets[name]
            rate = df["rate"].iloc[0]
            Xn, pr = preds(df)
            S_true = (df["T"] - ALPHA * df["D"]).values
            s_new, raw_new = P.multi_score(*pr["new"], ALPHA)
            s_nw, _ = P.multi_score(*pr["new_noweight"], ALPHA)
            s_old, raw_old = P.multi_score(*pr["old"], ALPHA)
            models = sorted({m.split("@")[0] for p in df["placement"] for m in p.split(";")})
            prof = B.latency_profile(rate, models)
            lat_src.append(prof)
            tg = B.latency_targets(prof)
            tg_stored = B.stored_rule_targets(rate)
            cand = cand_of(df)
            per = {
                "Latency-based": single_per_workload(df, B.choose_by_targets(cand, tg), ALPHA, "Latency-based"),
                "Ours (retrained)": ours_per_workload(df, s_new, ALPHA, "Ours (retrained)"),
                "Ours (existing artifact + new evaluator)": ours_per_workload(df, s_old, ALPHA, "Ours (existing artifact + new evaluator)"),
                "All-GPU": single_per_workload(df, B.choose_all_device(cand, "all_gpu"), ALPHA, "All-GPU"),
                "Oracle": ours_per_workload(df, S_true, ALPHA, "Oracle"),
                "All-CPU": single_per_workload(df, B.choose_all_device(df.assign(all_cpu=df["all_cpu"])[["group", "comb_num", "all_cpu"]], "all_cpu"), ALPHA, "All-CPU"),
                "Latency-based (stored rule replay)": single_per_workload(df, B.choose_by_targets(cand, tg_stored), ALPHA, "Latency-based (stored rule replay)"),
                "Ours (retrained, D without weight)": ours_per_workload(df, s_nw, ALPHA, "Ours (retrained, D without weight)"),
            }
            bo5 = B.best_of_k(cand, ALPHA)
            mae = {"Ours (retrained)": {"rows": E.mae_table(df, *pr["new"], "new")},
                   "Ours (existing artifact + new evaluator)": {"rows": E.mae_table(df, *pr["old"], "old")},
                   "Ours (retrained, D without weight)": {"rows": E.mae_table(df, *pr["new_noweight"], "new_noweight")}}
            results[name] = {"df": df, "per": per, "bo5": bo5, "mae": mae, "pr": pr, "targets": tg, "targets_stored": tg_stored,
                             "s_new": s_new, "raw_new": raw_new, "Xn": Xn}
            pd.DataFrame({"row_uid": df["row_uid"], "group": df["group"], "combination": df["combination"], "N": df["N"], "T": df["T"], "D": df["D"],
                          "T_hat": pr["new"][0], "D_hat": pr["new"][1], "D_hat_noweight": pr["new_noweight"][1],
                          "S": S_true, "S_hat_raw": raw_new, "S_hat": s_new,
                          "T_hat_old_artifact": pr["old"][0], "D_hat_old_artifact": pr["old"][1], "S_hat_old_artifact": s_old,
                          "all_gpu": df["all_gpu"]}).to_csv(OUT / f"predictions_{name}.csv", index=False)
            log(f"  [{name}] latency targets: {tg} | stored-rule targets: {tg_stored}")
            log(f"  [{name}] S_hat clipped rows: {int((raw_new != s_new).sum())}/{len(df)}; workloads with tie at first place (N>=3): "
                f"{int((per['Ours (retrained)'].query('N >= 3')['n_tied_with_first'] > 1).sum())}")
            main_names = ["Latency-based", "Ours (retrained)", "Ours (existing artifact + new evaluator)", "All-GPU", "Oracle"]
            extra = ["All-CPU", "Latency-based (stored rule replay)", "Ours (retrained, D without weight)"]
            for mn in (3, 1):
                tb = table_rows({k: per[k] for k in main_names + extra}, bo5_summary(df, ALPHA, mn, bo5), mae, mn)
                tb.insert(0, "set", name)
                tb.insert(1, "min_N", mn)
                all_tables[(name, mn)] = tb
        pd.concat(lat_src, ignore_index=True).to_csv(OUT / "latency_profile_sources.csv", index=False)

        r3 = results["x3"]
        pw = pd.concat(list(r3["per"].values()), ignore_index=True)
        g, S_sel, top1, contains = r3["bo5"]
        pw_b = pd.DataFrame({"strategy": "Best-of-5 (1,000 trials)", "group": g, "S_selected": S_sel.mean(axis=0), "S_selected_std_over_trials": S_sel.std(axis=0),
                             "top1_rate": top1.mean(axis=0), "sample_contains_oracle": contains.mean(axis=0)})
        pw_b = pw_b.merge(r3["df"].groupby("group").agg(N=("N", "first"), n_candidates=("row_uid", "size")).reset_index(), on="group")
        pd.concat([pw, pw_b], ignore_index=True).to_csv(OUT / "per_workload_x3.csv", index=False)

        order = ["Latency-based", "Ours (retrained)", "Ours (existing artifact + new evaluator)", "Best-of-5 (1,000 trials)", "All-GPU", "Oracle",
                 "All-CPU", "Latency-based (stored rule replay)", "Ours (retrained, D without weight)"]
        t3 = all_tables[("x3", 3)].set_index("strategy").loc[order].reset_index()
        t3["manuscript_Table_3a"] = [MANUSCRIPT.get(s.split(" (")[0], "") if "existing" not in s and "replay" not in s and "without" not in s else "" for s in t3["strategy"]]
        t3.drop(columns=["set", "min_N"]).to_csv(OUT / "table3a.csv", index=False)
        (OUT / "table3a.md").write_text("# Table 3(a) - CPU-GPU, 3x hold-out, N>=3, alpha=0.3\n\n" + md_table(t3.drop(columns=["set", "min_N"])))
        all_tables[("x3", 1)].set_index("strategy").loc[order].reset_index().drop(columns=["set", "min_N"]).to_csv(OUT / "table_all_n.csv", index=False)
        ti = pd.concat([all_tables[(n, 3)].set_index("strategy").loc[order].reset_index() for n in ("x1-5", "x2-5", "x3-5")], ignore_index=True)
        ti.to_csv(OUT / "table_intermediate.csv", index=False)
        (OUT / "table_intermediate.md").write_text("# Intermediate rates, N>=3, alpha=0.3\n\n" + md_table(ti))
        (OUT / "table_all_n.md").write_text("# 3x hold-out, all N (29 workloads), alpha=0.3\n\n" + md_table(all_tables[("x3", 1)].set_index("strategy").loc[order].reset_index()))
        log("\n" + t3[["strategy", "workloads", "avg_score", "avg_score_std", "top1", "top5", "regret", "mae_T", "mae_D", "sample_contains_oracle"]].to_string(index=False))

        # duplicates kept (reference, Ours and Oracle only): candidates made unique by (combination, run order)
        xr = raw["x3"].sort_values(["group", "comb_num", "timestamp"]).reset_index(drop=True)
        xr["comb_num"] = xr["comb_num"] * 10 + xr.groupby(["group", "comb_num"]).cumcount()
        _, prr = preds(xr)
        sr, _ = P.multi_score(*prr["new"], ALPHA)
        ref = [dict(E.aggregate(ours_per_workload(xr, sr, ALPHA, "Ours (retrained), duplicates kept"), 3)),
               dict(E.aggregate(ours_per_workload(xr, (xr["T"] - ALPHA * xr["D"]).values, ALPHA, "Oracle, duplicates kept"), 3))]
        pd.DataFrame(ref).to_csv(OUT / "reference_duplicates_kept.csv", index=False)

        # D weight ablation
        ab = []
        for nm, key in (("T + D (weighted, primary)", "Ours (retrained)"), ("T + D_noweight", "Ours (retrained, D without weight)")):
            for mn in (3, 1):
                a = E.aggregate(r3["per"][key], mn)
                mm = next(r for r in r3["mae"][key]["rows"] if r["subset"] == ("N>=3" if mn == 3 else "N>=1"))
                ab.append({"variant": nm, "min_N": mn, "workloads": a["workloads"], "mae_D": round(mm["mae_D"], 4),
                           "mae_D_where_D>0.01": round(mm["mae_D_where_D>0.01"], 4), "candidates_D>0.01": mm["candidates_D>0.01"],
                           "mae_T": round(mm["mae_T"], 4), "avg_score": round(a["avg_score"], 4), "top1": E.frac(a["top1_hits"], a["top1_n"]),
                           "top5": E.frac(a["top5_hits"], a["top5_n"]), "regret": round(a["regret"], 4),
                           "cv_mae_D": round(meta_m["models"]["D" if "weighted" in nm else "D_noweight"]["cv_mae_mean"], 4),
                           "best_params_D": json.dumps(meta_m["models"]["D" if "weighted" in nm else "D_noweight"]["best_params"])})
        pd.DataFrame(ab).to_csv(OUT / "ablation_d_weight.csv", index=False)
        pd.DataFrame([r for k in r3["mae"] for r in [dict(x, strategy=k) for x in r3["mae"][k]["rows"]]]).to_csv(OUT / "mae_x3.csv", index=False)

    # ------------------------------------------------------------------ 3-9 Fig.3 data
    with stage("fig3"):
        df = r3["df"]
        old3 = {}
        p_old = D.REPO / "revision_audit" / "tables3" / "fig3_data.csv"
        if p_old.exists():
            D.note_input(p_old)
            for _, r in pd.read_csv(p_old).iterrows():
                old3[(int(r["N>="]), r["strategy"])] = r["fig3_value(stored CSV)"]
        rows = []
        for n in (3, 4, 5, 6, 7, 8):
            b = bo5_summary(df, ALPHA, n, r3["bo5"])
            row = {"N>=": n, "workloads": b["workloads"]}
            for label, key in (("Oracle", "Oracle"), ("Ours", "Ours (retrained)"), ("Latency-based", "Latency-based"), ("All-GPU", "All-GPU"),
                               ("Ours_existing_artifact", "Ours (existing artifact + new evaluator)")):
                a = E.aggregate(r3["per"][key], n)
                row[label] = round(a["avg_score"], 4)
                if label == "All-GPU":
                    row["All-GPU_workloads"] = a["workloads"]
            row["Best-of-5"], row["Best-of-5_std"] = round(b["avg_score"], 4), round(b["avg_score_std"], 4)
            for k in ("Oracle", "Ours", "Best-of-5", "Latency-based"):
                row[f"old_figure_{k}"] = old3.get((n, k), "")
            rows.append(row)
        f3 = pd.DataFrame(rows)
        f3.to_csv(OUT / "fig3_data.csv", index=False)
        log("\n" + f3.to_string(index=False))

    # ------------------------------------------------------------------ 3-10 Fig.4 data
    with stage("fig4"):
        Xn = r3["Xn"]
        t_hat, d_hat = r3["pr"]["new"]
        rows = []
        for a in TS.ALPHAS:
            S_true = (df["T"] - a * df["D"]).values
            sm, _ = P.multi_score(t_hat, d_hat, a)
            ss, _ = P.single_score(single[a], Xn)
            pm = E.aggregate(ours_per_workload(df, sm, a, "multi"), 3)
            ps = E.aggregate(ours_per_workload(df, ss, a, "single"), 3)
            po = E.aggregate(ours_per_workload(df, S_true, a, "oracle"), 3)
            row = {"alpha": a, "workloads": pm["workloads"], "oracle_avg": round(po["avg_score"], 4)}
            for tag, p in (("multi", pm), ("single", ps)):
                row.update({f"avg_score_{tag}": round(p["avg_score"], 4), f"top1_{tag}": round(p["top1_hits"] / p["top1_n"], 4),
                            f"top1_hits_{tag}": p["top1_hits"], f"top5_{tag}": round(p["top5_hits"] / p["top5_n"], 4), f"top5_hits_{tag}": p["top5_hits"],
                            f"top5_legacy_rule_hits_{tag}": p.get("top5_legacy_hits"), f"regret_{tag}": round(p["regret"], 4),
                            f"selected_all_gpu_{tag}": p.get("selected_all_gpu")})
            row["single_best_params"] = json.dumps(meta_s["models"][str(a)]["best_params"])
            rows.append(row)
        f4 = pd.DataFrame(rows)
        p_old = D.REPO / "experimental_results" / "alpha_comparison_plots.csv"
        D.note_input(p_old)
        o4 = pd.read_csv(p_old)
        f4 = f4.merge(o4.rename(columns={"avg_score_ge3_fixed": "old_figure_avg_multi", "top1_acc_ge3_fixed": "old_figure_top1_multi",
                                         "top5_acc_ge3_fixed": "old_figure_top5_multi", "avg_score_ge3_trained": "old_figure_avg_single",
                                         "top1_acc_ge3_trained": "old_figure_top1_single", "top5_acc_ge3_trained": "old_figure_top5_single"})
                      [["alpha", "old_figure_avg_multi", "old_figure_top1_multi", "old_figure_top5_multi", "old_figure_avg_single",
                        "old_figure_top1_single", "old_figure_top5_single"]], on="alpha", how="left")
        f4.to_csv(OUT / "fig4_data.csv", index=False)
        log("\n" + f4[["alpha", "oracle_avg", "avg_score_multi", "top1_hits_multi", "top5_hits_multi", "avg_score_single", "top1_hits_single", "top5_hits_single"]].to_string(index=False))

    with stage("plots"):
        import plots as PL
        PL.fig3(f3, OUT / "figs" / "fig3_model_count_vs_score.pdf", OUT / "figs" / "fig3_model_count_vs_score.png")
        PL.fig4(f4, OUT / "figs" / "fig4_alpha_comparison.pdf", OUT / "figs" / "fig4_alpha_comparison.png")

    # ------------------------------------------------------------------ bookkeeping
    with stage("bookkeeping"):
        n = D.write_manifest()
        (LOGS / "git_status_after.txt").write_text(git_status())
        hash_after = protected_hashes()
        changed = sorted(k for k in set(hash_before) | set(hash_after) if hash_before.get(k) != hash_after.get(k))
        (LOGS / "protected_hash_after.json").write_text(json.dumps(hash_after, indent=0))
        (LOGS / "protected_hash_diff.txt").write_text("\n".join(changed) + ("\n" if changed else "NO DIFFERENCE\n"))
        pyc = [str(p) for p in D.EXP.rglob("__pycache__")]
        log(f"  manifest: {n} input files | protected files: {len(hash_after)} hashed, changed: {len(changed)} | __pycache__ under revision_exp: {pyc}")
    pd.DataFrame(STAGES).to_csv(LOGS / "stage_times.csv", index=False)
    log(f"total wall time {time.time() - _T0:.1f}s")


if __name__ == "__main__":
    main()

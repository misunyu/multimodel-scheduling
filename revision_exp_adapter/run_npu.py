#!/usr/bin/env python3
"""Phase N2-B: retrain and evaluate the UNMODIFIED revision_exp pipeline on the CPU-NPU dataset (Phase N2 v1.3 section B).

  MPLCONFIGDIR=/tmp .venv/bin/python3 -B revision_exp_adapter/run_npu.py

Data: ../mms_npu_results (detached worktree of origin/revision-profiling-npu at 4070ce2e), revision_profiling_npu/dataset/.
Outputs: revision_exp/out_npu/ only.

Same modules as run_reprofiled.py (data, folds, train_multi, evaluate, baselines, predict, run_all helpers) imported as they are;
only D.DS / D.SETS / D.OUT and D.note_input (paths outside this repository) are changed at run time.
Per-instance baseline: revision_exp/phase3/baselines/ imported as is; the two functions that hard-code the CPU/GPU device set
(rows_from_schedules, iso_capacity) are re-stated here with the NPU device set and profile path (baselines_npu.md).

3x isolation: training / grid search / folds use 1,2,4x only. 3x features, predictions and selections are built from the 3x
SCHEDULE file only and written to disk before any 3x label file is opened (GUARD); then the labels are read, the features rebuilt
by the unchanged loader are asserted equal to the blind ones, and the evaluation runs.
"""
import sys

sys.dont_write_bytecode = True

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
sys.path.append(str(REPO / "revision_exp_adapter"))
os.environ.setdefault("MPLBACKEND", "Agg")

import baselines as B  # noqa: E402
import data as D  # noqa: E402
import evaluate as E  # noqa: E402
import folds as F  # noqa: E402
import predict as P  # noqa: E402
import run_all as R  # noqa: E402  (helpers only)
import run_reprofiled as RR  # noqa: E402  (helpers only)
import train_multi as TM  # noqa: E402
import train_single as TS  # noqa: E402  (ALPHAS list only)

NPU_WT = REPO.parent / "mms_npu_results"
NPU = NPU_WT / "revision_profiling_npu"
DS = NPU / "dataset"
OUT = REPO / "revision_exp" / "out_npu"
LOGS, ART = OUT / "logs", OUT / "artifacts"
EXPECTED_COMMIT = "4070ce2e"
EXPECTED_SHA16 = {"labels_train.csv": "7f5acf8534fd58e4", "labels_test.csv": "fed522320da95488"}
ALPHA = D.ALPHA
assert ALPHA == 0.3
WINDOW_SEC = 30.0
DEVICES = ("CPU", "NPU0", "NPU1")
INSTANCE_ORDER = ["resnet50_big", "resnet50_small", "yolov3_big", "yolov3_small"]
PRIMARY = {"rbrsyb_x3.yaml", "rbybys_x3.yaml", "rsybys_x3.yaml", "rbrsybys_x3.yaml"}
GUARD = {"labels_open": False}
_T0 = time.time()

# ------------------------------------------------------------------------------------------------ data module set-up
_orig_note_input = D.note_input


def note_input(path):
    """Same as data.note_input, but also accepts files outside this repository (the NPU worktree): key = absolute path."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO / path
    try:
        path.relative_to(REPO)
        return _orig_note_input(path)
    except ValueError:
        pass
    if path.name in ("labels_test.csv", "test_x3.csv", "labels_instances_all.csv"):
        assert GUARD["labels_open"], f"3x label file {path.name} opened before the blind predictions were written"
    key = str(path)
    if key not in D.MANIFEST:
        D.MANIFEST[key] = (os.path.getsize(path), sha(path))
    return path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


D.note_input = note_input
D.DS = DS
D.SETS = {"train": ("train_x3.csv", "train_schedules_x3.csv", []), "x3": ("test_x3.csv", "test_schedules_x3.csv", [])}
D.OUT = OUT

# ------------------------------------------------------------------------------------------------ per-instance baseline (device set / path only)
import phase3.baselines.instance_rows as IR  # noqa: E402

NPU_FEATURES = [f"id_model_hash_{k}" for k in range(IR.B_M)] + ["dev_cpu", "dev_npu0", "dev_npu1", "lam", "iso_cap", "co_same_n", "co_same_lam",
                                                                  "co_other_n", "co_other_lam", "N"]
IR.FEATURES = NPU_FEATURES


def iso_capacity_npu(profile_path):
    """(instance, DEVICE) -> isolated throughput_fps of that variant on that device (CPU: the common CPU executable)."""
    ent = json.loads(Path(note_input(profile_path)).read_text())["entries"]
    out = {}
    for inst in INSTANCE_ORDER:
        model, variant = inst.rsplit("_", 1)
        out[(inst, "CPU")] = float(ent[f"{model}_cpu"]["throughput_fps"])
        for dev in ("npu0", "npu1"):
            e = ent[f"{model}_{variant}_{dev}"]
            assert e["variant"] == variant and e["device"] == dev and e["valid"]
            out[(inst, dev.upper())] = float(e["throughput_fps"])
    return out


def rows_from_schedules_npu(sched_csv, iso):
    """phase3 instance_rows.rows_from_schedules with the device set {CPU, NPU0, NPU1}; everything else unchanged."""
    s = D.suite()
    idx = s._index_schedules_from_csv(sched_csv)
    out = []
    for name, doc in sorted(idx.items()):
        for comb, _blob in s._iter_combos_from_schedule(doc):
            inf = s._build_infps_lookup(doc, comb)
            assert inf, f"empty schedule entry {name} {comb}"
            n = len(inf)
            for (model, dev), lam in sorted(inf.items()):
                assert lam > 0 and dev in DEVICES
                same = [v for (m2, d2), v in inf.items() if m2 != model and d2 == dev]
                other = [v for (m2, d2), v in inf.items() if m2 != model and d2 != dev]
                r = {"group": name, "family": D.family_key(name), "rate": D.rate_key(name), "combination": str(comb), "comb_num": D.comb_num(comb),
                     "model": model, "device": dev, **{c: 0.0 for c in NPU_FEATURES[:IR.B_M]}, **IR.model_onehot(model),
                     "dev_cpu": float(dev == "CPU"), "dev_npu0": float(dev == "NPU0"), "dev_npu1": float(dev == "NPU1"), "lam": lam,
                     "iso_cap": iso[(model, dev)], "co_same_n": float(len(same)), "co_same_lam": float(sum(same)), "co_other_n": float(len(other)),
                     "co_other_lam": float(sum(other)), "N": float(n)}
                out.append(r)
    df = pd.DataFrame(out)
    assert df[NPU_FEATURES].notna().all().all()
    assert not df.duplicated(["group", "comb_num", "model"]).any()
    return df


IR.rows_from_schedules = rows_from_schedules_npu   # training_rows() calls it through the module global
from phase3.baselines import compose_select as CS  # noqa: E402
from phase3.baselines import fit_instance as FI  # noqa: E402

assert FI.FEATURES is NPU_FEATURES and CS.FEATURES is NPU_FEATURES, "per-instance modules did not pick up the NPU device set"


# ------------------------------------------------------------------------------------------------ helpers
def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    with open(LOGS / "run_npu.log", "a") as f:
        f.write(line + "\n")


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def isolated_latency():
    """(instance, DEVICE) -> service_ms_p50 of the isolated closed-loop profile (CPU = common CPU executable)."""
    ent = json.loads(Path(note_input(DS / "offline_profile.json")).read_text())["entries"]
    out = {}
    for inst in INSTANCE_ORDER:
        model, variant = inst.rsplit("_", 1)
        out[(inst, "CPU")] = float(ent[f"{model}_cpu"]["service_ms_p50"])
        for dev in ("npu0", "npu1"):
            out[(inst, dev.upper())] = float(ent[f"{model}_{variant}_{dev}"]["service_ms_p50"])
    return out


def latency_assignment(instances, lat):
    """baselines_npu.md: instances in family-key order; shortest isolated latency device, occupied NPU -> next; ties cpu<npu0<npu1."""
    rank = {d: i for i, d in enumerate(DEVICES)}
    used, out = set(), {}
    for inst in sorted(instances, key=INSTANCE_ORDER.index):
        prefs = sorted(DEVICES, key=lambda d: (lat[(inst, d)], rank[d]))
        for d in prefs:
            if d == "CPU" or d not in used:
                out[inst] = d
                if d != "CPU":
                    used.add(d)
                break
    return out


def schedule_candidates(sched_csv):
    """{group: {comb_num: {instance: DEVICE}}} and the blind feature dicts, from the schedule file only."""
    s = D.suite()
    idx = s._index_schedules_from_csv(sched_csv)
    plac, feats = {}, []
    for name, doc in sorted(idx.items()):
        plac[name] = {}
        for comb, blob in s._iter_combos_from_schedule(doc):
            views = s._rows_from_combo_struct(blob)
            pm = {v["model"]: s._norm_exec(v["execution"]) for v in views}
            assert len(pm) == len(views)
            plac[name][D.comb_num(comb)] = pm
            window = {"combination": str(comb), "schedule_file": name,
                      "models": {f"view{i + 1}": {"model": v["model"], "execution": v["execution"]} for i, v in enumerate(views)}}
            x, _y, _m = s.featurize_window(window, s._build_infps_lookup(doc, comb))
            feats.append({"group": name, "comb_num": D.comb_num(comb), "N": len(pm), "_x": x})
    return plac, pd.DataFrame(feats)


def latency_choice(plac, lat):
    out, detail = {}, []
    for g, combs in plac.items():
        insts = list(next(iter(combs.values())).keys())
        want = latency_assignment(insts, lat)
        hit = [c for c, pm in combs.items() if pm == want]
        assert len(hit) == 1, f"latency-based placement {want} not among the candidates of {g}"
        out[g] = hit[0]
        detail.append({"group": g, "latency_combination": f"combination_{hit[0]}", "latency_placement": ",".join(f"{i}:{want[i].lower()}" for i in sorted(want, key=INSTANCE_ORDER.index))})
    return out, pd.DataFrame(detail)


def placement_str(pm):
    return ",".join(f"{i}:{pm[i].lower()}" for i in sorted(pm, key=INSTANCE_ORDER.index))


# ------------------------------------------------------------------------------------------------ main
def main():
    for d in (OUT, LOGS, ART / "multi", ART / "per_instance", OUT / "blind", OUT / "manuscript"):
        d.mkdir(parents=True, exist_ok=True)
    assert not (LOGS / "config_frozen.json").exists(), "out_npu already holds a frozen run; refusing to overwrite"
    log(f"command: {' '.join(sys.argv)} | data={DS} | out={OUT}")
    (LOGS / "git_status_before.txt").write_text(subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout)

    # ---- B1 input pinning
    head = subprocess.run(["git", "-C", str(NPU_WT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head.startswith(EXPECTED_COMMIT), f"NPU worktree at {head}, expected {EXPECTED_COMMIT}"
    wt_status = subprocess.run(["git", "-C", str(NPU_WT), "status", "--short"], capture_output=True, text=True).stdout
    sha16 = {}
    for fn, exp in EXPECTED_SHA16.items():
        sha16[fn] = sha(DS / fn)[:16]   # hash only; contents not parsed here
        assert sha16[fn] == exp, f"{fn}: sha256 {sha16[fn]} != {exp}"
    log(f"input pinned: commit {head}, sha16 {sha16}, worktree clean={wt_status == ''}")

    import sklearn
    import xgboost
    env = {"python": platform.python_version(), "xgboost": xgboost.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
           "pandas": pd.__version__, "cpu_count": os.cpu_count(), "master_seed": D.MASTER_SEED, "alpha": ALPHA, "drop_definition": "macro"}
    (LOGS / "environment.json").write_text(json.dumps(env, indent=1))
    log(E.self_test())

    # ---- training data (1, 2, 4x) only
    train = RR.load("train", "macro")
    assert len(train) == 279 and set(train["rate"]) == {"x1", "x2", "x4"} and train["family"].nunique() == 11
    X = D.build_features(train)
    cols = list(X.columns)
    (OUT / "feature_columns.txt").write_text("\n".join(cols) + "\n")
    bad_tokens = ("f_mean", "f_max", "r_max", "r_agg", "d_macro", "_rep", "_se", "completed", "dropped", "generated", "throughput", "drop", "score")
    assert all(D.ALLOWED_FEATURE_RE.match(c) for c in cols), "feature column outside the schedule-feature pattern"
    assert not [c for c in cols if any(t in c.lower() for t in bad_tokens)] and not set(cols) & {"T", "D", "S", "F", "R"}
    npu_cols = [c for c in cols if "npu0" in c or "npu1" in c]
    e0, e1 = X["views.sum.view.exec_npu0"].values, X["views.sum.view.exec_npu1"].values
    npu_check = {"feature_columns": len(cols), "npu_columns": len(npu_cols), "npu0_columns": sum("npu0" in c for c in npu_cols),
                 "npu1_columns": sum("npu1" in c for c in npu_cols), "gpu_columns": sum("gpu" in c for c in cols),
                 "rows_exec_npu0_nonzero": int((e0 != 0).sum()), "rows_exec_npu1_nonzero": int((e1 != 0).sum()),
                 "rows_exec_npu0_ne_npu1": int((e0 != e1).sum()), "rows": len(X),
                 "npu0_npu1_columns_identical": bool(np.array_equal(e0, e1))}
    assert npu_check["rows_exec_npu0_nonzero"] > 0 and npu_check["rows_exec_npu1_nonzero"] > 0 and not npu_check["npu0_npu1_columns_identical"]
    hashes = {i: int(hashlib.md5(i.encode()).hexdigest(), 16) % D.suite().MODEL_HASH_BUCKETS for i in INSTANCE_ORDER}
    npu_check["instance_hash_buckets"] = hashes
    npu_check["instance_hash_buckets_distinct"] = len(set(hashes.values())) == len(hashes)
    log(f"features: {npu_check}")

    fold = F.make_folds(train)
    for k in range(F.N_SPLITS):
        assert set(train["family"][fold == k]).isdisjoint(train["family"][fold != k])
    fam, per_fold = F.describe(train, fold)
    fam.to_csv(OUT / "folds.csv", index=False)
    per_fold.to_csv(OUT / "folds_summary.csv", index=False)
    family_fold = {f_: int(k) for f_, k in zip(train["family"], fold)}

    # label reconstruction (training rates): instance q_i mean -> D_macro
    lab_tr = pd.read_csv(DS / "labels_train.csv")
    pre = {"train_rows": len(train), "train_label_rows": len(lab_tr), "families": train["family"].nunique(), "folds": per_fold.to_dict(orient="records"),
           "D_equals_D_macro_mean_max_abs": float((train["D"] - train["D_macro_mean"]).abs().max()),
           "T_equals_throughput_norm_max_abs": float(np.abs(train["T"].values - lab_tr.set_index(["schedule_file", "combination"]).loc[
               list(zip(train["schedule_file"], train["combination"])), "throughput_norm"].values).max())}
    assert pre["D_equals_D_macro_mean_max_abs"] <= 1e-12 and pre["T_equals_throughput_norm_max_abs"] <= 1e-12

    # ---- B2 training
    t0 = time.time()
    meta_m, rec_m = TM.train_multi(X, train, fold, ART / "multi", log)
    meta_m.update({"env": env, "train_rows": len(train), "fold_scheme": "GroupKFold(3) by family", "drop_definition": "macro"})
    (ART / "multi" / "meta.json").write_text(json.dumps(meta_m, indent=1))
    rec_m.to_csv(OUT / "hparam_search.csv", index=False)
    t_train = time.time() - t0
    mT, mD = P.load_model(ART / "multi" / "T.json"), P.load_model(ART / "multi" / "D.json")

    iso = iso_capacity_npu(DS / "offline_profile.json")
    rows_tr = IR.training_rows(DS / "train_schedules_x3.csv", DS / "train_x3.csv", DS / "labels_train.csv", iso, WINDOW_SEC)
    fold_i = FI.instance_folds(rows_tr, family_fold)
    t0 = time.time()
    m_inst, rec_i, info_i = FI.fit_instance_models(rows_tr, fold_i, ART / "per_instance", log)
    t_inst = time.time() - t0
    rec_i.to_csv(OUT / "hparam_search_instance.csv", index=False)
    tr_ratio = {"rows": len(rows_tr), "t_ratio_min": float(rows_tr["t_ratio"].min()), "t_ratio_max": float(rows_tr["t_ratio"].max()),
                "q_i_max": float(rows_tr["q_i"].max()), "q_i_rows_above_0.01": int((rows_tr["q_i"] > 0.01).sum())}

    # training-data (in-sample) performance, kept apart from the 3x table
    t_in, d_in = P.predict_multi(mT, mD, X)
    s_in, _ = P.multi_score(t_in, d_in, ALPHA, clip=False)
    tp = []
    for rate, sub in train.assign(_s=s_in).groupby("rate"):
        per = R.ours_per_workload(sub, sub["_s"].values, ALPHA, "Ours")
        ag = E.aggregate(per, 1)
        tp.append({"data": "TRAINING DATA (in-sample)", "rate": rate, "workloads": ag["workloads"], "avg_score": round(ag["avg_score"], 4),
                   "oracle_avg": round(ag["oracle_avg_same_workloads"], 4), "top1": E.frac(ag["top1_hits"], ag["top1_n"]), "regret": round(ag["regret"], 4)})
    pd.DataFrame(tp).to_csv(OUT / "train_performance.csv", index=False)

    frozen = {"frozen_at": datetime.now().isoformat(sep=" ", timespec="seconds"), "input_commit": head, "input_worktree_clean": wt_status == "",
              "labels_sha256_16": sha16, "alpha": ALPHA, "drop_definition": "D_macro", "s_hat_clip": False, "n_features": len(cols), "grid": TM.GRID,
              "base_params": TM.BASE_PARAMS, "d_weight": {"threshold": TM.D_WEIGHT_THRESHOLD, "value": TM.D_WEIGHT_VALUE},
              "q_weight": {"threshold": FI.Q_WEIGHT_THRESHOLD, "value": FI.Q_WEIGHT_VALUE},
              "fold_scheme": "GroupKFold(3) by family, shuffle=False", "family_fold": dict(sorted(family_fold.items())),
              "train_rates": sorted(train["rate"].unique()), "npu_feature_check": npu_check, "prechecks": pre,
              "best_params": {k: v["best_params"] for k, v in meta_m["models"].items()},
              "cv_mae": {k: v["cv_mae_mean"] for k, v in meta_m["models"].items()}, "per_instance": info_i, "per_instance_target_range": tr_ratio,
              "artifact_sha256": {str(p.relative_to(OUT)): sha(p) for p in sorted(ART.rglob("*.json")) if p.name != "meta.json"},
              "train_csv_sha256": sha(DS / "train_x3.csv"), "training_seconds": {"multi": round(t_train, 1), "per_instance": round(t_inst, 1)},
              "env": env,
              "statement": "written before any 3x label file is opened by this process; nothing below changes models, features, grids, rules or alpha"}
    (LOGS / "config_frozen.json").write_text(json.dumps(frozen, indent=1, default=str))
    log("configuration frozen")

    # ---- blind 3x stage: schedule file only
    plac, bf = schedule_candidates(DS / "test_schedules_x3.csv")
    assert len(bf) == 93 and bf["group"].nunique() == 11
    Xb = pd.DataFrame(list(bf["_x"])).fillna(0.0).reindex(columns=cols, fill_value=0.0)
    unseen = sorted(set().union(*[set(x) for x in bf["_x"]]) - set(cols))
    tb, db = P.predict_multi(mT, mD, Xb)
    sb, _ = P.multi_score(tb, db, ALPHA, clip=False)
    ids = bf[["group", "comb_num"]].reset_index(drop=True)
    order_ours, ties_ours = E.select(ids, sb)
    lat = isolated_latency()
    lat_choice, lat_detail = latency_choice(plac, lat)
    rows_te = IR.rows_from_schedules(DS / "test_schedules_x3.csv", iso)
    assert set(rows_te["rate"]) == {"x3"}
    Xi = rows_te[NPU_FEATURES].values
    pt, pq = m_inst["t_ratio"].predict(Xi), m_inst["q_i"].predict(Xi)
    comp = CS.compose(rows_te[sorted(CS.ALLOWED_INPUT & set(rows_te.columns))], pt, pq)
    s_inst, order_inst, ties_inst = CS.select_candidates(comp, ALPHA)
    comp["S_hat"] = s_inst
    blind = bf[["group", "comb_num", "N"]].assign(placement=[placement_str(plac[g][c]) for g, c in zip(bf["group"], bf["comb_num"])],
                                                   T_hat=tb, D_hat=db, S_hat=sb)
    blind = blind.merge(comp[["group", "comb_num", "T_hat_select", "D_hat", "S_hat"]].rename(
        columns={"T_hat_select": "inst_T_hat", "D_hat": "inst_D_hat", "S_hat": "inst_S_hat"}), on=["group", "comb_num"], how="left", validate="one_to_one")
    blind.to_csv(OUT / "blind" / "predictions_x3_blind.csv", index=False)
    sel = pd.DataFrame([{"group": g, "N": len(next(iter(plac[g].values()))), "ours": f"combination_{order_ours[g][0]}",
                         "ours_top5": ";".join(f"combination_{c}" for c in order_ours[g][:5]), "ours_ties_at_first": ties_ours[g],
                         "per_instance": f"combination_{order_inst[g][0]}", "latency_based": f"combination_{lat_choice[g]}",
                         "all_cpu": "combination_1"} for g in sorted(plac)]).merge(lat_detail, on="group")
    assert all(all(d == "CPU" for d in plac[g][1].values()) for g in plac), "combination_1 is not all-CPU"
    sel.to_csv(OUT / "blind" / "selections_x3_blind.csv", index=False)
    blind_sha = {p.name: sha(p) for p in sorted((OUT / "blind").glob("*.csv"))}
    (LOGS / "blind_stage.json").write_text(json.dumps({"written_at": datetime.now().isoformat(sep=" ", timespec="seconds"), "sha256": blind_sha,
                                                       "unseen_feature_names_in_3x": unseen, "latency_ms": {f"{k[0]}@{k[1]}": v for k, v in lat.items()},
                                                       "statement": "predictions and selections written before any 3x label file is opened"}, indent=1))
    log(f"blind 3x predictions and selections written: {blind_sha}")

    GUARD["labels_open"] = True
    final_eval(train, cols, (mT, mD), bf, Xb, blind, ids, order_ours, ties_ours, lat_choice, lat_detail, comp, order_inst, ties_inst, lat, frozen, npu_check)
    D.write_manifest()
    (LOGS / "git_status_after.txt").write_text(subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout)
    log(f"total wall time {time.time() - _T0:.1f}s")


def final_eval(train, cols, models, bf, Xb, blind, ids, order_ours, ties_ours, lat_choice, lat_detail, comp, order_inst, ties_inst, lat, frozen, npu_check):
    mT, mD = models
    test = RR.load("x3", "macro")
    assert len(test) == 93 and set(test["rate"]) == {"x3"}
    Xt = D.build_features(test, cols)
    key_t = list(zip(test["group"], test["comb_num"]))
    key_b = {k: i for i, k in enumerate(zip(bf["group"], bf["comb_num"]))}
    pos = np.array([key_b[k] for k in key_t])
    assert np.array_equal(Xt.values, Xb.values[pos]), "features rebuilt by the loader differ from the blind schedule-only features"
    t_hat, d_hat = P.predict_multi(mT, mD, Xt)
    assert np.array_equal(t_hat, blind["T_hat"].values[pos]) and np.array_equal(d_hat, blind["D_hat"].values[pos])
    s_hat = blind["S_hat"].values[pos]
    S_true = (test["T"] - ALPHA * test["D"]).values
    assert np.allclose(S_true, test["T"] - ALPHA * test["D_macro_mean"])
    lab_te = pd.read_csv(DS / "labels_test.csv")
    inst = pd.read_csv(DS / "labels_instances_all.csv")
    rec = inst.groupby(["workload", "combination"])["q_i_i"].mean()
    lab_all = pd.concat([pd.read_csv(DS / "labels_train.csv"), lab_te])
    rec_d = float(np.abs(rec.loc[list(zip(lab_all["workload"], lab_all["combination"]))].values - lab_all["D_macro_mean"].values).max())
    assert rec_d <= 1e-12, rec_d
    log(f"3x labels read; blind features == loader features; label reconstruction max diff {rec_d:.2e}")

    n_of = test.groupby("group")["N"].first()
    g3 = set(n_of[n_of >= 3].index)
    assert g3 == PRIMARY and int((test["group"].isin(g3)).sum()) == 60
    g_all = set(n_of.index)
    cmp_ = comp.merge(test[["group", "comb_num", "N", "T", "D"]], on=["group", "comb_num"], how="left", validate="one_to_one")
    s_inst_t = cmp_.set_index(["group", "comb_num"]).loc[key_t, "S_hat"].values
    cand = R.cand_of(test)
    NAME = {"orc": "Oracle", "ours": "Predictive (Ours)", "lat": "Latency-based", "inst": "Per-instance predictor", "bo5": "Best-of-5", "cpu": "All-CPU (reference)"}
    per = {"orc": R.ours_per_workload(test, S_true, ALPHA, NAME["orc"]),
           "ours": R.ours_per_workload(test, s_hat, ALPHA, NAME["ours"]),
           "lat": R.single_per_workload(test, lat_choice, ALPHA, NAME["lat"]),
           "inst": R.ours_per_workload(test, s_inst_t, ALPHA, NAME["inst"]),
           "cpu": R.single_per_workload(test, B.choose_all_device(test[["group", "comb_num", "all_cpu"]], "all_cpu"), ALPHA, NAME["cpu"])}
    for k in ("ours", "inst"):   # selections must equal the blind ones
        o = {g: f"combination_{(order_ours if k == 'ours' else order_inst)[g][0]}" for g in g_all}
        assert all(per[k].set_index("group").loc[g, "selected"] == o[g] for g in g_all)
    bo5 = B.best_of_k(cand, ALPHA)

    def mae(t, d, m):
        return float(np.abs(np.asarray(t)[m] - test["T"].values[m]).mean()), float(np.abs(np.asarray(d)[m] - test["D"].values[m]).mean())
    m3 = test["group"].isin(g3).values
    mall = np.ones(len(test), bool)
    it = cmp_.set_index(["group", "comb_num"]).loc[key_t]
    maes = {"ours": {"g3": mae(t_hat, d_hat, m3), "all": mae(t_hat, d_hat, mall)},
            "inst": {"g3": mae(it["T_hat_select"].values, it["D_hat"].values, m3), "all": mae(it["T_hat_select"].values, it["D_hat"].values, mall)}}

    def table(groups, tag):
        rows = []
        for k in ("orc", "ours", "lat", "inst", "bo5", "cpu"):
            if k == "bo5":
                b = RR.bo5_row(bo5, test, ALPHA, groups)
                rows.append({"method": NAME[k], "workloads": b["workloads"], "avg_score": b["avg_score"], "avg_score_std_over_trials": b["avg_score_std_over_trials"],
                             "regret": b["regret"], "top1": f"{float(b['top1'].split()[0]):.3f} (mean hit rate)", "top5": "N/A", "mae_T": "N/A", "mae_D": "N/A",
                             "sample_contains_oracle": b["sample_contains_oracle"]})
                continue
            p = per[k][per[k]["group"].isin(groups)]
            a = E.aggregate(p, 1, NAME[k])
            mm = maes.get(k, {}).get(tag)
            rows.append({"method": NAME[k], "workloads": a["workloads"], "avg_score": round(a["avg_score"], 4), "avg_score_std_over_trials": "",
                         "regret": round(a["regret"], 4), "top1": f"{a['top1_hits']}/{a['top1_n']}",
                         "top5": f"{a['top5_hits']}/{a['top5_n']}" if a["top5_hits"] is not None else "N/A",
                         "mae_T": round(mm[0], 4) if mm else "N/A", "mae_D": round(mm[1], 4) if mm else "N/A", "sample_contains_oracle": ""})
        return pd.DataFrame(rows)

    t3 = table(g3, "g3")
    t3.to_csv(OUT / "table3b.csv", index=False)
    (OUT / "table3b.md").write_text("# Table 3(b) - CPU-NPU, 3x hold-out, N>=3 (4 workloads, 60 candidates), alpha = 0.3, D = D_macro\n\n" + md_table(t3)
                                    + "\nBest-of-5: 1,000 candidate-sampling trials (5 of the measured candidates, best measured taken); Top-1 is the mean hit rate "
                                      "(not n/4). All-CPU is a reference row. MAE over the 60 primary candidates. Per-instance MAE(T) uses its own "
                                      "T_hat (F_hat / predicted maximum).\n")
    ta = table(g_all, "all")
    ta.to_csv(OUT / "table3b_all11.csv", index=False)
    log("\n" + t3.to_string(index=False))
    log("\n" + ta.to_string(index=False))

    # ---- per-workload
    rows = []
    for k in ("orc", "ours", "lat", "inst", "cpu"):
        for _, r in per[k].iterrows():
            rows.append({"method": NAME[k], "group": r["group"], "N": r["N"], "primary": r["group"] in g3, "n_candidates": r["n_candidates"],
                         "selected": r.get("selected"), "S_selected": r.get("S_selected"), "T_selected": r.get("T_selected"), "D_selected": r.get("D_selected"),
                         "oracle_S": r["oracle_S"], "oracle_combination": r["oracle_combination"], "n_oracle": r["n_oracle"], "regret": r.get("regret"),
                         "top1": r.get("top1"), "top5": r.get("top5")})
    g_, S_sel, top1, contains = bo5
    orc = per["orc"].set_index("group")
    for i, g in enumerate(g_):
        rows.append({"method": NAME["bo5"], "group": g, "N": int(n_of[g]), "primary": g in g3, "n_candidates": int((test["group"] == g).sum()),
                     "selected": "1,000 samplings", "S_selected": float(S_sel[:, i].mean()), "oracle_S": float(orc.loc[g, "oracle_S"]),
                     "oracle_combination": orc.loc[g, "oracle_combination"], "regret": float(orc.loc[g, "oracle_S"] - S_sel[:, i].mean()),
                     "top1": float(top1[:, i].mean()), "S_selected_std_over_trials": float(S_sel[:, i].std()), "sample_contains_oracle": float(contains[:, i].mean())})
    pw = pd.DataFrame(rows)
    plc = test.set_index(["group", "combination"])["placement"]
    pw["selected_placement"] = [plc.get((g, s), "") if isinstance(s, str) and s.startswith("combination_") else "" for g, s in zip(pw["group"], pw["selected"])]
    pw.to_csv(OUT / "per_workload.csv", index=False)

    # ---- paired
    pds, summ = [], []
    for label, a, b in (("Ours - Latency-based", "ours", "lat"), ("Ours - Oracle", "ours", "orc"), ("Ours - Per-instance", "ours", "inst")):
        p, s = RR.paired_difference(test, per[a], per[b], ALPHA, "macro", sorted(g3), label)
        pds.append(p)
        summ.append(s)
    pd.concat(pds, ignore_index=True).to_csv(OUT / "paired.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / "paired_summary.csv", index=False)
    log("\n" + pd.DataFrame(summ)[["comparison", "workloads", "mean_diff", "se_across_workloads", "n_positive", "n_zero", "n_negative"]].to_string(index=False))

    # ---- Top-1 tie statistics
    sr = RR.s_rep(test, ALPHA, "macro")
    tie = []
    for g in sorted(g3):
        m = (test["group"] == g).values
        sub = test[m]
        S = S_true[m]
        o = np.sort(S)[::-1]
        reps = {}
        for k in range(3):
            own = (sub[f"F_rep{k}"] / sub[f"F_rep{k}"].max() - ALPHA * sub[f"D_macro_rep{k}"]).values   # each repeat normalised by itself
            reps[f"oracle_rep{k}_own_norm"] = f"combination_{int(sub['comb_num'].values[int(np.argmax(own))])}"
            reps[f"oracle_rep{k}_mean_Fmax"] = f"combination_{int(sub['comb_num'].values[int(np.argmax(sr[m][:, k]))])}"
        tie.append({"group": g, "oracle": f"combination_{int(sub['comb_num'].values[int(np.argmax(S))])}", "gap_1st_2nd": float(o[0] - o[1]),
                    "gap_le_0.0002": bool(o[0] - o[1] <= 0.0002), **reps,
                    "repeat_oracles_agree_own_norm": len({reps[f'oracle_rep{k}_own_norm'] for k in range(3)} | {f"combination_{int(sub['comb_num'].values[int(np.argmax(S))])}"}) == 1,
                    "repeat_oracles_agree_mean_Fmax": len({reps[f'oracle_rep{k}_mean_Fmax'] for k in range(3)} | {f"combination_{int(sub['comb_num'].values[int(np.argmax(S))])}"}) == 1})
    tie = pd.DataFrame(tie)
    tie.to_csv(OUT / "top1_tie_stats.csv", index=False)

    # ---- alpha sensitivity (Multi only, no retraining)
    f4 = []
    for al in TS.ALPHAS:
        S_a = (test["T"] - al * test["D"]).values
        sm, _ = P.multi_score(t_hat, d_hat, al, clip=False)
        pm_ = R.ours_per_workload(test, sm, al, "multi")
        po_ = R.ours_per_workload(test, S_a, al, "oracle")
        pm, po = E.aggregate(pm_[pm_["group"].isin(g3)], 1), E.aggregate(po_[po_["group"].isin(g3)], 1)
        f4.append({"alpha": al, "workloads": pm["workloads"], "oracle_avg": round(po["avg_score"], 4), "avg_score_multi": round(pm["avg_score"], 4),
                   "regret_multi": round(pm["regret"], 4), "top1_hits_multi": pm["top1_hits"], "top5_hits_multi": pm["top5_hits"]})
    pd.DataFrame(f4).to_csv(OUT / "fig4_npu_data.csv", index=False)

    # ---- latency vs oracle: 3x primary + training split (labels only, descriptive)
    lvo = []
    for split, lab in (("3x evaluation", lab_te), ("training (labels only)", pd.read_csv(DS / "labels_train.csv"))):
        for (sf, rate), sub in lab.groupby(["schedule_file", "rate"]):
            pm_rows = {r["comb_num"]: dict(x.split(":") for x in r["placement"].split(",")) for _, r in sub.iterrows()}
            insts = list(next(iter(pm_rows.values())).keys())
            want = {k: v.lower() for k, v in latency_assignment(insts, lat).items()}
            hit = [c for c, pm in pm_rows.items() if pm == want]
            assert len(hit) == 1
            S = dict(zip(sub["comb_num"], sub["throughput_norm"] - ALPHA * sub["drop_rate_norm"]))
            assert np.allclose(list(S.values()), sub["S_macro_a0.3"].values)
            mx = max(S.values())
            oset = sorted(c for c, v in S.items() if v >= mx - E.TIE_TOL)
            lvo.append({"split": split, "rate": rate, "workload": sf.replace(".yaml", ""), "N": int(sub["N"].iloc[0]), "primary_3x_N>=3": sf in PRIMARY,
                        "latency_combination": f"combination_{hit[0]}", "latency_placement": placement_str({k: v.upper() for k, v in want.items()}),
                        "S_latency": float(S[hit[0]]), "oracle_combinations": ";".join(f"combination_{c}" for c in oset),
                        "oracle_placement(first)": sub.set_index("comb_num").loc[oset[0], "placement"], "S_oracle": float(mx),
                        "regret_latency": float(mx - S[hit[0]]), "latency_is_oracle": hit[0] in oset})
    lvo = pd.DataFrame(lvo).sort_values(["split", "rate", "N", "workload"])
    lvo.to_csv(OUT / "latency_vs_oracle.csv", index=False)

    # ---- facts for the report
    b3 = RR.bo5_row(bo5, test, ALPHA, g3)
    exp_top1 = (3 * 5 / 13 + 5 / 21) / 4
    lt = lvo[lvo["split"] == "training (labels only)"]
    summary = {"tables": {"table3b": t3.to_dict(orient="records"), "table3b_all11": ta.to_dict(orient="records")},
               "paired": summ, "top1_ties": {"gap_le_0.0002": int(tie["gap_le_0.0002"].sum()), "repeat_oracle_disagree_own_norm": int((~tie["repeat_oracles_agree_own_norm"]).sum()),
                                             "repeat_oracle_disagree_mean_Fmax": int((~tie["repeat_oracles_agree_mean_Fmax"]).sum()), "gaps": tie["gap_1st_2nd"].round(4).tolist()},
               "best_of_5_top1_primary": float(b3["top1"].split()[0]), "best_of_5_top1_expected_uniform": exp_top1,
               "latency_vs_oracle_3x_primary": int(lvo[(lvo["split"] == "3x evaluation") & lvo["primary_3x_N>=3"]]["latency_is_oracle"].sum()),
               "latency_vs_oracle_training_mismatch_by_rate": {r: int((~s["latency_is_oracle"]).sum()) for r, s in lt.groupby("rate")},
               "latency_vs_oracle_training_workloads_by_rate": {r: len(s) for r, s in lt.groupby("rate")},
               "mae": maes, "npu_feature_check": npu_check, "cv_mae": frozen["cv_mae"], "best_params": frozen["best_params"],
               "per_instance_cv": frozen["per_instance"], "label_reconstruction_max_diff": rec_d, "s_hat_ties_at_first_primary": {g: ties_ours[g] for g in sorted(g3)}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log(json.dumps({k: summary[k] for k in ("top1_ties", "best_of_5_top1_primary", "latency_vs_oracle_3x_primary", "latency_vs_oracle_training_mismatch_by_rate")}, default=float))


if __name__ == "__main__":
    main()

"""Per-instance baseline, inspired by Kim et al. [14] (instance-wise interference prediction). NOT a reproduction of their system.
Spec: revision_profiling/SPEC_v2.2.md section 7, variant (b) D_macro.

One row = (workload, placement, instance). Features (fixed in advance, no placement-wide pairwise information):
  id_model_hash_<k>   model identity one-hot (existing _get_model_features, 128 buckets)
  dev_cpu, dev_gpu    device one-hot
  lam                 scheduled request rate of the instance (schedule infps)
  iso_cap             isolated capacity of the model on its device (offline_profile.json, throughput_fps, closed loop)
  co_same_n, co_same_lam, co_other_n, co_other_lam, N   co-runner summary
Features come from the schedule CSV and the offline profile only. Targets (training rates only) come from the training CSV.
"""
import sys

sys.dont_write_bytecode = True

import json

import numpy as np
import pandas as pd

import data as D

B_M = 128
FEATURES = [f"id_model_hash_{k}" for k in range(B_M)] + ["dev_cpu", "dev_gpu", "lam", "iso_cap", "co_same_n", "co_same_lam", "co_other_n", "co_other_lam", "N"]
TARGET_SOURCES = {"train_x3.csv"}            # instance labels may be built from these files only


def iso_capacity(profile_path):
    prof = json.loads(open(profile_path).read())["models"]
    return {(m, dev.upper()): float(v[dev]["throughput_fps"]) for m, v in prof.items() for dev in ("cpu", "gpu")}


def model_onehot(model):
    s = D.suite()
    assert s.MODEL_HASH_BUCKETS == B_M
    f = s._get_model_features(model)
    assert len(f) == 1
    return {f"id_{k}": v for k, v in f.items()}


def rows_from_schedules(sched_csv, iso):
    """Feature rows for EVERY combination of every schedule in the CSV (no measured value is read).
    Returns DataFrame[group, family, rate, combination, comb_num, model, device, <FEATURES>]."""
    s = D.suite()
    idx = s._index_schedules_from_csv(sched_csv)
    out = []
    for name, doc in sorted(idx.items()):
        for comb, _blob in s._iter_combos_from_schedule(doc):
            inf = s._build_infps_lookup(doc, comb)            # {(model, DEV): infps}
            assert inf, f"empty schedule entry {name} {comb}"
            n = len(inf)
            for (model, dev), lam in sorted(inf.items()):
                assert lam > 0 and dev in ("CPU", "GPU")
                same = [v for (m2, d2), v in inf.items() if m2 != model and d2 == dev]
                other = [v for (m2, d2), v in inf.items() if m2 != model and d2 != dev]
                r = {"group": name, "family": D.family_key(name), "rate": D.rate_key(name), "combination": str(comb), "comb_num": D.comb_num(comb),
                     "model": model, "device": dev, **{c: 0.0 for c in FEATURES[:B_M]}, **model_onehot(model),
                     "dev_cpu": float(dev == "CPU"), "dev_gpu": float(dev == "GPU"), "lam": lam, "iso_cap": iso[(model, dev)],
                     "co_same_n": float(len(same)), "co_same_lam": float(sum(same)), "co_other_n": float(len(other)), "co_other_lam": float(sum(other)),
                     "N": float(n)}
                out.append(r)
    df = pd.DataFrame(out)
    assert df[FEATURES].notna().all().all()
    assert not df.duplicated(["group", "comb_num", "model"]).any()
    return df


def targets_from_csv(perf_csv, window_sec):
    """(group, comb_num, model, device) -> completed, dropped, generated, q_i from the json_content of a TRAINING csv."""
    assert perf_csv.name in TARGET_SOURCES, f"instance labels must not be read from {perf_csv.name}"
    raw = pd.read_csv(perf_csv)
    out = []
    for _, r in raw.iterrows():
        w = json.loads(r["json_content"])
        for v in w["models"].values():
            out.append({"group": w.get("schedule_file") or w.get("schedule file"), "comb_num": D.comb_num(w["combination"]), "model": v["model"], "device": str(v["execution"]).upper(),
                        "completed": float(v["completed"]), "dropped": float(v["dropped"]), "generated": float(v["generated"]), "q_i": float(v["q_i"])})
    t = pd.DataFrame(out)
    t["window_sec"] = window_sec
    return t


def training_rows(sched_csv, perf_csv, labels_csv, iso, window_sec):
    """Instance rows of the training rates with targets t_ratio = completed/window/lam and q_i."""
    lab = pd.read_csv(labels_csv)
    assert (lab["n_reps"] == 1).all(), "training candidates have one repeat; per-repeat averaging is therefore the identity here"
    rows = rows_from_schedules(sched_csv, iso)
    tg = targets_from_csv(perf_csv, window_sec)
    n = len(rows)
    rows = rows.merge(tg, on=["group", "comb_num", "model", "device"], how="inner", validate="one_to_one")
    assert len(rows) == n == len(tg), "every scheduled instance must have exactly one measured row"
    if (rows["generated"] <= 0).any():
        raise SystemExit("generated_i = 0 in a training instance row: stopping as instructed")
    rows["t_ratio"] = rows["completed"] / rows["window_sec"] / rows["lam"]
    assert np.allclose(rows["q_i"], rows["dropped"] / rows["generated"], atol=1e-12)
    assert set(rows["rate"]) == {"x1", "x2", "x4"}
    return rows

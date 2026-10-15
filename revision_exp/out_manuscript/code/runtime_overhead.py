"""Runtime overhead of the placement decision (the only new measurement for the manuscript assets).

  .venv/bin/python3 -B revision_exp/out_manuscript/code/runtime_overhead.py

Code path = revision_exp: features from the existing builder (featurize_window via data.suite()), column order from
out_reprofiled/feature_columns.txt (reindex as data.build_features), prediction with predict.load_model / predict_multi on the
Phase 2 artifacts (one batch call per model), score S_hat = T_hat - 0.3 D_hat (no clip, final rule), ordering as evaluate.select
(descending S_hat, ties -> lower combination number). No measured value is used: the decision only needs the schedule (models, rates).
Stages: (a) enumerate 2^N placements, (b) feasibility check, (c) features, (d) T_hat and D_hat, (e) score and sort.
100 warm repetitions (median, p10, p90) after one cold run that also loads the two models. CPU only.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "revision_exp"))

import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

t_import0 = time.perf_counter()
import data as D  # noqa: E402
import predict as P  # noqa: E402

S = D.suite()          # imports xgboost_model/deploy_selector_xgb_suite.py (unchanged)
t_import = time.perf_counter() - t_import0

OUT = REPO / "revision_exp" / "out_manuscript"
ART = REPO / "revision_exp" / "out_reprofiled" / "artifacts" / "multi"
COLS = (REPO / "revision_exp" / "out_reprofiled" / "feature_columns.txt").read_text().split()
SCHED = REPO / "revision_profiling" / "schedules"
ALPHA = 0.3
DEVICES = ("CPU", "GPU")
REPS = 100


def workload(family):
    y = yaml.safe_load((SCHED / f"{family}_x3.yaml").read_text())
    inst = sorted(next(iter(y["combinations"].values()))["instances"].values(), key=lambda i: i["model"])
    return [(i["model"], float(i["rate"])) for i in inst]


def enumerate_placements(inst):
    n = len(inst)
    return [tuple(DEVICES[(mask >> i) & 1] for i in range(n)) for mask in range(2 ** n)]   # bit i = 1 -> instance i on GPU


def feasible(inst, placements):
    """CPU-GPU platform: every instance may run on either device; no per-device instance limit."""
    return [p for p in placements if len(p) == len(inst) and all(d in DEVICES for d in p)]


def features(inst, placements):
    rows = []
    for p in placements:
        window = {"models": {f"view{i + 1}": {"model": m, "execution": d} for i, ((m, _), d) in enumerate(zip(inst, p))}}
        infps = {(m, d): r for (m, r), d in zip(inst, p)}
        x, _, _ = S.featurize_window(window, infps)
        rows.append(x)
    X = pd.DataFrame(rows).fillna(0.0).reindex(columns=COLS, fill_value=0.0)
    return X


def predict(mT, mD, X):
    return P.predict_multi(mT, mD, X)


def score_sort(t, d):
    s = t - ALPHA * d
    order = np.lexsort((np.arange(len(s)), -s))
    return s, order


def run_once(inst, mT, mD):
    tm = {}
    t0 = time.perf_counter(); pl = enumerate_placements(inst); tm["a_enumerate"] = time.perf_counter() - t0
    t0 = time.perf_counter(); pl = feasible(inst, pl); tm["b_feasibility"] = time.perf_counter() - t0
    t0 = time.perf_counter(); X = features(inst, pl); tm["c_features"] = time.perf_counter() - t0
    t0 = time.perf_counter(); t, d = predict(mT, mD, X); tm["d_predict"] = time.perf_counter() - t0
    t0 = time.perf_counter(); s, order = score_sort(t, d); tm["e_score_sort"] = time.perf_counter() - t0
    tm["total"] = sum(tm.values())
    return tm, pl[order[0]], len(pl)


def measure(family):
    inst = workload(family)
    t0 = time.perf_counter()
    mT, mD = P.load_model(ART / "T.json"), P.load_model(ART / "D.json")
    load = time.perf_counter() - t0
    cold, best, n_cand = run_once(inst, mT, mD)
    cold = {**cold, "model_load": load, "total_with_load": cold["total"] + load}
    warm = [run_once(inst, mT, mD)[0] for _ in range(REPS)]
    w = pd.DataFrame(warm)
    single = []
    for _ in range(REPS):
        pl = [enumerate_placements(inst)[-1]]
        t0 = time.perf_counter()
        X = features(inst, pl)
        predict(mT, mD, X)
        single.append(time.perf_counter() - t0)
    return {"family": family, "N": len(inst), "candidates": n_cand, "selected": "".join(d[0] for d in best),
            "cold_ms": {k: v * 1e3 for k, v in cold.items()},
            "warm_median_ms": (w.median() * 1e3).to_dict(), "warm_p10_ms": (w.quantile(0.1) * 1e3).to_dict(), "warm_p90_ms": (w.quantile(0.9) * 1e3).to_dict(),
            "single_candidate_feature_plus_predict_ms": {"median": float(np.median(single) * 1e3), "p10": float(np.percentile(single, 10) * 1e3),
                                                         "p90": float(np.percentile(single, 90) * 1e3)}}


def validate(family):
    """The schedule-only feature path must give the same T_hat, D_hat as the Phase 2 evaluation (out_reprofiled/predictions_x3.csv)."""
    inst = workload(family)
    pl = enumerate_placements(inst)
    mT, mD = P.load_model(ART / "T.json"), P.load_model(ART / "D.json")
    t, d = predict(mT, mD, features(inst, pl))
    ref = pd.read_csv(REPO / "revision_exp" / "out_reprofiled" / "predictions_x3.csv")
    ref = ref[ref["group"] == f"{family}_x3.yaml"].copy()
    ref["mask"] = ref["combination"].str.rsplit("_", n=1).str[1].astype(int)
    ref = ref.set_index("mask").sort_index()
    assert list(ref.index) == list(range(len(pl)))
    return {"family": family, "max_abs_diff_T_hat": float(np.abs(t - ref["T_hat"].values).max()), "max_abs_diff_D_hat": float(np.abs(d - ref["D_hat"].values).max()),
            "selected_unclipped": "".join(x[0] for x in pl[score_sort(t, d)[1][0]])}


def main():
    import xgboost
    m = P.load_model(ART / "T.json")
    cfg = json.loads(m.get_booster().save_config())
    env = {"python": platform.python_version(), "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__,
           "cpu": [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0], "logical_cpus": os.cpu_count(),
           "xgb_n_jobs_param_after_load": m.get_params().get("n_jobs"), "xgb_booster_nthread": cfg["learner"]["generic_param"].get("nthread"),
           "xgb_device": cfg["learner"]["generic_param"].get("device"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "loadavg_before": os.getloadavg(), "import_suite_ms": t_import * 1e3, "reps": REPS,
           "prediction_call": "one batch call per model over all candidates (predict.predict_multi -> XGBRegressor.predict(X.values))",
           "feature_build": "per-candidate loop calling featurize_window, then one DataFrame + reindex to the 204 training columns"}
    env["validation"] = [validate("gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19"), validate("gpt2+mnasnet+resnet50")]
    assert all(v["max_abs_diff_T_hat"] < 1e-6 and v["max_abs_diff_D_hat"] < 1e-6 for v in env["validation"]), env["validation"]
    res = [measure("gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19"), measure("gpt2+mnasnet+resnet50")]
    env["loadavg_after"] = os.getloadavg()
    (OUT / "runtime_overhead.json").write_text(json.dumps({"env": env, "results": res}, indent=1))
    rows = []
    for r in res:
        for st in ("a_enumerate", "b_feasibility", "c_features", "d_predict", "e_score_sort", "total"):
            rows.append({"family": r["family"], "N": r["N"], "candidates": r["candidates"], "stage": st, "warm_median_ms": r["warm_median_ms"][st],
                         "warm_p10_ms": r["warm_p10_ms"][st], "warm_p90_ms": r["warm_p90_ms"][st], "cold_ms": r["cold_ms"][st]})
        rows.append({"family": r["family"], "N": r["N"], "candidates": r["candidates"], "stage": "model_load (cold only)", "cold_ms": r["cold_ms"]["model_load"]})
        rows.append({"family": r["family"], "N": r["N"], "candidates": r["candidates"], "stage": "total incl. model load (cold)", "cold_ms": r["cold_ms"]["total_with_load"]})
        rows.append({"family": r["family"], "N": r["N"], "candidates": 1, "stage": "single candidate: features + predict",
                     "warm_median_ms": r["single_candidate_feature_plus_predict_ms"]["median"], "warm_p10_ms": r["single_candidate_feature_plus_predict_ms"]["p10"],
                     "warm_p90_ms": r["single_candidate_feature_plus_predict_ms"]["p90"]})
    pd.DataFrame(rows).round(4).to_csv(OUT / "runtime_overhead.csv", index=False)
    print(json.dumps(env, indent=1))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

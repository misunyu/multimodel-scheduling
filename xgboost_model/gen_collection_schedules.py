"""Generate the full re-collection schedules for CPU-NPU and CPU-GPU.

One combination_N schedule per platform, covering all 15 working sets from
working_sets.yaml. Sets with >=5 freely-placed models are stratified on the
"models on the accelerator" axis; the rest are enumerated in full. Both platforms
get the SAME free-placement set so best-combo can be compared across them (only
qwen2_vl's device differs: npu vs gpu). A .meta.json records rate/workload/stratum.

Deterministic: the stratified pick is seeded, so re-running reproduces the schedule.
"""
import json
import itertools
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import model_registry as reg
PROF = json.loads((ROOT / "xgboost_model/performance_data/sample_profiling_data/"
                   "sample_profiling_data.json").read_text())
BASE = {r["model"]: r.get("baseline_rate") for r in PROF["total_data"]}
W = yaml.safe_load((ROOT / "working_sets.yaml").read_text())
SETS = W["sets"]
STRAT_SETS = set(W["stratified_sets"])
STRAT_TARGET = {int(k): v for k, v in W["stratified_target_by_free"].items()}
SEED = 20260715


def rate_for(n):
    if n <= 2:
        return W["rate_factors_by_size"]["small"]["factors"]
    if n <= 5:
        return W["rate_factors_by_size"]["medium"]["factors"]
    return W["rate_factors_by_size"]["large"]["factors"]


def free_models(models):
    # qwen2_vl (no CPU) is pinned to the accelerator -> not a free axis.
    return [m for m in models if "cpu" in reg.allowed_devices(m)]


def all_placements(models):
    free = free_models(models)
    fixed = [m for m in models if m not in free]
    out = []
    for combo in itertools.product(["cpu", "accel"], repeat=len(free)):
        p = dict(zip(free, combo))
        for m in fixed:
            p[m] = "accel"
        out.append(p)
    return out


def stratify(models, target):
    """Pick `target` placements spread over strata = #free-models on accelerator.

    Extremes (all-cpu, all-accel) always in; middle strata filled round-robin so each
    reaches >=2 before any is over-sampled.
    """
    free = free_models(models)
    rng = random.Random(SEED + len(models))
    by_k = {}
    for p in all_placements(models):
        k = sum(1 for m in free if p[m] == "accel")
        by_k.setdefault(k, []).append(p)
    for k in by_k:
        rng.shuffle(by_k[k])
    strata = sorted(by_k)
    chosen, tags = [], []
    # extremes
    chosen.append(by_k[strata[0]][0]); tags.append(strata[0])
    chosen.append(by_k[strata[-1]][0]); tags.append(strata[-1])
    taken = {strata[0]: 1, strata[-1]: 1}
    mids = strata[1:-1]
    # round-robin the middle strata until target reached
    i = 0
    while len(chosen) < target:
        k = mids[i % len(mids)]
        t = taken.get(k, 0)
        if t < len(by_k[k]):
            chosen.append(by_k[k][t]); tags.append(k); taken[k] = t + 1
        i += 1
        if i > 100000:
            break
    return chosen[:target], tags[:target]


def combos_for(sid, models):
    n_free = len(free_models(models))
    if sid in STRAT_SETS:
        return stratify(models, STRAT_TARGET[n_free])
    ps = all_placements(models)
    return ps, [None] * len(ps)


def entry(models, placement, accel, rate):
    combo = {}
    for j, m in enumerate(models, 1):
        dev = accel if placement[m] == "accel" else "cpu"
        combo[f"{m}_{dev}"] = {"model": m, "execution": dev, "display": f"view{j}",
                               "infps": round(float(BASE[m]) * rate, 3)}
    return combo


def build(accel, out_dir):
    doc, meta = {}, {}
    idx = 0
    # Deterministic set order matters only for combination numbering; the collection
    # runner decides run order. Keep working_sets.yaml order.
    for sid, models in SETS.items():
        n = len(models)
        places, tags = combos_for(sid, models)
        for rate in rate_for(n):
            for p, tag in zip(places, tags):
                idx += 1
                name = f"combination_{idx}"
                doc[name] = entry(models, p, accel, rate)
                meta[name] = {
                    "set": sid, "rate_factor": rate,
                    "workload": next((m for m in models if reg.get(m).get("task") == "detection"), models[0]),
                    "stratum": tag,
                    "n_accel": sum(1 for m in free_models(models) if p[m] == "accel"),
                }
    path = Path(out_dir) / f"collect_cpu_{accel}.yaml"
    path.write_text(yaml.dump(doc, sort_keys=False))
    (Path(out_dir) / f"collect_cpu_{accel}.yaml.meta.json").write_text(json.dumps(meta, indent=2))
    return doc, meta, path


if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT)
    docs = {}
    for accel in ("npu", "gpu"):
        doc, meta, path = build(accel, out_dir)
        by_set = {}
        for m in meta.values():
            by_set[m["set"]] = by_set.get(m["set"], 0) + 1
        docs[accel] = doc
        print(f"{path.name}: {len(doc)} combinations  {by_set}")

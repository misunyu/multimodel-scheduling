#!/usr/bin/env python3
"""Concurrency-sweep schedules for the CPU+GPU platform.

Scales the number of concurrently running models N from 4 up to 8 by adding
YOLO11 detectors smallest-first (n, s, m, l, x) on top of a fixed base
(resnet50 + llama1b + qwen2_vl). For each N, samples device placements (each
model -> cpu or gpu; qwen2_vl fixed to gpu) stratified by GPU-count so the full
all-CPU..all-GPU contention spectrum is covered, then replicates across the
input-rate sweep. Workload id = "N<k>"; a meta sidecar carries rate_factor and
workload for the train/hold-out split and per-group normalization.

Usage:
    python generate_concurrency_schedules.py --out schedules_concurrency.yaml
"""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path
import yaml
import model_registry as reg

BASE = ["resnet50", "llama1b", "qwen2_vl"]
YOLO_ORDER = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
PLATFORM = ["cpu", "gpu"]


def model_set(n: int):
    """N models: base (3) + the (n-3) smallest YOLO detectors."""
    return BASE + YOLO_ORDER[: n - 3]


def _stratified_placements(models, cap):
    """All model->{cpu,gpu} assignments (qwen fixed gpu), sampled by gpu-count."""
    free = [m for m in models if "cpu" in reg.allowed_devices(m)]  # qwen -> gpu only
    fixed = {m: "gpu" for m in models if m not in free}
    all_pl = []
    for combo in itertools.product(*[PLATFORM for _ in free]):
        assign = dict(zip(free, combo)); assign.update(fixed)
        all_pl.append(assign)
    if len(all_pl) <= cap:
        return all_pl
    # stratify by number of models on gpu; round-robin pick across strata
    by_g = {}
    for p in all_pl:
        g = sum(1 for v in p.values() if v == "gpu")
        by_g.setdefault(g, []).append(p)
    ordered = []
    strata = [by_g[g] for g in sorted(by_g)]
    i = 0
    while len(ordered) < cap and any(strata):
        s = strata[i % len(strata)]
        if s:
            ordered.append(s.pop(0))
        if all(len(x) == 0 for x in strata):
            break
        i += 1
    # always include all-cpu-free and all-gpu extremes
    return ordered[:cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="schedules_concurrency.yaml")
    ap.add_argument("--static_json", default="xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json")
    ap.add_argument("--n_min", type=int, default=4)
    ap.add_argument("--n_max", type=int, default=8)
    ap.add_argument("--cap", type=int, default=10, help="max placements sampled per N level")
    ap.add_argument("--rate_factors", nargs="*", type=float, default=reg.RATE_FACTORS)
    args = ap.parse_args()

    baseline = {r["model"]: r.get("baseline_rate") for r in
                json.loads(Path(args.static_json).read_text()).get("total_data", [])}

    schedules, meta = {}, {}
    idx = 0
    for n in range(args.n_min, args.n_max + 1):
        models = model_set(n)
        placements = _stratified_placements(models, args.cap)
        for placement in placements:
            for rf in args.rate_factors:
                idx += 1
                name = f"combination_{idx}"
                entry = {}
                for j, (model, dev) in enumerate(placement.items()):
                    base = baseline.get(model) or 1.0
                    entry[f"{model}_{dev}"] = {
                        "model": model, "execution": dev,
                        "display": f"view{j + 1}", "infps": round(rf * base, 3),
                    }
                schedules[name] = entry
                meta[name] = {"rate_factor": rf, "workload": f"N{n}", "n_models": n}

    outp = Path(args.out)
    with open(outp, "w") as f:
        f.write("# concurrency sweep (CPU+GPU): N=4..8, YOLO added smallest-first\n\n")
        f.write(yaml.dump(schedules, default_flow_style=False, sort_keys=False))
    Path(str(outp) + ".meta.json").write_text(json.dumps(meta, indent=2))
    from collections import Counter
    print(f"[OK] {len(schedules)} combos -> {outp}")
    print("  by N:", dict(Counter(m['n_models'] for m in meta.values())))


if __name__ == "__main__":
    main()

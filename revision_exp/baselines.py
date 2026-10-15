"""Baselines: Latency-based, Best-of-5 (1,000 seeded trials), All-GPU / All-CPU, plus a replay of the stored latency rule."""
import sys

sys.dont_write_bytecode = True

import glob
import json
import re

import numpy as np
import pandas as pd

from data import MASTER_SEED, REPO, note_input, rate_key

OFFLINE_PROFILE = "sample_profiling_data/sample_profiling_data_ubuntu_5090.json"
RESULT_DIR_FOR_RATE = {"x1": "results_recompute", "x2": "results_recompute", "x3": "results_recompute", "x4": "results_recompute",
                       "x1-5": "results_recompute_additional", "x2-5": "results_recompute_additional", "x3-5": "results_recompute_additional"}


def _single_instance_entries(rate):
    """All N=1 measurements at the given rate: list of dicts (model, device, latency, score, timestamp, source_file)."""
    out = []
    d = RESULT_DIR_FOR_RATE[rate]
    for p in sorted(glob.glob(str(REPO / d / "*.json"))):
        doc = json.load(open(p))
        sf = doc.get("schedule file", "")
        data = doc.get("data", [])
        if rate_key(sf) != rate or not data or any(len(e.get("models", {})) != 1 for e in data):
            continue
        note_input(p)
        for e in data:
            m = next(iter(e["models"].values()))
            out.append({"model": m["model"], "device": str(m["execution"]).upper(), "latency_ms": float(m["avg_inference_time_ms"]),
                        "stored_score": e.get("score"), "timestamp": e.get("timestamp"), "entries_in_file": len(data),
                        "source_file": str(p).replace(str(REPO) + "/", "")})
    return out


def _offline_profile():
    p = note_input(REPO / OFFLINE_PROFILE)
    d = json.load(open(p))
    out = {}
    for dev, key in (("CPU", "cpu_data"), ("GPU", "gpu_data")):
        for r in d.get(key, []):
            name = re.sub(r"(/model)?\.onnx$", "", r["model"])
            out[(name, dev)] = float(r["infer"])
    return out


def latency_profile(rate, models):
    """(model, device) -> latency with provenance. Source: N=1 runs at the same rate (latest run if repeated);
    missing pairs fall back to the offline profile."""
    ent = _single_instance_entries(rate)
    off = _offline_profile()
    rows = []
    for model in sorted(models):
        for dev in ("CPU", "GPU"):
            c = [e for e in ent if e["model"] == model and e["device"] == dev]
            if c:
                e = max(c, key=lambda x: x["timestamp"])
                rows.append({"rate": rate, "model": model, "device": dev, "latency_ms": e["latency_ms"], "source_type": "single-instance run (N=1)",
                             "source_file": e["source_file"], "run_timestamp": e["timestamp"], "runs_available": len(c), "fallback_used": False})
            elif (model, dev) in off:
                rows.append({"rate": rate, "model": model, "device": dev, "latency_ms": off[(model, dev)], "source_type": "offline profile",
                             "source_file": OFFLINE_PROFILE, "run_timestamp": "", "runs_available": 0, "fallback_used": True})
            else:
                raise RuntimeError(f"no latency for {model} on {dev} at {rate}")
    return pd.DataFrame(rows)


def latency_targets(profile):
    """Device with the smaller single-instance latency; ties -> GPU."""
    t = {}
    for model, sub in profile.groupby("model"):
        lat = dict(zip(sub["device"], sub["latency_ms"]))
        t[model] = "GPU" if lat["GPU"] <= lat["CPU"] else "CPU"
    return t


def stored_rule_targets(rate):
    """Replay of scripts/latency_based_selector.py: per (model, PU) the highest stored single-instance score over ALL N=1 files,
    PU = argmax with ties -> CPU (dict order CPU, GPU). Used only to explain the manuscript value."""
    best = {}
    for e in _single_instance_entries(rate):
        k = (e["model"], e["device"])
        if e["stored_score"] is not None and (k not in best or e["stored_score"] > best[k]):
            best[k] = e["stored_score"]
    t = {}
    for model in sorted(set(k[0] for k in best)):
        c, g = best.get((model, "CPU"), 0) or 0, best.get((model, "GPU"), 0) or 0
        t[model] = "CPU" if c >= g else "GPU"
    return t


def _placement_map(p):
    return dict(x.split("@") for x in p.split(";"))


def choose_by_targets(cand, targets):
    """{group: comb_num} of the candidate whose device assignment equals the per-model targets (None if not measured)."""
    out = {}
    for g, sub in cand.groupby("group", sort=True):
        hit = [int(c) for c, p in zip(sub["comb_num"], sub["placement"]) if all(targets[m] == d for m, d in _placement_map(p).items())]
        out[g] = min(hit) if hit else None
    return out


def choose_all_device(cand, flag):
    out = {}
    for g, sub in cand.groupby("group", sort=True):
        hit = sub.loc[sub[flag], "comb_num"]
        out[g] = int(hit.min()) if len(hit) else None
    return out


def best_of_k(cand, alpha, k=5, trials=1000, tol=1e-9):
    """Seeded Monte-Carlo. Workloads in sorted group order, candidates in ascending combination order.
    Seed of (trial t, workload index w) = numpy SeedSequence([42, t, w]). Sampling without replacement; the sampled candidate
    with the highest measured S is taken (ties -> lower combination number). Workloads with <= k candidates use all of them.
    Returns (groups, S_sel[trials, groups], top1[trials, groups], contains_oracle[trials, groups])."""
    groups = sorted(cand["group"].unique())
    arrs = []
    for g in groups:
        sub = cand[cand["group"] == g].sort_values("comb_num")
        arrs.append((sub["T"] - alpha * sub["D"]).values)
    S_sel = np.zeros((trials, len(groups)))
    top1 = np.zeros((trials, len(groups)), dtype=bool)
    contains = np.zeros((trials, len(groups)), dtype=bool)
    for t in range(trials):
        for w, S in enumerate(arrs):
            n = len(S)
            if n <= k:
                idx = np.arange(n)
            else:
                rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, t, w]))
                idx = np.sort(rng.choice(n, size=k, replace=False))
            j = idx[np.argmax(S[idx])]            # idx ascending -> first maximum = lowest combination number
            mx = S.max()
            S_sel[t, w] = S[j]
            top1[t, w] = S[j] >= mx - tol
            contains[t, w] = bool((S[idx] >= mx - tol).any())
    return groups, S_sel, top1, contains

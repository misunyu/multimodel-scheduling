#!/usr/bin/env python3
"""Generate ranked candidate-placement lists for the paper §IV-A working sets
using the fsrr in-repo three-target predictor (xgboost_model/artifacts/{cpu_gpu,
cpu_npu}). Option (b) of v21/v22 task 91: rankings are exported as committed,
hash-stamped artifacts so "which ranking a run used" is permanently auditable.

This is a pure predictor call -- NO experiment re-run, NO scenario-schedule
generation. It enumerates the device placements of a working set and ranks them.

Vocabulary is validated at generation time (task 92-2): every model name must
resolve through model_registry with NO substring aliasing (removed in v21 task 87).

NOTE on rates: infps/slo below are a documented REPRESENTATIVE overload regime,
not the final per-scenario rates. Final rates are fixed by scenario-schedule
generation (v21 task 85, human-confirmed). Rankings are therefore provisional on
rates; the structural finding used by task 94 (rank of the recovery placement) is
checked for rate-sensitivity in the report.
"""
import argparse, hashlib, itertools, json, os, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_registry as reg
from deploy_predictor_logic import DeployPredictor

# Canonical scoring weights are stated HERE explicitly -- we do NOT inherit
# deploy_predictor_logic.DEFAULT_* implicitly. Implicit inheritance is exactly how
# beta drifted to 0.5 in v22 (docs/beta_canonicalization_report.md). beta=1.0 is the
# value at which the predictor bundle's quality was evaluated (mobilint
# evaluate_model.py --beta 1.0), so the deployed ranker matches the reported one.
CANONICAL_ALPHA = 0.3
CANONICAL_BETA = 1.0

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = {
    "cpu-gpu": "xgboost_model/artifacts/cpu_gpu/deploy_cpu_gpu",
    "cpu-npu": "xgboost_model/artifacts/cpu_npu/deploy_cpu_npu",
}
ACCEL = {"cpu-gpu": "gpu", "cpu-npu": "npu"}

# Representative overload regime (see module docstring). display = view key.
FG = [
    {"model": "yolo11s",      "display": "view1", "infps": 30, "slo_ms": 33},
    {"model": "yolo11m",      "display": "view2", "infps": 20, "slo_ms": 50},
    {"model": "resnet50",     "display": "view3", "infps": 60, "slo_ms": 17},
    {"model": "mobilenet_v2", "display": "view4", "infps": 90, "slo_ms": 11},
]
GEN1 = [{"model": "llama1b",  "display": "bg1", "infps": 2, "slo_ms": 500}]
GEN2 = GEN1 + [{"model": "qwen2_vl", "display": "bg2", "infps": 1, "slo_ms": 1000}]

# Gate-B recovered working set: the one the paper's Q3/Q6/Q5 runs ACTUALLY executed --
# llama1b (background) + THREE vision models (no yolo11m), uniform infps=80 (burst) and
# uniform slo_ms=15. Recovered from the raw logs plus the surviving experiment reports
# (docs/gate_b_schedule_provenance.md, docs/q3_q5_misprediction_report.md). 4 models ->
# 16 candidates, which is the space the paper's "rank five" refers to.
WS_PAPER_V3 = [
    {"model": "yolo11s",      "display": "view1", "infps": 80, "slo_ms": 15},
    {"model": "resnet50",     "display": "view2", "infps": 80, "slo_ms": 15},
    {"model": "mobilenet_v2", "display": "view3", "infps": 80, "slo_ms": 15},
    {"model": "llama1b",      "display": "view5", "infps": 80, "slo_ms": 15},
]

SCENARIOS = {
    "Q3Q6_paper_v3": {"ws": WS_PAPER_V3, "platforms": ["cpu-gpu"]},
    "Q5_paper_v3":   {"ws": WS_PAPER_V3, "platforms": ["cpu-npu"]},
    "Q1_3_steady":   {"ws": FG,          "platforms": ["cpu-gpu", "cpu-npu"]},
    "Q2_4b_load":    {"ws": FG,          "platforms": ["cpu-gpu", "cpu-npu"]},
    "Q3_mispred_1gen": {"ws": FG + GEN1, "platforms": ["cpu-gpu"]},
    "Q3_mispred_2gen": {"ws": FG + GEN2, "platforms": ["cpu-gpu"]},
    "Q5_npu_1gen":   {"ws": FG + GEN1,   "platforms": ["cpu-npu"]},
    "Q6_ablation_1gen": {"ws": FG + GEN1, "platforms": ["cpu-gpu"]},
}

# v28: the remaining scenarios. Composition confirmed from logs/reports, not assumed --
# Q1.3 (q13) and section-4b run vision-3 with NO background generative (gpu_/npu_Static
# logs; c2_reactive_baseline_report contrasts "llama1b bg + vision3" for Q3 against plain
# "vision3" for 4b), and Q4 uses the heavy-4 vision set with no background.
WS_V3_ONLY = [w for w in WS_PAPER_V3 if w["model"] != "llama1b"]
WS_HEAVY4 = [
    {"model": "yolo11x",  "display": "view1", "infps": 90, "slo_ms": 70},
    {"model": "yolo11l",  "display": "view2", "infps": 90, "slo_ms": 52},
    {"model": "yolo11m",  "display": "view3", "infps": 90, "slo_ms": 44},
    {"model": "resnet50", "display": "view4", "infps": 90, "slo_ms": 12},
]
SCENARIOS["Q4_heavy4"] = {"ws": WS_HEAVY4, "platforms": ["cpu-gpu"]}
SCENARIOS["Q4b_control_v3"] = {"ws": WS_V3_ONLY, "platforms": ["cpu-gpu"]}
SCENARIOS["Q1_3_persist_v3"] = {"ws": WS_V3_ONLY, "platforms": ["cpu-gpu", "cpu-npu"]}


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def enumerate_placements(ws, platform):
    # Enumerate every {cpu, accelerator} placement for every model -- this MATCHES the
    # runtime, which does not enforce model_registry.DEVICE_CONSTRAINTS anywhere in the
    # placement path (it is a generator-only annotation). Applying that constraint here
    # (v22/v23 did, via allowed_devices) wrongly excluded qwen2_vl-on-CPU and made the
    # paper's 2-gen recovery placement (both generatives on CPU, measured V(t)=0, "rank
    # twelve") unrepresentable. See docs/gate_a_device_constraints.md (Gate A, branch 2).
    devs = ["cpu", ACCEL[platform]]
    per = [list(devs) for _ in ws]
    combos = {}
    for i, assign in enumerate(itertools.product(*per)):
        blob = {}
        for w, dev in zip(ws, assign):
            blob[w["display"]] = {"display": w["display"], "execution": dev,
                                  "model": w["model"], "infps": w.get("infps"),
                                  "slo_ms": w.get("slo_ms")}
        combos[f"combination_{i:04d}"] = blob
    return combos


def device_of(blob, model):
    for v in blob.values():
        if v.get("model") == model:
            return v.get("execution")
    return None


def generate(scenario, platform, alpha, beta):
    spec = SCENARIOS[scenario]
    combos = enumerate_placements(spec["ws"], platform)
    names = {v["model"] for blob in combos.values() for v in blob.values()}
    bad = reg.unresolved_models(names)
    if bad:  # task 92-2: fail at generation time
        raise RuntimeError(f"{scenario}/{platform}: unresolved models {bad}")
    dp = DeployPredictor(log_callback=lambda m: None)
    best, df = dp.predict_best_combination(
        schedule_data=combos, model_input_path=BUNDLE[platform],
        alpha=alpha, beta=beta)
    df = df.sort_values("pred_score", ascending=False).reset_index(drop=True)
    ranked = []
    for rank_i, row in df.iterrows():
        blob = combos[row["combination"]]
        ranked.append({
            "rank": int(rank_i) + 1,
            "placement": {w["display"]: {"model": w["model"],
                          "device": device_of(blob, w["model"])} for w in spec["ws"]},
            "pred_norm_throughput": round(float(row["pred_norm_throughput"]), 6),
            "pred_deadline_miss_rate": round(float(row["pred_deadline_miss_rate"]), 6),
            "pred_norm_tokens": round(float(row["pred_norm_tokens"]), 6),
            "has_generative": bool(row["has_generative"]),
            "pred_score": round(float(row["pred_score"]), 6),
        })
    bundle_dir = os.path.join(PROJECT, "xgboost_model", "artifacts",
                              platform.replace("-", "_"))
    prov = {
        "predictor_source": {
            "repo": "fsrr-multimodel-scheduling (in-repo, committed)",
            "training_origin": "../multimodel-scheduling-mobilint",
            "bundle_prefix": BUNDLE[platform],
            "bundle_sha256": {f: _sha(os.path.join(bundle_dir, f))
                              for f in sorted(os.listdir(bundle_dir))
                              if f.endswith(".json")},
        },
        "trained_on": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
                       "resnet50", "mobilenet_v2", "llama1b", "qwen2_vl"],
        "targets": {"y1": "norm_throughput", "y2": "deadline_miss_rate",
                    "y3": "norm_tokens", "alpha": alpha, "beta": beta,
                    "score": "y1 + beta*y3 - alpha*y2 (y3 dropped for vision-only)"},
        "platform": platform,
        "workload": {"foreground": [w["model"] for w in FG],
                     "working_set": [w["model"] for w in spec["ws"]],
                     "rates_representative": {w["model"]: w["infps"] for w in spec["ws"]},
                     "rates_note": "REPRESENTATIVE overload regime, NOT final scenario "
                                   "rates (those come from v21 task 85)."},
        "generated": datetime.date.today().isoformat(),
        "vocabulary_validated": True,
        "n_candidates_total": len(ranked),
    }
    return {"scenario": scenario, "_provenance": prov, "ranking": ranked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(PROJECT, "rankings"))
    ap.add_argument("--alpha", type=float, default=CANONICAL_ALPHA)
    ap.add_argument("--beta", type=float, default=CANONICAL_BETA)
    ap.add_argument("--allow-noncanonical", action="store_true",
                    help="permit alpha/beta != canonical (experimental only)")
    args = ap.parse_args()
    # task 97-3: refuse non-canonical weights unless explicitly acknowledged.
    if (args.alpha != CANONICAL_ALPHA or args.beta != CANONICAL_BETA) and not args.allow_noncanonical:
        raise SystemExit(
            f"Refusing to generate rankings at non-canonical (alpha={args.alpha}, "
            f"beta={args.beta}). Canonical is (alpha={CANONICAL_ALPHA}, beta={CANONICAL_BETA}); "
            f"pass --allow-noncanonical for an intentional experiment "
            f"(see docs/beta_canonicalization_report.md).")
    print(f"[gen] scoring weights: alpha={args.alpha}, beta={args.beta}"
          + ("" if (args.alpha, args.beta) == (CANONICAL_ALPHA, CANONICAL_BETA)
             else "  [NON-CANONICAL]"))
    os.makedirs(args.outdir, exist_ok=True)
    for scenario, spec in SCENARIOS.items():
        for platform in spec["platforms"]:
            art = generate(scenario, platform, args.alpha, args.beta)
            out = os.path.join(args.outdir, f"ranking_{scenario}_{platform}.json")
            with open(out, "w") as f:
                json.dump(art, f, indent=2)
            top = art["ranking"][0]
            print(f"[{scenario}/{platform}] {art['_provenance']['n_candidates_total']} "
                  f"candidates -> {os.path.basename(out)} (top score {top['pred_score']})")


if __name__ == "__main__":
    sys.exit(main())

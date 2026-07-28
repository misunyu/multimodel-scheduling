#!/usr/bin/env python3
"""v26 lambda search: find the interval where a scenario holds.

For a given lambda and a given PLACEMENT TAKEN FROM THE RANKING ARTEFACT (never
hardcoded), run that placement statically (--adaptive-mode 3) and classify the
resulting V(t) trace with the canonical metric definitions (docs/metric_definitions.md):

    violated  : any sample with v_score > eps            (t0 exists)
    recovered : v_score <= eps from some point to run end (strict t_r exists)

Every artefact lands in a tracked runs/<utc>_<tag>/ directory with a run_manifest.json
(v24 convention) -- session scratch has been lost three times.

This is a SEARCH harness, not the main re-run.
"""
import argparse, csv, hashlib, itertools, json, os, subprocess, sys, time
from datetime import datetime, timezone

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
PYTHON = os.path.join(PROJECT, ".venv", "bin", "python3")
EXECUTOR = os.path.join(PROJECT, "schedule_executor_main.py")
RUNS = os.path.join(PROJECT, "runs")

ACCEL = {"cpu-gpu": "gpu", "cpu-npu": "npu"}
# L_SLO of the paper-configuration (vision-4) run, per docs/q3_experiment_report.md:
# "L_SLO(모델별, 5xTableI): yolo11s=155, yolo11m=209, resnet50=56, mobilenet_v2=32 ms".
SLO_MS = {"yolo11s": 155, "yolo11m": 209, "resnet50": 56, "mobilenet_v2": 32,
          "llama1b": 2000}
LLAMA_FPS = 2
EPS = 1.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def load_placement(ranking_path, which):
    """Return (placement_dict, rank, ranking_sha). `which` is 'top1' or 'recovery'.
    The placement comes from the ranking file -- we never write an order by hand."""
    art = json.load(open(ranking_path))
    plat = art["_provenance"]["platform"]
    acc = ACCEL[plat]
    ws = art["_provenance"]["workload"]["working_set"]
    gens = [m for m in ws if m in ("llama1b", "qwen2_vl")]
    R = art["ranking"]

    def dev(entry, model):
        for v in entry["placement"].values():
            if v["model"] == model:
                return v["device"]

    if which == "top1":
        e = R[0]
    else:
        e = next(x for x in R
                 if all(dev(x, g) == "cpu" for g in gens)
                 and all(dev(x, m) == acc for m in ws if m not in gens))
    return ({m: dev(e, m) for m in ws}, e["rank"], sha256(ranking_path), plat, ws)


def write_schedule(path, placement, lam, ws):
    """Configuration (a) as it was actually run on 07-22: the four vision models occupy
    view1..view4 (the monitored foreground, so V(t) covers exactly them), and the
    generative model runs headless (display: none) as background contention. Only
    view1..view4 are valid display labels, and a generative model in a view slot would
    be re-routed to headless anyway."""
    vision = [m for m in ws if m not in ("llama1b", "qwen2_vl")]
    gens = [m for m in ws if m in ("llama1b", "qwen2_vl")]
    lines = ["combination_probe:"]
    for i, m in enumerate(vision):
        lines += [f"    m{i+1}:",
                  f"        display: view{i+1}",
                  f"        execution: {placement[m]}",
                  f"        infps: {lam}",
                  f"        slo_ms: {SLO_MS.get(m, 15)}",
                  f"        model: {m}"]
    for j, m in enumerate(gens):
        lines += [f"    bg{j+1}:",
                  f"        display: none",
                  f"        execution: {placement[m]}",
                  f"        infps: {LLAMA_FPS}",
                  f"        slo_ms: {SLO_MS.get(m, 2000)}",
                  f"        model: {m}"]
    open(path, "w").write("\n".join(lines) + "\n")


def classify(csv_path, eps=EPS):
    """(violated, recovered, maxV, lastV, n) per docs/metric_definitions.md."""
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except Exception:
        return None
    v = [float(r["v_score"]) for r in rows if r.get("v_score") not in (None, "")]
    if not v:
        return None
    violated = any(x > eps for x in v)
    # strict t_r: <= eps from some index to the end
    recovered = False
    if violated:
        for i in range(len(v)):
            if v[i] <= eps and all(x <= eps for x in v[i:]):
                recovered = True
                break
    return {"violated": violated, "recovered": recovered,
            "maxV": round(max(v), 4), "lastV": round(v[-1], 4), "n": len(v)}


def one_run(ranking_path, which, lam, duration, tag_extra=""):
    placement, rank, rsha, plat, ws = load_placement(ranking_path, which)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"{stamp}_lam{lam}_{which}{tag_extra}"
    rd = os.path.join(RUNS, tag)
    os.makedirs(rd, exist_ok=True)
    sched = os.path.join(rd, "schedule_snapshot.yaml")
    write_schedule(sched, placement, lam, ws)
    metrics = os.path.join(rd, "metrics.csv")
    log = os.path.join(rd, "executor.log")
    cmd = [PYTHON, EXECUTOR, "--schedule", sched, "--duration", str(duration),
           "--adaptive-mode", "3", "--metrics-csv", metrics, "--auto_start_all"]
    # The generative model only actually runs when --background is passed; without it the
    # schedule's generative entry is parsed but never launched, so there is no LLM
    # contention and the misprediction scenario cannot reproduce.
    if any(m in ("llama1b", "qwen2_vl") for m in ws):
        cmd += ["--background"]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["FSRR_RATE_REPLICATE"] = "1"      # lambda = infps (documented load knob)
    env["FSRR_FRAME_BUFFER"] = os.environ.get("V26_BUFFER", "30")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=PROJECT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             timeout=duration * 6 + 180)
    res = classify(metrics)
    manifest = {
        "run_dir": tag, "scenario_probe": which, "lambda": lam,
        "placement": placement, "rank_in_ranking": rank,
        "ranking_file": os.path.relpath(ranking_path, PROJECT), "ranking_sha256": rsha,
        "platform": plat, "slo_ms": {m: SLO_MS.get(m, 15) for m in ws},
        "llama_fps": LLAMA_FPS, "duration_s": duration, "eps": EPS,
        "env_knobs": {k: v for k, v in env.items() if k.startswith("FSRR_")},
        "executor_rc": rc, "classification": res,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [{"file": f, "sha256": sha256(os.path.join(rd, f))}
                  for f in sorted(os.listdir(rd)) if os.path.isfile(os.path.join(rd, f))],
    }
    json.dump(manifest, open(os.path.join(rd, "run_manifest.json"), "w"), indent=2)
    return res, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--lambdas", required=True, help="comma-separated")
    ap.add_argument("--duration", type=int, default=20)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    print(f"{'lambda':>7} {'placement':>9} {'rank':>4} {'violated':>8} {'recovered':>9} "
          f"{'maxV':>9} {'lastV':>8}  run_dir")
    for lam in [int(x) for x in a.lambdas.split(",")]:
        for which in ("top1", "recovery"):
            for rep in range(a.reps):
                extra = f"_r{rep}" if a.reps > 1 else ""
                res, tag = one_run(a.ranking, which, lam, a.duration, extra)
                _, rank, _, _, _ = load_placement(a.ranking, which)
                if res is None:
                    print(f"{lam:>7} {which:>9} {rank:>4}  (no metrics)   {tag}")
                else:
                    print(f"{lam:>7} {which:>9} {rank:>4} {str(res['violated']):>8} "
                          f"{str(res['recovered']):>9} {res['maxV']:>9} {res['lastV']:>8}  {tag}")


if __name__ == "__main__":
    sys.exit(main())

"""Rebuild performance-window JSON files from the collected JSONL, so the existing
build_dataset pipeline can consume them unchanged.

featurize_window sums per-view throughput_fps for y1 and per-view tokens_per_s for y3,
and reads total.deadline_miss_rate for y2. The JSONL kept per-view throughput_fps
(so y1 rebuilds exactly) and the window-total y3 (per-view tokens were not stored).
We attach the window-total y3 to the single view that actually generated -- the one
LLM/VLM view on an accelerator -- so the per-view sum equals the measured total. That
view is also exactly the one that makes the combination y3-valid, so featurize_window's
y3_valid flag comes out right too. Vision-only / CPU-LLM combos get y3 spread as 0,
which is correct: their y3 is masked anyway.
"""
import json, sys
from pathlib import Path

SC = Path("/tmp/claude-1001/-home-msyu-PycharmProjects-multimodel-scheduling-mobilint/"
          "6f0af394-5a71-4f54-bead-adacfbd8c6b9/scratchpad")
sys.path.insert(0, "/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
import model_registry as reg

platform = sys.argv[1]                     # npu | gpu
jsonl = SC / f"collect/collect_{platform}_results.jsonl"
meta = json.loads((SC / f"collect_cpu_{platform}.yaml.meta.json").read_text())
out_dir = SC / "collect" / f"windows_{platform}"
out_dir.mkdir(parents=True, exist_ok=True)

windows = []
for line in open(jsonl):
    d = json.loads(line)
    combo = d["combo"]
    m = meta.get(combo, {})
    # find the accelerator-placed generative view to carry the total y3
    gen_view = None
    for model, (execu, fps, cnt) in d["models"].items():
        if reg.kind_of(model) in ("llm", "vlm") and execu.lower() in ("npu", "gpu"):
            gen_view = model
            break
    views = {}
    for j, (model, (execu, fps, cnt)) in enumerate(d["models"].items(), 1):
        v = {"model": model, "execution": execu, "throughput_fps": fps,
             "inference_count": cnt, "tokens_per_s": 0.0}
        if model == gen_view:
            v["tokens_per_s"] = d["y3"]     # window total on the one valid view
        views[f"view{j}"] = v
    windows.append({
        "combination": combo, "workload": m.get("workload"),
        "rate_factor": m.get("rate_factor"), "window_sec": d["window"],
        "models": views,
        "total": {"deadline_miss_rate": d["y2"], "total_throughput_fps": d["y1"],
                  "total_tokens_per_s": d["y3"]},
    })

out = out_dir / f"performance_{platform}_all.json"
out.write_text(json.dumps(windows, indent=2))
print(f"{platform}: {len(windows)} windows -> {out}")

# sanity: y1 (sum of per-view fps) and y3 (sum) match the recorded totals
import numpy as np
bad_y1 = bad_y3 = 0
for w in windows:
    s1 = sum(v["throughput_fps"] for v in w["models"].values())
    s3 = sum(v["tokens_per_s"] for v in w["models"].values())
    if abs(s1 - w["total"]["total_throughput_fps"]) > 0.05:
        bad_y1 += 1
    if abs(s3 - w["total"]["total_tokens_per_s"]) > 0.05:
        bad_y3 += 1
print(f"  y1 sum mismatch: {bad_y1}, y3 sum mismatch: {bad_y3} (both should be 0)")

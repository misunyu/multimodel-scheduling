"""B.5 platform comparison: for each working set + rate, find the empirically best
placement (measured-score argmax) on CPU-NPU vs CPU-GPU, and report where the two
platforms disagree on which models to put on the accelerator. Uses the collected
windows (measured y1/y2/y3), not the predictor, so it answers "does the hardware
actually prefer a different layout" — the question the two separate predictors exist for.
"""
import json, sys
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0, "/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
from xgboost_model.deploy_selector_xgb_suite import score_combo

SC = Path("/tmp/claude-1001/-home-msyu-PycharmProjects-multimodel-scheduling-mobilint/"
          "6f0af394-5a71-4f54-bead-adacfbd8c6b9/scratchpad")
ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
SETS = yaml.safe_load(open(ROOT / "working_sets.yaml"))["sets"]
sid_of = {",".join(sorted(v)): k for k, v in SETS.items()}
ALPHA, BETA = 0.3, 0.5
GEN = {"llama1b", "qwen2_vl"}


def load(platform):
    ws = json.loads((SC / "collect" / f"windows_{platform}" / f"performance_{platform}_all.json").read_text())
    rows = []
    for w in ws:
        models = sorted(v["model"] for v in w["models"].values())
        sid = sid_of.get(",".join(models))
        accel = tuple(sorted(v["model"] for v in w["models"].values()
                             if v["execution"].lower() in ("npu", "gpu")))
        has_gen = any(m in GEN for m in models)
        t = w["total"]
        s = score_combo(t["total_throughput_fps"], t["deadline_miss_rate"],
                        t.get("total_tokens_per_s", 0.0) if has_gen else 0.0, has_gen, ALPHA, BETA)
        rows.append({"sid": sid, "rate": w["rate_factor"], "combo": w["combination"],
                     "accel": accel, "score": s})
    return rows


npu = load("npu")
gpu = load("gpu")


def best_by_group(rows):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[(r["sid"], r["rate"])].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = max(rs, key=lambda r: r["score"])
    return out


bn, bg = best_by_group(npu), best_by_group(gpu)
order = ["S1", "S2", "S3", "S4", "base1", "base2", "base3", "base4", "base5",
         "S5", "S6", "S7", "S9", "S8", "S10"]

print("=== B.5 플랫폼별 최적 배치 비교 (측정 점수 argmax, set×rate) ===")
print(f"{'set':6s} {'rate':>4s}  {'NPU 최적 가속기 배치':40s} {'GPU 최적 가속기 배치':40s} 동일?")
n_same = n_diff = 0
diffs = []
for sid in order:
    for rate in sorted({k[1] for k in bn if k[0] == sid}):
        kn, kg = bn.get((sid, rate)), bg.get((sid, rate))
        if not kn or not kg:
            continue
        an, ag = kn["accel"], kg["accel"]
        same = set(an) == set(ag)
        n_same += same; n_diff += (not same)
        if not same:
            diffs.append((sid, rate, an, ag))
        na = ",".join(m.replace("yolo11", "y") for m in an) or "(none)"
        ga = ",".join(m.replace("yolo11", "y") for m in ag) or "(none)"
        print(f"{sid:6s} {rate:>4}  {na:40s} {ga:40s} {'✓' if same else '✗ 다름'}")

print(f"\n동일 배치 {n_same} / 다른 배치 {n_diff}  (총 {n_same+n_diff} set×rate 그룹)")
if diffs:
    print("\n플랫폼이 다른 최적 배치를 고른 그룹 (하드웨어 특성차 → 별도 예측기 필요 근거):")
    for sid, rate, an, ag in diffs:
        print(f"  {sid} @rate{rate}: NPU=[{','.join(an)}]  GPU=[{','.join(ag)}]")

"""repo_status.md §5 예비 계산 (읽기 전용).

P6 예비: 540창 데이터에서 뷰 단위 저하율 target
    r = throughput_fps / min(infps, capacity_fps)
을 만들 재료가 전부 있는지, 뷰별 결측이 있는지, 값 분포를 플랫폼별로 계산한다.
재학습·수집 없음 — 저장된 JSON/YAML만 읽는다.

infps: schedules/collection/collect_cpu_{gpu,npu}.yaml 의 combination별 (model, device) 행
capacity_fps: sample_profiling_data.json 의 s_infer로 1000/s_infer (suite와 동일 정의)
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
sys.path.insert(0, str(ROOT))
import yaml
from xgboost_model.deploy_selector_xgb_suite import (
    _build_infps_lookup, load_static_profiles, _device_static, _norm_exec)

STATIC = ROOT / "xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json"
S = load_static_profiles(STATIC)

for plat in ("gpu", "npu"):
    sched = yaml.safe_load((ROOT / f"xgboost_model/schedules/collection/collect_cpu_{plat}.yaml").read_text())
    wins = json.loads((ROOT / f"xgboost_model/full_collection_540/cpu_{plat}/performance_{plat}_full540.json").read_text())
    ratios, miss_infps, miss_cap, miss_thr = [], 0, 0, 0
    n_views = 0
    llm_ratios = []
    for w in wins:
        infps_map = _build_infps_lookup(sched, w["combination"])
        for v in w["models"].values():
            n_views += 1
            m, dev = v["model"], _norm_exec(v["execution"])
            thr = v.get("throughput_fps")
            infps = infps_map.get((m, dev))
            s_infer, _, _ = _device_static(m, dev, S)
            s_infer = float(s_infer) if np.isfinite(s_infer) else 0.0
            cap = (1000.0 / s_infer) if s_infer > 0 else None
            if thr is None: miss_thr += 1; continue
            if infps is None or infps <= 0: miss_infps += 1; continue
            if cap is None: miss_cap += 1; continue
            # infps in the collection schedule may already bake in rate_factor —
            # verified separately below before interpreting the ratio.
            eff = min(infps, cap)
            r = thr / eff
            (llm_ratios if m in ("llama1b", "qwen2_vl") else ratios).append(r)
    def q(a):
        a = np.array(a)
        return (f"n={len(a)} min={a.min():.3f} p25={np.percentile(a,25):.3f} "
                f"med={np.median(a):.3f} p75={np.percentile(a,75):.3f} max={a.max():.3f}")
    print(f"[{plat}] views={n_views} miss_thr={miss_thr} miss_infps={miss_infps} miss_cap={miss_cap}")
    print(f"  vision ratio: {q(ratios)}")
    if llm_ratios:
        print(f"  llm/vlm ratio: {q(llm_ratios)}")

# infps에 rate_factor가 이미 반영돼 있는지 검증: 같은 set의 rate 1.0 vs 2.0 조합에서
# 동일 (model,device)의 infps 비교
sched = yaml.safe_load((ROOT / "xgboost_model/schedules/collection/collect_cpu_gpu.yaml").read_text())
wins = json.loads((ROOT / "xgboost_model/full_collection_540/cpu_gpu/performance_gpu_full540.json").read_text())
by_combo = {w["combination"]: w for w in wins}
seen = {}
print("\n[infps vs rate_factor 검증] (같은 모델셋, 다른 rate의 infps)")
shown = 0
for w in wins:
    models = ",".join(sorted(v["model"] for v in w["models"].values()))
    key = (models, tuple(sorted((v["model"], v["execution"].upper()) for v in w["models"].values())))
    seen.setdefault(key, []).append(w)
for key, ws in seen.items():
    rates = sorted({w["rate_factor"] for w in ws})
    if len(rates) >= 2 and shown < 3:
        shown += 1
        for w in sorted(ws, key=lambda x: x["rate_factor"]):
            im = _build_infps_lookup(sched, w["combination"])
            ex = sorted(im.items())[:3]
            print(f"  {key[0][:40]} rate={w['rate_factor']} combo={w['combination']} infps(앞3)={ex}")

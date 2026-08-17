"""지시문 #26 Phase A: CPU 배치 vision 모델의 ORT intra-op 스레드 민감도.

원고 App G 는 "pinning the intra-op pool to one thread raises CPU inference latency
by $4.2\\times$ to $10.1\\times$ across the seven vision models" 라고 쓴다. 그 배율은
지시문 #25 Phase 4 에서 측정됐지만 당시 지시가 "파일에 저장하지 말 것" 이어서 산출물이
남지 않았고, 재현 스위트가 대조할 대상이 없었다. 여기서 같은 조건으로 다시 재고 저장한다.

조건 (#25 Phase 4 와 동일):
    ORT CPU EP, FP32, models/onnx/*.onnx 7종, intra_op in {1, 4, 24, default(미설정)},
    warm-up 3회 후 8회 평균. 입력은 모델별 native 크기의 난수 텐서.

재학습·재수집 없음 — 저장된 ONNX 파일의 추론만 한다. 원고·기존 산출물 미수정.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent

# model_registry.py 의 input_size 와 같은 값 — 여기서 재선언하지 않고 읽어 온다
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
import model_registry as reg  # noqa: E402

MODELS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
          "resnet50", "mobilenet_v2"]
SETTINGS = [1, 4, 24, 0]          # 0 = 미설정(런타임 기본값) — 수집 때 실제로 쓰인 설정
WARMUP, ITERS = 3, 8
SEED = 42


def bench(onnx_path: Path, size: int, intra_op: int) -> dict:
    """intra_op 설정 하나에 대한 ITERS 회 지연시간(ms). 0 이면 설정하지 않는다."""
    rng = np.random.RandomState(SEED)
    x = rng.rand(1, 3, size, size).astype(np.float32)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if intra_op:
        so.intra_op_num_threads = intra_op
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    for _ in range(WARMUP):
        sess.run(None, {inp: x})
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter()
        sess.run(None, {inp: x})
        ts.append((time.perf_counter() - t) * 1000.0)
    a = np.asarray(ts, float)
    return {"mean_ms": float(a.mean()), "sd_ms": float(a.std(ddof=1)),
            "min_ms": float(a.min()), "max_ms": float(a.max()),
            "n": ITERS, "samples_ms": [round(v, 4) for v in ts]}


def main() -> None:
    out = {
        "scope": {
            "purpose": "원고 App G 의 intra-op 스레드 배율 주장을 재현 가능한 산출물로 고정",
            "no_retraining": True, "no_recollection": True,
            "provider": "CPUExecutionProvider", "precision": "FP32",
            "models_onnx": "models/onnx/*.onnx (수집 때와 같은 파일)",
            "settings_intra_op": SETTINGS,
            "settings_note": "0 = intra_op_num_threads 미설정. 수집 코드"
                             "(model_processors.py:75-85)가 실제로 쓰는 설정.",
            "warmup": WARMUP, "iters": ITERS, "input": "난수 텐서, RandomState(42)",
        },
        "host": {"python": platform.python_version(),
                 "onnxruntime": ort.__version__,
                 "available_providers": ort.get_available_providers(),
                 "cpu_count": __import__("os").cpu_count()},
        "session_defaults": {
            "intra_op_num_threads": ort.SessionOptions().intra_op_num_threads,
            "inter_op_num_threads": ort.SessionOptions().inter_op_num_threads,
            "execution_mode": str(ort.SessionOptions().execution_mode)},
        "per_model": {},
    }

    for m in MODELS:
        size = int(reg.get(m).get("input_size", 640))
        p = ROOT / "models" / "onnx" / f"{m}.onnx"
        row = {"input_size": size, "onnx": str(p.relative_to(ROOT)), "settings": {}}
        for nt in SETTINGS:
            row["settings"][str(nt)] = bench(p, size, nt)
        d = row["settings"]["0"]["mean_ms"]
        row["ratio_1_over_default"] = row["settings"]["1"]["mean_ms"] / d
        row["ratio_1_over_24"] = (row["settings"]["1"]["mean_ms"]
                                  / row["settings"]["24"]["mean_ms"])
        # capacity = 1000/latency (fps) — feature 로 들어가는 값
        row["capacity_fps"] = {k: 1000.0 / v["mean_ms"] for k, v in row["settings"].items()}
        out["per_model"][m] = row
        print(f"  {m:14} 1={row['settings']['1']['mean_ms']:9.2f} "
              f"4={row['settings']['4']['mean_ms']:8.2f} "
              f"24={row['settings']['24']['mean_ms']:8.2f} "
              f"def={d:8.2f}  ratio={row['ratio_1_over_default']:5.2f}x", flush=True)

    ratios = {m: r["ratio_1_over_default"] for m, r in out["per_model"].items()}
    lo_m, hi_m = min(ratios, key=ratios.get), max(ratios, key=ratios.get)
    out["headline"] = {
        "n_models": len(MODELS),
        "ratio_1_over_default": {k: round(v, 4) for k, v in ratios.items()},
        "min": {"model": lo_m, "value": ratios[lo_m], "rounded_1dp": round(ratios[lo_m], 1)},
        "max": {"model": hi_m, "value": ratios[hi_m], "rounded_1dp": round(ratios[hi_m], 1)},
        "manuscript_cites": {"min": 4.2, "max": 10.1,
                             "sentence": "raises CPU inference latency by $4.2\\times$ "
                                         "to $10.1\\times$ across the seven vision models"},
        "matches_manuscript_1dp": (round(ratios[lo_m], 1) == 4.2
                                   and round(ratios[hi_m], 1) == 10.1),
    }
    # 측정 편차: 각 설정의 상대표준편차 중 최댓값 (배율의 신뢰도 판단용)
    rsd = [(v["sd_ms"] / v["mean_ms"]) for r in out["per_model"].values()
           for v in r["settings"].values()]
    out["headline"]["max_relative_sd"] = float(max(rsd))
    out["headline"]["median_relative_sd"] = float(np.median(rsd))

    (HERE / "thread_sensitivity.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    md = ["# CPU intra-op 스레드 민감도 (지시문 #26 Phase A)", "",
          f"ORT {ort.__version__}, CPU EP, FP32, warm-up {WARMUP} + {ITERS}회 평균. "
          f"`0` = 미설정(수집 때의 실제 설정).", "",
          "| 모델 | intra=1 (ms) | intra=4 | intra=24 | default | 1/default |",
          "|---|---|---|---|---|---|"]
    for m, r in out["per_model"].items():
        s = r["settings"]
        md.append(f"| {m} | {s['1']['mean_ms']:.2f} | {s['4']['mean_ms']:.2f} | "
                  f"{s['24']['mean_ms']:.2f} | {s['0']['mean_ms']:.2f} | "
                  f"**{r['ratio_1_over_default']:.2f}×** |")
    h = out["headline"]
    md += ["", f"최소 **{h['min']['value']:.2f}×** ({h['min']['model']}), "
               f"최대 **{h['max']['value']:.2f}×** ({h['max']['model']}).",
           f"원고 인쇄값 4.2× ~ 10.1× 와 소수 첫째 자리 일치: "
           f"**{'예' if h['matches_manuscript_1dp'] else '아니오'}**.",
           f"측정 편차: 상대표준편차 중앙값 {h['median_relative_sd']*100:.1f}%, "
           f"최대 {h['max_relative_sd']*100:.1f}%."]
    (HERE / "report.md").write_text("\n".join(md) + "\n")

    print()
    print(f"min {h['min']['value']:.3f}x ({h['min']['model']}) -> {h['min']['rounded_1dp']}")
    print(f"max {h['max']['value']:.3f}x ({h['max']['model']}) -> {h['max']['rounded_1dp']}")
    print(f"원고 4.2 / 10.1 과 1자리 일치: {h['matches_manuscript_1dp']}")
    print(f"상대 SD 중앙값 {h['median_relative_sd']*100:.2f}%, 최대 {h['max_relative_sd']*100:.2f}%")


if __name__ == "__main__":
    main()

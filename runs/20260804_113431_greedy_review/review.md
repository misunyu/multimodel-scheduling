# capacity greedy baseline 구현 검토 — VLM load의 누적 반영 시점

- 작성: 2026-08-04
- 대상: `xgboost_model/full_collection_540/scripts/greedy_baseline.py` (+ 실제 호출 경로 전부)
- **읽기 전용.** 코드·데이터·원고 미수정. 산출물은 이 run 디렉토리뿐이다.
- 대조한 원고 리비전: `manuscript/mlforsys_main.tex` **r4** (`REVISION: 2026-07-31 r4`,
  343행, sha256 `d31d17a0…c7af`). 직전 작업들이 인용한 r2에서 갱신되어 있어 행 번호가
  달라졌으므로, 아래 인용은 전부 r4 기준이다.

## 결론

**C — 경로 의존.** 정확히는 **정렬 변형(variant)에 의해 결정론적으로 갈린다.**

| 변형 | qwen2_vl 정렬 위치 | VLM load가 다른 instance 배치 전에 누적에 포함되는가 | 판정 |
|---|---|---|---|
| `speedup` | **60/60 그룹에서 1번째** (GPU 30/30, NPU 30/30) | **예** — 다른 어떤 instance보다 먼저 `cum`에 들어간다 | **A** |
| `load_factor` | 1번째인 경우 **0/60** (GPU 위치 2–6, NPU 1–3) | **아니오** — GPU 1–3개 · NPU 1–2개 instance가 이미 가속기에 배정된 뒤에 들어간다 | **B** |

즉 "고정은 되지만 미포함"(B)도, "처음부터 포함"(A)도 단독으로는 맞지 않는다. VLM은 **사전
고정도 후처리도 아닌 루프 내 예외**로 처리되므로, 그 load가 언제 `cum`에 반영되는지는 전적으로
정렬 키가 VLM을 어디에 놓느냐에 달려 있다.

논문 Table 1의 헤드라인 수치(`.424/.333`)는 **speedup 변형**이므로, 그 수치에 한해서는 A로
동작한다. 부록 B의 `load_factor` 변형(`.341/.250`)은 B로 동작한다.

### 의사코드 (실제 흐름)

```
rows ← 각 model의 (cpu_infer, accel_infer, capacity=1000/accel_infer,
                   load_factor=infps/capacity, speedup=cpu_infer/accel_infer)
order ← sort(rows, key = −rows[variant], tie-break: 모델명 오름차순)
cum ← 0.0;  placed ← []                        # VLM 선반영 없음
for r in order:                                 # VLM도 이 루프 안에서 처리
    if r.model ∈ FORCED_ACCEL:                  # qwen2_vl
        placed += r;  cum += r.load_factor      # 용량 검사 없이 무조건, 누적엔 반영
    elif cum + r.load_factor ≤ 1.0:
        placed += r;  cum += r.load_factor
    else:
        r.decision ← "cpu"                      # skip: 루프 계속, 되돌리기 없음
return sorted(placed), order, cum
```

---

## 확인 항목별 근거

### 1. VLM이 accelerator에 고정되는 시점 — **루프 내 예외**

`greedy_baseline.py:48-51`, 함수 `greedy_place()`:

```python
for r in order:
    if r["model"] in FORCED_ACCEL:
        placed.append(r["model"]); cum += r["load_factor"]
        r["decision"] = "accel (forced)"
```

`FORCED_ACCEL`은 `greedy_baseline.py:30`의 모듈 상수 `{"qwen2_vl"}`. 루프 전 사전 고정
(`placed` 초기화는 `:47`에서 빈 리스트)도, 루프 후 후처리도 없다 — 함수는 `:57`에서 바로
반환한다.

**부수 발견**: `FORCED_ACCEL`은 **하드코딩 리터럴**이며 `model_registry.DEVICE_CONSTRAINTS`
에서 파생되지 않는다. `:30`의 주석과 `:7-8`의 docstring이 `DEVICE_CONSTRAINTS`를 근거로
언급하지만, `greedy_baseline.py`는 `model_registry`를 **import하지 않는다**
(import는 `:17-25`: `sys`, `pathlib`, `numpy`, `yaml`, `analysis_common`,
`deploy_selector_xgb_suite`의 `_device_static`·`load_static_profiles`뿐). 제약이 바뀌면
greedy만 따라가지 못한다.

### 2. cumulative accelerator load의 초기값 — **0.0, VLM 선반영 없음**

`greedy_baseline.py:47`:

```python
cum, placed = 0.0, []
```

VLM load를 미리 빼두거나 더해두는 코드는 함수 전체에 없다.

### 3. VLM load가 cumulative에 더해지는 줄 — **존재함 (`:50`)**

`cum += r["load_factor"]`가 forced 분기(`:50`)와 일반 분기(`:53`) 양쪽에 있다. 따라서
"더하는 코드가 없어서 실질 B"는 아니다 — 더해지되, **더해지는 시점이 정렬 위치에 종속**된다.

### 4. capacity 조건식 원문 — `greedy_baseline.py:52`

```python
elif cum + r["load_factor"] <= 1.0:
```

여기서 `load_factor = infps / capacity_fps` (`:40`), `capacity_fps = 1000.0 / acc_infer`
(`:39`). `acc_infer`가 비유한/0이면 `cap = nan` → `lf = np.inf`(`:40`)라 해당 모델은 항상
CPU로 간다(단, FORCED_ACCEL이면 `cum`이 inf가 되어 이후 전부 CPU). 정렬은 `:45-46`:

```python
key = "speedup" if variant == "speedup" else "load_factor"
order = sorted(rows, key=lambda r: (-r[key], r["model"]))
```

### 5. capacity 초과 시 동작 — **skip (현 instance만 CPU). backtrack·break 아님**

`greedy_baseline.py:55-56`:

```python
else:
    r["decision"] = "cpu"
```

`placed`에서 제거하는 코드도, `break`도 없다. 루프는 남은 instance를 계속 검사한다.

**skip과 break가 실제로 다른 결과를 내는지 실측 확인** (180개 (플랫폼, 변형, 그룹) 셀 중
"CPU 판정 이후에 다시 accel 판정이 나오는" 셀 수):

| 플랫폼/변형 | skip ≠ break 인 그룹 | 종료 시 `cum > 1.0` |
|---|---|---|
| gpu/speedup | **0** / 45 | 0 |
| gpu/load_factor | **26** / 45 | 2 |
| npu/speedup | **15** / 45 | 0 |
| npu/load_factor | **23** / 45 | 18 |

합계 64/180 셀에서 두 의미가 갈린다. 실례 (gpu/load_factor, rate 2.0):

```
mobilenet_v2  cpu            lf=1.2671   ← 용량 초과로 거절
resnet50      accel          lf=0.5384   ← 그럼에도 이후 instance는 계속 배정됨
yolo11s       accel          lf=0.3530
yolo11l       cpu            lf=0.2198
yolo11m       cpu            lf=0.2019
yolo11x       cpu            lf=0.1583
qwen2_vl      accel (forced) lf=0.0433   ← VLM은 7번째에야 cum에 반영
llama1b       accel          lf=0.0026
```

이 예는 §5(skip)와 결론(C)을 동시에 보여준다: `resnet50`·`yolo11s`는 VLM load가 `cum`에
들어오기 **전에** 가속기에 admit되었다.

### 6. GPU/NPU 분기 — **배치 로직에는 없음**

`greedy_place()`는 `accel` 인자를 `_device_static(m, accel, S)` (`:38`) 한 곳에만 쓴다.
capacity 정의식, load 추정식, VLM 처리, 조건식, skip 동작 모두 플랫폼과 무관하게 동일하며,
차이는 정적 프로파일의 **값**뿐이다. `main()`의 `:88`에 `platform == "gpu"` 조건이 있으나
이는 `md_caps`(요약 문서의 생성 모델 capacity 표) 수집용이고 배치에 영향이 없다.

단, 값 차이가 크기 때문에 **결과적 민감도는 플랫폼별로 다르다** (§추가 확인).

---

## 추가 확인 — VLM load 추정치의 출처와 실제 값

**정의되어 있다.** 값의 출처 체인:

1. `analysis_common.py:38` `STATIC_JSON =
   xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json`
2. `deploy_selector_xgb_suite.load_static_profiles()` (`:101-123`)가
   `total_data[]`의 `{dev}_infer` 필드를 `{model: {device: {"infer":…}}}`로 적재.
   해당 키가 없으면 `np.nan`.
3. `_device_static(model, dev, S)` (`:126-133`)가 `(infer_ms, load_ms, tokens_per_s)` 반환.
4. `greedy_baseline.py:39-40`이 `capacity_fps = 1000/accel_infer`,
   `load_factor = infps/capacity_fps`.
5. `infps`는 `main()`의 `:81-84`에서 그룹 첫 combo의 스케줄 infps 맵
   (`ac.infps_map_for(platform)`) 중 모델별 최대값 — rate 배율이 이미 반영된 값이다.

실제 수치 (프로파일 원본 확인):

| 모델 | cpu_infer(ms) | gpu_infer(ms) | npu_infer(ms) |
|---|---|---|---|
| qwen2_vl | **12302.185** | 99.806 | 613.778 |
| llama1b | 154.739 | 9.5 | 81.192 |

→ **qwen2_vl은 CPU 프로파일이 결측이 아니라 실측값(12.3초)을 갖는다.** 따라서
`speedup = cpu_infer/accel_infer`가 유한하며, GPU 123.3 / NPU 20.0으로 **모든 세트에서 최대**다.
이것이 speedup 변형에서 VLM이 항상 1번째로 정렬되는 이유다(추정이나 NaN 폴백이 아니라 실제
값 때문). `:41-42`의 `else np.inf` 폴백은 qwen2_vl에 대해 발동하지 않는다.

VLM의 실제 `load_factor` (전 그룹 관측값):

| 플랫폼 | qwen2_vl load_factor |
|---|---|
| GPU | 0.0217 / 0.0325 / 0.0433 / 0.0650 (rate별) |
| NPU | **0.1332 / 0.2001 / 0.2664 / 0.3996** |

**NPU에서 VLM 혼자 용량의 13–40%를 차지한다.** 따라서 `load_factor` 변형에서 VLM이 뒤늦게
반영되는 문제는 NPU에서 실질적 영향이 크다(먼저 admit된 1–2개 instance가 VLM 몫을 모르는 채
용량을 소진). GPU에서는 2–7%라 영향이 작다. 이는 NPU/load_factor에서 종료 시 `cum > 1.0`인
그룹이 18/45로 가장 많은 것과 정합적이다.

---

## 원고 §3 서술과 코드 동작의 일치 여부

원고 r4 `:122` (`\textbf{Capacity greedy baseline.}`):

> It orders instances by their isolated accelerator speedup and assigns them to the
> accelerator **while the estimated cumulative load remains within capacity**. **Remaining
> instances are assigned to the CPU, except for the VLM, which must remain on the
> accelerator.**

**한 줄 판정: VLM 예외 서술은 코드와 일치하나("except for the VLM" = `FORCED_ACCEL` 분기,
`:49-51`), 같은 문장의 "while … remains within capacity"는 break 의미로 읽히는 반면 코드는
skip이므로 부분 불일치다.**

부연(수정 아님, 보고만): 원고가 기술한 것은 speedup 변형이며, 그 변형에서 skip≠break인 그룹은
GPU 0/45 · **NPU 15/45**다. 따라서 GPU 수치(.424)는 두 해석이 같은 결과를 주지만, NPU
수치(.333)는 skip 구현에서만 재현된다. 또한 원고는 VLM load가 누적에 포함된다는 점을 언급하지
않는데, speedup 변형에서는 VLM이 항상 첫 배정이라 이 사실이 결과를 좌우한다.

---

## 판단에 추가로 필요했던 파일 (전부 읽기만 함, 수정 없음)

| 파일 | 참조 이유 |
|---|---|
| `xgboost_model/full_collection_540/scripts/greedy_baseline.py` | 주 대상 |
| `xgboost_model/full_collection_540/scripts/analysis_common.py` | `STATIC_JSON`(`:38`), `load_dataset`, `infps_map_for`, `GEN_MODELS` |
| `xgboost_model/deploy_selector_xgb_suite.py` | `load_static_profiles`(`:101`), `_device_static`(`:126`), `_build_infps_lookup` |
| `xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json` | `*_infer` 실측치 (VLM load 추정치의 출처) |
| `model_registry.py` | `DEVICE_CONSTRAINTS`(`:45-47`) — 주석이 참조하나 코드는 import하지 않음을 확인하기 위해 |
| `xgboost_model/full_collection_540/analysis/greedy_summary.md` | 기존 산출 수치와의 대조 |
| `manuscript/mlforsys_main.tex` (r4) | §3 서술 대조 (읽기 전용) |

추가로 필요한 파일은 없다 — 위 목록으로 6개 항목과 추가 확인 모두 확정되었다.

## 보고 (수정하지 않음)

1. `FORCED_ACCEL`(`greedy_baseline.py:30`)이 `model_registry.DEVICE_CONSTRAINTS`에서
   파생되지 않는 하드코딩 리터럴이다. 주석·docstring은 파생인 것처럼 읽힌다.
2. 원고 `:122`의 "while … remains within capacity"가 skip 구현과 어긋난다(위 판정).
3. `load_factor` 변형에서 VLM이 늦게 반영되는 구조적 문제 — NPU에서 VLM load가 용량의
   13–40%라 영향이 크다. 부록 B가 두 변형을 비교하지만 이 비대칭은 서술되어 있지 않다.

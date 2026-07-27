# 신예측기 통합 · 랭킹 재생성 · 결정론 재확인 (v22)

작성 2026-07-27. 성격: 예측기 통합 + 랭킹 생성 + 검증. **스케줄 생성·재실행 없음. 커밋만(푸시 없음).**

---

## 작업 91 — 신예측기 통합 경로 (게이트) → **(b) 아티팩트 인터페이스**

### 결정
신예측기(MLA100 3-target 번들)는 **이미 fsrr에 커밋돼 있다**:
`xgboost_model/artifacts/{cpu_gpu,cpu_npu}/deploy_*_{y1,y2,y3,features,coverage}.json`
(`git ls-files`로 추적 확인). 소비 코드는 `deploy_predictor_logic.DeployPredictor`.
따라서 **형제 저장소를 import 하지 않는다**((a) 기각 — 교차 저장소 드리프트가 이번 사고의 부류).
랭킹을 **커밋된 아티팩트로 export**((b))하면 fsrr 저장소만으로 완전 재현·감사된다. 형제 저장소
`../multimodel-scheduling-mobilint`는 **학습 origin**으로만 기록한다.

### 91-2 인터페이스·파라미터
- 호출: `predict_best_combination(schedule_data|schedule_yaml_path, model_input_path, alpha, beta)`
  → `(best, df)`. df per-combo: `pred_norm_throughput(y1), pred_deadline_miss_rate(y2),
  pred_norm_tokens(y3), has_generative, pred_score`.
- 입력 = 워크로드 배치(모델·디바이스·infps). 세 타깃은 **워킹셋 내에서** [0,1] 정규화 → 점수는 한 셋
  내부 순위만 의미. 셋 간 비교 불가.
- 학습 대상(coverage) = **§IV-A와 정확히 일치**: `yolo11n/s/m/l/x, resnet50, mobilenet_v2, llama1b,
  qwen2_vl`. 옛 어휘(resnext50 등) 없음.
- 타깃/파라미터: `S = y1 + β·y3 − α·y2`, **α=0.3, β=0.5**(vision-only는 y3 항 제거).
- **양 플랫폼 별도 번들**: `cpu_gpu`, `cpu_npu`.

---

## 작업 93 — 결정론 재확인 → **유지 (진행)**

### 93-1 피처 벡터 (코드 판독)
`featurize_from_combo → _view_features/_aggregate`가 쓰는 필드 전량:
`infps`(스케줄), `exec_{cpu,gpu,npu}`(디바이스 배치), `is_vision/is_llm`(모델 종류),
`static_infer/load/tokens_sel`(오프라인 프로파일 S), `capacity_fps`·`load_factor`(정적 파생),
`infps×static_infer` 교호항. sum/mean/max/count로 집계.
소스 주석 명시: *"Measured per-view dynamics are NOT features — unavailable at predict time and using
them leaks the target."* → **런타임 상태 없음**(V(t)·큐 점유·관측 지연·시각·난수 전무). 유일한
`np.random`은 학습 데이터셋 split 경로(라인 631)로 추론 경로 밖.

### 93-2 반복 호출
동일 입력 3회 호출, **전체 순위+점수(소수 6자리)까지 동일** — 양 플랫폼:
```
[cpu-gpu] combos=32  3x-identical-order=True
[cpu-npu] combos=32  3x-identical-order=True
```

### 판정 (사전 고정, 첫 칸)
런타임 상태 없음 + 반복 호출 동일 → **Q6 progress 논증 유지.** §II·§IV Q6·§V의 결정론 서술은
신예측기에서도 성립한다.

---

## 작업 92 — 랭킹 생성 (커밋됨)

### 생성물
`rankings/generate_rankings.py`(커밋된 생성기) + 시나리오×플랫폼별 전체 랭킹 JSON 8종
(`rankings/ranking_<scenario>_<platform>.json`). 목록은 **자르지 않고 전체**(N_cand는 사람이 정함).

| 시나리오 | 플랫폼 | 워킹셋 | 후보 수 |
|---|---|---|---|
| Q1_3_steady, Q2_4b_load | cpu-gpu, cpu-npu | fg4 (vision-only) | 16 |
| Q3_mispred_1gen, Q6_ablation_1gen | cpu-gpu | fg4 + llama1b | 32 |
| Q3_mispred_2gen | cpu-gpu | fg4 + llama1b + qwen2_vl | 32 |
| Q5_npu_1gen | cpu-npu | fg4 + llama1b | 32 |

fg4 = **YOLO11s, YOLO11m, ResNet50, MobileNet-v2**(§IV-A foreground 고정).

### 92-2 어휘 검증 (생성 시점)
생성기가 랭킹에 등장하는 전 모델명을 `model_registry`로 해소(v21 작업 86 경로). 미해소 시
`RuntimeError`. **별칭 의존 없음**(v21 작업 87로 부분문자열 별칭 제거 — 정확 이름만 통과). 8종 전부 통과.

### 92-3 provenance (각 아티팩트 `_provenance`)
`predictor_source`(repo·bundle_prefix·**bundle sha256 전량**·training_origin), `trained_on`(9모델),
`targets`(y1/y2/y3, α=0.3, β=0.5, 점수식), `platform`, `workload`(foreground·working_set·대표 rate),
`generated`(2026-07-27), `vocabulary_validated=true`, `n_candidates_total`.

> **rate 주의**: infps/slo는 **문서화된 대표 overload 레짐**이며 **최종 시나리오 rate가 아니다**(최종은
> v21 작업 85 스케줄 생성에서 확정). 랭킹은 rate에 대해 잠정. 단, 작업 94의 구조적 결론은 rate 민감도
> 검사로 견고성 확인(아래).

---

## 작업 94 — N_cand 재검토 입력 (**값은 사람이 정함**)

"회복 배치" = 논문의 회복(경쟁 유발 생성 모델을 가속기에서 CPU로 이전, foreground는 가속기 유지)의
새 랭킹 순위:

| 시나리오 | 플랫폼 | 예측기 #1 (오예측) | 회복 배치 순위 | 회복 위 순위들 |
|---|---|---|---|---|
| Q3/Q6 1-gen | cpu-gpu | llama1b **on GPU** (miss 0.030) | **rank 3 / 32** | r1·r2 모두 llama1b GPU 유지 |
| Q5 1-gen | cpu-npu | llama1b **on NPU** (miss 0.0) | **rank 2 / 32** | r1 llama1b NPU 유지 |
| Q3 2-gen | cpu-gpu | 둘 다 GPU (miss 0.002) | **해당 없음** (아래) | — |

**옛 대비**: 옛 랭킹은 GPU 회복 rank 5 / NPU rank 2 → N_cand=5 정당화. **새 랭킹: GPU rank 3, NPU rank 2.**
NPU 불변, GPU는 5→3. → N_cand 예산 재검토의 직접 근거(값 결정은 사람).

**rate 민감도(견고성)**: foreground rate를 0.25×~4×(16배 범위)로 흔들어도 회복 순위 **불변**
(cpu-gpu 항상 rank 3, cpu-npu 항상 rank 2). 구조적 결론은 rate에 견고.

**구조(옛 워킹셋-크기 의존성 재확인)**: 회복 위의 상위 순위들은 옛 랭킹과 **같은 구조** — 생성 모델을
가속기에 두는 배치들이 상위를 차지. 옛 "생성 모델을 가속기에 + vision 하나를 CPU로"의 사다리가 새
랭킹에도 존재.

**2-gen 특이사항 (새로 드러난 제약)**: `qwen2_vl`은 레지스트리 `DEVICE_CONSTRAINTS`상 **CPU 불가**
(gpu/npu 전용). 따라서 "모든 생성 모델을 CPU로"라는 회복은 2-gen에서 **구조적으로 불가능** — 회복은
`llama1b`만 이전 가능(qwen2_vl은 가속기 상주). 2-gen의 N_cand·회복 정의는 이 제약을 반영해 사람이
재검토해야 한다.

---

## 작업 95 — legacy 예측기 격리

- `deploy_predictor_logic_legacy.DeployPredictor.__init__`에 **loud 경고**(stderr) 추가 — 신랭킹에
  쓰지 말 것, `deploy_predictor_logic` 사용 안내. `FSRR_ALLOW_LEGACY_PREDICTOR=1`로 의도적 재현 시
  침묵(확인 완료: 경고 발화 / opt-in 침묵 둘 다 동작).
- **STALE 표시**: `xgboost_model/artifacts/gpu/STALE.md`(legacy 2-target 번들 = pre-MLA100, 옛 워킹셋,
  신랭킹 금지, 감사용 보존). 옛 `xgboost_model/prediction_result/`는 이전 커밋에서 이미 삭제됨(복원 안 함).
- **옛 시나리오 YAML 보존 확인**: `tests/`에 ml_misprediction_*, bounded_recovery, dynamic_load,
  qos_recovery, steady_state 전부 존재. 삭제 안 함 — v21 fail-fast가 이 옛 어휘 스케줄을 실행 시
  중단시킨다(보존 + 무해화).

---

## 커밋 (푸시 없음)

커밋 대상: `rankings/`(생성기 + 8 아티팩트), `deploy_predictor_logic_legacy.py`(guard),
`xgboost_model/artifacts/gpu/STALE.md`, 본 리포트. `git ls-files rankings/` 출력은 커밋 절에 기재.
**푸시하지 않음**(§0.3 — 사용자 명시 시에만).

---

## 하지 않은 것 / 다음 결정 (사람 몫)

- **N_cand 값 결정** — 재료만 제공(GPU rank 3, NPU rank 2, rate 견고, 2-gen 제약).
- **스케줄 생성(v21 85)** — 랭킹 확정 후. 최종 rate·활성 background·옛→신 대응은 사람 확인.
- **재실행** — 범위·예산 확정 후.
- **푸시** — 커밋까지만.

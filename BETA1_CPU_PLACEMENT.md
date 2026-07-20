# β = 1.0 통일 + β=1.0에서 CPU 배치 조합 분석

> PART 1(수정): 점수식 β를 0.5 → **1.0**으로 변경(양 플랫폼 동일). PART 2(조사): β=1.0에서
> top-1이 CPU에 무엇을 두는지 본수집 540창 × 2 + 15세트 예측으로 확정.
> 점수식(변경 후): 생성 조합 `S = y1 + 1.0·y3 − 0.3·y2`, vision-only `S = y1 − 0.3·y2`(불변).

---

## PART 1 — β = 1.0 변경 (완료)

### 변경 위치 (β 하드코딩 전수)
| 파일:줄 | 변경 | 역할 |
|---|---|---|
| `deploy_selector_xgb_suite.py` (score_combo 앞) | **`DEFAULT_BETA = 1.0`** 상수 신설 (+ `DEFAULT_ALPHA=0.3`) | 정책값 단일 소스 |
| `deploy_selector_xgb_suite.py:767` | `--beta default=0.5 → DEFAULT_BETA` | CLI predict 경로 |
| `best_deploy_finder_executor.py:763` | `predict_best_combination(..., beta=0.5 → 1.0)` | **GUI 스코어링(양 플랫폼 공용)** |
| `evaluate_model.py:190` | `--beta default=0.5 → 1.0` | 오프라인 평가 |

- **양 플랫폼 커버**: `predict_best_combination`은 플랫폼-무관(β는 스칼라, 플랫폼은 로드하는
  `_y*.json`만 다름) → 이 한 곳 변경이 cpu_npu·cpu_gpu 양쪽에 동일 적용됨.
- **재학습 없음**: β는 런타임 스코어링 가중치. y1/y2/y3 예측기(.json) 무변경.
- **y3 마스킹 불변**: `combo_has_generative`/`with_y3` 로직 그대로 — LLM/VLM 없는·마스킹 조합은
  여전히 y3 항 제외. β 값만 0.5→1.0.
- α = 0.3 유지.

### 검증
- **손계산 (S3, β=1.0)**: top-5 전부 `y1 − 0.3·y2 + 1.0·y3`가 화면 score와 정확 일치.
  예) combination_8: y1=0.7642, y2=0.0196, y3=0.9974 → **1.75573** (@0.5였다면 1.257). **y3 항이 1.0배.**
- **vision-only 불변 (S1)**: β=1.0과 0.5의 score 동일(y3 항 없음) — combination_4 = 0.99899 양쪽.
- **기본값 확인**: `predict_best_combination` 기본 best(S3) = combination_8 = β=1.0 top-1
  (β=0.5의 combination_4 아님) → **기본 스코어링이 β=1.0로 동작.**

---

## PART 2 — β=1.0에서 CPU 배치 조합 분석

### 1. β 0.5 → 1.0 으로 top-1이 바뀐 세트
| 플랫폼 | 바뀐 세트 | 이전→새 top-1 | LM/VLM 배치 변화 |
|---|---|---|---|
| CPU-NPU | **S3** | combination_4 → 8 | **llama1b: cpu → npu** (핵심) |
| CPU-NPU | base3 | combination_6 → 8 | 불변 (llama·qwen 이미 npu; vision 재배치) |
| CPU-NPU | S5 | combination_12 → 16 | 불변 (llama·qwen 이미 npu; vision 재배치) |
| CPU-GPU | (없음) | — | — |

→ β=1.0의 실질 효과는 **S3에서 llama1b를 CPU에서 NPU로 이동**. base3·S5는 top-1이 바뀌었으나
LM/VLM은 이미 가속기였고 vision 배치만 재편(β 영향의 부수효과).

### 2. β=1.0에서 top-1이 CPU에 배치하는 모델 (전 세트 표)
| 세트 | 플랫폼 | CPU 배치 모델 | vision/LM | top-1 y1/y2/y3 |
|---|---|---|---|---|
| base4 | NPU | **resnet50** | vision | 0.87 / 0.30 / 0.91 |
| base5 | NPU | **resnet50** | vision | 0.95 / 0.11 / 0.92 |
| (그 외 13세트) | NPU | (없음) | — | — |
| (전 15세트) | GPU | **(없음)** | — | — |

- **LM/VLM이 CPU에 남은 세트: 0개 (양 플랫폼).** β=1.0에서 llama1b·qwen2_vl는 **항상 가속기**.
- CPU에 남는 건 **NPU의 base4·base5에서 resnet50(vision)뿐**. GPU는 CPU 배치 전무.
- qwen2_vl: 디바이스 제약({npu,gpu})으로 애초에 CPU 불가 — 전 세트 가속기 확인.

### 3. LM/VLM이 CPU에 남은 강한 케이스
**없음.** β=0.5에서 유일하게 llama를 CPU에 두던 S3가 β=1.0에서 NPU로 이동. β를 2배로 올리면
LM을 CPU에 두는 배치는 어떤 세트·플랫폼에서도 top-1이 아니다. (LM QoS 가중이 vision 처리량
이득을 넘어섬.)

### 4. β=1.0에서 CPU 배치(vision)의 이유 분해 — base4/base5(NPU)
- top-1 배치: **resnet50 → CPU**, llama1b·qwen2_vl·yolo11(l/x) → NPU.
- 의미: 무거운 yolo11l/x + 생성모델(llama,qwen)을 NPU에 두고, **CPU에서도 빠른 resnet50을
  CPU로 오프로드**해 NPU 경합을 줄인다. y3=0.91~0.92(LM 토큰 높음) 유지하면서 y1=0.87~0.95
  (vision 처리량)도 확보. **vision(y1) 이득이 이 배치를 top-1으로 만들며, LM은 가속기에 있어
  y3 손실이 없다.** → 정상적 vision 오프로드(문제 케이스 아님).

### 5. 플랫폼별 CPU 배치 패턴 (β=1.0) + B.5 재검토
- **CPU-NPU**: LM/VLM 전부 가속기. vision(resnet50)만 base4·base5에서 CPU. (경미)
- **CPU-GPU**: 전 모델 가속기, CPU 배치 0.
- **B.5 발견("NPU는 llama1b를 CPU에, GPU는 가속기에")의 변화**:
  - β=0.5: NPU S3가 llama를 CPU에 두는 유일 잔존 케이스였음(그마저 `LM_DEADLINE_AUDIT`에서
    B.5 원본은 raw-fps 스코어링 산물로 판명).
  - **β=1.0: 그 마지막 잔존(S3)도 사라짐 — NPU도 llama를 NPU에.** 따라서 **B.5의
    "NPU=llama on CPU" 패턴은 β=1.0에서 완전히 소멸.** 남는 NPU/GPU 차이는 vision 오프로드
    (NPU가 base4/5에서 resnet50을 CPU에) 정도로, LM 배치 차이는 없다.

---

## 요약

- **PART 1**: β 0.5→1.0, 단일 상수(`DEFAULT_BETA`)+3개 기본값, 양 플랫폼 동일, 재학습·마스킹 불변,
  손계산·기본값 검증 완료.
- **PART 2**: β=1.0에서 **LM/VLM은 어디서도 CPU에 남지 않는다.** CPU 배치는 NPU의 vision
  오프로드(resnet50, base4/5)뿐이고 GPU는 CPU 배치가 없다. β=1.0은 B.5의 "NPU CPU-llama"
  패턴을 완전히 제거한다.

### 산출 파일
- 코드: `deploy_selector_xgb_suite.py`(DEFAULT_BETA=1.0 + CLI), `best_deploy_finder_executor.py`,
  `evaluate_model.py`. 예측기(.json)·데이터 무변경.
- 분석: `<scratchpad>/beta1_analysis.py` (predictions.csv는 실행 후 원복).

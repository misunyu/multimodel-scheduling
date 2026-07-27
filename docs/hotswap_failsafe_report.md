# hot-swap fail-safe 화 (v19) — 작업 72 게이트 감사: **정지·보고**

작성 2026-07-27. 상태: **게이트에서 정지.** 코드 미변경(읽기만 수행). 작업 73~78 미착수.

---

## 요약 (한 문단)

v19 게이트(작업 72)는 "FATAL/fail-open 경로가 보고된 런에서 실제 발동한 적이 있는가"를 물었다.
**발동이 있다** — `results/` 725개 요약 중 6개 레코드에서 fail-open 증상(먹였는데 죽은 뷰, tp=0·
count=0·drops>0)이 관측됐다. 그러나 감사 도중 **더 근본적인 문제**가 드러났다: 그 증상을 낸 모델
(`mnasnet`, `resnext50`)을 포함해 **평가 스케줄이 참조하는 다수 vision 모델이 런타임에서 로드 불가**
(레지스트리 미등록 + 가중치 파일 부재)다. 런타임이 실제로 추론한 vision 모델은 `resnet50`과
`yolov4`(→`yolo11s` 별칭)**뿐**이며, 나머지는 전 런에서 **단 한 번도 추론을 만들지 못했다(alive=0)**.
사전 고정된 게이트 규칙("발동 있음 → 정지·보고, 영향 범위 먼저 판정")에 따라 여기서 정지한다.

---

## 작업 72 — 감사 상세

### 검색 대상
- `results/` 요약 JSON **725개** 전수(논문 수치 산출 런의 실제 출력물).
- `backup/**/*.log` 6개(pre-MLA100 stdout 캡처).
- 소스 내 FATAL 문자열 위치.

### 캡처 한계 (중요)
보고 실험 스크립트(`ml_misprediction_validation.py` 등)는 executor stdout를 `subprocess.PIPE`로 받아
**메모리에서 파싱만** 하고 디스크에 남기지 않는다(`results/`엔 `.json`만, stdout 로그 없음). 따라서
보고 런의 FATAL **문자열**은 영구 저장돼 있지 않다. → 문자열 대신 **증상**을 스캔했다:
전환/기동 후 **먹였는데(drops>0) 추론이 0(tp=0, count=0)** 인 뷰 = 죽은 워커가 계속 급식받는 상태.

### 관측: fail-open 증상 (vision)
`throughput_fps==0 ∧ inference_count==0 ∧ dropped_frames>0` 인 **vision** 뷰:

| 파일 | 스케줄 | combo | 뷰 | 모델 | dev | drops |
|---|---|---|---|---|---|---|
| performance_20260727_000038 | ml_misprediction_runtime_bg | combination_fallback | view1 | mnasnet | GPU | 3879 |
| performance_20260727_000038 | ml_misprediction_runtime_bg | combination_fallback | view2 | resnext50 | GPU | 3879 |
| performance_20260727_000547 | ml_misprediction_runtime_bg | combination_fallback | view1 | mnasnet | GPU | 3897 |
| performance_20260727_000547 | ml_misprediction_runtime_bg | combination_fallback | view2 | resnext50 | GPU | 3897 |
| performance_20260727_000736 | bounded_recovery_views | combination_offload | view1 | mnasnet | GPU | 2645 |
| performance_20260727_000736 | bounded_recovery_views | combination_offload | view3 | resnext50 | GPU | 878 |

- 6건 모두 **2026-07-27** = v18 계측 재실행분(관찰자 사이드카 부착). 게시 논문 그림의 출처 런은 아니다.
- **같은 GPU에서** `resnet50`·`yolov4`는 정상 추론(alive), `mnasnet`·`resnext50`만 정확히 0 →
  경합 아사(starvation)가 아니라 **모델별 결정적 로드 실패**. fail-open 경로의 서명.
- 생성계(`llama1b`/`qwen2_vl`) fed-but-dead 57건은 **구조적**(background 하위프로세스라 vision
  metric에 추론이 안 잡힘) — fail-open 아님, 집계에서 제외.

### 근본 원인 (감사 중 발견)
`mnasnet`, `resnext50`, `shufflenet-v2-12`, `squeezenet1.0-12`, `vgg19`는:
- `model_registry.MODELS`에 **미등록** — `reg.get()`이 `KeyError` (직접 확인).
- git 이력상 레지스트리에 **등재된 적 없음**(`git log -S` 공집합), onnx/mxq 파일도 **추적된 적 없음**.
- 디스크에 가중치 **부재**: `models/onnx/`엔 `resnet50`+`yolo11{n,s,m,l,x}`+`mobilenet_v2`만,
  `models/mobilint/`도 동일 집합. 이 5개 파일은 어디에도 없다.
- 이 이름들은 **XGBoost 예측기의 오프라인 프로파일/스코어 데이터**
  (`xgboost_model/prediction_result/...`)에만 존재. 즉 예측기 데이터셋의 모델명을 런타임 스케줄에
  그대로 옮겨 적었으나 런타임 실행 경로가 없다.
- `yolov4`가 도는 이유: `_normalize()`가 `"yolo"` 부분일치로 **`yolo11s`로 별칭** 치환하기 때문.
  `resnet50`은 실존.

전 런 통틀어 이 5개 모델의 `inference_count>0` 레코드 = **0** (alive=0):

| 모델 | alive | dead(count=0) | fed-but-dead | 관측 device |
|---|---|---|---|---|
| mnasnet | 0 | 3 | 3 | GPU |
| resnext50 | 0 | 3 | 3 | GPU |
| shufflenet-v2-12 | 0 | 2 | 0 | CPU |
| squeezenet1.0-12 | 0 | 2 | 0 | CPU |
| vgg19 | 0 | 2 | 0 | CPU |

### 영향 범위 — phantom 모델을 참조하는 스케줄
`tests/`에서 로드 불가 모델을 참조하는 YAML(런타임이 실제로 추론 가능한 것은 `resnet50`, `yolov4`뿐):

- `bounded_recovery_views_schedule.yaml` — mnasnet, resnext50 (§4b 회복 시나리오)
- `ml_misprediction_runtime_bg.yaml` / `_runtime_ml.yaml` — mnasnet, resnext50, shufflenet-v2-12, squeezenet1.0-12, vgg19 (Q3)
- `ml_misprediction_boundguard_runtime.yaml`, `_candidates.yaml`, `_mlonly_runtime.yaml`, `_rtvy_schedule.yaml`, `_runtime.yaml`
- `dynamic_load_views_schedule.yaml`, `steady_state_views_schedule.yaml`, `qos_recovery_schedule.yaml`,
  `adaptive_test_schedule.yaml`, `model_schedules_test.yaml`, 루트 `model_schedules.yaml`

---

## 왜 정지하는가

1. **게이트 규칙(사전 고정).** "발동 있음 → 정지·보고. 해당 런의 수치가 fail-open 경로를 탄 결과일 수
   있다. 영향 범위를 먼저 판정한다." — 발동을 확인했으므로 규정대로 정지.
2. **v19 수정만으로는 근본 문제가 안 풀린다.** fail-safe 화는 "로드 실패 시 옛 배치 유지 + 후보 기각"을
   보장하지만, 이 스케줄들에선 **애초에 로드 가능한 모델이 2개뿐**이라 전환 대상 자체가 phantom이다.
   fail-safe를 넣으면 phantom 후보가 "즉시 기각"될 뿐, 스케줄이 의도한 다중 모델 워크로드는 재현되지
   않는다.
3. **판단이 사람의 몫이다.** 게시된 Q3/§4b 수치가 이 런타임 실행에서 나온 것인지, 아니면 XGBoost
   예측(랭킹) 쪽에서 나온 것인지에 따라 영향이 완전히 갈린다. 이는 저장소만으로 단정할 수 없다.

---

## 결정이 필요한 갈림길

- **(A)** 게시 Q3/§4b 수치가 **XGBoost 예측/랭킹** 산출이고 런타임 실행은 보조라면 → phantom 모델은
  예측기 데이터셋에선 정상(프로파일 존재)이라 **게시 수치는 무탈**. 이 경우 v19 fail-safe만 진행하면 된다.
- **(B)** 게시 Q3/§4b 수치가 **런타임 실행 측정**(실제 V(t)·회복시간)이라면 → 대부분의 뷰가 죽은 채
  측정된 것이라 **수치 재검토가 필요**. 이건 v19 범위를 넘어서는 문제다.

**하지 않은 것:** 코드 수정, 스케줄/레지스트리 수정, 재실행. 전부 사용자 지시 대기.

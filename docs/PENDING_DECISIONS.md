# 사람 판단 대기 목록 (20260721_205708)

돌아오셨을 때 이 목록만 보면 다음 결정이 가능하도록 정리합니다.
각 항목: 무엇을 / 왜 멈췄나 / 관측 데이터 / 선택지.

## [C-1] 용어: "drop rate" vs "deadline miss" (논문 판단)
- 무엇: active 코드에 `drop_rate_fps` 메트릭 필드 다수. drop rate(큐-full 드롭)와 deadline miss는 별개 지표.
- 왜 멈춤: 이름을 바꾸면 필드명·다운스트림 파싱·기존 CSV 호환이 깨질 수 있고, 논문 용어 정합은 사람 판단.
- 데이터: drop_rate_fps는 unified_viewer 메트릭 CSV/results JSON의 필드. B2에서 drop은 miss의 성분으로 이미 집계됨(정합).
- 선택지: (a) 그대로 유지(권장 — drop은 miss의 부분집합, 별개 지표로 유효) / (b) 필드명 리팩터(호환성 위험, 비권장).

## [C-2] 휴면 legacy 데이터 (수정 불필요, 참고용)
- utils.py:175 `_LEGACY_TO_BASE`(resnet50_small/big, yolov3): 신규 경로는 reg.get 사용, 이 매핑은 dormant·무해.
- reactive_deploy.py:60 `_MODEL_GPU_MEM_MB`: 신규 9모델 추가됨 + 구 항목 보존(이미 문서화). 무해.
- scripts/q13·generate_comparison: legacy 재현 경로(신규 coverage 밖, 의도적 미이전).
- 조치: 없음. 제거는 legacy 재현 스크립트 영향 있어 사람 판단.

## [C-3] methods_and_runbook.html 부재
- 지시문 작업 C의 runbook 갱신 대상 파일이 저장소에 없음. 새로 만들지 여부는 사람 판단(SKIP).
## [B] Q5 CPU-NPU 재실험 — 시나리오 정의 필요 (SKIP)
- 무엇: Q5 그림을 단일 MLA100 신규 데이터로 재측정.
- 왜 멈춤: Q5의 정확한 시나리오(워킹셋·λ·페이즈 구조·infeasible 조건)를 모름. 임의 시나리오를 Q5로 라벨링하면 오염.
- 이미 확보된 CPU-NPU 4기법 데이터(재사용 가능):
  - `docs/phaseA_report.md` + phaseA CSV: 4기법 × cpu_npu V(t) 궤적(Static 18.4 / StopRestart 0 / Adaptive 2.5 / BoundGuard).
  - `docs/phaseA_boundguard_report.md`: BoundGuard cpu_npu bounded persistence(mode 1 + validate, 확정).
  - `docs/figures/c3_b2_metrics.pdf`(NPU 패널): 4기법 miss rate·drop/late·p99/p999.
- 선택지: (a) Q5 시나리오 정의를 주면 그 파라미터로 재측정+그림 / (b) 위 기존 데이터로 Q5 그림 구성(시나리오가 phaseA/B2와 부합하면). 사람이 Q5 정의를 확인해야 진행 가능.
## [A] Background LLM/VLM 격리 — 통합 결정 (SKIP, 조사·측정만 완료)
### 자동 완료(참고 데이터)
- mobilint background 코드: `runtime/llm_engine.py`(LLMEngine, transformers), `run_concurrency_study.py`, `model_processors.py`(run_llm/vlm_process). GPU=transformers, NPU=mobilint W8 체크포인트(mobilint/Llama 2.2G, mobilint/Qwen 3.2G 캐시됨).
- **GPU DRAM 예산**: GPU 32GB. llama1b 풋프린트 **2866 MiB**, vision ~1-2GB. 별 프로세스 공유 시 여유(qwen 추정 ~5-6GB 포함해도 32GB 내). → GPU background는 DRAM 제약 아님.
- **NPU DRAM**: smi 툴 부재로 직접 측정 불가(/dev/aries0만). maccel/mblt API로 조회해야 하나 조사 필요.
### SKIP한 판단 (사람 필요)
1. **Qwen2-VL conv3d cuDNN 우회**: 호환 cuDNN 설치 vs cuDNN 비활성화(현재 840ms). 방향 판단.
2. **vision(onnxruntime GPU) + LLM(torch GPU) 공존 = 프로세스 분리 필수**(cuDNN 충돌, 확정). background를 실제 실험 파이프라인에 통합하려면 워커 프로세스 분리가 선행 — 이건 실행모델 변경(되돌리기 어려움)이라 무인 금지. 통합 설계는 사람 확인.
3. **NPU DRAM 초과 시 모델 수 조정**: NPU DRAM 측정 방법(maccel API) 확보 후 vision+LLM mxq 동시 적재 가능 여부 판정 필요.
### 무회귀
- 격리 러너를 실제 파이프라인에 붙이지 않았으므로 vision GPU/NPU/BoundGuard 무영향. background 통합은 SKIP.

## [Q3/Q5] 오예측 시나리오 — **재검증 완료(2026-07-22): Q3 자연 발생 성립** ⇒ mixed셋 구현 정당화
- **⚠️ 갱신 (docs/llm_rate_misprediction_check.md)**: 아래 A/B 판정은 **틀린 축**(vision을 CPU로 이동)만 봐서 나온 것. **LLM을 CPU로 이동**하는 대안을 보면 결론이 뒤집힘:
  - **§3 예측 오차 = 있음**: 예측 y2=0.000 vs 실측 vision miss 0.23(λ60)~1.00(λ120). 학습이 저 vision-λ + 저 LLM율 수집(§1: LLM 0.19 req/s 단일스트림, rate_factor는 vision만) → 고λ·고율 경합 과소예측.
  - **§4 LLM-CPU 대안 = 있음(압도적)**: λ=120에서 (a)LLM-GPU vision miss=1.000(p50 5초 발산) vs (b)LLM-CPU miss=0.472(p50 8ms). 대안이 명백히 우수.
  - **예측기는 (a)를 (b)보다 높게 랭크**(S 1.473 vs 0.764): βy3 항 +0.314(최대, LLM 토큰 보상=QoS 무관) + y2 **반전**(예측기가 LLM-GPU miss를 더 낮다고 오판, GPU 경합 미모델링).
  - **판정 매트릭스: 오차있음 + 대안있음 → Q3 자연 발생 성립.** 원인 = 목적함수 불일치(y3 보상 vs vision-only QoS). **→ mixed 실행 구현 착수 정당화, Q3 실험 진행 가능.** (GPU 확정, NPU는 LLM 적재 선행 필요로 후속.)
- ~~**전제 불성립(구축)**~~ ↓ 아래는 vision-이동 축만 본 초기 판정(보존):
- **감사 결과 (docs/predictor_signal_audit.md + docs/contention_alternative_check.md, 측정/검사만)**:
  - **A. y2 신호 = 있음**: 학습 y2(deadline miss) 분포 0.15~1.0, median 0.86, zero비율 0%(degenerate 아님). 예측기가 실제로 사용, 학습 시 LLM-vision 경합 동시측정 반영. α 33× 스윕에도 top-1 불변은 신호부재가 아니라 all-GPU가 y1·y2 동시 지배 때문.
  - **B. 더 나은 대안 = 없음**: 2× LLM 최대 경합에서도 경합-GPU 처리율 > CPU 처리율(모든 vision 모델). 경계 mobilenet조차 268 vs 199=1.35× GPU 우위. "판정불가" 아님.
  - **판정 매트릭스(사전 고정)**: A=신호있음 + B=대안없음 → **예측기 정확, Q3 오예측 전제 불성립.** 고λ QoS위반(V=342)은 배치오류가 아니라 용량 문제(모든 배치가 함께 나쁨, GPU가 덜 나쁨).
  - **결정**: **혼합셋 대규모 구현을 Q3 오예측 시연 수단으로 진행하지 않음.** 오예측을 만들려면 학습분포 밖 이질 경합 또는 CPU가 실제 유리한 초경량/저해상도 모델 필요 — 현 모델셋엔 없음.
- (이하 기존 조사 기록 보존)
- 무엇: BoundGuard의 distinctive 우위(top-1 실패 → 대안 회복)를 Q3(cpu_gpu)·Q5(cpu_npu)로 실증.
- 왜 멈춤: **vision-only에선 자연 오예측(top-1 mis-rank)이 구조적으로 안 생김.** 예측기 top-1이 항상 all-accelerator(진짜 최적), 실패 모드는 용량 contention=Q4-like(대안 없음)이지 Q3 아님. 조작 금지 원칙(§0.1)으로 억지 실패 안 함.
- 데이터: docs/q3_q5_misprediction_report.md (coverage 가드/미학습조합/런타임 V=342 확인).
- 선택지:
  (1) **Mixed 셋(LLM/VLM)** = authentic Q3/Q5(LLM이 가속기 점유→vision CPU mis-rank). **프로세스 분리 선행 필수**(cuDNN). 후속.
  (2) **Controlled injection** = top-1 강제 대체, 논문에 injection 명시. 설득력 약함(대안 자명).
  (3) **NPU 단일-상주 제약 runtime 구현** = Q5 자연 오예측용. 실행모델 변경.
- 권장: (1) mixed 경로가 정공법이나 프로세스 분리 선행 필요. 어느 경로로 갈지 사람 판단.

## [Mixed] LLM/VLM background 실행 — **구현 완료(GPU, 2026-07-22)** ⇒ Q3 다리 확보
> **구현 결과 (docs/mixed_exec_report.md)**: GPU background를 격리 subprocess로 구현·검증 완료.
> - 신규 `runtime/{llm_engine,bg_entry,background_llm}.py` + `schedule_executor_main.py`에 `--background` 배선. **기본 OFF=완전 no-op**(기존 결과 무회귀 확인).
> - **거친 전환** 동작: LLM GPU→CPU 전환 시 GPU mem 3893→900MiB 해제 + **vision 98→220fps(2.2x) 회복** end-to-end 실증. → **§7 대안 도달 가능 확정, Q3 실험 다리 확보.**
> - **NPU 정정**: 이전 "런타임 미설치"는 **import 이름 오류**(`mobilint_qb_runtime`❌ → `qbruntime`✅). NPU LLM은 이 박스에서 **실행 가능 확인**(W8 .mxq를 aries0에 5.8s 적재+생성). DRAM 공개 API 없음→경험적. **Q5(NPU)는 실현 가능한 후속.**
> - **남은 사람 판단**: (1) Q3 실험 착수 시 후보 시퀀스에 LLM-CPU 후보 포함 방식(예측기 top-1은 LLM-GPU만 반환), (2) Q5 NPU background 구현, (3) qwen2_vl 추가(현재 llama1b만).
> ↓ 아래는 구현 전 설계 노트(보존):
### 확인된 findings (측정)
- **간섭 실증 (GPU)**: vision yolo11s 단독 5.12ms → LLM(llama1b) background 프로세스 실행 중 **11.11ms (2.2x 저하)**. 별 프로세스라 vision 프로세스에 torch 미유입(cuDNN 격리, GPU 정상). GPU mem 4128MiB(둘 다 적재).
- **오프라인 예측기**: mixed 셋(vision+LLM/VLM)에서도 top-1이 **항상 all-accelerator**(vision CPU 잔류 0), miss=0 예측. 예측기가 **가속기 용량/경합을 모델링 안 함** → mis-rank는 오프라인이 아니라 **런타임에 발현**(all-GPU가 LLM 경합으로 실패, vision-CPU가 더 나은 대안일 수 있음). ← Q3 구조의 씨앗.
### 설계 (§0, 확인 필요)
- LLM/VLM = **고정 배치 background**(hot-swap/QoS/ready_event 통합 없음). vision만 hot-swap. V(t)는 vision만.
- **GPU**: 별도 프로세스(검증됨). **NPU**: .mxq/HF 동시 적재(qbruntime 경유).
### 사람 판단 필요
1. **구현 착수 여부**: background 실행을 실험 파이프라인에 통합(프로세스 러너 + 생명주기 + NPU LLM 포팅). 대규모 코드 변경(vision 워커·디스패치·QoS는 불변).
2. **NPU background 범위**: DRAM 16GB에 vision(≤4) + LLM/VLM 동시 적재 가능한지 미측정(qbruntime DRAM API 모듈명 불명확). 둘 다 vs llama1b만 → 경험적 적재 시도 필요. 착수 시 측정.
3. **NPU LLM 경로**: fsrr엔 vision NPU(mblt_model_zoo.vision)만 있음. LLM NPU는 mobilint llm_engine(transformers+npu_hf, trust_remote_code, single-core)을 포팅해야 — 신규.
4. **Qwen2-VL**: conv3d cuDNN mismatch로 non-cuDNN 경로(GPU). background라 성능 최적 불필요.
### 권장
- 간섭이 실증됐고(2.2x) 오프라인 예측기가 경합을 모델링 안 하므로 **Q3 런타임 mis-rank 실험 가치 있음**. 다만 구현이 대규모라 착수 확인 필요.

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

## [Q3/Q5] 오예측 시나리오 — **Q3 실험 완료(2026-07-22): 자연 발생 실증** / Q5 게이트
> **Q3 완료 (docs/q3_q5_misprediction_report.md + figures/q3_misprediction.pdf)**: 예측기 진짜 top-1=gggg(all-GPU incl LLM)가 고부하(λ=80)에서 실패, 4기법 비교 — Static/Stop-restart/Adaptive는 V≈4 지속(persist 78-79s, 회복✗), **BoundGuard만 예측기 top-5 순회 후 cand_5=cggg(LLM→CPU)에서 V=0.54<ε 회복**(persist 35s). injection 아님(후보 조작 없음), D5(mode1+validate) 확인. **단 cand_5=제로마진 경계**(candidate_ranking_check.md).
> **Q5 게이트**: NPU co-load feasible 확인(vision .mxq + llama1b .mxq가 aries0 co-resident 성공). 인프라 준비됨(bg_entry --device npu). 전체 Q5 실험은 NPU 전용 λ 튜닝 필요 → 이전 GPU-only 결정 범위 밖, **착수 승인 대기**.
> ↓ 이하 사전 검증 기록(보존):
## [Q3/Q5-분석] 오예측 시나리오 — 재검증(2026-07-22): Q3 자연 발생 성립

> **[2026-07-24 후속] Q5(CPU–NPU) 완료** → `docs/q5_experiment_report.md` + `figures/q5_npu_generalization.pdf`.
> NPU background LLM 실증(vision3+llama1b 동시 상주, DRAM 여유), cpu_npu 예측기 top-1=nnnn 실패·LLM-CPU
> 대안 cnnn이 랭킹 2위, BoundGuard가 cand_2에서 회복(lastV 0.00, persist 15s) vs baseline 114–158s 고착.
> 메커니즘·유계성·δ(~2.9s)가 GPU와 동일 → **일반화 성립**. **논문 Q5 본문의 "NPU 단일 상주·reload" 서술은
> MLA100과 불일치 → 교체 필요**(하드웨어 절은 이미 정정됨).
- **⚠️ 갱신 (docs/llm_rate_misprediction_check.md)**: 아래 A/B 판정은 **틀린 축**(vision을 CPU로 이동)만 봐서 나온 것. **LLM을 CPU로 이동**하는 대안을 보면 결론이 뒤집힘:
  - **§3 예측 오차 = 있음**: 예측 y2=0.000 vs 실측 vision miss 0.23(λ60)~1.00(λ120). 학습이 저 vision-λ + 저 LLM율 수집(§1: LLM 0.19 req/s 단일스트림, rate_factor는 vision만) → 고λ·고율 경합 과소예측.
  - **§4 LLM-CPU 대안 = 있음(압도적)**: λ=120에서 (a)LLM-GPU vision miss=1.000(p50 5초 발산) vs (b)LLM-CPU miss=0.472(p50 8ms). 대안이 명백히 우수.
  - **예측기는 (a)를 (b)보다 높게 랭크**(S 1.473 vs 0.764): βy3 항 +0.314(최대, LLM 토큰 보상=QoS 무관) + y2 **반전**(예측기가 LLM-GPU miss를 더 낮다고 오판, GPU 경합 미모델링).
  - **판정 매트릭스: 오차있음 + 대안있음 → Q3 자연 발생 성립.** 원인 = 목적함수 불일치(y3 보상 vs vision-only QoS). **→ mixed 실행 구현 착수 정당화, Q3 실험 진행 가능.** (GPU 확정, NPU는 LLM 적재 선행 필요로 후속.)
- **후보 순위 확인 (docs/candidate_ranking_check.md, 2026-07-22, 계산만)**: BoundGuard $N_{cand}=5$(q13 `df.head(5)`, 점수순). Q3 세트(llama1b+vision3)에서 **LLM-CPU 최고 순위 = 정확히 5위**(cand_5). 2~4위는 전부 "LLM=GPU 유지+vision 하나 CPU"(y3 보존). → **경계 사례 = 판정 불가**: 기술적으로 예산 이내지만 cand_5(마지막)에서 제로 마진 회복이고, vision 5개면 6위로 밀려 도달 불가(워킹셋 크기 의존). **사람 판단**: N_cand 근거 재검토 / QoS 기준 순위 재정렬 / "ML 배치 구조적 위험"으로 서술 — 후보 조작은 금지.
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

## [Q4-1] ~~전 후보 실패 시 fallback 부재~~ — **해결됨 (2026-07-24, best-so-far 복귀 구현)**
> 옵션 (b) 관측 최선 배치로 복귀를 채택·구현 → `docs/best_so_far_report.md`. 서비스율(완료 프레임/초, 포화
> 무관) 기준으로 최선 후보 선택, backlog 보존 hot-swap으로 복귀. Q4: BoundGuard 743→55(baseline 74–76
> 아래), (iv) 충족. bound = T+N_cand(T_v+δ)+δ. Q3·§4b 무회귀(commit 먼저라 복귀 미발동). 남은 것:
> 논문 §III 서술(알고리즘 추가). 아래는 결정 당시 기록(보존).
- **무엇**: 예측기 top-N 후보가 **모두** 검증 실패했을 때, 현 구현은 C1 규약(`cand_tail = 마지막 후보
  연장관찰`)에 따라 **마지막 후보 위에서 실행이 끝난다**. 원배치나 관측 최선으로 복귀하는 단계가 없다.
- **왜 멈춤**: fallback 추가는 알고리즘 변경(파라미터 불변 원칙 밖)이라 사람 판단.
- **데이터** (`docs/q4_experiment_report.md`): 결정적 과부하(heavy-4, λ=90)에서 전 후보 infeasible.
  BoundGuard는 12.0s 안에 다섯 후보를 모두 정당하게 기각(**탐색 유계 bound 20s 충족**)했으나, 종료
  배치가 사전측정 최악 후보(cand_5=gggc, V=688)여서 **최종 V=735.8 — 아무것도 안 한 Static(384)의 약 2배**.
  → **bound는 탐색 *시간*만 보장하고 종료 배치 *품질*은 무보장**임이 실증됨. Q4 성립조건 (iv) 불충족.
- **선택지**:
  (a) 전 후보 실패 시 **원배치 복귀**(rollback) — mode 4에 유사 경로 존재.
  (b) 전 후보 실패 시 **탐색 중 관측된 최선 후보로 복귀** — `V_postswap`을 이미 기록하므로 추가 비용 0.
  (c) C1 규약 변경: 마지막 후보도 검증 대상에 포함하고 실패 시 (a)/(b) 적용.
  (d) 현행 유지 + 논문에 "탐색 유계, 종료 배치 무보장"으로 한정 기술.

## [Q4-2] vision-4 paper-set으로 Q3/Q4를 보이려면 부하 재설계 필요 (2026-07-24)
- **무엇**: paper-set(yolo11s/11m/resnet50/mobilenet_v2, per-model `L_SLO`=5×Table I)은 λ=52에서
  **네 기법 모두 위반 미진입**(persist=0s, maxV 0.19–0.40). 실측 지연이 SLO보다 한 자릿수 낮다.
- **왜 멈춤**: λ 상향은 실험 조건 변경이라 사람 판단. 조건을 원하는 그림이 나올 때까지 반복 조정하는 것은
  금지 사항이므로 **한 번에 정하고 그대로 보고**해야 함.
- **데이터**: `docs/hotswap_buffer_bias_fix.md` §6c. 과거 이 시나리오의 `V≈7.5–13.8`은 `ℓ_i` lifetime
  누적평균 아티팩트였고, Δ-창 수정 후 소멸. 그림 `q3_paperset_q4like.pdf`는 `_RETIRED`로 폐기.
- **선택지**: (a) paper-set은 §4b(feasible) 전용으로 쓰고 Q3/Q4는 heavy-set으로 보고(현행) /
  (b) paper-set λ를 상향해 재설계 / (c) `L_SLO` 배수(현 5×)를 낮춤.

## [B2-1] ~~hot-swap의 상속 backlog 폐기 semantics~~ — **해결됨 (2026-07-24, 옵션 a 채택)**
> 선택지 (a) 이관을 채택·구현·검증 완료 → `docs/backlog_preserve_report.md`. 실측 `moved=754 dropped=0`.
> 아래는 결정 당시 기록(보존).
- **무엇**: `adaptive_deploy._hot_swap_view()`가 교체 시 옛 입력 큐를 **폐기**(`self._drain(old_frame_q)`)하는데,
  `unified_viewer.stop_execution()`(mode 0/3/4)은 입력 큐를 **의도적으로 보존**한다(코드 주석에 명시:
  "draining them here would discard exactly the backlog we want the next phase to see").
  → 적응 기법(mode 1/2)만 배치 변경 때마다 **무상 큐 플러시**를 받는다.
- **왜 멈춤**: hot-swap 정책 변경이라 알고리즘 semantics 결정. **B2 재실행을 막는 게이트** — 정하지 않고
  재실행하면 큐 깊이만 맞춘 채 이 편향이 남아 또 무효가 된다.
- **데이터** (`docs/buffer_bias_impact_audit.md` §4–§5): B2에서 전환 직후 `view1_q`가 Static/StopRestart는
  745–754 유지, Adaptive/BoundGuard는 **2→0 붕괴**. 폐기 규모 ~700프레임 ≈ λ=123 기준 **5.7초 분량**
  (cand_1 구간 18s 대비 지배적). mean e2e 7773ms → 147ms(53배)의 실체.
- **선택지**: (a) **상속 backlog를 새 큐로 이관**(mode 0과 일치, 권장) / (b) mode 0도 폐기하도록 통일
  (unified_viewer의 명시 원칙을 뒤집고 C3 근거와 충돌) / (c) 현행 유지 + 논문에 "hot-swap은 stale frame
  폐기 정책"으로 명시하고 Stop-restart에도 동일 정책 부여.

## [B2-2] 논문 `tab:b2`·`fig:b2`·Q1.5 수치 무효 — **재실행 완료, 서술 판단 대기 (2026-07-24)**
> 재실행 완료 → `docs/backlog_preserve_report.md` §4a-2(시간대응, 정본)·§5.
> **남은 사람 판단**: Q1.5 서술 방향. miss rate 격차는 소멸(0.963/0.964/0.968), Static 대비 p99 5배
> 우위는 유지, Stop-restart 대비는 tail 약 18%만 남는다. 대비 축을 "적응 vs 비적응"에서
> "재배치 vs 고정"으로 옮길지 결정 필요. "24s→5s" 서술은 폐기 대상.
> 아래는 재실행 전 기록(보존).

## [B2-3] buffer 스윕(β_B∈{1.5,1.0,0.5}) 완료 — 지표 선택 판단 (2026-07-24)
> 논문 §IV가 약속한 스윕 이행 완료 → `docs/b2_buffer_sweep_report.md` + `docs/figures/b2_buffer_sweep.pdf`.
> **결과**: miss rate는 세 레짐 전체에서 분해능 없음(24셀 중 23셀이 1.000; deadline 8ms가 burst backlog
> 형성 후 구조적으로 도달 불가). 분해능은 **p99/mean**이 제공하며 서열 `Static≪Stop-restart≲Adaptive≈
> BoundGuard`가 6개 (레짐×플랫폼) 셀 전부에서 유지. β_B는 초과분을 drop(Static)/late(적응)로 낼지만 바꿈.
> **남은 사람 판단**: (a) B2 표의 주 지표를 miss→p99로 교체하고 3레짐 제시(권장), (b) 0.5만 본문·나머지
> 부록. 함께 명시: Static은 무손실 레짐에서도 drop 지속(CPU-only 구조적), tight buffer가 miss엔 역설적
> 유리(§3.3).

## [B2-4] deadline을 Table I L_SLO로 정합 — **완료 (2026-07-24)**
> B2 deadline이 폴백(1000/infps=8.13ms)이던 것을 논문 Table I의 L_SLO(YOLO11s=31.0ms)로 정합하고
> 3레짐 재실행 → `docs/b2_slo_rerun_report.md`. **정성 결론 전부 불변**(miss 포화·p99 분해능·서열·
> BoundGuard≈Adaptive). 유일 변화: 0.5×-GPU 적응 기법 miss 0.97→0.86(8–31ms 밴드 ~11% on-time 재분류).
> p99/mean은 deadline 독립(실측 확인). 코드 수정 2건(unified_viewer·adaptive_deploy의 slo_ms 전달 누락).
> `b2_buffer_sweep.pdf`·`b2_metrics.pdf`는 SLO 데이터로 재생성. → [B2-3]의 판단은 그대로 유효.
- **무엇**: B2는 `FSRR_FRAME_BUFFER`=754(GPU)/569(NPU) + mode 1 hot-swap 조합이라 큐 깊이 편향과
  backlog 폐기 편향을 **둘 다** 맞았다. 두 편향 모두 적응 기법에만 유리하게 작용한다.
- **영향**: `tab:b2` 8행 전부 / `fig:b2` / Q1.5의 "adaptive가 tail을 24s→5s로 낮춘다" 서술이 **무효**.
  (Q1.5의 §4b "no penalty" 주장은 buffer=2라 **유효** — 두 주장을 분리할 것.)
- **무영향 확인**: C3/`fig:fluid_validation`/Q1.4는 전 실행 mode 0·hot-swap 0회로 **유효**.
- **재실행 계획**: `[B2-1]` 결정 후 `c3/b2_run_stay.sh` 조건 그대로 8런 ≈ 12분. 이어서 Q4(11분)·Q3(9분)
  재검증. 예상 방향은 격차 축소이나 잔존 규모는 실행 전 단언 불가.


## [S4-1] §IV 잔여 그림·재실행 완료 — 3개 판단 대기 (2026-07-24)
> `docs/section4_remaining_report.md`. A부(재생성): Fig1 qos_score_validation(**ε=1.0 수정**),
> Fig3 bounded_recovery(**새 bound T+N_cand(T_v+δ)+δ**), Fig6/7 Q1.3(baseline 4기법). B부(재실행):
> Q1.1 runtime_overhead(양 플랫폼, **오버헤드 무시**), Q1.2 detection_sensitivity(**새 V(t) 스윕**),
> Q2 dynamic_load_adaptation(§4b, 예측정합 부하변동).
> **사람 판단 (갱신)**: (1) ~~Q1.3 6조건 스윕~~ → **재실행 안 함(확정)**: 메시지 불변, 우위서사는 Q3 담당. (2) **Q1.2 — 새 V(t)에서 T=1~2가
> T=3보다 검출지연·오탐 우수**(현 T=3도 안전범위, 변경 시 전 실험 파급), (3) Q2 background 변형 추가.
> 셋 다 현 설정으로 결론 유효 — 최적성/완전성 결정.
>
> **[Stop-restart V(t) 판정 추가]** 코드·CSV 확정: 재시작 downtime에서 **아티팩트 미발생**(서명 0건, V=0 샘플은 fps 1400+·backlog 0 = 진짜 회복). Q1.3 본문의 "measurement artifact" 서술 **삭제 대상** — Stop-restart는 정당한 비교 대상. 코드는 전 뷰 동시 다운 시 V=0 잠재 거동(a) 보유하나 이 데이터선 미발현(재시작<1s, T창 평활).

## [Qwen-1] Qwen2-VL background 추가 (2-gen Q3/Q5) — **완료: 케이스 B (2026-07-24)**
> `docs/qwen_background_report.md` + `figures/q{3,5}_*_2gen.pdf`(기존 1-gen 그림 보존).
> **결과**: 생성모델 둘(llama1b+qwen2_vl)이면 **회복배치(둘 다 CPU)가 예측기 랭킹 12위**로 밀려 N_cand=5 밖.
> 예측기가 β·y3를 생성모델마다 보상 → 이중보상. BoundGuard는 top-5 소진 후 best-so-far 복귀(유계),
> baseline 이하로 영향 축소하나 **회복 못 함**(NPU lastV 24.9 vs baseline 54–60; GPU 3.39, 온건). = **Q4 성격**.
> DRAM 여유 확인(vision3+llama+qwen 동시 적재). 코드: `bg_entry.py`(VLM 더미프레임+GPU cuDNN 비활성).
> **논문 반영**: workload 서술을 "실험별 활성 background 명시"로, 2-gen은 한계 실증으로 병기. 1-gen Q3/Q5 불변.
> **완화책(향후)**: N_cand 확대 또는 β를 활성 생성 job 수로 정규화.

## [Rep-1] 반복 측정 (B2 Q3/Q5 · B3 §4b/Q1.3) — **완료 (2026-07-24)**
> `docs/repetition_report.md`. **Q3·Q5 회복 5/5**(회복 후보 매번 동일: Q3 cand_5, Q5 cand_2; 랭킹 안정),
> T_valid Q3 24.6±0.5s·Q5 14.2±0.4s. **Q1.3 4기법×2플랫폼 5회**: Static 59.4±0.5s(cumV ~530) ≫ 회복 3기법
> 9–12s(cumV 76–91), 회복 3기법 오차범위 내 구별 불가. BoundGuard 판정경로 5회 일관(cand_1 commit).
> 이상치 없음 — 단일 런 결론이 반복으로 확증. **논문 반영**: Q1.1·Q3·Q5·Q1.3 수치를 평균±SD로 교체,
> q13·runtime_overhead 그림에 오차막대. 서술 불변(회복 유지, Static≫나머지).

## [C2-1] reactive baseline ablation (2×2) — **완료: A2 뒷받침+정밀화 (2026-07-25)**
> `docs/c2_reactive_baseline_report.md` + `figures/c2_reactive_comparison.pdf`. BoundGuard의 2×2 ablation
> (dwell×progress): A=no-dwell, B=re-invoke, C=hybrid. 예측기 결정성 코드 확인(정적 순수함수). 3시나리오×5회.
> **결과**: BoundGuard 세 조건 전부 property 유지(§4b·Q3 5/5, Q4 유계). **dwell 제거(A)는 §4b(1/5)·Q3(3/5)
> 양쪽 실패**(고분산 thrashing) → 보편 필수. **progress 제거(B·C)는 §4b 5/5·Q3 0/5** → 오예측 한정 실패
> (재호출이 같은 top-1만 줌). **A2 반증 아님 — 두 메커니즘 역할이 조건별로 분리되어 강화됨.**
> **논문 반영**: A2를 2×2로 제시, "dwell=관찰없는 선택 불안정 방지(보편), progress=오예측 대안탐색(예측기
> 결정적이라 재호출만으론 불가)"로 정밀화. 코드 변경: no-dwell 경로 best-so-far, auto-advance FSRR_NO_AUTOADVANCE 게이트.
## [C2-2] C2/Q6 데이터 무결성 (지표통일·대조·∑δ·정본화) — **완료: 조건부 확정본 (2026-07-25)**
> `docs/c2_integrity_report.md` + `metric_definitions.md`. 작업 0~4 완료. **질적 2×2 결론은 정정 재실행으로
> 전부 재확인.** 무결성 결함은 수치·라벨 차원(main.tex 재작성 불요).
> **발견 1 — §4b config 오류**: C2 리포트 §4b가 burst=all-accel(feasible→위반 없음)이라 "A 대조군 실패"가
> duration-순회 아티팩트였음. burst=all-CPU로 수정 재실행 → **같은 결론(A 1/5·hotswaps 13 폭주)을 맞는 이유
> (feasible offload 조기 이탈)로** 재확립. 나머지 5/5.
> **발견 2 — B≡Adaptive**: 결정적 예측기 하에서 re-invoke=top-1 적용=Adaptive baseline. B는 2×2의 "둘 다 제거"
> 모서리=baseline(격자 일관). 이전 B "persist 63"은 **censored 런길이 아티팩트** → "미회복(=baseline)"으로 정정.
> **발견 3 — Q3 수치 3중 불일치 해소**: main.tex 20.0=persist·c2 18.8=persist·정정본 21.2=persist(전부 정합,
> SD내) vs repetition **24.6=T_valid(탐색, 다른 양)**. persist(회복)와 T_valid(탐색)를 한 표에 섞지 말 것.
> **정본: 정정 매트릭스 Q3 persist 21.2±3.2s**(고정 120s·단일 지표·5회).
> **∑δ 실측**(공칭값 미사용): vision in-place ~50–180ms, LLM 거친 전환 ~2.9s. A는 vision churn 2.4배(3367 vs
> 1396ms). Σδ_bg로 BoundGuard/A만 LLM전환(회복시도) 확증.
> **논문 반영**: (1) §4b 수치를 정정 매트릭스로 교체, (2) B를 "≡Adaptive"로 라벨·persist 삭제, (3) Q3에서
> persist/T_valid 분리 표기. 코드: `adaptive_deploy.py` δ 타임스탬프, `c2_fixed_scenario` 고정길이·§4b 수정.

## [C2-3] C2/Q6 무결성 v2 (런길이 게이트·배치분산·완성표) — **완료: 확정본 (2026-07-25)**
> `docs/c2_integrity_report_v2.md`. 후속 지시문(미해결 2건+결측치). 게이트 통과, Q6 §7 완성표 확정본.
> **작업 5 (게이트)**: "T_run=120 고정"은 실제로 적용된 적 없었음 — 런 길이가 최종 held combo 착지에 따라
> 변형별 **32s** 산포. strict t_r 정의상 회복 판정 오염. **수정(2층)**: 실행측 고정 윈도우(`FSRR_HARD_WINDOW_S`,
> 첫 cand에서 `_fire_window_end` 독립 QTimer) + 분석측 공통 평가창(`FSRR_EVAL_WINDOW`). dur 32s→0s, 75런 전량
> 통과, 그림 마커 정렬, Q4 cens 균일(~99). **정직 정정 2건**(둘 다 이전값이 런길이 아티팩트): C hybrid Q3
> **1/5→0/5**(progress 필요성 강화), A no-dwell §4b **1/5→4/5**(dwell은 "대조군 실패방지"가 아니라
> 신뢰성·thrashing 억제; A는 §4b 불안정 회복+thrash 5.8배, Q3 실패).
> **작업 6 (배치분산)**: BoundGuard Q3 persist 25런(5배치×5회). **배치 효과 없음**(F=0.64<1, σ²_between≈0,
> 배치평균 산포 1.4s). 이전 "배치 간 6.4s 산포"(18.8/20.0/21.2/25.2)는 진짜 배치효과 아니라 런길이+persist/
> T_valid 혼동 아티팩트. **정본 persist = 23.6±2.3s(관측 최대 28, 최소 15, 이상치 미폐기)**. 경미한 첫-런
> 워밍업 ~2s(within SD 이내), 써멀 신호 없음.
> **작업 7**: 5변형×3시나리오 완성표 결측치 전량 충전(Q3 A/B/C lastV, §4b 전열, Q4 lastV, no_op·skip_guard, Σδ).
> **논문 반영(사람 판단)**: (a) Q3 persist 20.0 → 정본 **23.6±2.3(또는 max 28)** 갱신, (b) §IV "dwell 대조군도
> 실패" → "대조군 불안정·과churn, 오예측 실패"로 수정, (c) B를 "≡Adaptive"로 명기, (d) persist/T_valid 분리.
> main.tex 재작성(3행) 불필요. 코드: `schedule_executor_main.py`·`metrics_lib.py`·`c2_fixed_scenario.py`.

## [C2-4] censored persist 논문 전면 감사 (v3) — **완료: main.tex §IV 표기 결함 확정 (2026-07-26)**
> `docs/c2_integrity_report_v3.md`. 재실험 없음(기존 로그 판독+통계). v2가 C2에서 찾은 결함("미회복 변형의
> persist는 거동이 아니라 런길이")이 **논문 헤드라인에도 존재**함을 확인.
> **작업 8 (게이트) = 표기 결함 확정**: main.tex §IV baseline persist가 전부 censored — Q3 Static/SR/Adaptive
> **105–106s**(lastV 2.8, 0/5회복, censdur≈dur), Q5(lastV 33), Q1.3 Static **59.4s**(lastV 9.0). 회복시간(BoundGuard)과
> 런길이(baseline)를 나란히 놓음 + 서로 다른 윈도우(baseline 107s vs BG 83s) 비교. **"105 vs 20"·"59.4 vs 9–12"
> 배수 대비 불성립**(런길이÷회복시간). 단 "baseline 회복못함 vs BoundGuard 회복"은 참·더 강함.
> **main.tex 표기 수정 목록(사람 적용, 재실험 금지)**: baseline persist → **"회복 없음"(lastV>ε, ≥창길이)**;
> Q1.3 "5–6×" 삭제 → "Static 회복못함/3기법 11–13s 회복"; 각 실험 창 길이 명시; §IV 표 대비 축을 persist →
> **회복여부+lastV**로 이동(Q6 표도 연동).
> **작업 9-1**: 정본 persist **단일 확정 = 23.6±2.3s**(고정윈도우 30회, 최대28). §7의 21.4는 매트릭스 단일배치
> 부분집합. main.tex 20.0은 hot-swap 간격 누락 과소보고(wall-clock strict는 원본도 25.4).
> **작업 9-2**: §4b cens 107은 t0가 t_inject보다 7s 선행(§4b stable 자체가 위반)이라 발생, SD 0은 결정적(정상).
> **작업 10**: 전 6배치 ANOVA **F=1.55<2.62** → 유의 배치효과 없음(work6만 0.64보다 상승). 주장을 "동일 세션
> 6배치 간 미관측"으로 좁힘. 15s(matrix rep2)는 정상 빠른 회복(유지). 정본 SD=풀링 2.3s.
> **사람 판단**: (a) §IV baseline persist를 "회복없음"으로 전면 교체, (b) Q3 persist 20.0→23.6±2.3(max 28),
> (c) Q1.3 배수 표현 삭제, (d) §IV/Q6 표 대비 축 이동. main.tex 재작성 아님(표기 정정).

## [C2-5] §IV wall-clock 재계산·상한 여유 (v4) — **완료: bound 생존, Q3 레이블 반전 발견 (2026-07-26)**
> `docs/c2_integrity_report_v4.md`. 재실험 없음(로그 재판독+통계). v3 작업9-1이 밝힌 "전환 간격 누락→persist
> 과소보고"의 파급 범위 확정.
> **작업 11 (게이트) = 정합, 상한 위반 없음**: Q4 첫위반→reversion = **24s = bound 24s(tight)**, c2 5회 22–23s.
> **샘플 간격 누락 = 0s**(Q4는 background LLM 없음 → 거친 전환 없음 → sample-gap 미발생). 작업 11-1 우려
> (누락→28s 초과) **미발현**. **§II bound 서술 불변.**
> **작업 12/13**: 같은 런(b2rep)으로 재계산 시 **part≤whole 성립·분해 정확**. v3 "부분>전체" 모순은 하네스 혼용
> +레이블 반전 아티팩트. **Q3 정본: persist(t0→회복)=25.4±1.4, validation(cand1→회복)=19.4±1.4, 분해
> 25.4=6.0+19.4.** main.tex 720 "persist 20.0/T_valid 24.6"은 **레이블 반전**(20.0=validation, 24.6=persistence).
> bound~26 비교 대상은 persistence 25.4(여유 +0.6). **v3의 23.6 단일정본 철회** → 정본 25.4(b2rep, §IV 내부 정합),
> 23.6은 C2 교차재현(Threats).
> **작업 14**: Q5·§4b는 stable 단계가 이미 위반(Q5 stable V=6.9>ε) → t0가 t_inject보다 6.8s 앞섬. main.tex Q5
> 14.2는 burst 기준(정확). 본문에 stable 위반 미기재 → 문장 초안 제시(v4).
> **사람 판단**: (a) Q3 persist/T_valid 레이블·수치 정정(25.4/19.4), (b) Q4 24s tight 유지(정본 확인), (c) Q5/§4b
> stable 위반 각주 추가, (d) v3 §8-4 censored 표기와 통합 적용. §II bound 불변.

## [C2-6] 레이블 반전 코드검증·Q3 런별 상한·Q5 T_detect (v5) — **완료: v3/v4 권고 4건 자기정정 (2026-07-26)**
> `docs/c2_integrity_report_v5.md`. 재실험 없음(로그·코드 판독). v4가 남긴 3건을 닫고 §IV 수정 목록 확정.
> **작업 15 (게이트) = 반전 아님**: main.tex 720의 persist 20.0·T_valid 24.6 산출 코드(`b2_analyze.py`) 판독 —
> persist=`V>ε 샘플 개수`(작업9-1 과소집계), T_valid=`burst→회복후보 적용`(결정적, SD 0.5). 내 25.4(t0→V≤ε,
> SD 1.4)와 SD가 달라 다른 양. **v4 "레이블 교체" 권고 철회.**
> **작업 16 = 상한 위반 없음**: bound는 탐색(회복후보 적용까지) 경계인데 persist(=탐색+배수)를 비교한 게 범위
> 오류. **탐색(t0→적용) max 23s < 상한 24–26s, 전 35런 이내.** b2rep 외견상 초과는 vision δ 미로깅(under-count).
> **v3 "28s>26" 우려 해소**(28s는 persist).
> **작업 17 = 가설 A**: Q5 T_detect 18.6s는 감지 지연 아니라 고정 스케줄(stable+burst)+stable 자체 위반(V≈7>ε
> 처음부터). 감지 정상(cand_2 3.1s validate) → STOP 아님. **t0 기준 21.0 미사용(v4 철회), burst 기준 14.2만.**
> **작업 18 = Q1.3 "5–6× 삭제" 철회**: 6배는 누적위반 ∫V(531 vs 85)이고 Static 창이 더 짧아 보수적. 삭제 대상은
> persistence 59.4 하나뿐.
> **§IV 확정 목록**(사람 적용): Q3 720 레이블 유지+persist를 샘플수로 명시·bound 비교는 탐색(≤23s)로; Q4 24s
> tight 유지; Q5 burst기준 유지+stable 각주; Q1.3 5–6× 유지+"창 짧음에도"·59.4만 "회복없음"; baseline persist
> censored. **§II bound 불변.** 자기정정 4건 명시.

## [C2-7] 샘플 기반 지표 편향 감사·∫V 검증 (v6) — **완료: §IV 최종 확정본 (2026-07-26)**
> `docs/c2_integrity_report_v6.md`. 재실험 없음. v5가 찾은 `persist=샘플개수` 편향의 파급을 같은 분석 파일
> 전체로 확대 감사.
> **작업 19**: b2_analyze/b3_aggregate에서 **샘플 기반 편향 지표는 persist 하나뿐**. cumV(∫V)=시간가중(dt),
> T_valid/T_stable=wall-clock, lastV/maxV=점값. persist 편향은 **전환 변형에만**(Q3: BoundGuard −5샘플,
> Static/SR/Adaptive 0) → "20 vs 105" 대비 분자만 축소(편향 방향 확정).
> **작업 20**: Q1.3 6배는 ∫V(시간가중, 무편향). 창 정합(58s)에도 ~5.7×, full 6.2×. Static 창이 더 짧아 보수적.
> **6배 유지** + "Static 창 더 짧음에도" 추가.
> **작업 21**: (1) persist 20.0(샘플수)→wall-clock **25.4 = search 22.6 + drain 2.8**(정확 분해). 교체안/유지안
> 문장 초안 제시(사람 택일). (2) v5 "b2rep 7런" 오기 정정 → **7/35 = b2rep 5 + C2 2**(paper 상한 대비 0건 초과).
> (3) Q5 문장을 하네스 제약→"실험 설계상 burst에서 controller 활성"으로 정정.
> **§IV 최종 확정 목록**(사람 적용): Q3 persist 교체/유지 택일·상한비교는 search≤23s; Q4 24s tight; Q5 burst기준
> +stable 각주; Q1.3 6배 유지+"창 짧음에도"·59.4만 회복없음; baseline persist censored. **§II bound 불변.**

## [C2-8] Q5 출처확인·Q1.3 wall-clock·수치 확정표 (v7, 최종) — **완료 (2026-07-26)**
> `docs/c2_integrity_report_v7.md`. 재실험 없음. §IV 전 구간을 같은 규약(wall-clock)으로 통일하고 copy-ready
> 수치 확정표 완성.
> **작업 22**: Q5 774 "15.2"=샘플수 persist(pre-burst stable 위반 포함 과다), "14.2"=burst→적용(wall-clock). Q5는
> cand_2 즉시 회복(drain≈0)이라 **수렴=T_valid=14.2±0.4 단일값**. 15.2 폐기(샘플카운트 아티팩트).
> **작업 23**: Q1.3 회복 3기법 wall-clock **11–13s**(SR 10.8–12.6·Ad 11.0–11.8·BG 11.2–12.0). BG/Ad 누락0(샘플=wc),
> SR만 −2s. 샘플 "9–12" 대체. Static 미회복→"회복없음".
> **작업 24**: v6 Q3 wall '114'는 full-CSV(stable 포함); from-burst 106.8/106.8/107.0/83.0(v3 정합). 누락 결론 불변.
> **작업 25 = 수치 확정표**(§IV copy-ready): Q3 persist 25.4±1.4(=search22.6+drain2.8)·T_valid 24.6·상한 search≤23
> vs 24–26·baseline 회복없음; Q4 24s tight; Q5 14.2 단일값·baseline 회복없음; Q1.3 회복 11–13s·Static 회복없음·
> ∫V 6.2/6.3×(보수적); Q5/§4b stable 위반 문장; 관측창·t0표; Q6 완성표(A §4b 4/5·C Q3 0/5·Adaptive·Σδ).
> **§II bound 불변.** 이 표에 없는 수치는 논문에 넣지 않음.

## [C2-9] 그림 재생성 — 확정표 정합·censored 규약 (v8) — **완료 (2026-07-26)**
> `docs/figure_regeneration_report.md`. 재실험 없음(기존 로그 재플롯). 본문(v7 §25 확정표)과 어긋난 그림 정정.
> **재생성 5종**: q13_failure_persistence(wall-clock 11–13 + Static censored 해칭), q3_misprediction(x축
> 샘플인덱스→wall-clock, 회복 ~28s), q5_npu_generalization(x축 wall-clock, converge=T_valid 14.2), 
> bounded_recovery_analysis(Q5 16→14.2·Q3 drain 5→2.8·Q4 censored revert), q13_cumulative_violation(∫V 6.2/6.3×).
> **censored 규약**(전 그림 동일): 해칭 open 막대 + ↑ + "no recovery(lastV)" + 관측 창 명시. 궤적은 open square.
> 대안(축을 회복여부+종점V로) 장단 병기, 사람 판단.
> **핵심 산출물 = 사이드카+자동대조**: `fig/confirmed_values.json`(단일 소스), 그림별 `.values.json`, 
> `fig/check_figures.py`(확정표 대조, 불일치시 exit1). **이번 대조 13값/5사이드카 불일치 0건.** 본문·그림 
> 불일치(반복 실패모드) 재발 자동 감지.
> **작업 30**(c2 2패널화)은 선택이라 미착수. 코드(main app) 무변경, 그림·플롯스크립트만.

## [C2-10] 그림 정정 — 기준점 통일·checker 강화 (v9) — **완료 (2026-07-26)**
> `docs/figure_fix_report.md`. 재실험 없음. 제공 파일(confirmed_values.json·check_figures.py) 그대로 사용.
> **작업 31**: bounded_recovery_analysis 축을 detection onset으로 통일. Q3 막대 24.6(burst)→**22.6(t0)** →
> 스택 22.6+2.8=25.4가 bound 26 아래로(이전 27.4로 선 관통). Q5 14.2(burst, stable 위반), Q4 24=24 tight.
> 24.6은 그림에서 제거(본문 부가설명 전용).
> **작업 32**: checker를 값+quantity+reference+unit+censored+window_s 동시 대조로 강화, 필드누락=실패. 음성
> 테스트로 "값 맞고 기준점 틀림(24.6/burst)" 검출 확인. emit_sidecars.py가 confirmed를 figure로 필터링해
> 사이드카 생성(값 복사 → 구조적 보장).
> **작업 33/34**: q13_cumulative 범례·문구(비문) 정정(값 불변), q13_failure 라벨 캡 위로, q3 캡션 "rep 3 of 5".
> v8 plot 스크립트가 옛 confirmed 구조 참조라 q13/q3를 flat 제공본 id 조회로 재작성.
> **최종 대조: 31값/5사이드카 0 mismatch·0 warn.**
> **남은 사람 판단**: v8 작업27 대안(축을 회복여부+종점V로) 미결 — Static 막대 58.4 높이가 여전히 "5배"로
> 읽힐 수 있음. 이번 라운드 결정 안 함.

## [C2-11] 그림 정정 — censored/recovered 분리·사이드카 캔버스화 (v10) — **완료 (2026-07-26)**
> `docs/figure_fix_report_v10.md`. 재실험·값변경 없음. 제공 파일(recovered 필드 추가본) 사용.
> **작업 35/36**: censored(값이 창에 잘림)와 recovered(회복 여부) 독립 분리. Q4 search=24는 측정완료(censored
> false)·미회복(recovered false) → 채운 막대+테두리(해칭 제거)로 24=bound 24 tight 정확. 범례 통일:
> "censored(value truncated at window,≥window)" + "no recovery(measured)". q13_cumulative Static cumV도
> censored:true라 해칭+≥531/≥530로 정정(값 불변).
> **작업 37 (핵심)**: v9의 emit_sidecars.py가 confirmed 복사 → checker 자기검사 결함. sidecar_util.py로
> **캔버스 아티스트(bar 높이·hline y·주석 텍스트)에서** 값 추출하도록 재작성, emit_sidecars.py 삭제. **음성
> 테스트**(Q3 막대 +2.0 → 캔버스 24.6 vs confirmed 22.6): checker가 `value 24.6 != 22.6` 잡음(v9는 못 잡던 케이스).
> **최종: 31값/5사이드카 0 mismatch·0 warn, 값 불변.**
> **남은 사람 판단**: v8 작업27 대안(축을 회복여부+종점V로) 여전히 미결.

## [C2-12] 그림 파이프라인 문서화 HTML (v11) — **완료 (2026-07-26)**
> `docs/figure_pipeline.html`(단일 파일·외부의존0·오프라인) + 생성기 `fig/build_pipeline_doc.py` + 요약
> `docs/figure_pipeline_report.md`. 재실험·값변경 없음.
> main.tex \includegraphics 파싱 → 그림 14개(라벨·캡션·절). 그림별 계보(생성스크립트 현행세대·입력데이터·
> 사이드카·confirmed id)를 저장소에서 확인. **검증 5 / 미감사 9 뱃지 구분.** 규약(quantity·reference·censored/
> recovered)은 confirmed _meta에서 렌더. base64 PNG 미리보기 인라인. 폐기값 목록(20.0/59.4/15.2/114/105–106/
> 21.4/23.6/24.6-as-persist) + 삭제파일(emit_sidecars.py) 기록. 감사이력 v2–v10 표. 자체검증 문서 내 포함
> (main.tex==카드 14 일치, confirmed id 실재, checker 0 mismatch, 하드코딩 없음). 손으로 안 쓰고 스크립트 생성.
> **생성스크립트 미상 2개**(architecture·c3_fluid_validation) §9에 명시. 미감사 9개 감사는 범위 밖(계보만).

## [D2] Trace-driven 유체 시뮬레이션 — bounded persistence 스케일링 (v12) — **완료 (2026-07-26)**
> `docs/d2_simulation_report.md` + `fig/sim_d2.py`·`sim_d2_sweep.py`·`plot_d2.py`. 이전 "착수 금지"였으나 v12가
> 설계 게이트 갖춘 착수 지시. 하드웨어 재실험·main.tex 수정 없음. **자유 파라미터 0**(상수 전부 측정에서).
> **게이트(작업 43) 통과**: 통과기준을 시뮬레이터 실행 전 사전 고정 후, 세 측정점 재현 — Q3 search 20.9(측정
> 22.6)·Q5 13.9(14.2)·Q4 22.0(24), 전부 ±20% 이내·분류 일치·회복후보 일치·순서 보존. 체계적 ~8% 과소추정(정직 보고).
> **스윕(작업 44)**: (1) dwell 용량반응 — 회복률 T_v=0 0.40→τ_settle(2s)에서 평탄 1.0, 그 위는 persist만 증가.
> **변형 A=곡선 한 끝**(strawman 반론 종결). τ 민감도: 무릎=τ(구조적)이나 형태는 robust. (2) (N_cand,k*) 평면 —
> 회복 k*≤N_cand, persist가 k*에 선형; Q3(5,5)=margin 0 경계, 2-gen(k*=12)=미회복 영역. (3) 부하: search 부하무관
> (~0.91×bound), persist는 λ→μ*에서 발산(C3 재현).
> **모형 한계 명시**: 유체 수준, 간섭 미시구조·스케줄러·확률변동 미모델링, 결과는 경향·임계점이지 절대예측 아님.
> **스키마 확장(작업 45-1)**: confirmed_values.json·check_figures.py에 `source_kind`(hardware/simulation) 추가,
> 강제 검증(31값 0 mismatch). D2 그림 3종은 SIMULATION 명시.
> **논문 반영안(작업 45-2)**: Q7 신설 권장, Q6 limitation 3 교체 문안·Q4 rank-5 연결 각주·모형한계 문장 초안 제시.
> **사람 판단**: D2를 Q7로 넣을지 vs Threats/appendix. τ_settle=2s 측정 추가값의 문서화.

## [D2-2] D2 후속 — 난수원 공개·대조점·회복률 곡선 정지 (v13) — **완료 (2026-07-26)**
> `docs/d2_followup_report.md` + d2_simulation_report.md 갱신. 시뮬 스윕만 재실행, 게이트 통과분 재조정 없음.
> **작업 46**: 난수원은 μ 분포가 아니라 **판정**(judge_correct Bernoulli(f=T_v/τ)+오판 시 동전). 실패후보 μ는
> feasibility boolean에만 쓰여 회복률 무관. §2 서술 정정(판정노이즈 공개). 분포 (a)~(e) 민감도: 회복률 동일
> (μ 무관), 형태 robust.
> **작업 47 (정지)**: 하드웨어 2점 대조 — 예측정합 k*=1 측정 0.80/sim 0.48, 오예측 k*=5 측정 0.20/sim 0.57.
> **순서 뒤집힘**(sim k*=1<k*=5, 측정 반대). 사전 고정 47-4대로 **회복률 곡선 논문 미등재**, 파라미터 추가 안 함.
> 원인: 판정모형이 A의 배회·고착(k* 의존성) 미포함.
> **유지**: budget×rank·load(T_v=3 결정론적), persist 추세, 형태(정성). **회복률 절대값은 그림에서 제거**.
> **작업 48**: 캡션 잘림 수정 — SIMULATION을 제목+범례+각주에(한 곳 잘려도 남게).
> **작업 49**: D2 검증값 3개 confirmed 등재(source_kind=simulation), 사이드카 캔버스 추출, checker 34값/7사이드카
> 0 mismatch. budget×rank 하드웨어 마커는 source_kind=hardware로 구분.
> **작업 50**: §6 문안 정정(무릎=모형 입력이지 발견 아님; 5-2=상한식 시각화, 기여는 하드웨어 점 위치).
> **사람 판단**: D2를 논문에 넣을 때 회복률 곡선 제외(형태·persist·평면만), Q7 배치 여부.

## [C2-13] 그림 재라벨·(win 58s)·checker 텍스트검사 (v14) — **완료 (2026-07-26)**
> `docs/figure_fix_report_v14.md`. 하드웨어/논문 값 불변. 재실험 없음.
> **작업 51**: d2_budget_rank을 해석적 지도로 재라벨(SIMULATION 제거, 제목 "measured cases", 마커 measured).
> main.tex 캡션과 정합. 평면 데이터를 sim-persist→**상한식 T+N_cand(Tv+δ) 평가**로 정합(라벨/데이터 모순 해소,
> 튜닝 아님). source_kind=analytic 추가, 평면값 2개 analytic·마커 measured. dwell/load는 리포트 전용→confirmed
> 미등재(사이드카 제거).
> **작업 52**: q13_failure 그룹라벨 `(win 58s)` 제거 — 58.4는 Static 창일 뿐(회복 3기법 62–69s). Static 막대 안
> +캡션에 창 정보 유지.
> **작업 53 (핵심)**: checker가 그림 전체 텍스트(제목·축·범주형 틱·범례·주석·하단문구) 스캔. 각 숫자는 confirmed
> 값(±tol) 또는 allowlist(정확·사유필수). `$...$`·수치축틱 제외. **음성 테스트**: `(win 58s)` 되돌리면 `text
> number 58.0` 검출(legit 58.4에 미흡수). **양성**: 33값/6사이드카 0 mismatch. allowlist 23항목 전부 사유.
> **작업 54**: cumV_ratio censored→true(비율=하한). 표시 변경 없음, 값 불변.
> **작업 55 (v8 작업27 닫음)**: q13_failure_persistence를 2패널("회복 여부 + 종점 V")로 전환 — 절단값 58.4가
> 시간축 막대 높이로 안 나타나게. 좌=회복결과(5/5 vs 0/5, Static V≈9), 우=회복변형 persistence(11–13s, 오차막대,
> Static 부재). 요건 1–5 충족. main.tex 서술 정합(수정 불요), 값 불변. **v8 작업27 PENDING 종결.**

## [C2-14] 마커 가림·δ 인스턴스화 표기 (v15) — **완료 (2026-07-26)**
> `docs/figure_fix_report_v15.md`. 값 불변. 재실험 없음.
> **작업 55**: q13 축 전환은 v14 마지막 턴에 이미 완료(2패널) — 확인만. v8 작업27 종결 유지.
> **작업 56**: d2_budget_rank 범례가 2-gen 마커(5,12)를 가리던 것 → lower-right 이동, 마커 확대+검은테두리.
> 두 마커(Q3 별, 2-gen X) 육안 식별 확인. main.tex "outside the shaded region"가 가리키는 점이 이제 보임.
> **작업 57**: 평면 하단 문구를 "single representative delta (1 s vision swap); per-transition instantiation
> (LLM move) raises it"로 정정 — main.tex 캡션(δ=1s 하한 포락선, 전환별 시 Q3 24–26s)과 정합. 컬러바 유지. 값 불변.
> checker 33값/6사이드카 0 mismatch(δ=1 allowlist 추가).

## [C2-15] figure_pipeline.html 재생성 (v16) — **완료 (2026-07-26)**
> `docs/figure_pipeline.html`(재생성) + `docs/figure_pipeline_report_v16.md` + `fig/build_pipeline_doc.py`(갱신).
> 재실험·값변경·그림재생성 없음. 문서는 스크립트 생성.
> v11(14그림) → **15그림**(d2_budget_rank 신규), 검증 5→6/미감사 9. 스키마 recovered·source_kind 규약 추가
> (Q4 반례, hardware/simulation/analytic, d2_budget=analytic+hardware 혼합 유일사례). 프로그램 갱신(sidecar_util
> 등재, emit_sidecars 삭제이유, check_figures 텍스트스캔). **D2 계보 §3b 본문**에 dwell/load 제외 이유(회복률
> 순서 뒤집힘) 기록. 미사용 PDF §9(21개 중 7 미사용, 성격분류, *_2gen 삭제금지, 접미사 오집 경고). 감사이력
> v11–v15, 폐기값에 D2 회복률 곡선·v13 budget 평면 추가. 자체검증: 파싱14⊆카드15(주입 d2_budget), checker
> 0 mismatch, 하드코딩 없음.
> **한계(정직)**: main.tex 갱신본 미보유(파서 14개, d2_budget는 지시문 근거 주입). 지시문 작업60 예시 미사용
> 파일들은 이 docs/figures에 부재 — 실존 21개 기준 목록화.

## [C2-16] main.tex 저장소 제거·매니페스트 인터페이스 (v17) — **완료 (2026-07-26)**
> `docs/figure_pipeline.html`(매니페스트 기반 재생성) + `docs/figure_pipeline_report_v17.md` +
> `fig/paper_figure_manifest.json`(신규) + `fig/build_pipeline_doc.py`(파싱·주입 삭제). 재실험·값변경 없음.
> **"main.tex 불변" 규칙 종료** — 실험 끝. 정본 main.tex는 **저장소 밖 한 곳**, 저장소엔 사본 미보유(v16 분기
> 재발 방지). 무결성 문구: **"main.tex 부재(설계) — 정본 외부, 인터페이스 fig/paper_figure_manifest.json"**.
> **작업 63**: 저장소 main.tex 0개(직전 삭제), 백업 불요(원본은 업로드). architecture.pdf(루트 자산) 유지.
> main.tex 읽던 코드는 build_pipeline_doc.py뿐 → 매니페스트로 교체(잔여 참조 0).
> **작업 64/65**: 생성기가 매니페스트만 읽음(파싱·주입 코드 삭제), sha16+추출일 표기, 정합성 검사 3종, staleness
> 자동검증 불가 명시(해시 기록·사람 대조). 재생성: 15카드 전부 매니페스트, **주입 0**, self-check 15==15.
> **작업 66**: 그림 디렉터리 선언(docs/figures=정본 21, 루트=architecture 1; 33-PDF 소스트리 저장소 부재).
> 미사용 7개 경로 포함, *_2gen 삭제금지. 파일 이동·삭제 없음.
> **작업 67**: 정본 관리 규약 §6b 기재.
> **빌드 한계(정직)**: pdflatex 미설치·bib/bst 부재로 4패스 빌드 실행 불가(문서생성은 무관); 정적확인만.

## [R4] hot-swap 전환 자원 소모 측정 (v18) — **완료 (2026-07-27)**
> `docs/hotswap_resource_report.md` + `scratchpad/v18/`(사이드카·런너·분석·raw). ISSRE R4 미해결분
> ("transient resource exhaustion during hot-swap under contention") 답변.
> **작업 68 게이트=없음**: 기존 로그(results/ 722개)는 조합별 집계 요약만, 메모리 시계열 0 → 최소 재실행.
> **작업 70**: Q3(4-phase) + bounded-recovery(3-phase) 2건만, 10 Hz 관찰자 사이드카(≥5 Hz 충족). 워크로드·
> 알고리즘·파라미터 불변. 샘플러 부하 ≤6 %(throughput; 메모리 풋프린트엔 무영향). 게시 results/·그림 불변.
> **핵심 발견(구조)**: (1) 전 저장소 스케줄에 **같은-디바이스 모델 교체 0건** — 모든 전환이 GPU↔CPU 교차이동.
> (2) generative relocation은 **kill-then-start**(background_llm.py) — 옛 인스턴스 미생존. → 옛+새가 같은
> 디바이스를 동시 점유하는 상황 자체가 없어 **VRAM 전환 증분 0**(실측). 겹침은 vision hot-swap뿐(작은 모델).
> **실측(RTX5090 32GB/62GiB)**: GPU VRAM 증분 전 전환 0(여유 ∞); host RSS 최악 +201 MiB(4-모델 CPU→GPU
> 회복 스왑, 여유 308×). 전역 VRAM 최대 1950 MiB=6.0 %. → 판정 **위험 미미**(사전 매트릭스 여유>2 충족).
> **69-3 실패 경로(코드, 실질 잔여 위험)**: 사전 자원 점검 0건(R4의 "assumes residual resources" 확증);
> 로드 실패 시 예외 억제·ready 강제 set → 죽은 워커로 큐 교체 + 옛 배치 정지 = **조용한 뷰 정지**, 전방
> 전환 롤백 없음; 실패는 외곽 V(t) 검증이 **T_v 뒤** 간접 감지(사각지대). NPU DRAM 초과·generative 실패도
> 동일 정지 경로. → §III 설계서술 보완·Threats 문안 초안 제시(사람 적용, main.tex 미수정).
> **한계(정직)**: NPU DRAM 전환 피크 직접 계측 안 함(GPU 케이스라 NPU 워커 없음, 코드로만); 같은-디바이스
> 중복은 워크로드 부재라 상한만(2× 워킹셋=3900 MiB, 32GB의 12 %/16GB의 24 %); 새 그림 없음(표로 충분).

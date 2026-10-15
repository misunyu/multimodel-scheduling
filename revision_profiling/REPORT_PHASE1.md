# REPORT_PHASE1 — CPU–GPU 재profiling Phase 1 (사양 v2.2)

2026-10-13. 표기: [확인] 코드·데이터로 직접 확인, [추정], [확인 불가]. Phase 2는 실행하지 않았다.

## 결론

**β를 선택하지 못했다.** 사전 고정 규칙(§5.0)의 순서 5.1 → 5.2 → 5.3 중 5.2를 통과한 β가 없다. 세 후보 모두 같은 이유로 떨어졌으므로 β의 문제가 아니라 runtime의 문제다.

1. **반복 편차가 기준의 몇 배다.** 같은 배치를 두 번 돌린 36쌍에서 F의 상대 차이는 중앙값 9.2%, 최대 29.3%이고, 기준(5%) 안에 든 것은 14쌍이다.
2. **N=8 전부-CPU 배치에서 요청 생성이 밀린다.** 실현율 최저 0.82(기준 0.95). 다른 배치(N=4 전체, N=8 전부-GPU·반반)는 전부 0.96 이상이다.
3. **원인은 CPU thread 과다 사용이다** [확인: process CPU 시간]. CPU instance 하나가 ONNX Runtime thread 24개를 계속 돌리고, TinyLLaMA는 GPU에 두어도 그렇다. Pilot의 arrival run 72개 전부에서 process CPU가 10 s에 약 238 core·s(24 core 포화)였다.
4. Pilot 범위에서는 **부하를 바꿔도 throughput이 변하지 않는다.** N=4 전부-GPU의 F는 β=0.25·2×에서 253–259, β=1.0·4×에서 262–269 req/s다. 가장 낮은 설정에서도 이미 포화다.

따라서 §5.4(선택된 β의 반복 편차)는 실행하지 않았다. 사용자 결정이 필요한 것은 9절에 모았다.

## 0. 환경, branch, 변경

- Branch `revision-profiling`(from `ubuntu_gpu` @ `1f7e6bc`). 원본 tracked 파일 수정 없음, commit 없음. 산출물은 전부 `revision_profiling/`.
- 보드: Core Ultra 9 285K(24 thread), RTX 5090. Python 3.10.12, onnxruntime 1.23.2, numpy 2.2.6. `-B` 실행, 프로젝트에 새 `__pycache__` 없음(`vendor/` 제외 확인).
- 사양서 사본 `SPEC_v2.2.md`. 사양서 v1·v2와 `revision_map.md`는 저장소에 없다 [확인 불가]. v1의 JSON schema는 대조하지 못했고, 원고 줄 번호는 7절에 적지 못했다.
- 변경 요약(`CHANGES.md`): 원본 runtime 사본을 고치는 대신 headless executor를 새로 썼다. 사양에 없던 변경이 둘 있다.
  - 분류 전처리를 표준 ImageNet 변환으로 바꿨다(원본 전처리로는 예측이 맞지 않음). 교체 후 128장 top-1은 83–119장.
  - 추론 thread를 nice 10으로 낮췄다(feeder가 굶지 않게).
- 모듈 경로 검증: 정식 run 126개 전부에서 executor와 모든 feeder process의 프로젝트 모듈이 `revision_profiling/code/` 아래였다. 위반 0건 [확인].
- 실행 시간: 정식 pilot 1–4단계 41분(15:14–15:56). 그 전 개발 점검 약 30분.

## 1. Runtime·counter, smoke (§5.1)

구현: instance마다 feeder process 1개 + worker thread 1개, lock 없는 shared-memory queue(용량 2, drop-newest), 경계 읽기 전용 process. 이벤트마다 `CLOCK_MONOTONIC` timestamp를 남기고 실행 뒤 `[t0, t1)`로 집계한다. `window_sec_actual`은 모든 run에서 정확히 10 s다.

Feeder는 thread 구현이 실현율 0.35–0.78로 기준에 못 미쳐 사양대로 process로 분리했다(시도 6개의 기록: `CHANGES.md` 2절, `pilot/arrival/attempt*_dev/`).

**Conservation** (정식 run 126개, instance 486개):

| `enqueued − completed − failed − Δpending` | instance 수 |
|---|---|
| 0 | 456 |
| ±1 (경계 읽기 오차, 허용) | 29 |
| +2 (invalid) | 1 |

- Timestamp로 계산한 pending으로는 486개 전부 0이다. ±1은 reader process가 경계 시각보다 최대 약 1 ms 늦게 읽어서 생긴다.
- `generated = enqueued + dropped_full`은 전부 0, in-flight는 전부 0 또는 1, `failed`와 `consumer_failed`는 전부 0.

**Invalid run 1건**: `pilot/beta_1.0/single_gpu/runs/single_shufflenet-v2-12_b1.0_x1__combination_001__rep0.json`, 사유 `conservation`(+2). 823 req/s로 capacity에 붙은 run이고 읽기 지연이 0.97 ms였다. 실행 전체의 합계 항등식은 맞는다. 어떤 집계에도 넣지 않았다. 재실행 대상.

**Smoke 판정** (β=0.5, 1×, `pilot/smoke/smoke_5_1.csv`): 18개 중 15개 통과.

| 항목 | CPU | GPU |
|---|---|---|
| 모듈 경로·sha256 | 9/9 | 9/9 |
| failed = 0, consumer_failed = 0 | 9/9 | 9/9 |
| conservation 오차 0, in-flight ≤ 1 | **7/9** | 9/9 |
| 실현율 ≥ 0.95 | 9/9 | 9/9 |
| isolated latency ±20% | 9/9 | 9/9 |
| 완료율 | **7/9** | 9/9 |

미통과 3건:

| 모델 (CPU) | 미통과 항목 | 수치 |
|---|---|---|
| gpt2 | conservation | −1 (경계 읽기). 다른 항목 통과 |
| mnasnet | 완료율(비과부하), conservation | 요청 390 req/s < 0.8 × capacity 631. drop 56/3,881 = 1.4% (기준 < 1%). conservation +1 |
| squeezenet1.0-12 | 완료율(비과부하) | 요청 639 < 0.8 × 883. drop 215/6,352 = 3.4% |

- 비과부하인데 drop이 나오는 이유: CPU service time의 꼬리가 길다(p95/p50 = 1.4–1.5, thread feeder 때의 측정에서 최대 13–23 ms). Queue 용량이 2라 한 번 밀리면 넘친다 [확인: 이벤트 기록].
- "conservation 오차 0"을 문자 그대로 적용했다. §2.4의 ±1 허용을 smoke에도 적용하면 conservation 미통과는 0건이고 전체 통과는 16/18이다.
- CPU 구분: 비과부하 2(mnasnet, squeezenet), 경계 1(shufflenet, 과부하 기준으로 통과), 과부하 6.

Isolated capacity (`offline_profile/offline_profile.json`, closed loop, 전처리·후처리 포함):

| 모델 | GPU req/s | GPU p50 ms | CPU req/s | CPU p50 ms |
|---|---|---|---|---|
| mnasnet | 780.8 | 1.22 | 630.8 | 1.44 |
| resnet50 | 442.9 | 2.17 | 138.2 | 6.50 |
| resnext50 | 427.9 | 2.27 | 147.6 | 6.14 |
| shufflenet-v2-12 | 823.5 | 1.18 | 501.7 | 1.92 |
| squeezenet1.0-12 | 1,278.9 | 0.75 | 883.4 | 1.07 |
| vgg19 | 532.9 | 1.78 | 39.9 | 23.72 |
| yolov4 | 128.7 | 7.52 | 18.2 | 53.92 |
| gpt2 | 20.0 | 47.65 | 5.5 | 178.16 |
| tiny-llama-chat-onnx | 3.3 | 299.92 | 1.2 | 830.87 |

## 2. LM (§3) — `pilot/lm/LM_RESULT.md`

| | GPT-2 | TinyLLaMA |
|---|---|---|
| 양쪽 device 실행, 출력 정상 | 예 | 예 |
| `completed ≥ 20` (1×, β=0.5) | GPU 100, CPU 56 — 통과 | GPU 17, CPU 12 — **판정 보류** |
| 권장 | 포함 | 제외 후보 |

TinyLLaMA의 문제 셋:
- 파일이 int4 양자화본(909.5 MB)이라 원고의 2,488 MB와 다르다. 맞는 ONNX는 저장소에 없다.
- CPU capacity가 1.2 req/s라 10 s 구간에서 어떤 β로도 20건에 닿지 못한다.
- GPU에 두어도 attention node가 CPU에서 돌아 CPU thread 24개를 계속 쓴다.

**Family 재구성** (`family_scenarios.csv`, 대응표 `family_reconstruction.csv`):

| 시나리오 | family | N≥3 family | 후보 Σ2^N | run (4 rates) | 반복 추가 (3×, N≥3, 2회) | 총 run |
|---|---|---|---|---|---|---|
| 둘 다 포함 | 29 | 18 | 1,194 | 4,776 | 2,336 | 7,112 |
| TinyLLaMA 제외 | 22 | 12 | 504 | 2,016 | 960 | 2,976 |
| GPT-2 제외 | 24 | 11 | 772 | 3,088 | 1,472 | 4,560 |
| 둘 다 제외 | 17 | 7 | 290 | 1,160 | 528 | 1,688 |

TinyLLaMA만 제외하면 29개 중 16개 유지, 6개 변형 유지, 6개 중복 제거, 1개 빈 family 제거다. 평가 workload(N≥3)가 18개에서 12개로 줄어든다.

## 3. β pilot (§5.2, §5.3)

Family는 실행 전에 "해당 N에서 이름순 첫 번째"로 정했다: N=4 `g_m_r_t`(gpt2, mnasnet, resnet50, tiny-llama), N=8 `g_m_resnet50_resnext50_shufflenet-v2-12_squeezenet1.0-12_t_v`. 둘 다 LM 2개를 포함한다.

**기준별 통과표**

| β | 5.1 (β=0.5 고정) | 5.2 workload 통과 | 5.3 (A) 4× 전부-GPU | 5.3 (B) 1× 단일 GPU drop < 1% | 선택 가능 |
|---|---|---|---|---|---|
| 0.25 | 15/18 | **0/4** | 4/4 | 9/9 | 아니오 |
| 0.5 | 〃 | **0/4** | 4/4 | 9/9 | 아니오 |
| 1.0 | 〃 | **0/4** | 4/4 | 8/8 + invalid 1 | 아니오 |

**5.2 상세** (`pilot/arrival/arrival_5_2_workloads.csv`, run 72개 전부 valid):

| family | β | 배수 | 최소 실현율 | 최대 missed 비율 | F 상대 차이 (전부-CPU / 반반 / 전부-GPU) | generated 불일치 instance |
|---|---|---|---|---|---|---|
| N=4 | 0.25 | 2× | 0.989 | 1.2% | 5.6% / 25.1% / 2.3% | 0 |
| N=4 | 0.25 | 4× | 0.984 | 1.6% | 18.7% / 13.4% / 2.7% | 0 |
| N=4 | 0.5 | 2× | 0.988 | 1.2% | 0.7% / 6.6% / 11.6% | 0 |
| N=4 | 0.5 | 4× | 0.976 | 2.4% | 25.7% / 9.8% / 2.0% | 1 |
| N=4 | 1.0 | 2× | 0.991 | 0.9% | 29.3% / 4.2% / 0.5% | 0 |
| N=4 | 1.0 | 4× | 0.980 | 2.0% | 16.0% / 2.1% / 2.5% | 0 |
| N=8 | 0.25 | 2× | **0.868** | 13.2% | 4.9% / 17.7% / 5.8% | 4 |
| N=8 | 0.25 | 4× | **0.845** | 15.5% | 28.8% / 4.8% / 10.5% | 11 |
| N=8 | 0.5 | 2× | **0.815** | 18.5% | 3.6% / 15.6% / 0.5% | 11 |
| N=8 | 0.5 | 4× | **0.884** | 11.6% | 11.5% / 22.9% / 8.6% | 9 |
| N=8 | 1.0 | 2× | **0.888** | 11.2% | 13.5% / 14.6% / 0.7% | 7 |
| N=8 | 1.0 | 4× | **0.823** | 17.7% | 9.9% / 16.7% / 4.0% | 9 |

- 실현율 미달은 N=8 전부-CPU 배치에만 있다(instance 96개 중 35개). CPU session 8개가 thread 192개를 24 core에 올린 상태다.
- F 차이는 배치 종류를 가리지 않는다. 전부-GPU가 가장 안정적이지만(12쌍 중 8쌍이 5% 이내) 그래도 최대 11.6%다.
- Generated 불일치는 실현율 미달의 결과다(같은 workload의 전부-CPU 배치만 요청 수가 적다).
- Feeder process 분리 후의 재검증 결과가 이 표다. 더 고칠 수 있는 것은 9절 선택지에 적었다.

**5.3 상세** (`pilot/beta_*/gpu_load_5_3_*.csv`):
- (A) 전부-GPU run 24개 전부에서 모든 GPU instance에 drop이 있다. GPU utilization 평균 78–85%, latency p50 배율의 중앙값은 N=4에서 3.1–3.7×, N=8에서 8.5–9.6×다. Utilization이 80%에 못 미친 run도 latency 기준으로 통과한다.
- (B) β=0.25, 0.5는 9개 모델 전부 drop 0. β=1.0은 8개가 통과했고(drop 최대 28건, 0.5% 미만) shufflenet은 invalid run이다(drop 자체는 14/8,233 = 0.17%).
- 5.3만 보면 조건을 만족하는 가장 작은 β는 0.25다. 그러나 5.2 미통과로 선택 규칙이 여기까지 오지 않는다.
- 주의: latency 배율이 큰 것을 GPU 포화로만 읽으면 안 된다. 이 run들에는 TinyLLaMA가 들어 있어 CPU도 포화 상태다. GPU worker thread의 전처리와 Python 구간이 CPU 경합으로 늦어진 몫이 섞여 있다 [추정: 분리 측정은 하지 않았다].

**5.4**: 실행하지 않음(선택된 β 없음). 반복 편차의 근거는 5.2의 36쌍이다: F 상대 차이 중앙값 9.2%, 90% 지점 24.0%, 최대 29.3%.

## 4. Drop metric (§4, §5.5) — 보고용, β·α·D10 선택 근거 아님

대상: 5.2의 workload 12개, 각각 pilot 후보 3개(전부-GPU, 전부-CPU, 반반), 2회 평균. "best"는 pilot 후보 안에서의 best-measured다.

| 정의 | best-measured = throughput-max인 workload (α=0.3) |
|---|---|
| (a) aggregate `R/R_max` | 12/12 |
| (b) D_macro | 11/12 |
| (c) D_max | 11/12 |

- (a)는 α 0.1–1.0 전 구간에서 12/12다. 예상대로 α가 선택에 영향을 주지 않는다.
- (b)·(c)가 달라진 1건: N=4, β=0.5, 4×. Throughput-max는 반반(257.2 req/s), (b)·(c)의 best는 전부-GPU(256.2 req/s)이고 α 0.1부터 1.0까지 전부 그렇다. 전부-GPU에서 q_i가 내려간 instance는 resnet50(0.985 → 0.884)과 tiny-llama(0.985 → 0.803), 올라간 것은 mnasnet(0.849 → 0.907)과 gpt2(0.793 → 0.816)다. 두 배치의 F 차이(0.4%)는 반복 편차(9%)보다 훨씬 작으므로 이 1건을 정의 차이의 증거로 쓸 수 없다.
- `F + R` vs `Λ`: `(F + R − Λ)/Λ`의 중앙값 −0.55%, 범위 −9.9% ~ −0.03%. 음수는 missed tick 때문이다. 실제 generated 기준으로는 `(F + R − generated)/generated`가 −0.03% ~ +0.01%다(pending 변화분).
- q_i 분포: 후보 36개의 D_macro 범위는 0.40–1.00이고 대부분 0.85 이상이다. 예: N=4, β=0.5, 4×에서 전부-GPU 0.852, 반반 0.905, 전부-CPU 0.990. 모든 설정이 깊은 과부하여서 q_i가 1 근처에 몰린다.
- 모든 instance의 λ_i가 같아 (b)가 F에 단조가 되는 family: N=1 family 9개(LM 제외 시 7–8개). N≥2 family에는 같은 모델이 두 번 들어간 경우가 없다 [확인].
- Invalid run: 위 1건(`zero_generated` 0건, in-flight 위반 0건, 모듈 경로 위반 0건, `consumer_failed` 0건).

표: `pilot/drop_metric_5_5_{candidates, best_measured, alpha_sensitivity, protected_instances, conservation_F_R_Lambda, q_i}.csv`.

## 5. CPU–NPU 호환성 (§6) — `npu_compat/NPU_COMPAT.md`

**재계산 불가.** 근사 D_macro와 best-measured는 계산하지 않았다.

| 검증 | 결과 |
|---|---|
| (i) `inference_count` 구간 = 30 s | 불일치. 30 s는 process 생성·load를 포함한 전체 실행 시간이고 warm-up reset 함수는 호출된 적이 없다 |
| (ii) drop이 모든 instance에서 집계 | 아님. ResNet feeder는 `queue.Full`을 세지 않는다. ResNet instance 484개(47%) 전부 0. YOLO의 값도 요청 수가 아니라 재시도 횟수다 |
| (iii) headless 없음 | 통과 |
| (iv) throughput = `1000/latency` | 맞음. 예외 5개는 완료 0건인 NPU instance |

데이터 수: 측정 행은 372(= 93 placements × 4 rates), 학습 279 + 평가 93. 원고의 "279 placements"는 학습 표본 수이고 "1,116 samples"는 데이터에 없는 수다.

선택지: (1) NPU 보드 재측정 372 runs ≈ 3.2 h + runtime 이식, (2) 근사 재계산 — 이번 검증으로 근거가 없어짐, (3) NPU 평가 범위 축소.

## 6. Phase 2 예상

Run당 wall-clock(pilot 실측): N=1 16.3 s, N=4 20.2 s, N=8 23.5 s. 선형 보간 `15.8 + 0.96·N` s로 추정했다 [추정].

| 시나리오 | 1·2·3·4× | 3× N≥3 반복 2회 | 합계 |
|---|---|---|---|
| 둘 다 포함 | 29.7 h | 14.6 h | 44.2 h |
| TinyLLaMA 제외 | 12.0 h | 5.8 h | 17.8 h |
| GPT-2 제외 | 19.0 h | 9.1 h | 28.1 h |
| 둘 다 제외 | 6.8 h | 3.2 h | 10.0 h |

측정 순서: 1·2·4× → 3× hold-out → 3× N≥3 반복 2회.

**현재 runtime으로는 Phase 2를 권하지 않는다.** 반복 편차 9%(중앙값)는 placement 간 차이를 가리기에 충분히 크다. 4절의 1건이 그 예다.

## 7. 원고 영향

원고 파일과 `revision_map.md`가 저장소에 없어 줄 번호는 적지 못했다 [확인 불가]. 항목은 사양서가 지목한 위치 기준이다.

**(i) 새 정의 채택 시 교체 대상**
- Fig. 1(b), Fig. 4, Table 3(a)(b): 전부 새 데이터로 다시 만든다.
- §3 정의: throughput = 완료 건수 / Δ, drop = queue overflow 건수. D_macro를 채택하면 `D_macro = (1/N) Σ dropped_i / generated_i`와, 후보 간 generated가 같고 failed·Δpending이 무시될 때 `Σ w_i f_i` 최대화와 동치라는 조건을 함께 적는다.
- α 근거: (a)에서는 α가 선택을 바꾸지 않는다는 것이 pilot에서도 확인됐다(12/12). α 기본값은 Phase 2 학습 데이터에서 따로 정한다.
- Abstract·Contribution의 "multi-objective" 서술: (b)에서도 독립된 두 목표가 아니라 가중 throughput임을 반영한다.
- 실험 설정 서술: 모델별 실제 ONNX, 요청 rate 규칙(`β × GPU isolated throughput`), Δ = warm-up 5 s + 10 s, LM 요청 정의(prompt 32 + 16 tokens), TinyLLaMA 파일 크기.
- Per-instance baseline 사양(§7): target과 합성을 (b) 변형으로.
- CPU–NPU: 372 samples로 정정. 새 정의로 재계산할 수 없다는 점.

**(ii) 중간 rate 제외에 따라 삭제·수정할 것**
- §1의 generalization 서술(56행 근처).
- §5.1.3 "Profiling interval and generalization evaluation"의 54 schedules / 3,504 samples.
- §5.2 중간 rate 문단(Top-1 0.72 / 0.88 / 0.88, Top-5 0.94 / 1.0 / 0.94)과 "broad range of unseen workload scales" 결론.
- §7 "observed generalization to unseen workload intensities".
- R1-3 response의 해당 부분.

**(iii) 이번 pilot이 새로 드러낸 것** — 원고 서술에 영향
- TinyLLaMA의 "GPU 실행"은 CPU–GPU 혼합 실행이다.
- 요청 rate를 GPU isolated throughput 기준으로 잡으면 CPU 배치는 1×에서도 대부분 과부하다(9개 모델 중 6개).

## 8. §7 per-instance baseline

코드는 쓰지 않았다(사양대로 Phase 2 이후). 사양서 §7의 설계를 그대로 유지한다. Pilot에서 확인된 주의점 하나: λ_i는 schedule의 요청 rate이고 `generated_i`는 실측값이라, 실현율이 1에 못 미치는 run에서는 둘이 다르다. Target `q_i = d_i / generated_i`의 분모는 실측 generated를 쓴다.

## 9. 사용자 승인·결정 항목

| # | 항목 | 현재 상태 | 조사한 쪽의 의견 |
|---|---|---|---|
| 1 | **Runtime의 반복 편차·CPU 과다 사용** | 5.2 미통과의 원인 | 먼저 정해야 한다. 아래 선택지 |
| 2 | LM 포함 여부 | GPT-2 통과, TinyLLaMA 판정 보류 | GPT-2 포함, TinyLLaMA 제외. TinyLLaMA는 파일·완료 건수·CPU 사용 세 문제가 겹친다 |
| 3 | D10 drop metric | 셋 다 계산됨 | 채택은 연구 목적으로 정할 일. Pilot은 근거가 되지 못한다 |
| 4 | β | 선택 불가 | 1번 해결 후 5.2부터 재실행 |
| 5 | CPU–NPU | 재계산 불가 | (1) 재측정 또는 (3) 범위 축소 |
| 6 | Phase 2 | 미실행 | 1·2·4번 뒤에 |
| 7 | 사양 밖 변경 2건(분류 전처리, nice 10) 추인 여부 | 적용됨 | |

**1번의 선택지** (어느 것도 승인 없이 적용하지 않았다)

| | 내용 | 기대 | 부담 |
|---|---|---|---|
| A | CPU session의 `intra_op_num_threads`를 24에서 낮춘다(예: 4–8). ORT thread spinning을 끈다 | CPU 포화와 thread 192개 경합이 사라져 실현율·반복 편차가 함께 개선될 가능성이 크다 [추정] | 원본 runtime 설정과 달라진다. CPU isolated capacity가 바뀌므로 offline profile부터 다시 |
| B | Feeder·reader를 전용 core에 고정하고 추론을 나머지 core로 제한한다 | 실현율은 해결될 것이다 [추정] | 추론에 쓰는 core가 줄어든다. 반복 편차는 남을 수 있다 |
| C | TinyLLaMA를 제외하고 현재 설정으로 5.2를 다시 본다 | 전부-GPU·반반 배치의 CPU 포화가 없어진다 | CPU instance가 있는 배치는 그대로다. 전부-CPU N≥6은 여전히 미달일 수 있다 |
| D | 기준을 완화한다(F 차이 5% → 더 큰 값, 반복 횟수 증가) | — | 사양서가 금지한 방향. 권하지 않는다 |

의견: A와 C를 함께 적용하고 5.1부터 다시 도는 것이 가장 빠르다. 약 1시간이다.

## 10. 한계

- Pilot family 두 개가 모두 LM 2개를 포함한다. LM이 없는 workload의 반복 편차는 재지 않았다.
- 반복은 2회다. 편차의 분포를 말하기에는 적다.
- 5.3 (A)의 latency 배율에서 GPU 경합과 CPU 경합을 나누지 못했다.
- 분류 top-1 점검은 128장, 통과 기준이 아니다. YOLOv4 검출은 frame당 평균 8.3개가 나오는 것만 확인했다.
- 개발 점검 run(`attempt*_dev/`)은 집계에 넣지 않았다. Thread feeder로 돌린 첫 smoke와 offline profile은 `attempt1_thread_feeder/`에 남겼다.

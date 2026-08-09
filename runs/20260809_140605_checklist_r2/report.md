<!-- 작업 14 — 체크리스트 6항목. 기존 아티팩트·측정 데이터·원고 미수정.
     신규 산출물은 이 디렉터리 이하에만 생성. A 만 재학습(신규 경로 저장), 나머지는
     저장된 OOF·측정값 재사용. -->

# 체크리스트 r2 — 항목별 결과

| 항목 | 내용 | 상태 | 원고 반영 |
|---|---|---|---|
| **A** | joint paired-fold 재학습 (#7) | 완료 (재학습) | **필요 — 숫자 교체** |
| **B** | divergence score-gap (#9) | 완료 | 부록 표 추가 가능 |
| **C** | cluster bootstrap CI (#10) | 완료 | 부록 표 추가 가능 |
| **D** | greedy worst-case bound (#11) | 완료 | Table 5 열 추가 가능 |
| **E** | r3 집계식 확정 (#16) | 완료 | **필요 — 정의문 추가** |
| **F** | regret / oracle ratio (#20) | 완료 | 부록 표 추가 가능 |

---

## A. joint paired-fold 재학습 (#7) — **결론: 수치 유지 또는 개선. 숫자 교체 가능**

### 문제와 조치

현행 joint(P4)는 CV 그룹 키에 플랫폼을 포함해 90그룹으로 나눈다. 그래서 같은
(set, rate)의 GPU 행과 NPU 행이 서로 다른 fold 에 떨어질 수 있고, 그러면 한 플랫폼의
동일 워크로드가 학습에 들어간 채 다른 플랫폼의 같은 워크로드를 평가하게 된다.

**구 분할에서 실제로 갈라진 쌍: 45쌍 중 27쌍 (60%)** — 우연이 아니라 지배적 경우다.

신규 분할은 그룹 키에서 플랫폼을 뺀다: `(models, rate_factor)` → **45 paired 그룹**.
한 (set, rate)의 GPU·NPU 행 24개가 항상 같은 fold 에 들어간다. 그 외 프로토콜은 전부
동일하다(3-fold, seed 42, 그룹 인지 내부 CV, 동일 4조합 그리드, y3 마스킹, feature 37개).
`analysis_common` 헬퍼를 그대로 쓰고 새 함수를 만들지 않았다.

### 결과 (점수 수준 = 원고가 싣는 지표)

| 플랫폼 | 분할 | 그룹평균 Spearman | Top-1 | Top-5 |
|---|---|---|---|---|
| GPU | old (90그룹) | +0.9748 | 0.9333 | 1.000 |
| GPU | **new (45 paired)** | **+0.9755** | **0.9333** | 1.000 |
| NPU | old (90그룹) | +0.9742 | 0.8444 | 1.000 |
| NPU | **new (45 paired)** | **+0.9842** | **0.8667** | 1.000 |

- GPU: Spearman +0.0007, Top-1 변화 없음
- NPU: Spearman **+0.0100**, Top-1 **+0.0222** (38/45 → 39/45)
- **하락 없음.** 누수를 제거했는데 순위 품질이 오히려 올라간다.

구 분할 재실행이 저장된 `unified_metrics.json` 값을 **양 플랫폼 모두 정확히 재현**했으므로
(check `stored_unified_metrics_reproduced: {gpu: true, npu: true}`), 두 열은 같은 조건에서
비교된 것이다.

### 원고에 미치는 영향

| 원고 위치 | 현행 | paired-fold |
|---|---|---|
| Abstract / §4 joint Top-1 | GPU 93.3 / **NPU 84.4** | GPU 93.3 / **NPU 86.7** |
| Fig. transfer `joint` 막대 (Spearman) | GPU 0.975 / NPU 0.974 | GPU 0.976 / **NPU 0.984** |
| 대비되는 platform-specific | GPU 93.3 / NPU 86.7 · ρ 0.986 / 0.972 | (불변) |

**서술이 바뀌는 지점 두 곳:**

1. **NPU Top-1 이 platform-specific 과 정확히 같아진다** (0.8667 = 0.8667). 현행 원고는
   "joint approaches platform-specific"인데, paired fold 에서는 NPU 에 한해 **동일**하다.
2. **NPU 그룹 Spearman 은 joint 가 platform-specific 을 앞선다** (+0.9842 vs +0.9724).
   현행에서도 +0.002 앞섰지만 폭이 +0.012 로 커진다.

즉 이 재학습은 원고의 주장을 약화시키지 않고 **강화**한다.

### 주의: target 수준 지표는 내려간다 (누수 제거의 예상된 결과)

| 플랫폼 | 분할 | y1 ρ | y2 ρ | y3 ρ |
|---|---|---|---|---|
| GPU | old | +0.994 | +0.951 | +0.983 |
| GPU | new | +0.989 | +0.922 | +0.976 |
| NPU | old | +0.988 | +0.942 | +0.977 |
| NPU | new | +0.973 | +0.890 | +0.962 |

target 회귀 정확도는 전 항목에서 떨어진다 — 같은 워크로드의 상대 플랫폼 행을 더는
학습에서 볼 수 없기 때문이다. **떨어진 것은 회귀 정확도이고, 순위 품질은 떨어지지 않았다.**
원고가 싣는 것은 순위 지표이므로 표는 교체하되, 만약 target 수준 수치를 어디선가 인용
한다면 그쪽은 하락 폭을 명시해야 한다.

산출물: `A_joint_paired/paired_metrics.json`, `joint_oof_{old90,new45}.csv`, `run.log`.
**기존 `analysis/unified_*` 와 `artifacts/` 는 건드리지 않았다.**

---

## B. divergence score-gap (#9 1단계) — 수치만

11개 불일치 그룹 × 2플랫폼 = 22 셀. 각 셀에서
`gap = (자기 플랫폼 최고 측정 점수) − (상대 플랫폼 최적 배치가 자기 쪽에서 받은 점수)`.
기저는 선언 정규화(α=0.3, β=1.0), 점수 정의는 감사 모듈 import.

### 분포

| 통계 | gap 절대값 | gap / 그룹 내 점수 SD | gap / 그룹 점수 범위 |
|---|---|---|---|
| 최소 | 0.0026 | 0.0047 | 0.0015 |
| p25 | 0.0710 | 0.1239 | — |
| 중앙값 | 0.1553 | 0.3101 | — |
| p75 | 0.6715 | 1.0357 | — |
| 최대 | 0.9513 | 1.4247 | — |
| 평균 | 0.3470 | 0.5772 | — |

### tie 폭·노이즈 척도와의 비교

| 항목 | 값 |
|---|---|
| tie 판정 폭 | 1e-9 |
| 가장 작은 gap | **0.002628** = tie 폭의 **2.6×10⁶ 배** |
| gap < 1e-3 인 셀 | **0** / 22 |
| gap < 1e-2 인 셀 | **2** / 22 (S8@2.0 GPU 0.0026, base3@3.0 NPU 0.0068) |
| 45그룹×2플랫폼(90셀) 전체의 그룹 내 점수 SD | 중앙값 0.5382, 범위 0.3002–0.7391 |

### 그룹별 표 (min gap 오름차순, 상위 5)

| 세트@rate | gen | 후보 | GPU own_best | GPU cross | GPU gap | gap/sd | NPU own_best | NPU cross | NPU gap | gap/sd |
|---|---|---|---|---|---|---|---|---|---|---|
| S8@2.0 | Y | 24 | 1.6925 | 1.6899 | **0.0026** | 0.005 | 1.8615 | 1.7809 | **0.0806** | 0.164 |
| base3@3.0 | Y | 8 | 2.0000 | 1.1352 | **0.8648** | 1.343 | 1.6201 | 1.6133 | **0.0068** | 0.013 |
| S7@2.0 | Y | 20 | 1.8289 | 1.6844 | **0.1445** | 0.237 | 1.6956 | 1.6781 | **0.0175** | 0.036 |
| S10@2.0 | − | 24 | 0.8553 | 0.8231 | **0.0322** | 0.094 | 1.0000 | 0.6084 | **0.3916** | 1.236 |

전체 11행은 `B_divergence_gap/gap_table.md`, 기계 판독은 `gap_facts.json`.

> 판정 문장은 쓰지 않았다. 위 수치가 "측정 노이즈로 설명 가능한가"를 논할 재료다 —
> 특히 (i) 가장 작은 gap 도 tie 폭의 10⁶ 배라는 점, (ii) 22셀 중 2셀만 그룹 내 SD 의 5%
> 미만이라는 점, (iii) 두 플랫폼 중 **한쪽의 gap 이 작으면 다른 쪽은 큰** 비대칭이
> 반복된다는 점(S8@2.0, base3@3.0, S10@2.0 모두).

---

## C. cluster bootstrap 95% CI (#10)

리샘플링 단위 = **워킹셋 15개**(한 세트가 rate 3수준 = 3그룹을 통째로 이동). B=10,000,
seed 42, 백분위수 CI. 점추정이 저장된 P1 값·divergence 11/45 와 일치함을 검산했다.

| 지표 | 점추정 | 95% CI |
|---|---|---|
| direct 그룹평균 Spearman — GPU | +0.986 | [+0.982, +0.990] |
| direct 그룹평균 Spearman — NPU | +0.972 | [+0.953, +0.986] |
| direct Top-1 — GPU | 0.933 | [0.867, 1.000] |
| direct Top-1 — NPU | 0.867 | [0.756, 0.956] |
| **divergence 비율 (11/45)** | **0.244** | **[0.089, 0.422]** |
| decomposed Top-1 — GPU | 0.911 | [0.844, 0.978] |
| decomposed Top-1 — NPU | 0.622 | [0.422, 0.800] |
| greedy(speedup) Top-1 하한 — GPU | 0.311 | [0.156, 0.467] |
| greedy(speedup) Top-1 하한 — NPU | 0.244 | [0.111, 0.378] |

### paired 차이 (같은 리샘플에서 계산)

| 비교 | 차이 | 95% CI | CI가 0 제외 |
|---|---|---|---|
| direct − greedy(speedup) — GPU | +0.622 | [+0.489, +0.756] | **예** |
| direct − greedy(speedup) — NPU | +0.622 | [+0.422, +0.800] | **예** |
| direct − greedy(load_factor) — GPU | +0.622 | [+0.489, +0.756] | **예** |
| direct − greedy(load_factor) — NPU | +0.667 | [+0.511, +0.800] | **예** |
| direct − decomposed — NPU | +0.244 | [+0.044, +0.467] | **예** |
| direct − decomposed — GPU | +0.022 | [+0.000, +0.067] | **아니오** (하한이 정확히 0) |

**읽을 점**: greedy 대비 우위와 NPU 에서의 decomposed 대비 우위는 세트 단위 리샘플링에서도
CI 가 0 을 넘지 않는다. **GPU 에서의 decomposed 대비 우위(+0.022 = 1그룹)는 CI 하한이
0 에 붙어 유의하지 않다** — "direct가 decomposed를 이긴다"는 주장은 NPU 에서만 통계적으로
뒷받침된다. divergence 비율의 CI 는 [0.089, 0.422] 로 넓다(세트 15개 기준의 한계).

산출물: `C_bootstrap/bootstrap_facts.json`, `bootstrap_table.md`, `per_group_metrics.csv`.

---

## D. greedy worst-case bound (#11)

분자(맞힌 그룹 수)는 그대로 두고 분모만 45로 바꾼 하한.

| 플랫폼 | 정렬 | 보고치 (측정분) | **하한 (45그룹)** | 미측정 | 상한 (참고) |
|---|---|---|---|---|---|
| GPU | load-factor | 0.341 (14/41) | **0.311 (14/45)** | 4/45 | 0.400 (18/45) |
| GPU | speedup | 0.424 (14/33) | **0.311 (14/45)** | 12/45 | 0.578 (26/45) |
| NPU | load-factor | 0.250 (9/36) | **0.200 (9/45)** | 9/45 | 0.400 (18/45) |
| NPU | speedup | 0.333 (11/33) | **0.244 (11/45)** | 12/45 | 0.511 (23/45) |

네 셀 모두 보고치가 원고 Table 5 와 일치함을 검산했다. **하한에서는 speedup 과
load-factor 의 GPU 우열이 사라진다**(둘 다 14/45) — 현행 본문의 "speedup 정렬이 두
플랫폼 모두에서 더 높은 Top-1"은 측정분 분모에서만 성립하고, 하한에서는 GPU 동률·NPU 만
우세다. Table 5 에 "Top-1 (lower bound, /45)" 열을 하나 추가하면 그대로 실을 수 있다.

산출물: `D_greedy_bound/bound_table.md`, `bound_facts.json`.

---

## E. r3 집계식 확정 (#16) — **원고에 정의문 추가 필요**

### 판정 (측정 시점)

> **r3 = 창 안 모든 scheduled view 의 `avg_tokens_per_s` 단순 합.**
> 디바이스 필터도 모델 필터도 **없다** — CPU 에 배치된 생성 모델의 토큰도 그대로 더해진다.
> 생성 모델이 2개인 세트(base1–5, S5–S8)에서는 **두 값이 모두 더해진다**(특정 모델만이 아니다).
> vision 핸들러는 `avg_tokens_per_s` 속성이 없어 0.0 으로 들어갈 뿐이다.

근거:
- `unified_viewer.py:1292-1294` — `tokens_by_view = {v: getattr(<v>_handler,
  "avg_tokens_per_s", 0.0) for v in self.view_names}`
- `unified_viewer.py:1296` — `total_tokens_per_s = sum(tokens_by_view[v] for v in
  scheduled_views)` — **execution 조건이 전혀 없다**
- `unified_viewer.py:1337` — `total.total_tokens_per_s` 로 기록 (2자리 반올림)
- `view_handlers.py:357,428,439` — `avg_tokens_per_s` 는 생성 핸들러만 갖는다

### ⚠ 저장 파일의 **뷰별** 토큰은 측정값이 아니다

수집 JSONL 은 뷰별로 `(execution, throughput_fps, inference_count)` 만 저장하고
**뷰별 토큰을 버린다**(`collect_run.py:122-123`). `jsonl_to_windows.py:32-44` 가 창 총계
y3 를 **가속기 위 생성 뷰 하나에 통째로 얹고** 나머지 뷰를 0 으로 둔다(그 의도는 스크립트
docstring `:1-11` 에 명시). 그래서:

- 저장 파일에서 토큰이 0 이 아닌 뷰는 창당 **최대 1개**이고, 그것은 항상 가속기 생성 뷰다
- 생성 2모델 창 696개 중 **두 뷰가 모두 0 이 아닌 창은 0개** — 이는 "두 모델이 동시에
  생성하지 않았다"는 뜻이 **아니라** 재구성 산물이다
- **저장 파일의 뷰별 토큰으로 "어느 생성 모델이 얼마를 냈는지"는 알 수 없다.**
  창 총계만이 실측이다

### LLM 이 CPU 에 배치된 창

- llama1b 가 CPU 인 창: 390개(두 플랫폼 합). 저장 파일의 뷰별 토큰은 전부 0
- 그중 **가속기에 생성 모델이 하나도 없는 창**(= S3 에서 llama1b 가 CPU): 플랫폼당 12개.
  이 창들의 **창 총계 r3 는 0 이 아니다** — 0.0 ~ 2.02 tok/s (평균 0.535, 정확히 0 인 창은 24개 중 6개).
  즉 CPU 위의 LLM 도 실제로 토큰을 냈고 그 값이 총계에 들어가 있다

### 저장소에 r3 관례가 **두 개** 있다

| 경로 | r3 |
|---|---|
| 파이프라인 (P1·P4·P5·P6) | `featurize_window` 가 **뷰별 tokens_per_s 의 합** (`deploy_selector_xgb_suite.py:323`) |
| 기저 감사·Figure 2 | `total.total_tokens_per_s` (**창 총계**) (`build_basis_audit.py:84`) |

두 값이 갈리는 창은 **18개**(플랫폼당 9개)이고 **전부 S3** — 가속기 생성 뷰가 없어
재구성이 총계를 아무 뷰에도 얹지 못한 창이다. 갈린 창에서 뷰별 합은 0, 총계는 최대 2.02.

**수치 영향은 없다**: 이 창들이 속한 S3 그룹의 정규화 분모 `max(r3)` 는 두 관례에서
**동일하다**(같은 그룹에 가속기-LLM 후보가 있어 그쪽이 최댓값). 관례가 달라도 max 가
바뀌는 그룹은 **0개**. 따라서 published 수치는 어느 관례든 그대로다.

### y3 학습 마스킹과의 정합 — **일치**

`y3_valid` = 372/540 (양 플랫폼), 저장된 `groupkfold_*_metrics.json` 의
`y3.n_rows_evaluated = 372` 와 일치. 마스킹된 168창 = **vision-only 세트 156창 +
생성 세트인데 생성 모델이 전부 CPU 인 12창**(전부 S3). 완전히 분해된다.

### 원고에 넣을 정의문 (제안)

> $r_3$ is the sum of the token rates of every scheduled instance in the window; an
> instance that is not generative contributes zero. A generative instance placed on the
> CPU is included. The training target $y_3$ is masked for windows in which no generative
> instance runs on the accelerator (168 of 540 per platform: 156 vision-only workloads and
> 12 in which the LLM was placed on the CPU), because the measured token rate of a
> CPU-only generation is not a usable label.

산출물: `E_r3_definition/r3_facts.json`.

---

## F. regret / oracle ratio (#20)

`oracle ratio` = 그 방법이 고른 배치의 **측정** 점수 / 그룹 최고 측정 점수, `regret` = 1 − ratio.
점수 기저는 파이프라인과 동일한 그룹 정규화(α=0.3, β=1.0). direct·decomposed 의 Top-1 이
저장된 P1·P6 값과 정확히 일치함을 검산했다(합성 재현이 충실함).

| 방법 | 플랫폼 | 분모 | Top-1 | **Oracle ratio 평균** | 중앙값 | p10 | 최소 | regret 평균 | ratio<0.9 |
|---|---|---|---|---|---|---|---|---|---|
| direct (P1) | GPU | 45 | 0.933 | **0.996** | 1.000 | 1.000 | 0.921 | 0.004 | 0 |
| direct (P1) | NPU | 45 | 0.867 | **0.993** | 1.000 | 0.976 | 0.910 | 0.007 | 0 |
| decomposed (P6) | GPU | 45 | 0.911 | **0.980** | 1.000 | 1.000 | 0.249 | 0.020 | 1 |
| decomposed (P6) | NPU | 45 | 0.622 | **0.951** | 1.000 | 0.883 | 0.455 | 0.049 | 6 |
| greedy (speedup) | GPU | 33 측정분 | 0.424 | **0.734** | 0.782 | 0.397 | 0.089 | 0.266 | 19 |
| greedy (speedup) | GPU | 45 (미측정=0) | 0.311 | 0.538 | 0.567 | 0.000 | 0.000 | 0.462 | 31 |
| greedy (speedup) | NPU | 33 측정분 | 0.333 | **0.793** | 0.926 | 0.443 | 0.029 | 0.207 | 15 |
| greedy (speedup) | NPU | 45 (미측정=0) | 0.244 | 0.581 | 0.686 | 0.000 | 0.000 | 0.419 | 27 |
| greedy (load_factor) | GPU | 41 측정분 | 0.341 | **0.575** | 0.531 | 0.074 | −0.154 | 0.425 | 27 |
| greedy (load_factor) | GPU | 45 (미측정=0) | 0.311 | 0.524 | 0.522 | 0.000 | −0.154 | 0.476 | 31 |
| greedy (load_factor) | NPU | 36 측정분 | 0.250 | **0.590** | 0.569 | 0.021 | −0.156 | 0.410 | 23 |
| greedy (load_factor) | NPU | 45 (미측정=0) | 0.200 | 0.472 | 0.419 | 0.000 | −0.156 | 0.528 | 32 |

**Top-1 이 놓치는 것을 이 표가 보여준다:**

- direct 는 Top-1 을 틀린 그룹에서도 **최적의 99% 이상**을 얻는다(평균 ratio 0.996/0.993,
  최악 0.921/0.910, ratio<0.9 인 그룹 **0개**). "틀려도 거의 손해가 없다"가 정량화된다.
- decomposed 는 NPU Top-1 이 0.622 로 크게 낮지만 평균 ratio 는 0.951 이다 — 다만
  **최악이 0.455**이고 ratio<0.9 인 그룹이 6개라 꼬리가 direct 보다 두껍다.
- greedy 는 평균 ratio 가 0.47–0.79 로 뚜렷이 낮고, 정규화 점수 기준으로 **음수 ratio**
  까지 나온다(load_factor 최소 −0.156). 정규화 점수가 음수일 수 있어 ratio 하한이 0 이
  아니라는 점은 표 각주에 필요하다.

산출물: `F_regret/regret_table.md`, `regret_facts.json`, `per_group_{direct,decomposed}_{gpu,npu}.csv`.

---

## 원고 반영이 필요한 판정 (요약)

1. **A — joint 숫자 교체.** paired fold 로 재학습하면 NPU joint Top-1 0.844 → **0.867**,
   NPU 그룹 Spearman 0.974 → **0.984**. GPU 는 사실상 불변(0.933 / 0.975→0.976).
   하락 없음. 구 분할에서 45쌍 중 **27쌍이 갈라져 있었다**는 사실도 방법 절에 한 줄
   가치가 있다. 교체 시 Abstract 의 "84.4"와 Fig. transfer 의 joint 막대가 함께 바뀐다.
   부수 효과로 **NPU 에서 joint = platform-specific (둘 다 0.867)** 이 되므로 "approaches"
   를 "matches on the NPU"로 조정할 수 있다.
2. **E — r3 정의문 추가.** r3 는 창 안 모든 인스턴스 토큰율의 합이고 CPU 배치 생성
   인스턴스도 포함한다. 그리고 y3 마스킹 168창의 구성(vision-only 156 + CPU-LLM 12)을
   명시해야 "LLM 이 CPU 면 r3 는 무엇이냐"는 질문이 닫힌다. 제안 문장은 E 절에 있다.
   (저장 파일의 뷰별 토큰이 재구성 산물이라는 점은 원고에 쓸 필요는 없지만, 향후 뷰 단위
   토큰 분석을 하려면 **재수집이 필요하다**는 뜻이다.)
3. **D — Table 5 에 하한 열.** 현행 본문의 "speedup 이 두 플랫폼 모두에서 더 높다"는
   하한에서는 GPU 동률(14/45 = 14/45)이 되므로, 하한을 실으면 문장을 "on the NPU"로
   좁혀야 한다.
4. **C — 유의성 주의.** GPU 에서 direct 가 decomposed 를 앞서는 폭(+0.022 = 1그룹)은
   세트 단위 CI 하한이 0 이다. "direct outperforms the decomposed baseline"을 무조건
   서술하면 GPU 에서는 뒷받침되지 않는다 — NPU 는 CI [+0.044, +0.467] 로 명확하다.

## 실행·검산 요약

| 항목 | 검산 |
|---|---|
| A | 구 분할 재실행이 저장된 `unified_metrics.json` 을 양 플랫폼 재현 ✅ |
| B | 불일치 11그룹이 F1 README 의 기대 목록과 정확히 일치(assert) ✅ |
| C | 점추정이 저장 P1 Top-1·Spearman·divergence 11/45 와 일치 ✅ |
| D | 네 셀 보고치가 원고 Table 5 와 일치 ✅ |
| E | 13개 검산 전부 통과 (y3_valid 372, 마스킹 168 = 156+12 분해 포함) ✅ |
| F | direct·decomposed Top-1 이 저장 P1·P6 값과 정확히 일치 ✅ |

각 항목은 독립 실행 가능하다(`A_joint_paired/run_paired.py`,
`B_divergence_gap/gap.py`, `C_bootstrap/bootstrap.py`, `D_greedy_bound/bound.py`,
`E_r3_definition/check_r3.py`, `F_regret/regret.py`). C 는 F 의 per-group CSV 를 읽으므로
F 다음에 실행한다.

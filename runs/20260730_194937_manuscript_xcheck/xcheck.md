# 원고 미러 대조 — UNKNOWN 3건 종결 (읽기 전용)

- 작성: 2026-07-30
- 대상: `manuscript/mlforsys_main.tex` (353행, 25,926 B) — **읽기만 함. 수정 없음.**
- 산출물: 이 문서 + `run_manifest.json` + `r1_convention_recheck.json`
- 원고는 이 저장소에서 편집하지 않는다(파일 자체 정책, 아래 §1). 커밋 대상은 이 run 디렉토리뿐이다.

---

## §1. 미러 최신성 — **확인, 진행**

`mlforsys_main.tex:5-8`:

```
% REVISION: 2026-07-30 r2 -- declared-basis numbers (11/45), Fig.1 removed,
%   Appendix D/E added, 4-page body confirmed, fig_divergence v2
%   (sha256 3c13b1df...) embedded. Style file: neurips_2025.sty
%   (PENDING swap to neurips_2026.sty per CFP).
```

`REVISION: 2026-07-30 r2` 존재 → 중단 조건 미해당, 대조 진행.

**부가 검증 (요구 항목 아님, 자발적)**: r2 주석이 명시한 임베드 그림 해시가 저장소의 v2 그림과
일치한다.

```
sha256(analysis/figures/fig_divergence.pdf)
  = 3c13b1dfe82814c0df6a7a4b7b715b35cd08912c316eaa215e0a575c098f1fd5
```

앞 8자리 `3c13b1df`가 원고 주석과 **일치**. 즉 원고에 박힌 Figure 2는
`runs/20260730_192257_fig_divergence_v2`에서 생성한 선언 기저 그림(오렌지 셀 11개)과
동일 파일이다.

또한 파일 상단 정책 주석(`:1-4`)이 "이 .tex는 Claude 채팅 세션에서만 편집하며 저장소에서
편집하지 않는다"고 규정한다 — 본 작업의 읽기 전용 범위와 일치.

---

## §2. 식 (1) · divergence 기저 — **정규화 기준으로 확정. 생성기와 일치**

### 원고 인용

**식 (1) 정의부** (`:93-99`):

> **Normalization and ranking.** A workload group contains all candidate placements for the
> same model set and request rate configuration. Within each group, $r_1$ and $r_3$ are
> divided by their maxima, while $r_2$ is scaled using its minimum and maximum. We denote
> the normalized targets by $y_1$, $y_2$, and $y_3$. Candidates are ranked using
> $$S(\mathbf{x}) = y_1(\mathbf{x}) - \alpha y_2(\mathbf{x}) + \beta y_3(\mathbf{x})$$
> where $\alpha=0.3$ and $\beta=1.0$ are selected by the operator rather than learned. The
> $\beta$ term is omitted when a workload has no generative model.

**divergence가 이 기저를 쓴다는 명시 문장** (`:99`, 같은 단락 말미):

> **All measured-best placements and divergence counts below use this same normalized
> score; comparing raw measurements across platforms would instead let unit-scale
> differences dominate.**

**Abstract** (`:69`):

> **Under the same group-normalized score used for ranking**, the measured-best placement
> differs across the two accelerators in 11 of 45 workload groups, concentrated in
> workloads that include generative models (10 of 33 generative groups versus 1 of 12
> vision-only groups)

### 판정

원고는 divergence를 **정규화 점수 기준으로 서술한다** — 그것도 명시적으로, 그리고 raw 대조를
"단위 스케일이 지배한다"는 이유로 배제한다는 문장까지 포함한다. 이는
`runs/20260730_190154_score_basis_audit`의 §E-3 원인 분석(GPU y3 최대 180.8 tok/s vs NPU
22.6 tok/s, 8배 격차)과 동일한 논지다.

### 생성기 기저와의 일치

| 항목 | 원고 식 (1) | 생성기 |
|---|---|---|
| 그룹 정의 | same model set + request rate configuration | `(set, rate)` |
| y1, y3 | 그룹 최대로 나눔 | `r1/max(r1)`, `r3/max(r3)` |
| y2 | 그룹 min–max 스케일 | `(r2−min)/(max−min)`, max==min이면 0 |
| α, β | 0.3, 1.0 (운영자 선택, 학습 대상 아님) | `ALPHA=0.3`, baseline β=1.0 |
| β 항 | 생성 모델 없는 워크로드에서 생략 | `has_gen`일 때만 가산 |

생성기: `scripts/make_figures.py --fig f1 --basis declared_normalized` (Figure 2),
`scripts/compare_platforms_v2.py --basis declared_normalized` (수치·문서). 두 생성기 모두
점수 정의를 `runs/20260730_190154_score_basis_audit/build_basis_audit.py`의
`normalize()/score()/tie_sets()`에서 import하며 자체 재구현하지 않는다. **기저 일치 확인.**

### 원고 수치 전수 대조 결과 (전부 일치)

| 원고 위치 | 원고 값 | 대조 소스 | 일치 |
|---|---|---|---|
| Abstract, `:128`, Fig.2 캡션 | 11 of 45 | `part2a_declared_basis_divergence` | ✓ |
| Abstract, `:128`, `:136` | 10/33 생성, 1/12 vision | 동 | ✓ |
| `:128` | β=0.5 → 16 of 45 | 동 | ✓ |
| `:128`, `:284` | exhaustive 8 (8/24 생성, 0/6 vision) | `part2b_exhaustive` | ✓ |
| `:285` | 13/30 at β=0.5 | 동 | ✓ |
| `:126`, `:155` | ρ .986/.972, Top-1 .933/.867, Top-5 1.000 | `groupkfold_{gpu,npu}_metrics.json` | ✓ |
| `:128`, Appendix D | β>0에서 GPU 최대 3 / NPU 최대 9 | `part2d_beta_split` | ✓ |
| Appendix D 표 (`:344-348`) | 25개 셀 전부 (GPU/NPU) | `sweep_declared_basis.csv` | ✓ **25/25** |
| Appendix D | β=0 열 GPU 9 / NPU 11 | `part2d_beta_split` | ✓ |
| Appendix E 표 (`:300-310`) | 15세트 × (N, 2^N, feasible, measured, mode) | `appendix_coverage.csv` | ✓ **75/75 셀** |
| Appendix E (`:273-275`) | feasible = 2^(N−#VLM), VLM 없는 세트는 2^N | coverage §B-2 | ✓ |
| Appendix E (`:278-280`) | 측정 placement 집합 대칭차가 모든 (set, rate)에서 0 | `coverage_facts.A_all_zero` | ✓ |
| Appendix E (`:286-287`) | S9·S10이 6그룹 중 1개 불일치 (S10 최고 rate) | `S10@2.0` | ✓ |
| Table 1 (`:153-155`) | greedy .424/.333, decomposed .931/.911 · .911/.622 | `greedy_summary.md`, `decomposed_metrics.json` | ✓ |
| `:164` | 전이 .902/.927, Top-1 .800; 합동 .975/.974, .933/.844 | `cross_platform_metrics.json`, `unified_metrics.json` | ✓ |
| `:175` | 분해 per-view .993/.992, y2 재구성 .834/.763 | `decomposed_metrics.json` | ✓ |

**원고 헤드라인 수치와 저장소 산출물 사이 불일치는 0건이다.** 앞선 run들이 우려한 "raw 기저
수치가 원고에 남아 있을 가능성"은 해소되었다 — 원고는 이미 선언 기저 값(11/45 등)으로 개정되어
있고, 폐기된 29/45·30/45는 원고 어디에도 나타나지 않는다.

---

## §3. r1 규약 — **원고는 vision 뷰 합, 생성기 주 계산은 전 뷰 합. 논문 수치 무영향 (양 β 재확인)**

### 원고 인용 (`:91`)

> **Performance targets.** Each workload and placement pair uses a 20\,s warmup followed by
> a 180\,s measurement window. We record **aggregate vision throughput $r_1$**, mean
> deadline miss rate $r_2$, and the token rate $r_3$ of a generative model on the
> accelerator.

즉 원고의 $r_1$은 **vision 뷰의 처리량 합**이다. 생성 뷰(LLM/VLM)의 fps는 $r_1$에 들어가지
않는다 — 이는 `deploy_selector_xgb_suite.featurize_window`(`:320-321`, `_is_vision` 분기)의
`y1_vision_fps`와 정확히 같은 정의다.

### 생성기 주 계산은 전 뷰 합

`build_basis_audit.py`의 주 경로는 `normalize(rows, "r1_totals")`, 즉
`window["total"]["total_throughput_fps"]`를 쓴다. 이 값은 수집 JSONL의 `y1`으로,
`jsonl_to_windows.py`가 **전 뷰 fps 합**으로 기록한 것이다. 따라서 원고 정의(vision 뷰 합)와
생성기 주 계산(전 뷰 합)은 **정의상 다르다**. 차이가 나는 행: GPU 443/540, NPU 461/540,
최대 절대차 GPU 0.86 fps · NPU 0.45 fps (생성 뷰 fps가 작기 때문).

### 재확인 결과 — 양 β 모두 불일치 그룹 **집합**이 동일

지시대로 두 β 모두에서 재확인했다 (개수 일치가 아니라 **집합 동일성**):

| β | r1 = 전 뷰 합 | r1 = vision 뷰 합 | 집합 동일 | 차집합 |
|---|---|---|---|---|
| 1.0 | 11 (생성 10, vision 1) | 11 (생성 10, vision 1) | **동일** | 양방향 공집합 |
| 0.5 | 16 (생성 15, vision 1) | 16 (생성 15, vision 1) | **동일** | 양방향 공집합 |

감사 JSON(`basis_facts.json`의 `part2a_r1_vision_variant`,
`part2a_declared_basis_divergence`)과도 양 β에서 일치 → **중단 조건 미해당.**
근거 데이터: `r1_convention_recheck.json` (본 run).

### 판정

**논문 수치 무영향.** 원고의 r1 정의와 생성기 주 계산이 다르지만, 그룹 내 정규화 후
플랫폼 간 argmax 비교라는 사용 방식에서 두 규약은 동일한 불일치 그룹 집합을 낸다. 원고에
실린 11/45·16/45·10/33·1/12는 어느 규약에서도 성립한다.

단, 이는 "수치가 같다"는 사실 확인이지 "정의가 같다"는 뜻이 아니다. 재현 코드를 공개할
경우 `r1_totals`가 원고 문면과 다르다는 점이 드러나므로, 원고 §Performance targets에
반 문장을 넣거나(예: 생성 뷰 fps는 $r_1$에서 제외되며 그 크기가 무시할 수준임을 명시)
생성기 주 경로를 `r1_vision`으로 바꾸는 편이 안전하다. **어느 쪽을 택할지는 채팅 세션
결정 사항이며 본 작업 범위 밖이다** (원고 미수정, 생성기 기본값 미변경).

---

## §4. tie 규칙 — **원고에 정성적 서술 있음, 수치 허용오차는 코드 규약**

앞선 run들이 "원고 무서술"로 가정했으나, 실제로는 **두 곳에 tie 서술이 있다**:

**`:121` (Measured oracle 정의)**:

> **Measured oracle.** The oracle selects the candidate with the highest measured score $S$
> in each workload group and **treats tied maxima as correct**.

**`:319` (Appendix D)**:

> counts, per grid point, the working sets (of 15) whose group optimum leaves the baseline
> optimum set ($\alpha{=}0.3$, $\beta{=}1.0$; **tie-aware**).

### 판정

- 원고는 tie 처리를 **정성적으로 규정**한다: 동점 최대값 전부를 정답으로 간주(= tie set),
  그리드점 비교는 "baseline optimum **set**을 벗어나는지"로 판정. 이는 본 분석·생성기가 쓰는
  규칙(tie set 교집합이 비면 변경/불일치)과 **의미상 동일**하다.
- **수치 허용오차(1e-9)는 원고에 없다.** 코드 규약
  (`analysis_common.ranking_metrics`, `build_basis_audit.TIE_EPS`)이 유일한 출처다.
- 실측 영향: 선언 기저 45그룹 · 양 β에서 tie 발생 그룹은 **0개**이므로, 허용오차 값이
  현재 보고된 어떤 수치도 바꾸지 않는다.
- 따라서 기록은 **"원고: tie set 정성 서술 있음(`:121`, `:319`) / 수치 허용오차 1e-9는 코드
  규약, 원고 무서술"**이다. 부록에 반 문장(예: 동점은 1e-9 이내를 동일 최적으로 취급)을
  추가할지는 채팅 세션 결정 사항.

---

## §5. 앞선 4개 run의 UNKNOWN — 종결 현황

| # | 출처 run | UNKNOWN 내용 | 상태 | 근거 |
|---|---|---|---|---|
| 1 | `20260730_172053_coverage` | 원고 `.tex` 부재로 tie-aware argmax 문면 대조 불가 | **종결** | §4 — `:121` "treats tied maxima as correct", `:319` "tie-aware". 코드 규약과 의미 일치 |
| 2 | `20260730_184450_coeff_sweep` | 원고 문면·스윕 범위·"measured-score"의 스케일 정의 확인 불가 | **종결** | §2 — `:99` "use this same normalized score". 스윕 범위는 Appendix D(`:315-328`)가 α∈{0…0.6}×β∈{0…1.0}, β=0 열을 "degenerate, 비대칭 주장에서 제외"로 명시 |
| 3 | `20260730_184450_coeff_sweep` | "F3 rerun"이 지칭하는 계획 문서·그리드 확인 불가 | **부분 종결** | Appendix D 표가 본 작업의 사전 등록 그리드(α 5점 × β 5점)와 25/25 셀 일치. 단 "F3"이라는 식별자는 원고에도 없음 → 식별자 자체는 여전히 UNKNOWN |
| 4 | `20260730_190154_score_basis_audit` | 식 (1)의 r1 정의 (전 뷰 합 vs vision 뷰 합) | **종결** | §3 — `:91` "aggregate vision throughput $r_1$" = vision 뷰 합. 생성기 주 경로와 다르지만 양 β에서 수치 무영향 |
| 5 | `20260730_190154_score_basis_audit` | 원고가 divergence를 식 (1) 기준으로 서술하는지 | **종결** | §2 — `:99`가 명시. raw 대조를 이유와 함께 배제 |
| 6 | `20260730_190154_score_basis_audit` | 29/33·0/12 gen/vision 분해의 원 산출 스크립트 | **미종결** | 원고는 값을 쓰지 않고(선언 기저 10/33·1/12로 개정됨) 산출 경로도 밝히지 않음. 저장소 코드에도 분해 로직 없음 |
| 7 | `20260730_192257_fig_divergence_v2` | (UNKNOWN 없음 — 구판 참조 목록만 보고) | — | 작업 3(`88ab051`)에서 정리 완료 |

**종결 5건 / 부분 종결 1건 / 미종결 1건.**

미종결·부분 종결 2건은 모두 "원 산출 경로·식별자"에 관한 것으로, 현재 보고 수치의 정확성에는
영향이 없다(값은 모두 재현 확인됨).

---

## §6. 대조 중 발견한 원고 쪽 검토 후보 (수정하지 않음 — 보고만)

원고 미수정 범위이므로 목록만 남긴다. 수치 오류는 아니다.

| 심각도 | 위치 | 내용 |
|---|---|---|
| 중간 | `:101` | "device memory determines placement feasibility"라고 서술하지만, 후보 생성 코드(`gen_collection_schedules.all_placements`)에는 **메모리 제약이 존재하지 않는다**. feasible 수를 줄이는 요인은 VLM 가속기 고정 하나뿐이며 Appendix E(`:273`)도 `2^(N−#VLM)`으로 그렇게 쓴다. `:101`은 테스트베드 특성 서술로 읽히지만, feasible 정의의 근거로 오독될 수 있다 (coverage run §B-2가 동일 지적) |
| 낮음 | `:270`, `:315` | 부록 소절 순서가 A, B, C, **E, D**다. `\subsection*{E...}`가 `\subsection*{D...}`보다 앞에 온다. 참조는 이름 기반(`Appendix~D`/`~E`)이라 깨지지 않지만 독자에게는 역순으로 보인다 |
| 낮음 | `:21`, `:8` | `neurips_2025.sty` 사용 중이며 주석이 2026 CFP에 따른 교체를 PENDING으로 표시 |
| 정보 | `:15-16`, `:52-53` | `\RES`/`\TODO` 매크로 정의는 남아 있으나 본문·부록에 **사용처 0건** (grep 확인) → 제출 전 "red 없음" 조건 충족 |
| 정보 | `:61` | `\And \TODO{co-authors}`가 주석 처리된 상태 (저자 1인) |

---

## UNKNOWN (본 작업에서 새로 남는 것)

없음. §5의 미종결 2건은 본 작업 이전부터의 항목이며, 원고에서도 확인 불가함을 §5에 근거와
함께 기록했다.

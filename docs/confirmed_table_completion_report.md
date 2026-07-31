# 확정표 등재 완결 — 논문이 인용하는 값 (v37)

작성 2026-07-31. **커밋함, 푸시 없음.** 논문 문안은 쓰지 않았다 — 등재 목록과 주석 위치만 낸다.

산출물: `fig/confirmed_values.json` **+63항목**(409 → 472), 주석 목록 확장(16 → 17),
`scripts/{emit_v33_confirmed,aggregate_v33_campaign,classify_untracked}.py`,
`scripts/check_paper.py` 커버리지 집계 보강.

---

## 0. 한 줄 요약

**값이 없는 것이 아니라 등재가 안 된 것이었다.** 63항목을 등재했고 전부 `source`가 런 디렉터리를
가리킨다. 그 대가로 지워야 했던 Q4 문장은 이제 **쓸 수 있다** — 다만 재실행값은 게재본과 결론이
반대다(§2.1). 등재해도 남는 것은 **36개 수치, 커밋된 런으로 되짚을 수 없다**(§5). 그중 둘은
어느 리포트에도 없다.

---

## 1. 작업 194 — 미등재 목록과 등재 판정

`check_paper.py`의 미추적 목록에서 **논문이 인용하는 실측값**만 골랐다. 판정 기준은 하나다 —
**커밋된 런 디렉터리로 되짚을 수 있는가.** 리포트만 가리키는 값은 등재하지 않는다: 리포트는
파생물이고, `source`가 리포트를 가리키면 감사 체인이 그 자리에서 다시 끊긴다(`runs/README.md`).

| 대상 | 출처 | 런 디렉터리 | 등재 |
|---|---|---|---|
| Q4 5변형 종단 $V$·cumV·전환 횟수 | `main_rerun_report.md` §1 | `runs/*_q4_*` 25런 | **예** |
| Q4 5변형 `maxV` | 리포트 표에 **없음** | 같음 | **예**(재계산, 표시함) |
| Q4 포화 가드 기각 수 | `main_rerun_report.md` §3 | `runs/*_q4_*` executor.log | **예** |
| Q4 후보별 서비스율 | 게재값은 증발한 `q4_bsf_out` | `runs/*_q4_BoundGuard_*` 로그에 실측 | **예**(v33 값) |
| Q6 5변형 cumV·`maxV` | `main_rerun_report.md` §1 | `runs/*_q6_*` 25런 | **예** |
| §4b 5변형 cumV·전환 횟수 | 같음 | `runs/*_q4b_*` 25런 | **예** |
| Q3 baseline 종단 $V$·`maxV`·cumV | 같음 | `runs/*_q3_*` 20런 | **예** |
| Q5 baseline 종단 $V$·cumV | 같음 | `runs/*_q5_*` 20런 | **예** |
| Q5 컨트롤러 활성 전 구간 9.6 s | `main_rerun_report.md` §4 | `compute_bound.py` ← `runs/*_q5_BoundGuard_*` | **예** |
| Q1.1 전환 시 서비스 공백 | `switching_lossless_report.md`(07-24) | **없음** | 아니오 |
| Q4 YOLO11x 단독 ≈72 fps | **어느 리포트에도 없음** | 없음 | 아니오 |
| vision-4 구성 비교 5.2 / 3.4 | **어느 리포트에도 없음** | 없음 | 아니오 |
| 생성 모델 이전 $\delta\approx2.9$ s | `hotswap_resource_report.md` | 없음 | 아니오 |
| 2-gen 사례 수치 | 미상 | 없음 | 아니오 |
| Q5 NPU 경합 7 / 34–49 / 4.4× | `q5_experiment_report.md` §2.2 | 없음, **게다가 λ=80 사전측정**(본 실험 λ=45) | 아니오 |
| 유체 시뮬레이션 오차 8% | `d2_simulation_report.md` | 시뮬레이션 | 아니오 |

### 창 길이 94 s는 등재하지 않았다

지시문 §195-1이 열거했으나, 이 수는 **이미 확정표에 있다** — 모든 해당 항목의 `window_s` 필드가
그것이다. 값으로 한 번 더 등재하면 같은 사실이 두 곳에 생기고, 둘이 갈라질 수 있다. §194가
정의상의 상수를 제외하라고 한 것과 같은 이유로 상수 분류에 넣었다.

---

## 2. 작업 195 — 등재

**63항목.** 44건은 `main_rerun_report.md` 표에서 **전사**, 14건은 리포트 표에 없는 양이라
커밋된 런에서 **재계산**(`note`에 그렇게 적었다), 5건은 개별 사유를 단다.

생성기 `scripts/emit_v33_confirmed.py`. 재실행하면 **덮어쓰지 않고 거부**한다 — v37은 등재이지
정정이 아니다.

### 값이 되짚어지는지 먼저 확인했다

`main_rerun_report.md`의 표를 만든 스크립트는 커밋되지 않았다. `source`가 런을 가리키려면 그
런에서 값이 다시 나와야 하므로 `scripts/aggregate_v33_campaign.py`를 써서 30셀 전부를 다시 계산했다
(고정 94 s 창, `v_score` 시계열 + executor.log).

**종단 $V$·`maxV`·전환 횟수는 정확히 일치**하고, **cumV만 ≤2 V·s(≤0.4%) 낮게** 나온다 —
적분 끝점 규약 차이다. 확정표에는 **리포트 값을 그대로** 실었고(지시문 §195-3), 차이는 여기 적는다.

| 셀 | 리포트 cumV | 재계산 | 차 |
|---|---|---|---|
| Q4 BoundGuard | 17059 | 17060 | +1 |
| Q4 C | 17724 | 17725 | +1 |
| Q4 Adaptive | 23746 | 23748 | +2 |
| Q6 BoundGuard | 303 | 304 | +1 |
| Q6 B / C / Adaptive | 265 / 216 / 264 | 266 / 217 / 265 | +1 |
| Q3 BoundGuard / Adaptive | 280 / 292 | 281 / 291 | +1 / −1 |
| Q5 BoundGuard | 382 | 382 | 0 |
| 나머지 22셀 | — | — | **0** |

### 스키마

`censored`와 `recovered`는 독립이므로 셀마다 따로 판정했다. 특히:

- **Q4 전 변형** — 어느 것도 회복하지 않는다: `recovered=false`, 종단·누적은 `censored=true`.
  `maxV`는 창 안에서 관측이 완결되므로 `censored=false`다 — 잘린 값이 아니다.
- **§4b 변형 A** — 5런 중 2런만 회복한다. `recovered`는 셀의 성질이 아니므로 **`null`**로 두었다.
  `false`로 적으면 "회복하지 않았다"는 거짓 진술이 된다.
- **Q6·Q3·Q5 baseline** — 미회복이므로 `censored=true, recovered=false`.

`maxV`는 검사기 어휘에 새로 넣었다. `lastV`(창 끝 값)와 **다른 양**이며, Q4 본문이 인접한 문장에서
둘을 함께 인용한다(탐색 중 peak vs 종단) — 한 어휘로 묶으면 한쪽이 조용히 낡는다.

### 2.1 등재로 **되살아나는 문장**, 그리고 그 문장의 결론이 뒤집힌다

지워진 문장은 *"The three baselines settle at $V(t)\approx74$–$76$ … BoundGuard … ending at
$V(t)\approx55$, **the lowest of the four**"*였다. 이제 다섯 값이 전부 등재돼 있다.

| 변형 | 종단 $V$ | `maxV` | 전환 |
|---|---|---|---|
| **BoundGuard** | **362.5 ± 112.2** | **480.3 ± 46.4** | 9 |
| A (no dwell) | 352.1 ± 130.5 | 670.8 ± 148.4 | 9 |
| B (no dwell, no progress) | 294.2 ± 2.8 | 298.2 ± 3.0 | 0 |
| C (dwell only) | **292.6 ± 3.7** | 298.7 ± 3.2 | 0 |
| Adaptive | 315.1 ± 10.7 | 322.8 ± 10.9 | 0 |

**게재본의 "가장 낮다"는 재실행에서 성립하지 않는다.** 탐색한 두 변형(BoundGuard 362.5, A 352.1)이
가만히 있은 셋(292.6–315.1)보다 **높다.** 방향이 반대다.

이것은 §IV가 이미 하고 있는 주장과 모순되지 않는다 — 같은 문단이 *"Exploration is not free"*라고
적고, 새 `maxV`가 그 진술을 더 크게 뒷받침한다(BoundGuard 480.3 vs baseline 298–323). 바뀌는 것은
**"the lowest of the four"라는 우위 주장**뿐이다. **문안은 쓰지 않았다** — 값만 낸다.

### 2.2 등재하며 드러난 것: 45라는 수의 범위가 다르다

본문은 *"Across five runs all $45$ rejections took this path"*라 적는다. 실측은:

| | QoS-validate | saturated 경유 |
|---|---|---|
| Q4 BoundGuard (5런) | **20** | 20 |
| Q4 변형 C (5런) | 25 | 25 |
| **합 (10런)** | **45** | **45** |

45는 맞으나 **다섯 런이 아니라 열 런의 합**이다(`main_rerun_report.md` §3도 변형 합산으로 싣는다).
둘 다 등재해 구분이 진술 가능해졌다: `q4_bg_saturated_advances`=20, `q4_saturated_advances_all`=45.
**검사기는 값만 보므로 이 범위 오류를 잡지 못한다** — 사람이 고쳐야 한다.

### 2.3 후보 서비스율 — 게재값은 증발한 런 유래

본문의 237 fps / 186 fps는 `best_so_far_report.md`, 즉 **증발한 `q4_bsf_out`**에서 온다.
v33 재실행의 같은 측정은 로그에 남아 있다:

| 후보 | v33 재실측 | 게재값 |
|---|---|---|
| cand_2 (최중량 모델을 가속기에서 뺌) | **231.0 ± 2.5** | 237.3 |
| cand_5 (예측기 최하위) | **176.1 ± 4.0** | 186.2 |

**순서는 그대로**다 — 최선이 cand_2, 최악이 예측기가 마지막으로 순위매긴 cand_5. 본문의 논증
(*"the worst is precisely the candidate the predictor ranked last"*)은 유지되고 수치만 낡았다.

---

## 3. 작업 196 — 주석 위치 목록

`docs/paper_cv_annotation{s.json,_worklist.md}`. **배포본은 편집하지 않았다.**

v36의 16개에서 **17개**로 늘고 추적 값은 64 → **72**건이 됐다. 새로 등재한 63항목 중
**7건만 논문이 지금 인용한다** — 나머지 56건은 §2.1의 되살릴 문장을 포함해 **쓸 수 있게 된 값**이다.
등재는 커버리지를 바로 늘리는 작업이 아니라 **쓸 수 있게 만드는 작업**이다.

| 줄 | 추가/신설 | 새 id |
|---|---|---|
| 557 | 추가 | `q5_lead_pre_activation` (9.6) |
| **712** | **신설** | `q3_static_maxV` (4.1) — 위반 중 수준은 종단(3.5)과 다른 양이다 |
| 743 | 추가 | `q4_saturated_advances_all` (45) |
| 775 | 추가 | `q5_lead_pre_activation`, `q5_adaptive_lastV`, `q5_sr_lastV` |
| 817 | 추가 | `q6_a_cumV` (2226), `q6_adaptive_cumV` (264) |

각 항목은 **대상 줄의 sha256**을 담는다. v36 목록의 16줄은 새 `main.tex`(Q4 정정본)에서도
**전부 유효**했다 — 정정이 741·756줄에서 제자리로 이루어져 줄이 밀리지 않았다.

### 저장소 밖 검증 출력

```
$ python3 scripts/apply_cv_annotations.py --tex <scratch>/main.tex
17 applied, 0 skipped

$ python3 scripts/check_paper.py --paper <scratch>
paper.lock:      OK (main.tex + 13/13 figures)
tracked values:  73 checked, 0 mismatch
untracked:       224 numerals (192 on 57 lines without %!cv, 32 on 13 annotated lines that no id explains)
figures:         13 referenced, 13 present

[OK] PAPER MATCHES CONFIRMED TABLE (0 mismatches).
```

73 = 목록 72 + 배포본이 이미 달고 온 `q4_bg_lastV` 1건. **전부 통과한다.**

---

## 4. 작업 197 — 커버리지

### 4.1 검사기 보강 둘 — 집계가 커버리지를 부풀리고 있었다

**(가) 주석 줄의 미설명 수치를 따로 센다.** 줄 단위로만 세면 **주석을 하나 달았다는 이유로 그 줄의
낡은 수치까지 미추적 집계에서 사라진다.** 실제로 그랬다 — Q4 정정본이 L743에 주석을 달자 같은 줄의
낡은 237 / 186 / 146 / 75–78이 집계에서 조용히 빠졌다. 이제 별도 항목으로 나온다.

**(나) `sd`를 덮인 수치로 센다.** 논문은 `$30.4\pm0.5$`로 적으므로 산포도 그 줄의 수치다.
세지 않으면 "설명되지 않은 수치"가 산포로 부풀어(38 → 32) 실제 결함이 묻힌다.
**다만 검사하지는 않는다** — 산포를 인쇄하지 않는 문장이 많아 부재를 실패로 삼으면 거짓 경고가 된다.
`sd`가 낡아도 잡히지 않는다는 뜻이며, 이는 남는 한계다.

### 4.2 전후

| 상태 | 주석 줄 | 추적 값 | 미추적 (주석 없는 줄 / 주석 줄의 미설명) |
|---|---|---|---|
| 배포본 현재(주석 1) | 1 | 1 | **323** (323 / 0) |
| v36 목록만 적용 | 17 | 65 | **232** (194 / 38) |
| **v37 목록 적용** | **18** | **73** | **224** (192 / 32) |

등재로 늘어난 추적은 8건이다. 작아 보이지만 그 8건이 **이전에는 갈 곳이 없던 값**이다 — Q6의
누적 위반, Q5의 활성 전 구간, Q3의 위반 중 수준, Q4의 기각 수.

### 4.3 미추적 224건의 분류

`scripts/classify_untracked.py`가 산출한다(줄 단위 지도를 스크립트에 두어 산문 집계가 낡지 않게 했다).

```
 224 untracked numerals across 70 lines

  65  constant       정의상의 상수·설정 — 측정이 아니므로 낡을 수 없다
  27  table_i        Table~I 모델 지연 — 별도 출처, 런 디렉터리 아님
   5  hardware       장비 사양 — 측정이 아니다
  70  table_cell     표 셀 — 갱신 시 통째로 교체(우선순위 4)
  36  unresolvable   **되짚을 수 없음** — 커밋된 런으로 추적 불가. 실질적 잔여 위험
   4  derived        파생값 — 확정표 두 항목의 차/비
  17  not_yet        달 수 있으나 아직 달지 않음
```

- **원리적으로 불가**는 이 표에 아예 나오지 않는다. **글자로 쓴 수**(`two runs of five`,
  `fifteen transitions`, `seven times`)는 수치가 아니어서 검사기의 시야 밖이다. L627·L817에 있고,
  대응하는 `q4b_a_recoveries`·`q6_a_hotswaps`·`q6_bg_hotswaps`는 확정표에 있다 — **값은 추적되는데
  문장이 추적되지 않는** 상태다.
- **파생값** 4건: Δ CPU(L581, `q11_*_cpu` 두 항목의 차), L815의 3.3(B 3.2와 C 3.4의 중간).
- **아직 안 단 것** 17건: c3의 축적 오차 0.2%·앵커 오차 6.8/12.5%·λ/μ\*_eff 1.02,
  Q1.5의 배율 3.3–4.7×·26% tail 감소. 전부 리포트에 있으나 확정표 미등재 — **다음 라운드의 대상**.

---

## 5. **되짚을 수 없는 값** — 실질적 잔여 위험

논문이 인쇄하는데 **커밋된 런으로 되짚을 수 없는** 36개 수치다. 다음 사고의 자리는 여기다.

| 줄 | 개수 | 값 | 출처와 상태 |
|---|---|---|---|
| 462 | 2 | 5.2 / 3.4 | **어느 리포트에도 없다.** vision-4 구성이 왜 기각됐는지의 근거 수치 |
| 585 | 5 | 0.15–0.21 / 1.0 / 1.5–2.2 s | `switching_lossless_report.md`(07-24). 전환 실험 런이 `runs/`에 없다 |
| 712 | 1 | 3.7 | 셀 평균 최저는 3.8(`q3_sr_maxV`). 4.1은 등재됨 |
| 714·775 | 2 | $\delta\approx2.9$ s | `hotswap_resource_report.md`. v33 로그의 hot-swap delta는 **vision 뷰 0.08–1.08 s뿐** — 생성 모델 이전은 `BackgroundManager`가 처리해 로그가 없다 |
| 739 | 2 | $\lambda=90$ 대 ≈72 fps | **어느 리포트에도 없다.** Q4 시나리오 설계의 근거 |
| 743 | 5 | 237 / 186 fps, 146, 75–78 | **증발한 `q4_bsf_out`.** v33 재실측은 등재됨(§2.3) — **본문만 갱신하면 해소된다** |
| 745·747 | 9 | 2-gen 사례 전부 | 미상. 그림 `q3_misprediction_2gen.pdf`는 있으나 본문이 문장으로만 서술 |
| 752 | 5 | 봉투 24–26 / 23 | analytic. 런이 아니라 상한식 평가 |
| 773 | 5 | 7 / 34–49 / 4.4× | `q5_experiment_report.md` §2.2, **λ=80 사전측정**. 본 실험은 λ=45 — 다른 조건의 수치를 인용하고 있다 |

**성격이 셋으로 갈린다**:

1. **본문 갱신으로 해소되는 것** — L743(5건). 값은 이미 등재돼 있다.
2. **재실행해야 해소되는 것** — L585(전환 실험), L739·L462(사전측정), L745·747(2-gen), L773(λ 불일치).
   **이 라운드의 범위 밖**이며, 재실행 여부는 사람 판단이다.
3. **원리적으로 런이 아닌 것** — L752(analytic 봉투). `source_kind: analytic`으로 등재는 가능하나
   `source`가 가리킬 런이 없다. `d2_budget_rank`가 같은 성격이며 선례가 있다.

가장 나쁜 둘은 **L462와 L739** — 어느 리포트에도 없다. 논문이 근거로 드는 수인데 저장소 어디에도
대응물이 없으므로, 지금 물으면 답할 수 없다.

---

## 6. 검사

```
$ make check
checked 208 values across 11 sidecars
[OK] ALL FIGURE VALUES MATCH CONFIRMED TABLE (0 mismatches).

parsed 13 figures from main.tex
[OK] committed paper.lock reproduces from main.tex + figures/

paper.lock:      OK (main.tex + 13/13 figures)
tracked values:  1 checked, 0 mismatch
untracked:       323 numerals (323 on 74 lines without %!cv, 0 on 0 annotated lines that no id explains)
figures:         13 referenced, 13 present
[OK] PAPER MATCHES CONFIRMED TABLE (0 mismatches).
```

배포본에는 아직 주석이 1개뿐이므로 `tracked values: 1`이다. 목록을 정본에 반영하면 73이 된다(§3).
`scripts/testdata/paper_fixture`의 음성 테스트 4종은 v36과 동일하게 실패로 잡힌다(회귀 확인).

---

## 7. 커밋

| 커밋 | 내용 |
|---|---|
| `4ba4d66` | 확정표 +63항목 · 등재/집계/분류 스크립트 3종 · `check_paper.py` 커버리지 보강 · 주석 목록 확장 · 본 보고서 |

**푸시하지 않았다.**

---

## 8. 남은 것 — **사람 판단**

1. **Q4 우위 문장(§2.1).** 재실행값은 게재본과 방향이 반대다 — 탐색한 쪽이 더 높다. 다섯 값이
   등재됐으므로 서술은 가능하다. **어떻게 쓸지는 내가 정하지 않았다.**
2. **"45 rejections" 범위(§2.2).** 다섯 런이 아니라 열 런의 합이다. 검사기가 잡지 못한다.
3. **후보 서비스율 237/186 → 231.0/176.1(§2.3).** 논증은 유지되고 수치만 낡았다.
4. **되짚을 수 없는 36건(§5).** 특히 L462·L739는 어느 리포트에도 없다. 재실행할지, 문장을 뺄지.
5. **아직 안 단 17건(§4.3).** c3 오차와 Q1.5 배율 — 다음 라운드에 등재 가능하다.
6. **글자로 쓴 수(§4.3).** 값은 확정표에 있으나 문장이 수치를 쓰지 않아 검사 밖이다.
   아라비아 숫자로 바꾸면 추적된다 — 문체 결정이다.

# 그림 재라벨·(win 58s) 정정·checker 텍스트 검사 확장 (v14)

날짜: 2026-07-26 · 재실험 없음 · **하드웨어/논문 값 불변** · main.tex 불변(이미 반영)
선행: [figure_fix_report_v10.md](figure_fix_report_v10.md), [d2_followup_report.md](d2_followup_report.md)

> **요약**: (1) `d2_budget_rank`를 **해석적 지도**로 재라벨(SIMULATION 제거) — main.tex 캡션("evaluates the
> bound rather than reporting runs")과 정합. (2) `q13_failure_persistence` 그룹 라벨의 `(win 58s)` 제거(58.4는
> Static 창일 뿐). (3) **checker가 그림의 모든 텍스트(제목·축·범주형 틱·범례·주석·하단문구)를 스캔**하도록
> 확장 — 음성 테스트로 `(win 58s)` 재현이 잡힘 확인. (4) cumV_ratio `censored:true` 정정. **하드웨어 값 불변.**

## 작업 51 — `d2_budget_rank` 재라벨 (해석적 지도)
| 요소 | 현행(v13) | 수정(v14) |
|---|---|---|
| 제목 | `[SIM plane + HW points] ... where the hardware cases sit` | `Budget x rank: where the measured cases sit` |
| 하단 | `Plane = bound formula (SIMULATION). Stars/X = hardware ...` | `Plane: worst-case persistence from the bound, with measured delta. Markers: measured operating points. Recoverable iff k* <= N_cand.` |
| 범례 | `Q3 HARDWARE` / `2-gen HARDWARE` | `Q3 measured` / `2-gen measured` |
| 컬러바 | `worst-case persistence = T+N_cand(Tv+delta) [bound]` | 유지 |
- **SIMULATION 단어 완전 제거.** `main.tex` 캡션과 대조: 모순 없음(둘 다 "analytic map, evaluates the bound").
- **평면 데이터 정합(정직 기록)**: v13 평면은 **sim persist_mean**을 그려 컬러바 라벨("bound formula")·논문 캡션과
  모순이었다. v14는 평면을 **상한식 T+N_cand(T_v+δ)를 격자에서 평가**한 값으로 바꿔 라벨·캡션과 일치시켰다.
  이는 결과에 맞춘 튜닝이 아니라 **라벨/데이터 모순 해소**(하드웨어·논문 헤드라인 값은 불변).
- **51-3 source_kind=analytic**: `confirmed_values.json`·`check_figures.py`에 `"analytic"` 추가. 평면 두 값
  (`d2_budget_bound_q3`=23, `d2_budget_bound_max`=51)은 `analytic`. 마커 두 점은 measured(하드웨어 운영점).
- **51-4 논문 미등재 그림**: `d2_dwell_response`·`d2_load_tightness`는 리포트 전용 → confirmed의 `figure`에
  **미등재**(사이드카 제거), 거짓 경고 방지. 회복률 미검증 표기(v13) 유지.

## 작업 52 — `(win 58s)` 정정
- q13_failure 그룹 틱 라벨 `CPU–GPU (win 58s)` → **`CPU–GPU`**. 58.4s는 Static 창일 뿐, 회복 3기법 창은
  62–69s로 다르다 → 남의 창을 붙인 오류. 정보 손실 없음(Static 막대 안 `≥58.4s` + 캡션 `58.4s window` 유지).
- 반올림 통일: 막대 안 `≥58.4s`, 캡션 `58.4s`.
- 다른 그림 점검: q13_cumulative는 Static만 해칭·창 언급 캡션뿐(문제없음). q3/q5의 `window ~107s`는 전 변형
  공통이라 유지.

## 작업 53 — checker가 그림 텍스트를 본다 (핵심)
- **문제**: 작업 52 오류를 v10 checker가 못 잡음 — `num_from_text`가 주석만 보고 축·틱 라벨은 안 봄.
- **수정**: `sidecar_util._harvest_numbers(fig)`가 **제목·축/컬러바 라벨·범주형 틱 라벨·범례·주석·figure 텍스트**를
  전부 스캔(수치 축 틱과 `$...$` 수식 스팬은 제외 — 눈금·subscript/exponent는 주장이 아님). 사이드카에
  `text_numbers`+`text_allowlist` 기록. `check_figures.check_text_numbers`가 각 숫자를 **confirmed 값(±tol)** 또는
  **allowlist(정확 일치, 사유 필수)** 와 대조. **allowlist는 정확 일치**라 `(win 58s)`의 58이 legit 58.4에 tol로
  흡수되지 않는다.
- **53-4 음성 테스트** (그룹 라벨에 `(win 58s)` 되돌림):
  ```
  [FAIL] 2 mismatches:
    - q13_failure_persistence.values.json: text number 58.0 in 'CPU–GPU (win 58s)' is neither a confirmed figure value nor allowlisted
    - q13_failure_persistence.values.json: text number 58.0 in 'CPU–NPU (win 58s)' is neither a confirmed figure value nor allowlisted
  ```
  → 잡힌다. v10에선 못 잡던 케이스. 복원 후 0 mismatch.
- **53-4 양성 테스트**: `checked 33 values across 6 sidecars` → `[OK] ... 0 mismatches`, 경고 0건.
- **allowlist 전량(23항목, 전부 사유)** — 목록이 짧아 스캔 규칙이 과광범위하지 않음:

| 그림 | allowlist (값: 사유) |
|---|---|
| bounded | 2/3/4/5 (cand 인덱스·T=3s), 100 (Q4 창), 107 (Q3/Q5 창) |
| d2_budget_rank | 5 (Q3 마커), 12 (2-gen k*/축 max), 3 (T=3s 공식), 0 (margin 0), 2 (2-gen) |
| q13_cumulative | 5 (run 수) |
| q13_failure | 58.4 (Static 창, Static 막대에만), 5 (run 수) |
| q3_mispred | 3 (rep), 5 (run 수), 107 (창) |
| q5_npu | 2 (cand_2), 3 (rep), 5 (run 수), 107 (창), 0 (drain≈0) |

## 작업 55 — `q13_failure_persistence` 축 전환 (v8 작업 27 대안 채택)

### 55-1. 범주 오류 (방향 아님)
현행은 회복시간 축(0–60s)에 Static을 **58.4 높이 해칭 막대**로 그려, 독자가 58.4(창)와 12(회복시간)를 "5배"로
읽게 한다 — **다른 종류의 양**. Static의 persistence는 미회복이라 정의 안 됨. (∫V 그림은 대상 아님 — 거기선
531/82가 같은 양이고 창 절단이 비율을 낮추는 쪽이라 성립.)

### 55-2. 도안 (채택) — 2패널 "회복 여부 + 종점 V"
| 요건 | 충족 |
|---|---|
| 1 절단값(58.4)이 시간축 막대 높이로 안 나타남 | ✓ 58.4는 하단 footnote에만 |
| 2 회복 결과(5/5 vs 0/5) 읽힘 | ✓ 좌 패널 |
| 3 미회복 종점 V(≈9) 표시 | ✓ 좌 패널 Static "V≈9" |
| 4 회복 변형 persistence(11–13s)+오차막대 유지 | ✓ 우 패널(실측값) |
| 5 축이 절단값에 지배 안 됨 | ✓ 좌 0–5, 우 0–16s(세 막대 다 읽힘) |
- **좌**: 회복 결과(방법×플랫폼 n/5). 회복 3기법 5/5, Static 0/5+`no recovery V≈9`.
- **우**: 회복 변형만 persistence(0–16s, 오차막대). **Static 부재**(`no recovery (not shown)` 주석).
- 채택 사유: v8 작업27이 남긴 미결(해칭 막대가 여전히 시간축에 있어 "5배" 오독 여지). 이 도안은 미회복 변형을
  **애초에 시간축에 놓지 않아** 오독 원천 제거. 대안(단일 패널 축 밖 주석)보다 회복 결과·persistence를 동시에
  명확히 보임.

### 55-3. 파급 (값·main.tex 불변)
- **본문 정합**: Q1.3 서술(*"never returns below ε ... censored at the window length (58.4 s)"*, *"persistence
  of 11–13 s"*)은 새 도안과 그대로 맞음. **main.tex 수정 불필요**(확인 완료).
- 확정표·사이드카: `q13_static_lastV`(9.0, censored, window 58.4)가 Static 주 표현, 회복 6항목 유지. 값 불변.
- censored 규약: 해칭 막대를 쓰는 그림은 이제 `q13_cumulative_violation` 하나. 범례 문구는 전 그림 공통 유지.

## 작업 54 — 스키마 정정
- `q13_gpu_cumV_ratio`·`q13_npu_cumV_ratio`: 분자(Static cumV)가 censored이므로 비율도 하한 → **`censored:true`**
  (창 58.4s). `recovered:null`(파생값). 그림·캡션은 이미 "ratio is a lower bound"라 표시 변경 없음. 값 불변.
  플롯의 사이드카 방출도 `censored=True`로 맞춤. checker `censored∧recovered` 모순검사 미저촉(recovered null).

## 최종 검증
```
checked 33 values across 6 sidecars
[OK] ALL FIGURE VALUES MATCH CONFIRMED TABLE (0 mismatches).
```
경고 0건. 음성 테스트 통과(위). **하드웨어/논문 값 하나도 안 바뀜**(라벨·스키마·평면 데이터 정합만).

## 교체/신규 파일
- 재생성: `d2_budget_rank`(analytic 재라벨), `q13_failure_persistence`(win 제거), `q13_cumulative_violation`
  (스키마), 하드웨어 4종(text_numbers 추가). `d2_dwell_response`·`d2_load_tightness`(리포트 전용, 사이드카 없음).
- `fig/sidecar_util.py`(capture_text+_harvest_numbers), `fig/check_figures.py`(analytic·text-scan·allowlist),
  `fig/confirmed_values.json`(cumV_ratio censored, d2 analytic 2, report-only 3 제거), 사이드카 6종(text_numbers 포함).

## 무결성
- 값 불변. 재실험·main.tex·알고리즘 불변. checker가 못 잡던 오류(그림 텍스트)를 checker 확장으로 닫음(사람 재확인 아님).

# 그림 정정 v9 — 기준점 통일 · checker 스키마 강화

날짜: 2026-07-26 · 재실험 없음(값 교체 + 스크립트 수정) · main.tex 불변 · 파라미터 불변
값 출처: `fig/confirmed_values.json`(제공본, v7 §25) · 대조: `fig/check_figures.py`(제공본, 강화판)
선행: [figure_regeneration_report.md](figure_regeneration_report.md)(v8)

> **요약**: `bounded_recovery_analysis`의 축을 **detection onset**으로 통일해 Q3 막대를 24.6(burst)→**22.6(t0)**로
> 고쳤다 — 스택 22.6+2.8=**25.4가 bound 26 아래**로 들어간다(이전엔 27.4로 선을 뚫었다). checker를 **값+기준점
> +단위+censored+창** 동시 대조 스키마로 교체했고, 전 사이드카를 새 스키마로 갱신했다. **대조 0 mismatch·0 warn.**

## 작업 31 — bounded_recovery_analysis 기준점 통일
- 축: `time from first violation` → **`time from detection onset`**(전 막대 공통 원점).
- 막대(제공 confirmed에서 id로 조회):

| 막대 | search | drain | bound | onset |
|---|---|---|---|---|
| Q3 CPU–GPU | **22.6**(t0) | 2.8 | 26 | t0 (=burst+2s) |
| Q5 CPU–NPU | 14.2(burst) | ≈0 | 26 | burst(컨트롤러 활성) |
| Q4 CPU–GPU | 24(t0) | —(미회복) | 24 | t0 |

- **24.6(burst 기준)은 그림에서 완전 제거** — 본문 부가설명 전용. 그림에 두 기준점 혼재 방지.
- Q3 스택 **25.4 < 26**(선 아래), Q4 **24=24**(tight) 정확히 보임. Q5 detection onset=burst 이유(stable 위반)를
  축 각주에 명시. 미회복(Q4)은 공통 censored 규약(해칭+revert 표기).

## 작업 32 — checker 스키마 강화 (제공본 적용)
- `fig/check_figures.py`·`fig/confirmed_values.json`을 **제공본으로 교체**.
- 사이드카 필수 필드 `id·value·quantity·reference·unit·n·censored`(+censored면 `window_s`) — 누락 시 **실패**.
- `value` 외 `quantity·reference·unit·censored·window_s` **정확 대조** → "값은 맞고 기준점이 틀린" 오류를 잡는다.
- **음성 테스트**(고의로 Q3에 24.6/burst 주입 + q4 reference 삭제):
  ```
  [FAIL] 3 mismatches:
    - bounded_recovery_analysis.values.json[q3_bg_search]: value 24.6 != confirmed 22.6
    - bounded_recovery_analysis.values.json[q3_bg_search]: reference 'burst' != confirmed 't0'
    - bounded_recovery_analysis.values.json[q4_search]: 필수 필드 누락 ['reference']
  ```
  → 라운드 4~7을 잡아먹은 "값은 맞고 의미가 틀린" 유형이 이제 자동 검출된다.
- **사이드카 생성 방식**: `fig/emit_sidecars.py`가 confirmed의 각 항목을 `figure` 필드로 필터링해 사이드카를
  만든다 — 값이 confirmed에서 **복사**되므로 대조가 구조적으로 보장된다(그림 갱신 누락은 checker의 [warn]로 노출).

## 작업 32-3 — 기존 3종 사이드카 갱신 (그림 재생성 안 함, 값 불변)
- `q13_failure_persistence`(7항목)·`q3_misprediction`(4)·`q5_npu_generalization`(2) 사이드카를 새 스키마로 갱신.
- 값·기준점 불변. checker 0 mismatch 확인.

## 작업 33 — q13_cumulative_violation 레이아웃·문구
- 범례를 축 위로 이동(주석·530 라벨 충돌 해소). 비율 주석을 하단 캡션으로.
- 문구 정정(비문): *"Static ≈ 6.2× ... recovery techniques"* → **"Static accumulates 6.2× (GPU) / 6.3× (NPU)
  the cumulative violation of the recovering methods"**. 부연(∫V 시간가중·창 짧아 하한)은 캡션 유지. 값 불변.

## 작업 34 — 미세 정정
- `q13_failure_persistence`: 회복 막대 값 라벨을 **오차막대 캡 위로**(means+SD+1.4) — GPU Stop-restart "13" 겹침 해소.
- `q3_misprediction`: 캡션에 **"representative run (rep 3 of 5)"** 명시. 궤적·값 불변.
- (부수) v8 plot 스크립트가 옛 confirmed 구조를 참조했으므로, q13/q3 스크립트를 **flat 제공본을 id로 읽도록
  재작성**(`plot_q13_v9.py`·`plot_q3_v9.py`). 값은 confirmed에서만.

## 대조 결과 (최종)
```
checked 31 values across 5 sidecars
[OK] ALL FIGURE VALUES MATCH CONFIRMED TABLE (0 mismatches).
```
경고 0건(모든 figure-지정 값이 사이드카에 존재).

## 교체/신규 파일
- 재생성 PDF+PNG: `bounded_recovery_analysis`, `q13_cumulative_violation`, `q13_failure_persistence`, `q3_misprediction`.
  (`q5_npu_generalization`은 값 불변·재생성 안 함; 사이드카만 갱신.)
- `fig/confirmed_values.json`·`fig/check_figures.py`(제공본), `fig/emit_sidecars.py`, `fig/plot_*_v9.py`,
  `fig/*.values.json`(5, 새 스키마).

## 남은 사람 판단 (PENDING)
- v8 작업27 **대안**(축을 회복시간→"회복여부+종점 V") 미결. 현 규약도 방어 가능하나 Static 막대가 58.4 높이로
  서 있는 한 훑어보는 독자에겐 "5배"로 읽힐 수 있음(본문에서 지운 비교). 이번 라운드에서 결정 안 함 → PENDING.

## 무결성
- 재실험 없음. 값은 제공 confirmed 단일 소스에서만. 알고리즘·파라미터·main.tex·legacy/backup·메인앱 코드 불변.

# 그림 정정 v10 — censored/recovered 분리 · Q4 막대 · 사이드카를 캔버스에서

날짜: 2026-07-26 · 재실험 없음 · 값 불변 · main.tex 불변 · 파라미터 불변
제공본 사용: `fig/confirmed_values.json`(recovered 필드 추가본), `fig/check_figures.py`(recovered 검사 추가본)
선행: [figure_fix_report.md](figure_fix_report.md)(v9)

> **요약**: (1) `censored`(값이 창에 잘림)와 `recovered`(회복 여부)를 **독립 필드로 분리**. Q4 search=24는
> 측정 완료값(censored=false)이나 미회복(recovered=false) → **채운 막대+테두리**로 바꿔 해칭 제거, 24=bound 24
> tight가 정확히 보인다. (2) **사이드카를 실제 그린 아티스트에서** 뽑도록 재작성(v9의 confirmed-복사 경로
> 제거) — 음성 테스트로 검증. **값은 하나도 바뀌지 않았다.**

## 작업 35/36 — censored/recovered 분리 · 시각 부호 재할당
- 제공 confirmed에 `recovered` 필드(30항목; bound 등 비측정 항목은 `null`). 시각 부호 3상태:

| 상태 | 예 | 부호 |
|---|---|---|
| 회복함·측정됨 | Q3/Q5 search, q13 회복 3기법 | 채운 막대(색) |
| **회복 못 함·측정됨** | **Q4 24s** | **채운 막대(중립 채움 + bound색 테두리), 해칭 없음** |
| 회복 못 함·창에 잘림 | q13 Static, baseline lastV, Static cumV | 해칭 + ↑ + ≥window |

- **36-1 Q4 막대**: 해칭·화살표 제거, 중립 채움+빨간 테두리, 라벨 "no recovery (revert, bounded)" 유지 →
  24 = bound 24 tight가 시각적으로 정확.
- **36-2 범례 문구(전 그림 통일)**: 해칭 항목 = **"censored (value truncated at window, ≥window)"**;
  측정된 미회복 항목(Q4) = **"no recovery (measured)"**.
- **36-3 부호 점검**: 해칭이 붙은 항목이 전부 `censored:true`인지 확인. **q13_cumulative의 Static cumV는
  censored:true**(창 짧아 절단)인데 v9에서 솔리드였음 → 해칭+"≥531/≥530"으로 정정(값 불변, "하한" 메시지 강화).
  q3/q5 baseline open-square·q13 Static 해칭 모두 censored:true 확인.

## 작업 37 — 사이드카를 캔버스에서 (핵심)
- **v9 결함**: `emit_sidecars.py`가 confirmed에서 값을 복사 → checker가 `confirmed==confirmed`를 검사(자기검사).
  플롯이 22.6을 읽고 24.6을 그려도 사이드카엔 22.6이 적혀 통과됨.
- **수정**: `fig/sidecar_util.py`로 **그려진 아티스트에서** 값 추출 — `bar_h()`(막대 높이), `hline_y()`(bound 선 y),
  `num_from_text()`(주석/캡션 텍스트의 수치, 문구 오타까지 포착). 각 `plot_*_v10.py`가 이 값으로 사이드카 작성.
  **confirmed-복사 경로(`emit_sidecars.py`)는 삭제.**
- **37-3 음성 테스트** (Q3 search 막대에 +2.0 그림 → 캔버스 24.6, confirmed 22.6):
  ```
  [FAIL] 1 mismatches:
    - bounded_recovery_analysis.values.json[q3_bg_search]: value 24.6 != confirmed 22.6
  ```
  → 캔버스에 그린 값이 confirmed와 다르면 잡힌다. v9에선 못 잡던 케이스. 복원 후 0 mismatch.
- 제공 checker의 추가 검사도 유효: `recovered` 필수, `censored=true ∧ recovered=true` 모순 실패,
  `censored=true`면 `window_s` 필수.

## 최종 대조 (제공 checker, --figdir fig)
```
checked 31 values across 5 sidecars
[OK] ALL FIGURE VALUES MATCH CONFIRMED TABLE (0 mismatches).
```
경고 0건(figure-지정 값 전부 사이드카 존재). 사이드카 값은 전부 **캔버스에서** 추출.

## 교체/신규 파일
- 재생성 PDF+PNG: `bounded_recovery_analysis`(Q4 막대), `q13_failure_persistence`, `q13_cumulative_violation`
  (Static 해칭), `q3_misprediction`, `q5_npu_generalization` — 부호·사이드카만, **값 불변**.
- `fig/confirmed_values.json`·`fig/check_figures.py`(제공본), `fig/sidecar_util.py`(신규), `fig/plot_*_v10.py`,
  `fig/*.values.json`(5, recovered 필드 + 캔버스 추출). `fig/emit_sidecars.py` **삭제**.

## 남은 사람 판단 (PENDING, 유지)
- v8 작업27 대안(축을 회복시간→"회복여부+종점 V") 미결. Static 막대가 창 높이로 서 있는 한 훑어보는 독자에겐
  "5배"로 읽힐 수 있음. 이번 라운드도 결정 안 함.

## 무결성
- 재실험 없음. **값 불변**(부호·검증 방식만). 알고리즘·파라미터·main.tex·legacy/backup·메인앱 코드 불변.

# 축 전환 확인 · 마커 가림 · δ 인스턴스화 표기 (v15)

날짜: 2026-07-26 · 재실험 없음 · **값 불변** · main.tex 불변(캡션 1건은 이미 반영) · v14 작업 51~54 확정 유지
선행: [figure_fix_report_v14.md](figure_fix_report_v14.md)

> **요약**: (1) 작업 55(q13 축 전환)는 **v14 마지막 턴에 이미 완료**(2패널)됨을 확인. (2) `d2_budget_rank`의
> 범례가 2-gen 마커를 가리던 것을 lower-right로 이동해 **두 마커 모두 식별**. (3) 평면 하단 문구를 `main.tex`
> 캡션과 정합 — 단일 대표 δ=1s(vision), 전환별 인스턴스화 시 증가. **하드웨어/논문 값 불변, checker 0 mismatch.**

## 작업 55 — q13_failure_persistence 축 전환 — **이미 완료(확인)**
- v15는 "v14에서 미수행"으로 봤으나, 갱신본이 전달되기 전 시점 기준이다. 현재 그림은 **2패널**이다
  (`subplots(1,2)`, `recovery outcome` / `recovery time (recovered variants only)`; 58.4 시간축 막대 없음).
- 요건 재확인: (1) 58.4는 footnote에만(막대 높이 아님) ✓ (2) 좌 5/5 vs 0/5 ✓ (3) Static 종점 V≈9 ✓
  (4) 회복 persistence 11–13s + 오차막대 유지 ✓ (5) 우 축 0–16s, 세 막대 안 눌림 ✓.
- allowlist 정리: `58.4(Static 창, footnote)`·`5(run/recoveries)`·`0(0/5)`·`1(gridspec)`. main.tex Q1.3 서술 정합
  (수정 불요). **v8 작업 27 종결**(대안 채택) — PENDING [C2-13]에 반영됨.

## 작업 56 — d2_budget_rank 2-gen 마커 가림 정정
- **문제**: 범례(upper-left)가 x∈[1,6],y∈[11,12.5]를 덮어 2-gen 점(N=5,k*=12)을 가림. main.tex "the
  two-generative-model case lies outside the shaded region entirely"가 가리키는 점이 안 보였다.
- **수정**: 범례를 **lower-right**(흰 반투명 프레임)로 이동 — 미회복 영역(좌상단)의 마커를 안 가린다.
  2-gen 마커를 **키우고(s 95→130) 검은 테두리** 부여. Q3 별(5,5)·2-gen X(5,12) **둘 다 육안 식별 확인**.

## 작업 57 — δ 인스턴스화 표기 정정 (값 아님)
- **문제**: 평면 N=5에서 23s ⇒ 역산 δ=1s(vision 균일). §IV Q3 본문은 "measured transition overheads →
  24–26s"(회복 전환이 LLM δ≈2.9s)라 균일 δ=1이 과소평가. Q3 마커가 N=5에 있는데 맵은 23s.
- **수정(표기)**: 그림 하단 문구를 `with measured delta` → **"bound with a single representative delta (1 s,
  in-place vision swap); a per-transition instantiation (e.g. an LLM move) raises it"**로. main.tex 캡션(단일
  대표 δ=1s → 하한 포락선, 전환별 인스턴스화 시 Q3 24–26s)과 정합. 컬러바 라벨 `T+N_cand(Tv+delta) [bound]`
  유지(정확). 2줄로 나눠 잘림 없음. **값 불변**(평면 데이터 그대로, 표기만).

## checker (v14 텍스트 스캔이 새 도안에 적용)
```
checked 33 values across 6 sidecars
[OK] ALL FIGURE VALUES MATCH CONFIRMED TABLE (0 mismatches).
```
- 새 문구의 숫자 `1`(대표 δ)을 allowlist에 사유와 함께 추가. q13 2패널의 숫자(5/5,0/5,V≈9,persist)도
  confirmed 또는 allowlist. 경고 0건.

## 교체 파일
- `d2_budget_rank`(범례 이동·마커 확대·문구 정합) 재생성. `q13_failure_persistence`(2패널, v14 확정) 유지.
- `fig/plot_d2.py`(56·57), `fig/*.values.json`(text_numbers 갱신). confirmed·sidecar_util·check_figures 불변(v14).

## 무결성
- 값 불변(라벨·범례·문구만). 재실험·main.tex(캡션 기반영)·알고리즘 불변. checker 텍스트 스캔 유지, 0 mismatch.

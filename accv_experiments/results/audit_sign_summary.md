# EXP-AUDIT-SIGN — Table 3 (tab:persize-contention) ΔsAP 부호 규약 확인 (read-only)

질문: Table 3의 ΔsAP가 전부 양수(+0.095 등)인데 본문은 "lose"라고 한다. 부호 오류인가, 표기 모호인가,
그리고 Table 2(−0.098)와 규약이 같은가. 코드/raw 데이터로 확정. 측정 재실행 없음.

## Q1 — Table 3 Δ 계산 정의 → **Case (b)** (양수 = 손실 크기)
산출원: `accv_experiments/scripts/phase_rev25_persize.py` → `rev25_persize_under_contention.csv`
(per-(point,strategy,rep) raw per-size sAP 저장). Δ는 표 생성기에서 계산:
`paper/results_data/table4_persize/generate_table4.py`
- L7 docstring: "Delta(size) = baseline(skip0) - contended"
- L26: `base={s: g[g.point=="skip0"][sAP_s].mean() ...}`  (skip0 = 무경합 baseline)
- L29: `d[s] = base[s] - g[g.point==pt][sAP_s].mean()`  → **Δ = (무경합) − (경합)**
즉 ΔsAP = baseline − contended = **손실 크기(양수)**. 부호 오류 아님(Case a 아님). 표 값은 "얼마나
잃었는가"로 내부적으로 정확하나, 캡션 라벨 "ΔsAP relative to the uncontended point"는 (경합−무경합)으로
읽힐 수 있어 모호(Case b).

## Q2 — raw per-size sAP 산수 (large, All-GPU)
| point | gpu_skip(평균) | large sAP (3rep 평균) |
|---|---|---|
| skip0 (무경합) | 0.0% | (0.4587, 0.4587, 0.4587) → **0.4587** |
| skip~42 (≈31%) | 31.1% | (0.4062,0.3942,0.3980) → 0.3995 |
| skip~58 (≈59%) | 58.8% | (0.3619,0.3645,0.3641) → **0.3635** |
| skip~100 | 100% | (0.1326,0.1367,0.1291) → 0.1328 |
계산: ≈59% → 0.4587 − 0.3635 = **+0.0952 ≈ +0.095** (표 값과 일치). 100% → 0.4587 − 0.1328 =
**+0.3259 ≈ +0.326**. (small ≈59%: 0.0078−0.0063=+0.0015≈+0.002; medium: 0.1681−0.1285=+0.0396≈+0.040.)
→ contended large(0.3635) < baseline(0.4587)이므로 **large는 경합에서 sAP를 잃음**. 따라서 +0.095는
(baseline − contended) = **손실 크기**가 맞다.

## Q3 — Table 2(tab:decomp) staleness Δ 부호 → **반대 규약**
`paper/results_data/table2_decomp/generate_table2.py`
- L4-5 docstring: "staleness dAP = threads24 gap − threads4 gap" (threads24 = 경합/staleness-on,
  threads4 = 무경합/staleness-off)
- L25-26: `staleness_dAP` 값에 부호 그대로(`'+' if v>=0 else '-'`) → large = **−0.098**
즉 Table 2의 Δ = (staleness on) − (off) = **경합 − 무경합** → 손실이 **음수**.
**두 표의 부호 규약이 반대다**: Table 3은 Δ=baseline−contended(손실 양수), Table 2는 Δ=contended−baseline
(손실 음수). 같은 "ΔsAP" 라벨이지만 빼는 순서가 반대.

## Q4 — 본문 "lose / degrades" 정합
정합함. raw 값에서 All-GPU large는 0.4587(무경합) → 0.3635(≈59% skip)로 실제 하락(0.095 손실),
100%에서 0.1328로 더 하락. 본문 "All-GPU's per-size sAP degrades" 및 "large objects lose 0.095"는
데이터와 일치한다. (참고: All-NPU large는 0.431–0.433 범위로 경합 무관하게 거의 불변.)

## 요약
Table 3의 +값은 **부호 오류가 아니라 손실 크기**(Δ = 무경합 − 경합, code: generate_table4.py:29)이며,
데이터·본문("lose")과 정합한다. 다만 (1) 캡션의 "ΔsAP"는 (경합−무경합)으로 오해될 수 있어 "loss"
명시가 더 명확하고, (2) **Table 2는 손실을 음수(Δ=경합−무경합, generate_table2.py:4-5)로 적어 Table 3
(양수)와 부호 규약이 반대다** — 두 표를 나란히 보면 같은 "ΔsAP" 라벨의 부호가 어긋난다. (반영 여부는
별도 채널; 본 점검은 사실 확인까지.)

## 준수
읽기 전용; 저장 CSV + 표 생성기 코드 라인만; 재측정 없음; 코드 라인 인용으로 확정; 논문 .tex 미수정.

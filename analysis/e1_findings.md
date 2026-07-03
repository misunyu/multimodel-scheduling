# E1 — Oracle vs All-NPU mean 불일치 원인 규명 (Task 2)

대상: `tab:main`(Table 4) \Lvlm{} endpoint, N=4.
관측: Oracle worst 0.083 = All-NPU worst 0.083 (동률), 그러나 Oracle mean **0.120** ≠ All-NPU mean **0.126**.

## 1. run별 Oracle 선택 (원자료 `rev20_5strat_heavybg.csv`)

| rep | Oracle placement | Oracle worst | Oracle mean | All-NPU placement | All-NPU worst | All-NPU mean |
|---|---|---|---|---|---|---|
| 0 | **NNNN** (All-NPU) | 0.0834 | 0.1262 | NNNN | 0.0834 | 0.1262 |
| 1 | **NGNN** (mixed, 3×NPU/1×GPU) | 0.0835 | **0.1072** | NNNN | 0.0832 | 0.1259 |
| 2 | **NNNN** (All-NPU) | 0.0834 | 0.1258 | NNNN | 0.0834 | 0.1258 |
| run-avg | — | 0.0834 | **0.1197 (→0.120)** | — | 0.0833 | 0.1260 (→0.126) |

- 3회 중 2회(rep0, rep2)는 Oracle = All-NPU(NNNN)로 완전 동일.
- **rep1에서만 Oracle가 mixed placement `NGNN`를 선택**. 그 placement의 mean(0.1072)이 All-NPU(0.1259)보다 크게 낮아, 3-run 평균에서 Oracle mean이 0.120으로 내려감.
- 교차검증: `rev23_partial_placements.csv`의 L3_vlm/N4 Oracle 행 = worst 0.0834±0.0001, mean 0.1197, **gpu_skip 평균 33.3%** (= 100%×1/3, 한 run만 GPU 스트림 잔류) → rep1 mixed 선택과 정합.

## 2. 가설 판정 → **주로 (b) [run별 선택 변동]**, (a)의 외형을 동반

- **(b) 성립**: 선택 placement가 run마다 달라짐(2×NNNN, 1×NGNN). mean 차이는 전적으로 rep1의 NGNN(mean 0.1072)에서 발생. Oracle mean(0.120)은 "run별 선택 placement mean의 평균"으로 계산됨(rev21/rev23 모두 0.1197로 일치).
- **(a)는 엄밀히는 불성립하나 외형상 유사**: rep1의 NGNN worst(0.0835)는 All-NPU worst(0.0832)보다 **근소하게 높음** — 즉 정확한 동률이 아니라, Oracle의 "worst-stream 최대" 규칙이 raw 값 기준으로 NGNN을 **정당하게** 선택. 다만 3자리 반올림에서 둘 다 0.083이라 보고 정밀도에서는 "동률처럼" 보이며, Oracle이 mean을 고려하지 않으므로 mean이 더 낮은 placement가 뽑힐 수 있다. → tie-breaking **규칙 버그**가 아니라, (i) worst-only 목적함수 + (ii) run별 argmax 변동의 조합.

## 3. 권장 수정 문안 초안 (영문 1–2문장, 논문 미적용 — 저자 판단)

권장(가설 b, 정밀도 nuance 포함):
> "At the \Lvlm{} endpoint the Oracle selects All-NPU in two of three runs and a mixed
> placement (three streams on the NPU) in the third; because the Oracle maximizes only
> worst-stream sAP, that mixed run wins by a margin invisible at three-decimal precision
> (0.0835 vs.\ 0.0832) yet has a substantially lower mean, so the run-averaged Oracle mean
> (0.120) sits below All-NPU (0.126) while the worst-stream scores remain tied at 0.083."

대안(간결):
> "The Oracle and All-NPU share the same worst-stream sAP (0.083) but differ in mean
> (0.120 vs.\ 0.126) because in one of three runs the Oracle's worst-stream-maximizing rule
> picked a mixed placement whose worst-stream score edged All-NPU's but whose mean was lower."

주의: 문안에 "tie-breaking rule"이라 쓰면 부정확(정확한 동률 아님). "worst-stream만 최적화 + run별 선택 변동"으로 기술 권장.

## 4. L1CNN 확인 (Task 2.5)

- 논문 `tab:main` L1_CNN 행(Oracle mean 0.137 = All-GPU mean 0.137)의 출처는 **rev30 canonical**(rev20 아님).
- `rev30_oracle_by_contention.csv` RES1(k=1, gpu_skip 24.3): `oracle_ratio=1`, `pick_dist={"0":1,"1":9}` → **ratio=1(large-object-aware one-stream split)이 10회 중 9회 선택**. 원고의 "selected in nine of ten runs"(현행 L604 부근) 서술과 **정합**.
- Oracle worst = split1_worst = 0.1026(→0.103); Oracle mean은 저경합(k=1)에서 한 스트림만 NPU로 옮겨도 시스템 mean이 거의 불변이라 **All-GPU mean(0.137)과 동일하게 반올림**됨. worst만 0.098→0.103으로 개선. → 정합.
- ⚠️ 별개 주의(A1/문구): one-stream split이 옮기는 스트림은 sid 3이나, 패널 내 `pct_large_count` 최대는 sid 21 → "the single stream with the **highest** large-object fraction" 문구는 엄밀히는 부정확(`FINDINGS_oracle_largefrac.md` T3). E1 수치와는 무관.
- ⚠️ 데이터 출처 불일치: `rev20_5strat_heavybg`의 L1_light/N4 Oracle는 `NNNG`(0.109, gpu_skip~35)로 rev30(0.103, skip24)과 다름 — 재생성 패키지(`paper/results_data/table3_main`)가 rev20 기반이라 논문값과 어긋남(A1 감사 항목).

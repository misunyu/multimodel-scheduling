<!-- provenance
  generated_at: 2026-07-30T19:20+09:00 (KST)
  score_basis: declared_normalized (Eq. 1)
  basis_definition: group = (set, rate); y1 = r1/max(r1), y3 = r3/max(r3),
    y2 = (r2-min r2)/(max r2-min r2) with y2 = 0 when max == min;
    S = y1 - alpha*y2 + beta*y3 (beta term only for generative sets)
  alpha: 0.3
  betas: [1.0, 0.5]
  tie_rule: 1e-9 tie set; intersecting tie sets count as agreement
  predictor_involved: no (measured windows only)
  input_data: full_collection_540/{cpu_gpu,cpu_npu}/performance_{gpu,npu}_full540.json
  generator: scripts/compare_platforms_v2.py --basis declared_normalized
  score_definitions_from: runs/20260730_190154_score_basis_audit/build_basis_audit.py
  recomputation_run: runs/20260730_190154_score_basis_audit
  source_rows: runs/20260730_190154_score_basis_audit/divergence_by_group.csv
    (basis == declared_normalized)
  supersedes: platform_divergence_raw_v1.{md,csv}
  figure: analysis/figures/fig_divergence.{pdf,png}
    (scripts/make_figures.py --fig f1 --basis declared_normalized)
-->

# P2 (v2) — 플랫폼별 최적 배치 불일치, **선언 기저(식 1)** 기준

점수: `S = y1 − 0.3·y2 + β·y3` (β 항은 생성 모델 포함 세트만), **그룹 (세트, rate) 내
정규화된 측정값** 기준. 예측기 미사용.

## v1(raw 기저)과의 차이

v1은 같은 데이터에 **원시 측정 창 총계**를 그대로 넣어 점수화했다(β=1.0에서 29/45 불일치).
그 기저에서는 y1이 8–455 fps인데 α·y2 ≤ 0.3이므로 argmax가 사실상 `y1 + β·y3`로 결정되고,
**GPU의 y3 최대 180.8 tok/s vs NPU 22.6 tok/s — 약 8배의 단위 스케일 격차**가 그대로 플랫폼
간 argmax 차이로 환산되었다. 식 (1)의 그룹 정규화를 적용하면 y3가 그룹 내 [0,1]로 압축되어
이 격차가 사라지고, 불일치는 **29/45 → 11/45** 로 떨어진다. 즉 v1 수치의 상당 부분은 플랫폼
간 실제 배치 선호 차이가 아니라 y3 단위 스케일의 반영이었다. 기저 감사·재계산 전체 근거는
`runs/20260730_190154_score_basis_audit/basis_audit.md` (파트 1 기저 감사표, 파트 2a/2b).

v1 산출물은 `platform_divergence_raw_v1.{md,csv}`로 보존되어 있다 (`--basis raw`로 재생성).

## 요약

| β | 동일 배치 | 다른 배치 | 불일치율 | 생성 세트 | vision-only |
|---|---|---|---|---|---|
| **1.0** | 34 | **11** | **24.4%** | 10 / 33 | **1 / 12** |
| 0.5 | 29 | **16** | 35.6% | 15 / 33 | 1 / 12 |

(v1 raw 기저 대비: β=1.0 29/45 → 11/45, β=0.5 30/45 → 16/45.)

tie가 발생한 그룹은 양 β에서 0이므로 tie 규칙은 결과에 영향을 주지 않는다.

## β=1.0 불일치 11그룹

`S10@2.0`, `S5@3.0`, `S7@2.0`, `S8@2.0`, `base3@1.0`, `base3@3.0`, `base4@1.0`, `base4@3.0`, `base5@1.0`, `base5@2.0`, `base5@3.0`

모델 수 N별 분해: N=2 0/3, N=3 0/9, **N=4 7/15**, N=5 1/6, N=6 0/3, N=7 2/6, N=8 1/3.

**vision-only 0/12 구조는 이 기저에서 성립하지 않는다.** `S10@2.0`이 vision-only이면서
불일치한다 (GPU 최적 `cpu:mobilenet_v2` vs NPU 최적 `all-accel`). raw 기저에서는 α·y2가
y1에 압도되어 vision 세트의 argmax가 양 플랫폼에서 항상 같았지만, 정규화하면 α·y2가 실제로
작동하기 때문이다. v1의 "생성 29/33 · vision 0/12" 서술은 v2에서 "생성 10/33 · vision 1/12"로
교체된다.

## β-민감 그룹

**일치/불일치 판정이 뒤집히는 그룹 5개** (전부 β=1.0 일치 → β=0.5 불일치):

`S5@2.0`, `base1@3.0`, `base2@3.0`, `base3@2.0`, `base4@2.0`

참고로 **자기 optimum이 이동하는** 그룹(판정 뒤집힘과는 다른 양)은 GPU 1개(`S8@2.0`),
NPU 6개(`S5@2.0`, `S7@2.0`, `base1@3.0`, `base2@3.0`, `base3@2.0`, `base4@2.0`)다.

v1(raw 기저)의 판정 뒤집힘은 `S8@1.5` 1개였다 — **v2와 목록이 전혀 겹치지 않는다.** β 민감도
서술 역시 점수 기저에 의존한다.

## exhaustive 그룹 한정 (부록 E용)

mode == exhaustive인 10세트 30그룹(`runs/20260730_172053_coverage` 기준) 한정:

| β | raw 기저 (v1) | **선언 기저 (v2)** |
|---|---|---|
| 1.0 | 22/30 (73.3%) | **8/30 (26.7%)** |
| 0.5 | 22/30 | **13/30 (43.3%)** |

선언 기저 β=1.0의 8그룹: `S5@3.0`, `base3@1.0`, `base3@3.0`, `base4@1.0`, `base4@3.0`, `base5@1.0`, `base5@2.0`, `base5@3.0`. vision-only는 0/6 (raw와 동일 — `S10@2.0`은 sampled
세트라 이 30그룹에 포함되지 않는다).

주의: raw 기저에서는 "exhaustive 한정 불일치율이 전체보다 높다"(73.3% vs 64.4%)고 쓸 수
있었으나, 선언 기저에서는 26.7% vs 24.4%로 차이가 거의 없다.

## 그룹별 전체 내역

`platform_divergence.csv` (90행 = 2 β × 45그룹). 컬럼:
`basis, beta, set, rate, has_gen, gpu_best, npu_best, agree`.
`gpu_best`/`npu_best`는 tie set을 CPU 배치 모델 라벨로 표기한다
(`all-accel` = 전 모델 가속기, `cpu:X+Y` = X와 Y만 CPU).

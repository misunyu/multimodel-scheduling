<!-- provenance
  generated_at: 2026-07-30T19:35+09:00 (KST)
  score_basis: raw measured window totals
  alpha: 0.3
  betas: [1.0, 0.5]
  tie_rule: 1e-9 tie set; intersecting tie sets count as agreement
  predictor_involved: no (measured windows only)
  input_data: full_collection_540/{cpu_gpu,cpu_npu}/performance_{gpu,npu}_full540.json
  generator: scripts/compare_platforms_v2.py --basis raw
  score_definitions_from: runs/20260730_190154_score_basis_audit/build_basis_audit.py
  original_generation: 2026-07-29T18:32:09+09:00, commit 475a2089eb49a1906d663508055d3aeb5d5e2d43
  superseded_by: platform_divergence.{md,csv} (declared_normalized, Eq. 1)
  status: retained for continuity with the pre-audit reports; the manuscript's
    Eq. (1) basis is the declared_normalized variant
-->

# P2 (v1, 폐기) — 플랫폼별 최적 배치 불일치, **원시 측정 총계** 기준

점수: `S = y1 − 0.3·y2 + β·y3` (생성 세트만), **원시 측정 창 총계** 기준 — 예측기 미사용.

> **이 기저는 원고 식 (1)이 선언한 기저가 아니다.** 식 (1)은 그룹 (세트, rate) 내 정규화를
> 규정하므로, 논문 수치는 `platform_divergence.{md,csv}`(선언 기저)를 써야 한다. 이 파일은
> 감사 이전 보고서와의 연속성 확인용으로만 유지된다. 근거: `runs/20260730_190154_score_basis_audit`.

| β | 동일 배치 | 다른 배치 | 불일치율 | 생성 세트 | vision-only |
|---|---|---|---|---|---|
| **1.0** | 16 | **29** | 64.4% | 29 / 33 | 0 / 12 |
| 0.5 | 15 | **30** | 66.7% | 30 / 33 | 0 / 12 |

β에 민감한 그룹 (한쪽 β에서만 불일치): 1개 — `S8@1.5`

그룹별 전체 내역은 `platform_divergence_raw_v1.csv` (90행 = 2 β × 45그룹) 참조.

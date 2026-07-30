<!-- provenance
  generated_at: 2026-07-29T18:42+09:00 (KST)
  git_commit: 1511447 (pre-commit HEAD)
  alpha: 0.3, beta: 1.0, seed: 42
  input_data: full_collection_540/{cpu_gpu,cpu_npu}/performance_*_full540.json
-->

# P1~P6 실험 요약 (MLForSys 워크샵)

전 실험은 저장된 540창×2플랫폼 데이터 위의 계산이다. 하드웨어 실행·신규 수집 없음. seed=42.

## 핵심 수치 (실험당 한 줄)

| ID | 한 줄 요약 |
|---|---|
| P1 | 누수 제거 GroupKFold OOF: 점수 그룹-ρ **GPU +0.986 / NPU +0.972**, Top-1 0.933/0.867, Top-5 1.000/1.000 (레거시 infps=0 feature 재현 시 +0.967/+0.954 — 기존 보고서와 일치) |
| P2 | 플랫폼별 최적 배치 불일치(**선언 기저 = 식 (1) 그룹 정규화**): **β=1.0에서 11/45**, β=0.5에서 16/45, β 민감 그룹 5개. 생성 10/33 · vision-only 1/12. raw 기저(원시 측정 총계) 수치 **29/45 · 30/45**(β 민감 그룹 1개)는 `analysis/platform_divergence_raw_v1.{md,csv}` 참조 — 기저 재검토 근거는 `runs/20260730_190154_score_basis_audit` |
| P3 | 교차 전이(그룹-ρ): 대각선 +0.986/+0.972 → **gpu→npu +0.902, npu→gpu +0.927**; Top-1 gpu→npu 0.933→0.800; exec one-hot 스왑 변형은 ±0.002로 무의미 |
| P4 | 통합 예측기(1080행, 90그룹): 그룹-ρ **+0.975(gpu)/+0.974(npu)** vs 플랫폼별 +0.986/+0.972; Top-1 0.933/0.844 vs 0.933/0.867 |
| P5 | Capacity greedy: Top-1(수집분) **gpu 0.424/0.341, npu 0.333/0.250**(speedup/load_factor 순), 정규화 oracle비 0.58~0.79, 미수집 4~12/45그룹 |
| P6 | 분해 모델링: 뷰 저하율 OOF ρ **+0.993/+0.992**(MAE 0.017/0.020, clip 0건)이지만 합성 후 점수 그룹-ρ **+0.931/+0.911**, Top-1 **0.911/0.622** — 직접 회귀(P1)에 열세. y2 합성 근사 자체의 오차(실측 저하율 합성): raw MAE 0.149/0.169, ρ +0.834/+0.763. y1은 정확 재구성(상대오차 0) |

## 구현 결정 (사양이 정하지 않았던 것)

1. **feature의 infps를 실제 값으로 복원** (P0/P1): full540 창에는 `schedule file` 힌트가 없어 기존 `build_dataset` 경로에서는 view.infps 계열 feature가 전부 0으로 학습됐다(출하된 deploy_* 아티팩트가 그렇다). 예측 경로(GUI `featurize_from_combo`)는 실제 infps를 쓰므로, 본 실험은 수집 스케줄 YAML의 combination별 infps로 feature를 만든다. 레거시 재현용 infps=0 변형을 P1 지표에 병기했다.
2. **GroupKFold 구현** (P1): sklearn 미사용. 고유 그룹을 `RandomState(42).permutation`으로 섞어 라운드로빈 배정(그룹 수 균형; 행 수는 216/140/184로 비균등 — 그룹 크기가 4~12행으로 다르기 때문).
3. **내부(하이퍼파라미터) CV도 그룹 단위** (P1/P4/P6): 모듈로 fold 대신 그룹 fold — 선택 단계의 그룹 내 누수도 제거.
4. **y3 fold 부스터는 valid 행으로만 학습하되 held-out 전 행을 예측** (P1): 운영 예측기가 LLM-on-CPU 배치도 점수화하는 것과 동일.
5. **fold별 부스터 저장** (P1): P3가 재학습 없이 fold 3개 예측 평균을 쓰기 위함 (`gkf_*_foldK.json`).
6. **P2 점수는 선언 기저(식 (1) 그룹 정규화)** — 원고가 선언한 점수 기저와 P2를 일치시킨다.
   `compare_platforms_v2.py --basis declared_normalized`(기본값)가 산출하며, 점수 정의는
   `runs/20260730_190154_score_basis_audit/build_basis_audit.py`를 import해 공유한다.
   개정 근거: 같은 감사에서 P2만 원시 총계 기저였고 나머지 전 지표(P1·P3·P4·P5·P6)는 이미
   정규화 기저였음이 확인되었다 (`basis_audit.md` 파트 1). 원시 기저 결과는
   `--basis raw`로 `platform_divergence_raw_v1.*`에 별도 산출한다.
   - 개정 이력: 최초 결정은 "**P2 점수는 원시 측정 총계** (원본 compare_platforms.py와 동일)
     — 30/45 연속성 확인이 목적이므로"였다 (2026-07-29, commit 154f114).
7. **P5 oracle비는 정규화 점수 기준을 1차로, 원시 점수 기준을 병기** (CSV에 둘 다).
8. **P5의 그룹 infps**: 그룹 내 첫 combo의 스케줄 맵에서 모델별 최대값(디바이스 무관 동일함을 확인).
9. **P6 합성은 vision 뷰만** (예측·실측 합성 동일): 생성 뷰 제외를 명시. ŷ3는 P1 OOF 값을 그대로 공유.
10. **P6 정규화 통계는 그룹의 실측 통계** (Fmax, y2 lo/hi): 랭킹 평가의 순위 보존 변환.
11. **P4 통합 부스터는 artifacts로 내보내지 않음**: 후속 소비자 없음, OOF csv로 충분.
12. **analysis_common.py를 P6에서 소폭 수정**: xgb 헬퍼에 `feature_names` 인자 추가(기본값 불변) — 뷰 모델(18 feature)이 동일 프로토콜을 재사용하기 위함.

## 이상 발견

1. **출하된 deploy_cpu_{gpu,npu} 아티팩트는 infps feature가 전부 0인 채 학습**되어 있다(위 결정 1). 예측 시에는 실제 infps가 들어가므로 train/serve 불일치다. 본 실험 산출물에는 영향 없지만, 운영 아티팩트 재학습 시 수정할 가치가 있다.
2. P6에서 **NPU Top-1이 0.622로 크게 하락** (P1 0.867). 뷰 수준 예측은 거의 완벽(ρ +0.992)인데 합성에서 무너진다 — y2 합성 근사 오차(NPU raw MAE 0.169)가 주범으로 보인다. 분해 방식의 한계를 보여주는 결과로 논문에 유용.
3. P3에서 **exec one-hot 스왑이 결과를 거의 바꾸지 않는다**(±0.002) — 전이 격차는 one-hot이 아니라 static/capacity feature의 디바이스 값 차이에서 온다.
4. rate=4.0 그룹은 플랫폼당 1개(4행)뿐 — 그룹 Spearman이 4점 순위로 계산되는 극소 그룹이다.
5. 뷰 저하율 target의 clip(>1.0)은 vision 뷰에서 **0건** (이전 조사에서 본 최대 1.014는 생성 뷰였고, 생성 뷰는 학습 제외).

## 질문 (논문 쪽 결정 사항)

- **Q1 (P3)**: asis/swap 두 전이 변형의 차이가 무의미했다. 대표로 어느 쪽을 실을지 — 단순한 asis를 권하지만 결정 필요.
- **Q2 (P5)**: 미수집 그룹(최대 12/45)을 논문에서 어떻게 다룰지 — "수집분 Top-1"로 보고할지, 미수집을 실패로 집계할지.
- **Q3 (P4)**: 통합 vs 플랫폼별 판정 — 기준(그룹-ρ 차 ≤0.01, Top-1 차 ≤1그룹)을 md에 명시했고 판정은 하지 않았다. GPU는 그룹-ρ 기준 초과 하락(−0.011), NPU는 Top-1 1그룹 하락으로 경계선이다.
- **Q4**: 운영 deploy_* 아티팩트의 infps=0 결함을 별도 커밋으로 고칠지 (이번 실험 범위 밖).

## 산출물 경로

- `analysis/TRACE_BACKUP.md` — 트레이스 백업 기록 (백업: `/home/msyu/trace_backup/full_collection_540_traces.tar.zst`, 1.01GiB)
- `scripts/analysis_common.py` — 공통 유틸
- P1: `scripts/train_groupkfold.py`, `analysis/groupkfold_{gpu,npu}_metrics.json`, `analysis/groupkfold_{gpu,npu}_oof.csv`, `analysis/groupkfold_ranking.md`, `../../artifacts/gkf_cpu_{gpu,npu}_{y1,y2,y3}[_fold{0,1,2}].json` + features/coverage
- P2: `scripts/compare_platforms_v2.py` (`--basis {declared_normalized,raw}`),
  선언 기저 → `analysis/platform_divergence.{csv,md}`,
  raw 기저(폐기, 연속성용) → `analysis/platform_divergence_raw_v1.{csv,md}`
- P3: `scripts/cross_platform_transfer.py`, `analysis/cross_platform.md`, `analysis/cross_platform_metrics.json`
- P4: `scripts/train_unified.py`, `analysis/unified_metrics.json`, `analysis/unified_oof.csv`, `analysis/unified_vs_specialized.md`
- P5: `scripts/greedy_baseline.py`, `analysis/greedy_results.csv`, `analysis/greedy_summary.md`
- P6: `scripts/train_decomposed.py`, `analysis/decomposed_metrics.json`, `analysis/decomposed_view_oof.csv`, `analysis/decomposed_results.md`

## 커밋 목록 (push 전 검토용)

| 커밋 | 제목 | 파일 수 |
|---|---|---|
| a8728e6 | [P0] Analysis groundwork: common utils + trace backup | 2 |
| 475a208 | [P1] GroupKFold retraining without fold leakage | 34 |
| 154f114 | [P2] Platform divergence rerun at beta=1.0 and 0.5 | 3 |
| 42bfd0c | [P5] Capacity greedy placement baseline | 3 |
| d0e2644 | [P3] Cross-platform transfer evaluation | 3 |
| c37c5bd | [P4] Unified predictor on both platforms' 1080 rows | 4 |
| 1511447 | [P6] Decomposed (Kim-style) per-view degradation modeling | 5 |
| (본 커밋) | [SUMMARY] Experiment summary | 1 |

푸시하지 않았다 — 푸시 여부·시점은 사용자 결정.

# 논문 그림 provenance & 캡션 재료

생성 스크립트: `scripts/make_figures.py` (matplotlib only, Agg, pdf.fonttype=42,
난수 없음). 각 그림은 벡터 PDF(bbox tight) + 확인용 PNG(dpi 200).

## F1 — fig_divergence.{pdf,png}

- 생성: 2026-07-30 10:59 KST, HEAD `1ee0f8e` (pre-commit), 인자 `--fig f1`
- 입력: `analysis/platform_divergence.csv`의 **β=1.0 행 45개** (P2, commit 154f114)
- figsize (3.4, 2.6)in, 색: 일치 `#E8E8E8` / 불일치 `#DE8F05`, 셀 내 텍스트 없음

### 세트 순서 결정

y축은 **모델 수 N 오름차순, 동률은 이름순**: S1, S2, S3, S4, base1–5, S5, S9,
S6, S10, S7, S8. (지시문의 서술 순서 대신 N 오름차순을 택함 — N에 따른 패턴이
축에서 바로 읽히도록. 레이블에 `(N)` 병기.)

### 세트별 실제 rate 배율 (x축 low/mid/high의 실값 — 캡션용)

| 세트 | low | mid | high |
|---|---|---|---|
| S1 | 2.0 | 3.0 | 4.0 |
| S2, S3, S4, base1–5, S5, S9 | 1.0 | 2.0 | 3.0 |
| S6, S7, S8, S10 | 1.0 | 1.5 | 2.0 |

### 불일치 분포 (캡션 문구용)

- 전체 **29/45 불일치** (β=1.0)
- N별: N=2 0/3, N=3 4/9, **N=4 15/15 (base1–5 전부, 모든 rate)**, N=5 3/6,
  N=6 3/3, N=7 3/6, N=8 1/3
- 패턴은 "큰 N·고rate 집중"이 **아니라** 세트 단위 전부-또는-전무에 가깝다:
  4모델 base 세트와 S3·S5·S6·S7은 세 rate 모두 불일치, S1·S2·S9·S10은 모두
  일치. rate가 가르는 세트는 S4(low에서만 불일치)와 S8(high에서만 불일치) 둘뿐.

## F2 — fig_transfer.{pdf,png}

- 생성: 2026-07-30 10:59 KST, HEAD `1ee0f8e` (pre-commit), 인자 `--fig f2`
- 입력: `analysis/cross_platform_metrics.json`(asis 변형, commit d0e2644),
  `analysis/unified_metrics.json`(commit c37c5bd),
  `analysis/groupkfold_{gpu,npu}_metrics.json`(commit 475a208)
- figsize (5.0, 2.2)in 2패널, y축 절단을 축 레이블에 명시
  (Spearman [0.85,1.00] / Top-1 [0.5,1.0]), 막대 위 수치 7pt

### 정확한 값 (JSON에서 읽음 — SUMMARY 수치와 전부 일치, 불일치 없음)

| 평가 플랫폼 | 지표 | specialized | zero-shot transfer | unified |
|---|---|---|---|---|
| GPU | 그룹 Spearman | 0.9859 | 0.9272 (npu→gpu) | 0.9748 |
| NPU | 그룹 Spearman | 0.9724 | 0.9022 (gpu→npu) | 0.9742 |
| GPU | Top-1 | 0.9333 | 0.9333 (npu→gpu) | 0.9333 |
| NPU | Top-1 | 0.8667 | 0.8000 (gpu→npu) | 0.8444 |

### 구현 결정

1. **통합(unified) 막대 색을 지시문의 `#999999` 대신 `#CCCCCC`로 조정**:
   #999999는 주황 #DE8F05와 상대휘도가 거의 같아(1.09:1) 흑백 인쇄에서 인접
   막대가 구분되지 않는다. #CCCCCC는 명도 순서 파랑<주황<회색 단조를 만든다.
   (색맹 검증 스크립트는 node 부재로 실행 불가 — 파랑/주황 쌍은 표준 CVD-safe
   조합이며 휘도 분리를 수동 확인함.)
2. **동일 값 레이블 dodge**: GPU Top-1은 세 조건이 모두 0.9333이라 레이블이
   겹침 → 인접 레이블이 같은 높이일 때 가운데를 한 줄 위로 올리는 결정적
   stagger 적용.
3. F1 셀은 6~10% 인셋으로 그려 3개 rate 열이 열로 읽히게 함(연속 막대 방지).

### 주의 (캡션 작성 시)

- **GPU 평가의 Top-1은 세 조건 모두 0.933으로 동일** — "zero-shot 열화"는
  GPU 평가에서는 Spearman(0.986→0.927)에만 나타나고 Top-1에는 나타나지
  않는다. NPU 평가에서는 둘 다 열화(0.972→0.902, 0.867→0.800).
- unified의 NPU Spearman(0.9742)은 specialized(0.9724)보다 오히려 +0.002
  높다 — "통합은 회복"이 NPU에서는 완전 회복 이상.

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


# 논문 그림 provenance & 캡션 재료

생성 스크립트: `scripts/make_figures.py` (matplotlib only, Agg, pdf.fonttype=42,
난수 없음). 각 그림은 벡터 PDF(bbox tight) + 확인용 PNG(dpi 200).

## F1 — fig_divergence.{pdf,png}

- **generator**: `scripts/make_figures.py --fig f1 --basis declared_normalized`
  (`--basis`가 기본값이므로 `--fig f1`만으로도 동일. `--basis raw`는 폐기된 v1 재현용)
- **basis**: `declared_normalized` — 원고 식 (1)의 그룹 정규화 점수.
  그룹 = (세트, rate), `y1 = r1/max(r1)`, `y3 = r3/max(r3)`,
  `y2 = (r2−min)/(max−min)` (max==min이면 0), `S = y1 − 0.3·y2 + β·y3`(생성 세트만), β=1.0.
  점수 정의는 스크립트가 재구현하지 않고
  `runs/20260730_190154_score_basis_audit/build_basis_audit.py`의
  `normalize()/score()/tie_sets()`를 그대로 import해서 쓴다.
- **입력**: `full_collection_540/cpu_{gpu,npu}/performance_{gpu,npu}_full540.json`
  (측정 창 원본에서 재계산) + 검증 대조용
  `runs/20260730_190154_score_basis_audit/divergence_by_group.csv`
  (`basis == declared_normalized`, β=1.0 행 45개)
- **검증**: 스크립트가 그리기 전에 불일치 집합을 (a) 위 CSV, (b) 기대 11그룹 목록과
  대조하는 assert를 통과해야 한다. 불일치 시 그림을 쓰지 않고 실패한다.
- 재생성: 2026-07-30 19:1x KST, HEAD `6d068e2` (pre-commit)
- figsize (3.4, 2.6)in, 색: 일치 `#E8E8E8` / 불일치 `#DE8F05`, 셀 내 텍스트 없음
  (셀 배열·색·축 규약은 v1과 동일 — 캡션의 "orange cells" 참조가 유지된다)

> **기저 변경 이력**: 최초 판(2026-07-30 10:59, HEAD `1ee0f8e`)은
> `analysis/platform_divergence.csv`의 raw 기저 행을 읽어 **29/45**를 그렸다. 기저 감사
> (`runs/20260730_190154_score_basis_audit`)에서 그 CSV가 식 (1)이 아닌 원시 총계 기저임이
> 확인되어, 선언 기저로 재생성했다(**11/45**). raw 기저 입력은
> `analysis/platform_divergence_raw_v1.csv`로 보존.

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

### 불일치 분포 (캡션 문구용) — 선언 기저 β=1.0

- 전체 **11/45 불일치**
- N별: N=2 0/3, N=3 0/9, **N=4 7/15**, N=5 1/6, N=6 0/3, N=7 2/6, N=8 1/3
- 불일치 11그룹: `base3@1.0`, `base3@3.0`, `base4@1.0`, `base4@3.0`,
  `base5@1.0`, `base5@2.0`, `base5@3.0`, `S5@3.0`, `S10@2.0`, `S7@2.0`, `S8@2.0`
- 패턴: **4모델 base 세트(base3–5)에 집중**되고, 나머지는 각 세트의 **high rate 열**에
  단발로 나타난다(S5·S7·S8·S10). base1·base2·S3·S4·S6·S9·S1·S2는 세 rate 모두 일치.
  base5만 세 rate 전부 불일치, base3·base4는 low·high만(mid는 일치).
- v1(raw)의 "세트 단위 전부-또는-전무" 패턴은 이 기저에서 사라진다. 또한 v1에서 0/12였던
  **vision-only에 불일치 1건**이 생긴다(`S10@2.0`) — 캡션에서 "vision-only는 전부 일치"라고
  쓸 수 없다.

## F2 — fig_transfer.{pdf,png}

- **generator**: `scripts/make_figures.py --fig f2`
- 재생성: 2026-08-04, HEAD `11fc6c6` (pre-commit) — **범례 라벨만 변경**
- 최초 생성: 2026-07-30 10:59 KST, HEAD `1ee0f8e` (pre-commit)
- 입력: `analysis/cross_platform_metrics.json`(asis 변형, commit d0e2644),
  `analysis/unified_metrics.json`(commit c37c5bd),
  `analysis/groupkfold_{gpu,npu}_metrics.json`(commit 475a208)
- figsize (5.0, 2.2)in 2패널, y축 절단을 축 레이블에 명시
  (Spearman [0.85,1.00] / Top-1 [0.5,1.0]), 막대 위 수치 7pt

### 범례 용어 (2026-08-04 갱신)

원고 §4 Transfer 문단이 세 설정을 **platform-specific / zero-shot / joint**로 정의하므로
범례를 본문 용어에 정렬했다. 막대 색·순서는 불변(파랑=platform-specific,
주황=zero-shot, 회색=joint).

| 이전 | 현재 |
|---|---|
| `specialized (own platform)` | `platform-specific` |
| `zero-shot transfer` | `zero-shot` |
| `unified (both)` | `joint (both platforms)` |

**데이터 불변 검증** (재생성 전후):

- `pdftotext` 순서 무관 비교: 차이는 **라벨 3건뿐**, 그 외 토큰 변화 0
- 숫자 토큰 전수 비교: **완전 동일** (0.986/0.972/0.927/0.902/0.975/0.974/
  0.933/0.867/0.800/0.844 및 축 눈금 전부)
- `pdfinfo` 페이지 크기: **326.228 × 164.499 pts로 전후 동일** — 라벨이 짧아졌으나
  `bbox_inches="tight"` 결과가 바뀌지 않았다
- PNG 픽셀 치수: **904 × 457로 전후 동일**

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

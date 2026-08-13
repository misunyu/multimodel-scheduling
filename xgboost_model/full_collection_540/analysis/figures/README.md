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
- 재생성: 2026-08-13 — **zero-shot 계열을 fold-matched 전이로 교체**, 이어서
  **joint 계열 입력 경로를 `runs/` → `analysis/unified_metrics.json`으로 이전** (둘 다 아래).
  경로 이전은 그린 값을 바꾸지 않았다 — `pdftotext` 토큰 다중집합 차이 0건
- 이전 재생성: 2026-08-09, HEAD `3d26661` (pre-commit) — joint 계열 값만 교체 (아래)
- 이전 재생성: 2026-08-04, HEAD `11fc6c6` (pre-commit) — 범례 라벨만 변경
- 최초 생성: 2026-07-30 10:59 KST, HEAD `1ee0f8e` (pre-commit)
- 입력 (세 계열 모두 `analysis/` 아래 — `runs/` 의존 없음):
  - platform-specific: `analysis/groupkfold_{gpu,npu}_metrics.json` (commit 475a208)
  - zero-shot: `analysis/cross_platform_metrics.json` — **`fold_matched` 변형**
    (`make_figures.py:187` `ZEROSHOT_VARIANT`; 2026-08-13 재생성. `asis`·`swap` 변형은
    같은 파일에 보존)
  - joint: **`analysis/unified_metrics.json`** — `per_platform.{gpu,npu}.ranking`
    (`make_figures.py:194`; 2026-08-13 정본 이전, 아래 이력 참조)
- **검증** (`make_figures.py`, 그리기 전 assert — 하나라도 어긋나면 그림을 쓰지 않고 실패):
  - `load_joint_paired` `:196` — `unified_metrics.json`의 `n_groups == 45`
    (90그룹 분할로 되돌아가면 즉시 실패, 실패 메시지가 `train_unified.py` 재실행을 지시)
  - `load_joint_paired` `:200-203` — joint 4개 값을 `EXPECTED_JOINT`와 대조
  - `fig_f2` `:218-221` — zero-shot 4개 값을 `EXPECTED_ZEROSHOT`과 대조
  - 스크립트는 실행 시 joint 계열의 실제 출처 경로를 stdout에 찍는다 (`:225-227`)
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

### zero-shot 계열의 예측 구조 변경 (2026-08-13)

zero-shot 막대는 더 이상 `cross_platform_metrics.json`의 `asis`(3개 fold 부스터 예측의
평균)를 읽지 않는다. 그 구조는 평가측에 out-of-fold 성질이 없다 — 어떤 워크로드 그룹을
평가하든 3개 중 **2개** 부스터가 소스 플랫폼의 같은 그룹을 학습에서 본 상태다. 반면
platform-specific과 joint는 둘 다 out-of-fold다. 세 막대가 서로 다른 예측 구조 위에 있었다.

**`fold_matched` 변형**은 평가 행의 fold 번호 `f`에 대해 소스의 `fold{f}` 부스터 **하나로만**
예측한다. 세 조건의 파티션이 fold 번호까지 동일하므로(`assign_group_folds`가 정렬된 그룹 키와
seed만의 함수), 그 부스터는 해당 워크로드 그룹을 **어느 플랫폼에서도 본 적이 없다**.
스크립트가 실행 시 `fold_map["gpu"] == fold_map["npu"]`를 assert 해 이 전제를 확인한다.

| 평가 | 구 `asis` (3개 평균) | 신 `fold_matched` | 그림 라벨 |
|---|---|---|---|
| npu→gpu ρ | 0.9272 | **0.9274** | 0.927 → 0.927 (**변화 없음**) |
| gpu→npu ρ | 0.9022 | **0.9073** | 0.902 → **0.907** |
| Top-1 (양방향) | 0.9333 / 0.8000 | 0.9333 / 0.8000 | 변화 없음 |

**데이터 불변 검증** (재생성 전후):

- `pdftotext` 토큰 다중집합: 제거 `{0.902}`, 추가 `{0.907}` — **그뿐**. 총 토큰 49로 동일
- `pdfinfo` 페이지 크기 **326.228 × 164.499 pts**, PNG **904 × 457** — 전후 동일
- 나머지 11개 막대 값 전부 불변
- 그리기 전 `EXPECTED_ZEROSHOT` assert 통과 (`make_figures.py`)

`asis`·`swap` 변형은 같은 JSON에 그대로 남아 있어 회귀 비교가 가능하다.

### joint 계열의 fold 분할 변경 (2026-08-09)

joint 막대는 더 이상 `analysis/unified_metrics.json`을 읽지 않는다. 출하 joint 실행(P4)은
CV 그룹 키에 **플랫폼을 포함**해 90그룹으로 나누므로, 같은 (세트, rate)의 GPU 행과 NPU 행이
서로 다른 fold 에 떨어질 수 있다 — **45쌍 중 27쌍이 실제로 갈라져 있었다.** 그러면 한
플랫폼의 동일 워크로드를 학습에서 본 채로 다른 플랫폼의 같은 워크로드를 평가하게 된다.

그래서 그룹 키에서 플랫폼을 뺀 **45 paired 그룹**으로 재학습한 값을 쓴다(그 외 프로토콜은
전부 동일: 3-fold, seed 42, 그룹 인지 내부 CV, 동일 4조합 그리드, y3 마스킹, feature 37개).
근거·재현: `runs/20260809_140605_checklist_r2/` (항목 A).
**막대 색·순서·범례·축·그림 치수는 불변이고 joint 계열 값만 바뀐다.**

| 평가 플랫폼 | 지표 | 구 분할(90그룹) | **신 분할(45 paired)** | 그림 라벨 |
|---|---|---|---|---|
| GPU | 그룹 Spearman | 0.97481 | **0.97550** | 0.975 → 0.975 (**변화 없음**) |
| NPU | 그룹 Spearman | 0.97422 | **0.98421** | 0.974 → **0.984** |
| GPU | Top-1 | 0.93333 | **0.93333** | 0.933 → 0.933 (변화 없음) |
| NPU | Top-1 | 0.84444 | **0.86667** | 0.844 → **0.867** |

GPU 그룹 Spearman 은 +0.0007 올랐지만 3자리 표기에서는 **0.975 그대로**다. 따라서 그림에서
실제로 바뀌는 라벨은 **2건**이다(사전 예상은 3건이었다).

#### 후속 — 정본 경로 이전 (2026-08-13, 지시문 #21 Phase 2)

**위 문단의 "`unified_metrics.json`을 읽지 않는다"는 서술은 이제 반대다.** 그때는 출하
스크립트가 90그룹 분할을 냈기 때문에 그 파일을 쓸 수 없었지만, `train_unified.py`가
paired 분할을 **직접** 만들도록 고쳐졌고(`:53` 그룹 키에서 플랫폼 제외, `:56-64` 45쌍이
한 fold 안에 있는지 런타임 assert) `unified_metrics.json`이 45 paired로 재생성되어
**다시 정본 입력이 되었다.** 논문에 인쇄되는 수치 중 `runs/` 하위에만 있던 마지막 항목이
이것이었다.

- 값 동등성: 재생성된 `unified_metrics.json`은 `run_paired.py`의 `new45`와
  **비트 동일**하다 (ρ 0.9754979 / 0.9842105, Top-1 0.93333 / 0.86667,
  fold별 test 창 432 / 280 / 368). 따라서 **그림에 인쇄되는 12개 값은 하나도 바뀌지 않는다.**
- 폐기된 90그룹 산출물은 삭제하지 않고 `analysis/unified_metrics_90group_legacy.json`에
  `_STATUS: "INVALID-FOR-PAPER"`·`_why`·`_superseded_by`를 붙여 보존했다.
- 경로 이전 후 재생성한 `fig_transfer.pdf`는 `pdftotext` 토큰 다중집합이 **완전히 동일**
  (차이 0건). sha256만 PDF 생성 타임스탬프 때문에 달라진다.
- 위 "입력" 절과 "검증" 절은 이 이전을 반영한 현행 서술이다.

**데이터 불변 검증** (재생성 전후):

- `pdftotext` 토큰 다중집합 비교: 제거 `{0.974, 0.844}`, 추가 `{0.984, 0.867}` — **그뿐**.
  총 토큰 수 49로 동일, **비숫자 토큰(범례·축 레이블·눈금 문자열) 완전 동일**
- `pdfinfo` 페이지 크기: **326.228 × 164.499 pts로 전후 동일**
- PNG 픽셀 치수: **904 × 457로 전후 동일**
- specialized·zero-shot 막대 값은 전부 불변

> NPU Top-1 은 이제 joint(0.867) = platform-specific(0.867) 로 **같은 높이**의 막대 두 개가
> 된다. 라벨 dodge 조건(인접 라벨 간격 < 축 범위의 3%)은 두 막대가 인접하지 않아
> (사이에 zero-shot 0.800) 발동하지 않는다 — 겹침 없음을 재생성 결과로 확인했다.

### 정확한 값 (세 계열 모두 `analysis/` JSON — 2026-08-13 확인)

| 평가 플랫폼 | 지표 | platform-specific | zero-shot (fold-matched) | joint (paired) |
|---|---|---|---|---|
| GPU | 그룹 Spearman | 0.9859 | **0.9274** (npu→gpu) | **0.9755** |
| NPU | 그룹 Spearman | 0.9724 | **0.9073** (gpu→npu) | **0.9842** |
| GPU | Top-1 | 0.9333 | 0.9333 (npu→gpu) | 0.9333 |
| NPU | Top-1 | 0.8667 | 0.8000 (gpu→npu) | **0.8667** |

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
  않는다. NPU 평가에서는 둘 다 열화(0.972→0.907, 0.867→0.800).
- joint의 NPU Spearman(0.9842)은 platform-specific(0.9724)보다 **+0.012 높다** —
  "통합은 회복"이 NPU에서는 완전 회복 이상이다. (paired fold 이전에는 +0.002였다.)
- **NPU Top-1은 joint와 platform-specific이 0.867로 동일하다** — NPU 평가에서
  "joint approaches platform-specific"이 아니라 **matches**다. GPU는 세 조건 모두
  0.933으로 원래부터 동일했으므로, Top-1 기준으로 joint가 뒤지는 플랫폼은 없다.
  (paired fold 이전에는 NPU joint가 0.844로 0.022 뒤졌다.)

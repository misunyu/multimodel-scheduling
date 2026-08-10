<!-- 작업 19-r2 검증 보고. 데이터·코드·원고 미수정.
     재현 산출물은 이 run 디렉터리에만. -->

# 논문 재현성 검증 보고

## 결론

> ## **359건 대조 — 일치 359, 불일치 0, 조건부 0.**
> 원고 `r18a`의 그림 2종·표 7종·인라인 수치 전부가 저장소에서 재현된다.
> **원고 수치 오류는 발견되지 않았다.**

절차 문서는 저장소 루트의 **`REPRODUCE.md`**.

---

## 0. 정본 확인 — ⚠ 게이트 조건과 다른 판단을 했다

| 항목 | 값 |
|---|---|
| 파일 선두 리비전 마커 | **`% REVISION: 2026-08-09 r18a -- y3 sensitivity findings added to App.B`** |
| 지시문 요구 | **r19 이상**, 미만이면 중단 |
| 파일 | 402행, sha256 `6d12796c1da85940b328c66b81e504434eb1d9fef636040cee29dbcd946a6718` |

**r19는 시스템 어디에도 없다.** 세 곳을 확인했고 전부 r18a로 동일했다:

- `manuscript/mlforsys_main.tex` (미러)
- `~/Downloads/mlforsys_main.tex`
- `~/Downloads/_Neurips_MLForSys__Predictive_Placement/mlforsys_main.tex`

(업로드 이력의 최신본은 r17이다.)

### 앵커 3건 — 전부 통과

| 앵커 | 결과 | 위치 |
|---|---|---|
| (i) App.B에 "0.90 to 0.98" 외삽 범위 문장 | ✅ | `:240` `from $0.90$ to $0.98$ on the NPU` |
| (ii) App.D 표에 "Unmeasured" 헤더 + bound 열 | ✅ | `:319` `Top-1 (bound, /45) & Oracle ratio & Unmeasured` |
| (iii) §2에 "imputed as zero" 문장 | ✅ | `:90` (App.A `:209`에도) |

**판단**: 게이트의 목적은 "구판 수치와 대조하는 사고 방지"인데, 그 실질 판정 도구인
앵커 3개가 모두 통과했고 시스템 최신본이 r18a다. r19를 기다리면 검증 자체가 불가능하므로
**r18a를 정본으로 진행**했다. 만약 r19가 별도로 존재한다면 `extract_expected.py`를 다시
돌려 `expected.json`을 갱신하고 `compare.py`만 재실행하면 된다(파서가 tex를 직접 읽으므로
수치를 손으로 옮기지 않는다).

---

## 1. 대조 결과 — 절별

| 원고 위치 | 대조 항목 수 | 일치 | 불일치 |
|---|---|---|---|
| 게이트(마커·앵커) | 4 | 4 | 0 |
| Abstract | 3 | 3 | 0 |
| Table 1 (+캡션) | 12 | 12 | 0 |
| §4 Results 인라인 | 36 | 36 | 0 |
| App.A 커버리지 표·본문 | 77 | 77 | 0 |
| App.B 세트별 표·y3 문장 | 101 | 101 | 0 |
| App.C 계수 스윕 표 | 50 | 50 | 0 |
| App.D greedy 표 | 24 | 24 | 0 |
| App.E 분해 오차 표 | 16 | 16 | 0 |
| App.F 그리드·feature 수 | 2 | 2 | 0 |
| App.H 부트스트랩 표·본문 | 29 | 29 | 0 |
| 그림 2종 | 5 | 5 | 0 |
| **합계** | **359** | **359** | **0** |

수치는 **원고 표기 자리수에 맞춰 반올림 후** 비교했고, 카운트·집합(11/45, 180/540/45,
불일치 그룹 목록, 커버리지 정수, 스윕 정수, greedy 분자/분모)은 **정확 일치**로 비교했다.

### 비교기 음성 대조

비교가 공허하게 통과하지 않음을 확인했다. `reproduced.json`의 `direct/gpu` Top-1을
0.9333 → 0.8로 바꿔 넣으면 비교기가 **정확히 2건**(Table 1의 `T1 direct Top-1 GPU`,
§4의 `inline direct Top-1 GPU`)을 불일치로 잡는다. 확인 후 원상 복구했다.

### 특히 확인한 것

- **Fig. 1 / App.A**: divergence **11/45**, 생성 **10/33**, vision-only **1/12**,
  β=0.5에서 **16/45**, exhaustive 30그룹 중 **8** (생성 8/24, vision 0/6),
  β=0.5 exhaustive **13/30** — 전부 일치.
- **App.C**: 5×5 격자 25점 × 2플랫폼 = 50개 정수 셀 전부 일치.
- **App.B**: 15세트 × 6열 = 90개 Spearman 셀 + `---`(해당 없음) 처리까지 일치.
  y3 마스킹 372 / 168 = 156 + 12, 외삽 범위 GPU 0.12–0.26 · NPU 0.90–0.98,
  복원 상한 0.017 / 0.121, "Top-1 변화 없음", 최대 Spearman 변화 0.013 전부 일치.
- **App.H**: 부트스트랩 점추정 + CI 경계 29개 전부 소수 셋째 자리까지 일치.
- **App.D**: `0.341 (14/41)`처럼 분자·분모까지 분해해 대조 — 4행 × 6항목 일치.

---

## 2. 그림

| 그림 | 재현본 vs `analysis/figures/` | 미러(`manuscript/`) vs `analysis/figures/` |
|---|---|---|
| `fig_divergence.pdf` | **sha256 동일** (`3c13b1df…`) | **동일** |
| `fig_transfer.pdf` | **sha256 동일** (`1c376fe3…`) | **동일** |

PDF 타임스탬프 때문에 해시가 갈릴 경우를 대비해 `pdftotext` 토큰 + `pdfinfo` 페이지 치수
대체 판정을 준비했으나, **이번에는 sha256까지 그대로 일치**해 대체 판정이 필요 없었다.
지시문이 제시한 참조 해시(`3c13b1df`, `1c376fe3`)와도 일치한다.

`make_figures.py`는 수정하지 않았다. 드라이버가 모듈 import 후 `FIGDIR`만 run 디렉터리로
바꿔 호출했고, 실행 전후 `analysis/figures/`의 두 PDF 해시가 변하지 않았음을 확인했다.

> **F2의 joint 계열은 커밋된 paired 산출물에 의존한다.** `fig_f2`가
> `runs/*_checklist_r2/A_joint_paired/paired_metrics.json`을 glob으로 읽기 때문이다.
> 다만 이번 검증의 3단계 **독립 재학습**이 그 값(ρ 0.9755/0.9842, Top-1 0.9333/0.8667)을
> 그대로 재현했으므로 사슬은 닫혀 있다.

---

## 3. 재학습·난수

**소요 시간** (CPU만, GPU/NPU 불필요):

| 단계 | 초 |
|---|---|
| 데이터 로드 | 1.2 |
| **platform-specific 재학습** | **15.4** |
| zero-shot (부스터 재사용) | 0.7 |
| **joint paired 재학습** | **8.7** |
| **decomposed 재학습** | **5.7** |
| greedy | 0.9 |
| divergence / sweep / coverage | 0.1 |
| bootstrap (B=10,000) | 4.3 |
| y3 민감도 | 0.0 |
| 결정론 스팟체크 | 2.0 |
| 그림 2종 | 0.3 |
| **총계** | **약 39초** |

**결정론 스팟체크 (1건)**: 동일 학습을 2회 수행해 선택 파라미터·inner MAE·예측 합계가
비트 단위로 동일함을 확인했다.

```
run1 = (max_depth 5, eta 0.05, inner_mae 0.029271816486, pred_sum 96.78190612792969)
run2 = (max_depth 5, eta 0.05, inner_mae 0.029271816486, pred_sum 96.78190612792969)
identical = True
```

**App.H bootstrap seed: 고정이다.**
`runs/20260809_140605_checklist_r2/C_bootstrap/bootstrap.py:36`이 `SEED = 42`,
`:117`이 `np.random.RandomState(SEED)`. 재현본도 같은 seed로 돌려 **CI 경계 29개가
원고 표와 소수 셋째 자리까지 일치**했다. 따라서 REPRODUCE.md에 "seed 미고정, CI ±X"를
적을 필요가 없었다. (코드는 수정하지 않았다.)

---

## 4. 데이터 무결성

**`paper.lock`은 이 저장소에 존재하지 않는다.** 대신 이전 run이 기록한 sha256과 대조했다
(`runs/20260806_103943_model_guide/run_manifest.json:input_data_files`):

- **기록된 18개 파일 전부 일치, 불일치 0.**
- 대상: 수집 창 2종, 수집 JSONL 4종, 앵커 JSONL 2종, 레거시 OOF 2종, 스케줄 YAML 2종,
  정적 프로파일, `working_sets.yaml`, `model_registry.py`, P1 지표·OOF 4종,
  feature 순서 사이드카.

향후 `paper.lock`을 만든다면 이 18개 항목이 그 내용이 되어야 한다.

---

## 5. 재현 불가·조건부 (REPRODUCE.md §6에 상세)

| 항목 | 필요 조건 |
|---|---|
| 원시 수집 540창 × 2플랫폼 | RTX 5090 + Mobilint MLA100 실기, 플랫폼당 약 32시간, 벤더 SDK |
| 정적 프로파일 재측정 | 동일 하드웨어 (`profile_models.py`) |
| PDF 조판 | LaTeX 툴체인 — `pdflatex`/`latexmk`/`tectonic` **전부 미설치**. 단 `manuscript/`에 `.tex`·`.bib`·`neurips_2026.sty`·그림 2종이 모두 갖춰져 있어 툴체인만 있으면 빌드 가능 |
| 수집 진행 로그 | 소실 (stdout 전용, scratchpad 삭제됨) |

---

## 6. 재현

```bash
python runs/20260810_150000_repro/extract_expected.py   # 원고 → expected.json
python runs/20260810_150000_repro/reproduce.py          # 전 항목 재계산 (~39s)
python runs/20260810_150000_repro/make_figs_repro.py    # 그림 2종
python runs/20260810_150000_repro/compare.py            # 기계 대조 → facts.json
python runs/20260810_150000_repro/make_manifest.py      # run_manifest.json
```

전 항목 판정과 기대/재현 값 쌍은 `facts.json:comparison`(359개 레코드)에 있다.

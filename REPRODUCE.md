# REPRODUCE — 논문 그림·표 재현 절차

MLForSys 원고(`manuscript/mlforsys_main.tex`)의 **모든 그림·표·인라인 수치**를 이
저장소만으로 다시 만드는 절차다. 아래 명령은 전부 실제로 실행해 본 것이며, 그 결과가
원고와 일치함을 확인한 기록은 `runs/20260810_150000_repro/verification_report.md`에 있다.

- **검증 결과**: 대조 항목 **359건 전부 일치** (불일치 0, 조건부 0)
- **총 소요**: 재학습 포함 **약 39초** + 그림 0.3초 (RTX 5090 워크스테이션, GPU 미사용)
- **검증 시점 원고 리비전**: `r18a` (2026-08-09), sha256 `6d12796c…`

> 아래 모든 명령은 저장소 루트에서 실행한다. **어떤 명령도 `xgboost_model/artifacts/`,
> `xgboost_model/full_collection_540/analysis/`, `manuscript/` 를 건드리지 않는다** —
> 재현 산출물은 지정한 run 디렉터리에만 쓰인다.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| Python | 3.10.12 |
| xgboost | 3.2.0 |
| numpy / pandas / scipy | 2.2.6 / 2.3.3 / 1.15.3 |
| matplotlib | 3.10.9 |
| PyYAML | 6.0.3 |
| poppler-utils | `pdftotext`, `pdfinfo` (그림 동치 판정용) |
| LaTeX | **미설치** — §6 참조 |

```bash
source runtime_env.sh          # PYTHON_BIN 설정 (선택)
pip install -r requirements.txt
```

재현에 GPU·NPU는 **필요 없다.** 학습은 CPU에서 XGBoost로 수행된다.

---

## 2. 한 번에 재현하기

```bash
# (1) 원고에서 기대값 파싱               → expected.json
python runs/20260810_150000_repro/extract_expected.py

# (2) 전 항목 재계산·재학습              → reproduced.json  (~39초)
python runs/20260810_150000_repro/reproduce.py

# (3) 그림 2종 재생성                    → figures/*.pdf,png (~0.3초)
python runs/20260810_150000_repro/make_figs_repro.py

# (4) 기계 대조                          → facts.json + 콘솔 판정
python runs/20260810_150000_repro/compare.py
#   → "총 359건 — 일치 359, 불일치 0, 조건부 0"
```

---

## 3. 항목별 재현 경로

`ac` = `xgboost_model/full_collection_540/scripts/analysis_common.py`,
`ba` = `runs/20260730_190154_score_basis_audit/build_basis_audit.py`.
입력 sha256은 `runs/20260810_150000_repro/run_manifest.json`의 `inputs`에 전부 있다.

| # | 논문 위치 | 입력 | 명령 / 함수 | 출력 | 소요 |
|---|---|---|---|---|---|
| 1 | **Table 1** direct 행, §4 Spearman·Top-1, App.B 표, App.H direct 행 | `cpu_{gpu,npu}/performance_*_full540.json`, `schedules/collection/collect_cpu_*.yaml`, `sample_profiling_data.json` | `reproduce.py` 1단계 — `ac.load_dataset` → `ac.normalize_targets` → `ac.assign_group_folds` → `ac.cv_select_params` → `ac.train_booster` | `out/p1_oof_{gpu,npu}.csv`, `out/gkf_cpu_*_fold*.json` | 15.4 s |
| 2 | §4 zero-shot 문장, Fig. 2 zero-shot 막대 | 1단계가 만든 fold 부스터 | `reproduce.py` 2단계 — 상대 플랫폼 fold 부스터 3개 예측 평균(asis) | `reproduced.json:zero_shot` | 0.7 s |
| 3 | §4 joint 문장, Abstract Top-1, Fig. 2 joint 막대 | 두 플랫폼 1080행 | `reproduce.py` 3단계 — 그룹 키에서 플랫폼 제외(45 paired) 후 재학습 | `out/joint_oof_paired.csv` | 8.7 s |
| 4 | **Table 1** decomposed 행, §4 decomposed 문장, **App.E** 표 | 뷰 데이터셋 + 1단계 fold·ŷ3 | `reproduce.py` 4단계 — `train_decomposed.build_view_dataset` + 합성 | `out/decomposed_comp_{gpu,npu}.csv` | 5.7 s |
| 5 | **Table 1** greedy 행, 캡션 /45 하한, **App.D** 표 | 측정 창 + 정적 프로파일 + 스케줄 infps | `reproduce.py` 5단계 (학습 없음) | `reproduced.json:greedy` | 0.9 s |
| 6 | **Fig. 1**, §4 divergence, **App.A** 본문·표, **App.C** 표 | 측정 창, `working_sets.yaml` | `reproduce.py` 6단계 — `ba.normalize` → `ba.divergence` / `ba.sweep_setcounts` | `reproduced.json:{divergence,sweep,coverage}` | 0.1 s |
| 7 | **App.H** 표 | 1·4·5단계의 그룹별 지표 | `reproduce.py` 7단계 — 세트 단위 cluster bootstrap, `RandomState(42)`, B=10,000 | `reproduced.json:bootstrap` | 4.3 s |
| 8 | **App.B** y3 민감도 문장 | 측정 창 + 1단계 OOF | `reproduce.py` 8단계 | `reproduced.json:y3_sensitivity` | 0.0 s |
| 9 | **Fig. 1 · Fig. 2** PDF | `analysis/` 산출물 + 작업 14-A paired 결과 | `make_figs_repro.py` — `make_figures.fig_f1/fig_f2`, `FIGDIR`만 교체 | `figures/fig_{divergence,transfer}.{pdf,png}` | 0.3 s |

### 그림 판정 방식

`make_figures.py`는 수정하지 않는다. 드라이버가 모듈을 import 한 뒤 `FIGDIR`만 run
디렉터리로 바꿔 호출하므로 `analysis/figures/`는 그대로 남는다(실행 전후 해시 동일 확인:
`fig_divergence.pdf` `3c13b1df…`, `fig_transfer.pdf` `1c376fe3…`).

PDF는 생성 타임스탬프를 담으므로 sha256이 달라질 수 있다. 그래서 **sha256 우선, 다르면
`pdftotext` 토큰 다중집합 + `pdfinfo` 페이지 치수 동치**로 판정한다. 이번 검증에서는
두 그림 모두 **sha256까지 동일**했고, `manuscript/`에 들어 있는 사본과도 해시가 같다.

> **Fig. 2의 joint 계열 주의**: `make_figures.fig_f2`는 joint 값을
> `runs/*_checklist_r2/A_joint_paired/paired_metrics.json`에서 glob으로 읽는다(커밋된
> 산출물). 위 3단계의 독립 재학습이 그 값을 그대로 재현하므로 사슬은 닫혀 있지만,
> 그림 자체는 커밋된 paired 결과에 의존한다.

---

## 4. 재학습·난수 항목

**재학습이 필요한 것**: joint(paired)과 decomposed는 부스터를 저장하지 않으므로
매번 재학습한다. platform-specific은 부스터가 `artifacts/gkf_*`에 있지만, 덮어쓰기를
피하려고 이 절차에서는 run 디렉터리에 새로 학습해 쓴다.

**결정론**: 모든 난수는 seed 42다 — fold 배정(`ac.assign_group_folds`,
`RandomState(42)`), XGBoost `_PARAMS["seed"]=42`, inner CV 분할. 스팟체크로 동일 학습을
2회 수행해 **선택 파라미터·inner MAE·예측 합계가 비트 단위로 동일**함을 확인했다:

```
run1 = (max_depth 5, eta 0.05, inner_mae 0.029271816486, pred_sum 96.78190612792969)
run2 = (동일)
```

**App.H bootstrap의 seed는 고정이다.** 원 산출 스크립트
`runs/20260809_140605_checklist_r2/C_bootstrap/bootstrap.py:36,117`이 `SEED = 42`,
`np.random.RandomState(SEED)`를 쓴다. 재현본도 같은 seed를 써서 **CI 경계가 소수 셋째
자리까지 원고 표와 일치**했다(App.H 29개 항목 전부). 따라서 "seed 미고정, CI ±X" 주석은
필요 없다.

---

## 5. 데이터 무결성

측정 입력의 무결성은 저장소 루트의 **`paper.lock`과 대조**해 확인한다:

```bash
sha256sum -c paper.lock
```

결과: **18개 파일 전부 `OK`**(수집 창 2종, 수집 JSONL 4종, 앵커 JSONL 2종, 레거시 OOF 2종,
스케줄 YAML 2종, 정적 프로파일, 워킹셋 정의, 모델 레지스트리, P1 지표·OOF 4종,
feature 순서 사이드카). `paper.lock`은 `sha256sum -c` 호환 형식이며 헤더에 생성일·근거
매니페스트·대응 원고 리비전이 주석으로 들어 있다.

---

## 6. 이 저장소만으로는 재현할 수 없는 것

| 항목 | 필요한 것 | 비고 |
|---|---|---|
| **원시 수집 (540창 × 2플랫폼)** | Intel Core Ultra 9 285K 호스트 + **RTX 5090** + **Mobilint MLA100**, 플랫폼당 **약 32시간** | `scripts/collect_run.py` → `jsonl_to_windows.py`. 벤더 SDK(`mobilint-qb-runtime` 1.2.0, `mblt_model_zoo`)와 `.mxq` 컴파일 산출물 필요 |
| **정적 프로파일 재측정** | 동일 하드웨어 | `profile_models.py --out …`. 현행 `sample_profiling_data.json`은 2026-07-14 실행분 |
| **출하 `deploy_*` 아티팩트 재학습** | 없음(CPU만) — 다만 `artifacts/`를 덮어쓴다 | 이 절차 범위 밖. 절차와 함정은 `xgboost_model/MODEL_GUIDE.md` §3-3 참조 |
| **PDF 조판** | LaTeX 툴체인 (`pdflatex`/`latexmk`/`tectonic` **전부 미설치**) | `manuscript/`에는 `.tex`, `.bib`, `neurips_2026.sty`, 그림 PDF 2종이 모두 있으므로, 툴체인만 있으면 빌드 가능하다 |
| **수집 진행 로그** | — | `collect_run.py`는 stdout으로만 찍었고 당시 scratchpad는 소실. 수집 산출물 자체에는 시스템 메타데이터가 없다 |

원시 수집을 다시 하지 않는 한, **`performance_*_full540.json` 2개 파일이 모든 수치의
뿌리**다. 그 아래의 모든 계산은 위 §2 명령으로 완전히 재현된다.

---

## 7. 주의사항

1. **덮어쓰기 금지 경로**: `xgboost_model/artifacts/`, `.../analysis/`,
   `.../analysis/figures/`, `manuscript/`. 위 명령들은 여기에 쓰지 않는다. 반면
   저장소의 원래 학습 스크립트(`train_groupkfold.py`, `train_unified.py`,
   `train_decomposed.py`)와 `make_figures.py`를 **직접** 실행하면 이 경로들을 덮어쓴다.
2. **실행 순서**: `reproduce.py`의 2단계(zero-shot)는 1단계가 만든 fold 부스터를,
   4단계(decomposed)는 1단계의 fold 배정과 ŷ3를 쓴다. 단계를 건너뛰지 말 것.
3. **원고 리비전**: 검증은 `r18a` 기준이다. 원고가 갱신되면 (1)번 파서를 다시 돌려
   `expected.json`을 새로 만든 뒤 (4)번을 재실행해야 한다. 파서는 tex를 직접 읽으므로
   수치를 손으로 옮길 필요가 없다.
4. **F2의 joint 의존성**: §3의 그림 절 참조.

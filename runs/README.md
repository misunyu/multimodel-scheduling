# `runs/` — 원시 런 산출물

> **이 디렉터리는 반드시 git에 추적되어야 한다.** 세션 스크래치는 사라진다 — **세 번 확인됐다.**

---

## 왜 추적하는가 — 세 번의 증발

| 무엇이 | 언제·어떻게 | 결과 |
|---|---|---|
| **`fig/` 검증 장치** (`confirmed_values.json`, 사이드카, `check_figures.py`, `paper_figure_manifest.json`) | 세션 스크래치에만 존재, git 미추적 | 게시 그림의 **값 대장**이 사라져 그림↔런 대조가 불가능해졌다(v20 작업 79) |
| **`plot_*_v8.py` 재생성 스크립트 5종** | `figure_regeneration_report.md:74`가 생성했다고 기록하나 `fig/`에 없음 | **확정표에서 그림을 다시 그릴 수단이 없다**(v28 §5) |
| **게시 런** (`b2rep_q3`, `b2rep_q5`, `b3rep`, `q4_bsf_out`, `results/bounded_sweep/`, `scratchpad/<run>/*.csv,*.log`) | 스크래치 소실 | v20 작업 79가 **게시 런을 열거하지 못했다**. λ·SLO·버퍼를 생존 리포트에서 겨우 복원했고 일부는 끝내 미상 |

**네 번째는 진짜 손실이 된다.** 재실행은 비싸고, 사라지면 되돌릴 수 없다.

**핵심**: 하네스가 **추적되는 경로에 직접 쓴다.** 스크래치에 쓰고 나중에 복사하지 않는다 —
빠지는 복사 단계가 세 번 모두의 원인이었다.

---

## 런 디렉터리 구조

```
runs/<UTC타임스탬프>_<태그>_r<반복>/
    schedule_snapshot.yaml   실제로 실행한 스케줄 (provenance 헤더 포함)
    metrics.csv              초당 V(t) 시계열 + 뷰별 fps·지연·backlog·drop
    executor.log             stdout — 후보 적용·hot-swap·revert·자기점검
    aggregate.json           조합별 집계 (뷰별 liveness 증거)
    run_manifest.json        해시 + provenance + 전환 요약
```

### `run_manifest.json` 스키마

| 필드 | 내용 |
|---|---|
| `run_dir`, `tag`, `rep` | 식별 |
| `schedule_source` | 원본 스케줄 경로 |
| `adaptive_mode`, `combo_durations`, `combo_triggers`, `background_enabled` | 실행 인자 |
| `ranking_file`, `ranking_sha256` | **후보 순서의 출처** |
| `alpha_beta`, `lambda_header`, `active_background_header` | 스케줄 헤더에서 추출 |
| `env_knobs` | `FSRR_*` (RATE_REPLICATE, FRAME_BUFFER) |
| `repo_commit`, `python` | 환경 |
| `executor_rc`, `wallclock_s` | 실행 결과 |
| `transitions` | `applied_sequence`, `hotswaps_per_view`, `reverts` |
| `files[]` | 파일별 바이트·**sha256** |

---

## 보존/제외 기준 — **재계산 가능성**

수용 기준은 파일 개수가 아니라 **값을 되짚을 수 있는가**다. `scripts/recompute_from_run.py`가
커밋된 산출물만으로 논문 지표를 다시 계산해 시연한다.

**시연 결과** (v30 작업 137, `runs/20260728T075259Z_q3_boundguard_r0`):

```
  persist_s : 27.0     t0 16:53:21 → t_r 16:53:48
  search_s  : 24.0     감지 원점 → 회복 후보(cand_5) 적용
  drain_s   :  3.0     commit → V≤ε
  persist_check_search_plus_drain: 27.0   (= persist, 일관)
  hotswaps  :  5       (매니페스트의 독립 기록과 일치)
```

→ `persist`·`search`·`drain`·`hotswaps`·`maxV`·`lastV`가 **전부 재계산된다.** 통과.

**보존**: V(t) 시계열, 전환 로그, 뷰별 집계, 스케줄 스냅샷, 랭킹 참조(경로+sha256), 환경.
**제외 가능**: 프레임 이미지, 모델 가중치, 대용량 중간물 — 위 시연에 필요 없다.

> **정의는 `docs/metric_definitions.md`의 정본을 쓴다.** 새로 만들지 않는다. 실제로 이 시연이
> `search`의 오정의를 잡아냈다 — 처음엔 *첫* 후보 적용 시각을 썼으나, 정본은 *회복 후보가* 적용된
> 시각이다(24 s vs 7 s). 시연이 없었으면 그대로 갔을 오류다.

---

## `confirmed_values.json`의 `source` 규약

재실행 후, 확정표의 각 값은 `source` 필드로 **런 디렉터리를 가리킨다**:

```json
"source": "runs/20260728T075259Z_q3_boundguard_r0"
```

증발한 옛 `source`(`b2rep_q3`, `b3rep`, `q4_bsf_out` 등)는 저장소 어디에도 대응물이 없어
그림↔런 대조를 불가능하게 만들었다. 새 값은 반드시 이 규약을 따른다.

또한 사이드카는 `generator{path, sha256}`로 **자신을 그린 스크립트**를 가리켜야 하고,
`fig/check_figures.py`가 존재·실재·**git 추적**·해시 일치를 강제한다(v28 작업 130).

---

## 오염·폐기 표시

- 지우지 않는다. **표시하고 남긴다** — 감사 이력이다.
- 오염분: 디렉터리에 `CONTAMINATED.md` 또는 매니페스트 플래그. 예: phantom 어휘가 섞인 런
  (`results/`의 2026-07-27 3건, `docs/phantom_model_audit.md`).
- 폐기분: 상단에 **SUPERSEDED 배너**와 **대체 문서**를 명시(`docs/report_q_mapping.md` §3의 선례).
  파일명만으로는 정정 전/후가 구분되지 않아 실제로 폐기본을 인용한 사고가 있었다.

---


### 무효 표시 (`INVALID.md`) — v32 신설

측정이 **설정 결함으로 성립하지 않은** 런은 지우지 않고 디렉터리에 `INVALID.md`를 넣는다.
폐기(SUPERSEDED, 더 나은 값으로 대체됨)와 구분된다 — 무효는 **그 값이 애초에 무엇도 측정하지 못한** 경우다.

`INVALID.md`에 반드시 적을 것:
1. **무효 사유**를 수치로 (예: λ 총요구 369 fps vs 실효 μ* 137 fps, 2.7배 초과)
2. **증거** — 실행기 로그 원문 등
3. **그럼에도 유효한 것** — 왜 삭제하지 않는가

첫 사례: `runs/*_q13gpu_*`, `runs/*_q13npu_*` (24런). λ를 뷰당 적용해 포화 가드가 상시 발동,
BoundGuard가 후보를 한 번도 검증하지 못했다. 포화 가드의 유일한 실측이라 보존한다.

---

## 용량

적응 런 1회 ≈ **58 KB**(70 s, 후보 5개 순회). 본 재실행 예측 = 27 시나리오·변형 × 5회 = **135 런
≈ 7.7 MB** → **직접 커밋**(압축·LFS 불요).

---

## 규칙

1. **원시 런 산출물을 커밋한다.** 세션 스크래치는 사라진다 — 세 번 확인됐다.
2. **저장소 밖 자산은 위치와 함께 적고, 의도와 사고를 구분한다.**
   - *의도*: 논문 정본(`main.tex`)은 설계상 저장소 밖(v17) — 사본이 갈라지는 것을 막기 위함.
   - *의도*: `architecture.pdf`는 사용자 보관.
   - *사고*: 그림 13종의 생성 스크립트, `fig/` 장치, 게시 런 — **소실**. 재실행으로 대체 예정.
3. 하네스는 `runs/`에 **직접** 쓴다. 나중에 복사하지 않는다.
4. 새 지표를 보고하려면 **먼저 `recompute_from_run.py`로 재계산되는지 확인**한다. 안 되면 보존 대상이
   부족한 것이다.

# 원시 런 영속화 기제 검증 (v30) — 재실행 직전 관문

작성 2026-07-28. **본 재실행 아님.** 커밋함, 푸시 없음.

## 판정: **통과 — 본 재실행 착수 가능**

커밋된 산출물만으로 `persist`·`search`·`drain`·`hotswaps`·`maxV`·`lastV`가 전부 재계산되고,
내부 일관성(`search + drain = persist`)과 매니페스트의 독립 기록(`hotswaps`)이 일치한다.

---

## 작업 136 — `runs/` 상태 점검

- 런 디렉터리 **70개**, 총 **1.6 MB**. v26~v29의 탐색·조건 확인 런이 전부 남아 있다.
- **추적**: 실제 파일 275 = `git ls-files runs/` 275, **미추적 0**. `.gitignore`에 `runs/` 없음.
- 분포: top1 35 / recovery 16 / allcpu 13 / rank:1–5 각 1 / **적응 런 1**(작업 138에서 추가).

### 요구(v24 작업 101) 대비 보유

| 요구 | 탐색 런(v26–29) | 적응 런(v30 신규) |
|---|---|---|
| `V(t)` 시계열 CSV | ✓ `metrics.csv` | ✓ |
| 전환 로그(후보 진입·commit·revert) | — (정적이라 전환 없음) | ✓ `executor.log` + 매니페스트 `transitions` |
| 뷰별 집계 JSON | ✗ | ✓ `aggregate.json` |
| 스케줄 스냅샷 | ✓ | ✓ |
| 랭킹 참조(경로+sha256) | ✓ | ✓ |
| 환경(플랫폼·커밋·날짜·α·β·λ·buffer) | 부분 — **repo commit·α/β 없음** | ✓ 전부 |

→ **부족분 2건**(뷰별 집계, repo commit·α/β)을 작업 138에서 채웠다.

---

## 작업 137 — 재계산 시연 (핵심 수용 기준)

`scripts/recompute_from_run.py` — 정의는 `docs/metric_definitions.md` 정본을 그대로 쓴다.

### 1차: 기존 탐색 런 — **부분 통과**
`runs/20260728T065145Z_lam80_recovery`(Q3 회복 런):
```
  persist_s : 5.0    t0 15:51:57 → t_r 15:52:02
  maxV 2.1197 / lastV 0.2705 / violated True / recovered True
  --- agreement ---  violated MATCH · recovered MATCH · maxV MATCH · lastV MATCH
```
그러나 **`search`·`hotswaps`는 계산 불가** — 이 런들은 전부 **정적 프로브**(`--adaptive-mode 3`,
단일 고정 배치)라 후보 순회·전환이 애초에 일어나지 않았다. → 판정 매트릭스의 *"계산 불가 → 작업 138"*.

### 2차: 적응 런 (작업 138 후) — **통과**
`runs/20260728T075259Z_q3_boundguard_r0` (BoundGuard, mode 1, 배경 LLM 기동, 후보 5개 순회):

```
run      : runs/20260728T075259Z_q3_boundguard_r0
eps      : 1.0   (from run_manifest.json)
--- recomputed from committed artefacts ---
  n_samples           : 99
  maxV                : 50.1006
  lastV               : 0.0
  violated            : True
  t0                  : 2026-07-28 16:53:21
  t_r                 : 2026-07-28 16:53:48
  recovered           : True
  persist_s           : 27.0
  combination_changes : [('16:53:20','combination_burst'), ('16:53:28','cand_1'),
                         ('16:53:29','cand_2'), ('16:53:32','cand_3'),
                         ('16:53:36','cand_4'), ('16:53:45','cand_5')]
  commit_combo        : cand_5
  search_s            : 24.0
  drain_s             : 3.0
  persist_check_search_plus_drain: 27.0
  hotswaps            : 5
```

**대조 — 매니페스트의 독립 기록**:
`transitions.hotswaps_per_view = 5`, `applied_sequence = [stable, burst, cand_1..cand_5]`,
`ranking_sha256 = 418aa13a…`, `alpha_beta = 0.3 / 1.0`, `repo_commit = 0e999eefad`.

**일치 확인**
- `hotswaps` 재계산 5 = 매니페스트 5 ✓
- `search + drain = 24 + 3 = 27 = persist` ✓ (내부 일관)
- `commit_combo = cand_5` = 랭킹의 **회복 배치(rank 5)** ✓ (게시 런의 라벨과 동일)

### 시연이 잡아낸 오류 (기록)
처음 작성한 `search`는 **첫 후보**(cand_1) 적용 시각을 썼다 — **7 s**. 정본 정의는
*"감지 원점 → **회복 후보가** 적용된 시각"*이므로 cand_5 기준 **24 s**가 맞다. 정정 후
`search+drain=persist` 항등식이 성립했다. **시연이 없었으면 그대로 갔을 오류다** — 이것이 "파일이
있는 것"과 "값을 되짚을 수 있는 것"이 다르다는 증거다.

---

## 작업 138 — 부족분 보완

`scripts/run_scenario.py` 신규 — 시나리오 1회를 **추적 경로에 직접** 실행한다(복사 단계 없음).

보완 항목:
1. **`aggregate.json`** — 실행기가 `results/`에 쓰는 조합별 집계를 런 디렉터리로 가져와 **자족적**으로 만든다(뷰별 liveness 증거).
2. **`repo_commit`, `alpha_beta`, `lambda_header`, `active_background_header`** — 매니페스트에 추가.
   α/β·λ·배경은 스케줄 헤더에서 추출하므로 스케줄과 어긋날 수 없다.
3. **`transitions`** — `applied_sequence`, `hotswaps_per_view`, `reverts`를 매니페스트에 기록.

보완 후 짧은 적응 런 1회로 재시연 → **통과**(위 §137 2차).

---

## 작업 139 — 재실행 용량 예측

적응 런 1회 = **58 KB**(70 s, 후보 5개 순회; log 39 KB + metrics 11 KB + aggregate 2 KB + 스케줄 4 KB + 매니페스트 2 KB).

| 시나리오 | 변형 | ×5회 |
|---|---|---|
| Q1.3 GPU / NPU | 4 / 4 | 20 / 20 |
| §4b | 5 | 25 |
| Q3·Q6(공유) | 5 | 25 |
| Q4 | 5 | 25 |
| Q5 | 4 | 20 |
| **합계** | **27** | **135 런** |

**예측 총량 ≈ 7.7 MB** → **직접 커밋**(수백 MB 이하). 압축·git LFS 불요. 정지 조건 해당 없음.

> 실제 런은 이보다 길 수 있어(고정 100 s 창) 로그가 커질 수 있으나, 2–3배여도 20 MB대다.

---

## 작업 140 — `runs/README.md`

커밋 완료. 내용: **세 번의 증발 이력(무엇이 언제 왜)**, 디렉터리 구조와 매니페스트 스키마,
보존/제외 기준과 **작업 137 시연 결과 인용**, `confirmed_values.json`의 `source`가 런 디렉터리를
가리킨다는 규약, 오염·폐기 표시 방법, 용량, 규칙 4개(원시 산출물 커밋 / 저장소 밖 자산은 의도와 사고를
구분 / 하네스가 직접 쓴다 / 새 지표는 재계산 확인 후 보고).

---

## 작업 141 — 커밋

`scripts/recompute_from_run.py`, `scripts/run_scenario.py`, `runs/README.md`, 적응 런 산출물, 본 리포트.
**푸시하지 않음.** `git ls-files runs/ | wc -l` 결과는 커밋 절에 기재.

---

## 남은 것
**본 재실행.** 입력은 v29의 다섯 시나리오 확정표이고, 하네스는 `scripts/run_scenario.py`,
수용 확인은 `scripts/recompute_from_run.py`다.

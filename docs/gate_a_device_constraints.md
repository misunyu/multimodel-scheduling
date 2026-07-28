# 게이트 A — `qwen2_vl` CPU 금지 제약 규명

작성 2026-07-28. 성격: 코드·git·로그 조사(측정 아님). 결과: **분기 2 확정.**

---

## 판정 요약

v23 리포트의 "2-gen 회복 = llama1b만 이전 가능, rank 6, qwen2_vl CPU 금지"는 **틀렸다.** 그 제약은
**랭킹 생성기에만** 있고 **런타임·실험 어디에도 없다.** 논문 §Q4의 "both to CPU가 feasible(V=0)이고
rank twelve"는 **옳다.** → **분기 2**: 생성기가 실험 공간을 잘못 서술했다. 생성기를 런타임과
일치시키고(제약 미적용) 2-gen 랭킹을 재산출했다.

---

## A-1. 확인 결과

### A-1-1. 제약 도입 시점 (git)
`DEVICE_CONSTRAINTS = {"qwen2_vl": ["gpu","npu"]}`는 커밋 **f02331b (2026-07-20) MLA100 swap**에서
도입됐다. 근거: `git log -S '"qwen2_vl"'`. 즉 예측기 교체와 함께 들어온 **사후 주석**이며, 논문 수치를
낸 실험 방법론(생성모델 배경 배치 측정)보다 **뒤**다.

### A-1-2. 강제 위치 — **생성기에만, 런타임엔 없음**
`allowed_devices`/`DEVICE_CONSTRAINTS` 호출부는 `rankings/generate_rankings.py`(v22/v23)와 v22 스크래치
하네스뿐이다. `unified_viewer`·`adaptive_deploy`·`schedule_executor_main`·`best_deploy_finder_executor`
·`runtime/*` — **런타임 배치 경로 어디에도 없다**(grep 공집합). 즉 런타임은 스케줄이 지시하면 qwen2_vl을
CPU에 올린다. **제약은 생성기 전용 주석이다.**

### A-1-3. 논문의 "both to CPU, V=0" 측정 출처
`docs/qwen_background_report.md`가 실측을 기록한다. 워킹셋 `[llama, qwen, yolo, res, mob]`(생성 2 +
비전 3), 순위:

| rank | 배치 | V(t) | 판정 |
|---|---|---|---|
| 1 | both accel | 51→59 | 위반 |
| 3/5 | 생성 하나만 CPU (qwen_cpu / llama_cpu) | 23→26 / 0→40 | **여전히 위반** |
| **12** | **both CPU (ccaaa)** | **0→0** | **feasible** |

- both_cpu는 **정책 후보로 랭킹에 등장**(rank 12)했지 수동 강제가 아니다.
- CPU의 VLM이 느려도(≈13 s prefill) V(t)=0인 이유: **qwen2_vl은 background**이고 V(t)는 foreground
  vision만 감시한다. 생성모델을 가속기에서 빼면 foreground 경쟁이 해소돼 회복한다. 제약의 근거("VLM이
  CPU에서 느림")는 **feasibility를 부정하지 못한다** — 감시 대상이 아니기 때문.

### A-1-4. v23 "회복 배치 순위"의 의미
`generate_rankings.py`는 **순위만** 매기고 V(t)를 **측정하지 않는다.** 따라서 v23의 "회복 배치 rank 6"은
**실측 회복이 아니라 제약 하 랭킹 위치**였다. 실측 회복(both_cpu V=0)은 위 A-1-3에서 별도로 측정됐다.
용어 혼동(분기 3 요소)도 함께 있었으나, 근본 원인은 생성기의 제약 오적용(분기 2)이다.

---

## 분기 결정 — **분기 2** (생성기를 런타임과 일치)

분기 1(제약 실재→논문 both-CPU 철회→2-gen이 C1로 붕괴)은 **기각**한다: both-CPU는 실측 feasible(V=0)한
**실재 배치**이므로 철회 대상이 아니다. 제약은 hard feasibility가 아니라 soft 선호이며, 이를 hard로
취급해 후보 공간에서 배제한 것이 오류였다.

**조치**: `generate_rankings.py`의 `enumerate_placements`가 모든 `{cpu, accel}` 배치를 열거하도록 수정
(제약 미적용, 런타임과 동일). `DEVICE_CONSTRAINTS` 상수는 런타임 미사용이므로 그대로 두되 생성기에서
적용하지 않는다.

### 재산출 확인 (정본 β=1.0)
- 논문 워킹셋(비전 3 + 생성 2): both-CPU = **rank 12/32** — **논문 "rank twelve" 정확 재현.**
- v23 시나리오 워킹셋(§IV-A foreground 4 + 생성 2): both-CPU 최상(비전 전부 가속기) = **rank 16/64**.
- 어느 쪽이든 both-CPU는 N_cand=5 **밖**이고, 생성 하나만 이전한 배치는 여전히 위반 — **논문 §Q4 2-gen의
  논지(둘 다 CPU라야 feasible, 그 배치가 예산 밖) 구조적으로 성립.**

> **미결(작업 85)**: 2-gen 워킹셋의 비전 개수가 논문 실측(3)과 §IV-A foreground(4)에서 다르다. rank 12↔16
> 차이의 원인. 최종 2-gen 시나리오 구성은 스케줄 생성(작업 85, 사람 확인)에서 확정한다. 게이트 결론(both-CPU
> 재현·2-gen 논지 성립)은 어느 구성에서도 유지된다.

---

## v23 산출물 정정

- `rankings/ranking_Q3_mispred_2gen_cpu-gpu.json`: 제약 없이 재생성(64 후보). both-CPU가 이제 후보로
  등장(최상 rank 16).
- `docs/beta_canonicalization_report.md` 98-1 표의 "2-gen llama1b만 rank 6" 행: **오류, 본 문서로 정정**
  (both-CPU 표현 가능, rank 16(fg4)/12(논문 fg3)). 해당 리포트에 정정 주석 추가.

---

## 후속 (이 게이트 밖)

- Q3/Q6 재실행(§2)은 2-gen과 독립이며 분기 확정으로 착수 가능 상태다. 단 **미스프리딕션 워크로드가 옛
  어휘 스케줄(phantom, 현재 fail-fast로 중단)을 쓰므로, 재실행에는 정본 랭킹 기반 신어휘 스케줄이
  필요하다(작업 85, 사람 확인 대기).** 이 의존성은 별도 보고.

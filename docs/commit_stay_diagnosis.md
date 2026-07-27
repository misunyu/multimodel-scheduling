# "commit ≠ stay" 진단 — §4b NPU 7초 잔차

날짜: 2026-07-23 · **진단 전용, 코드 변경 없음** · 게이트 후 수정 판단

## §1. 의도 확인 (논문/설계) — commit-and-stay
- main.tex는 이 저장소에 없음. 코드 주석·설계 리포트로 의도 확인:
  - `schedule_executor_main.py:72` 주석: *"validate : ... if > epsilon advance, **otherwise stay** (up to
    combo_durations cap)"* → 검증 통과 시 **stay** 의도.
  - `docs/phaseA_boundguard_report.md:51-52` (원저자 명시):
    - *"스케줄이 커밋 후에도 combo-duration으로 다음 후보까지 순회(모든 후보가 양호해 위반 지표엔 무영향)."*
    - *""commit-and-stay" 정밀 재현은 Phase B에서 `--stop-after`/validate 커밋 시 잔여 페이즈 스킵으로 정리."*
  - `docs/q3_q5_misprediction_report.md:38`, `candidate_ranking_check.md:6`: BoundGuard는 후보를 순회하다
    **통과하는 배치에서 회복(=그 배치로 커밋)**하는 것으로 서술.
- **판정: 의도 = commit-and-stay**(검증 통과 시 그 배치에 머물고 잔여 후보 스킵). 원저자가 구현이 이에
  미치지 못함을 이미 **Phase B 후속으로 명시**.

## §2. 구현 실제 동작 — visit-all (스킵 없음)
- `check_validate` commit 경로: `self._qos_advance_fired=True; self._stop_qos_poll()` — **조기 advance만 막고
  duration timer는 취소 안 함.** 반면 advance 경로(`_qos_advance`)는 duration timer를 취소하고 즉시 다음 콤보.
- 따라서 commit 시: 그 후보에 **duration cap까지 머문 뒤**, cap이 만료되면 `_after_stop`→다음 콤보로 진행.
  스케줄에 cand_1..cand_tail이 모두 나열돼 있으므로 **커밋과 무관하게 전 후보를 순차 방문**.
- **판정: 구현 = visit-all.** commit은 "조기 advance 방지 + 해당 후보 duration 유지"일 뿐, **잔여 후보 스킵 안 함.**
- **불일치: 예** (의도 commit-and-stay ↔ 구현 visit-all). 단 원저자가 문서화한 알려진 미완성.

## §3. NPU 7초 잔차 정량 (§4b, validate 수정본, 실행순서 P1 확인: stable→burst→cand_1..cand_tail)
콤보별 V(t)>ε 지속(초) — 보고용 V(t)(누적) 기준:

| 콤보 | GPU V>ε | NPU V>ε | 비고 |
|---|---|---|---|
| combination_burst | 8 | 8 | 위반 진입(동일) |
| cand_1 (**커밋**: V_postswap 0.34/0.41) | 12 | 11 | burst backlog 배수 |
| cand_2 | 4 | 5 | 배수 잔여 |
| **cand_3** | **0** | **2** | 커밋 후 방문(worse 배치) |
| **cand_4** | **0** | **6** | 커밋 후 방문(worse 배치) |
| cand_tail | 0 | 0 | |
| **합(persistence)** | **24s** | **32s** | |

- **cand_1이 양 플랫폼 모두 커밋**(V_postswap≤ε)했으나 스케줄은 **cand_2/3/4/tail을 계속 순회.**
- **NPU 잔차 8s(≈7s) = cand_3(2)+cand_4(6)**: 커밋 후 방문한 **worse 배치**(모델 일부 CPU 이동)가 NPU에선
  더 느려 보고용 V(t)를 다시 ε 위로 밀어 올림.
- **GPU/NPU 차이의 원인**: **같은 순회가 양쪽 다 일어남**(GPU도 cand_2/3/4/tail 방문). 그러나 GPU에선 그
  방문들이 빠르게 배수돼 V>ε=0; NPU에선 CPU-오프로드 배치가 느려 V>ε 발생. → **순회 자체가 아니라, 순회가
  NPU에서 위반을 만든다는 것**이 차이.
- **정량 판정: 7초 잔차는 commit 후 순회(cand_3+cand_4)로 설명됨.** commit-and-stay면 cand_1 커밋 후 cand_2~4를
  스킵 → NPU도 ~24s로 수렴 예상(burst 8 + cand_1 11 + cand_2 배수 5 ≈ 24).

## §4. 게이트 표
| 확인 | 결과 |
|---|---|
| 논문/설계상 의도 | **commit-and-stay** (주석 L72 + phaseA 리포트 L51-52; 원저자 Phase B 후속으로 명시) |
| 구현 실제 동작 | **visit-all** (commit=조기advance 방지+duration 유지, 잔여 후보 스킵 안 함) |
| 불일치 여부 | **예** (문서화된 알려진 미완성) |
| 7초 잔차가 commit 후 순회로 설명되는가 | **예** — NPU cand_3(2)+cand_4(6)=8s, GPU는 그 방문들이 V>ε=0 |
| GPU/NPU 차이 원인 | 순회는 양쪽 동일; NPU에선 CPU-오프로드 후보가 느려 위반 유발, GPU는 빠르게 배수 |

## §5. 수정 방안 (구현 대기 — 게이트)
- **방안**: validate commit 시 `_qos_advance` 대신 **잔여 후보를 스킵하고 committed 배치에 머무는** 경로 추가
  (예: commit 시 다음 후보들의 combo-duration을 건너뛰고 긴 tail로 진입, 또는 `_index`를 tail로 점프).
- **정당성(중요)**: 이 수정은 **결과가 좋아져서가 아니라 논문 §III 서술(commit-and-stay)과 일치시키기 위함.**
  원저자가 이미 미완성으로 명시한 항목의 완성.
- **영향 범위(명시)**: commit-and-stay면 **§4b·B2·Q3/Q4 전부에서 BoundGuard가 유리해짐**(커밋 후 worse 후보
  방문 스파이크 제거). 특히:
  - §4b NPU: 32s→~24s(Adaptive 대등) 예상.
  - Q3/Q4: BoundGuard가 회복 후보 커밋 시 즉시 안정(순회 스파이크 제거) → 회복 더 뚜렷.
  - B2: 최종 배치 동일이라 경미할 수 있음.
- **주의**: 이 변경이 BoundGuard를 전반적으로 유리하게 하므로, 적용 시 **모든 4기법 실험을 재실행**하고
  Adaptive와의 대비를 정직하게 재보고해야 함.

## 무결
- 코드 변경 없음(진단·측정만). GPU 폴백 0(§4b 재실행 fallback=0). legacy/backup/main.tex 불변.

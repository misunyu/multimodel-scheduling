# 논문 산출물 ↔ 저장소 출처 매핑

**목적**: 게이트 A~G가 전부 출처 재발굴이었다. 이 문서는 그 재발굴을 **다시 하지 않게** 한다.
특히 **폐기본 인용 사고**(패치 v1이 정정 전 리포트를 정본으로 인용해 표 수치가 논문과 어긋난 일)를
막는 것이 최우선이다 — 못 찾는 것보다 **틀린 것을 찾는 것**이 위험하다.

**작성 규칙**: 한 칸이라도 확인되지 않으면 그 칸만 `미상`으로 적고 **행은 유지**한다. 행을 빼면 다음
사람에게 "볼 것이 없다"로 읽힌다. **이름 유사성으로 잇지 않는다** — 데이터 고리가 확인돼야 한다.

마지막 갱신 2026-07-28. 근거: `gate_a`~`gate_g` 리포트.

---

## 1. 그림 15개 (`main.tex`가 `\includegraphics` 하는 것)

| 논문 위치 | 산출물 | 생성 스크립트 | 입력 데이터 | 실행 설정 근거 | 리포트 | 상태 |
|---|---|---|---|---|---|---|
| §II 동기, Fig.~1 | `qos_score_validation.pdf` | `scripts/qos_recovery_validation.py` (07-24 **replot**, ε=1.0) | `backup/results_pre_mla100_20260720_185647/results/qos_recovery_*.csv` (2026-04-09/10) — **개별 런 미상** | `tests/stale_phantom/qos_recovery_schedule.yaml` — 전경 4종 mnasnet·squeezenet1.0-12·resnet50·resnext50(Table~I 밖 3종), 배경 없음, `slo_ms` 미설정 → deadline = 도착 간격 1개(≈333 ms). 네 뷰 전부 실측(phantom 아님) | `q1_workload_attribution.md` §2 | 정본 |
| §III 구조, Fig.~2 | `architecture.pdf` | 없음(수작업 도해로 추정) | — | — | — | **파일이 저장소에 없음** |
| §IV-B Q1, Fig.~3 | `bounded_recovery_analysis.pdf` | `fig/plot_bounded_v8.py` — **부재**(증발) | `fig/confirmed_values.json` 8개 값 = 3시나리오×(search+drain+bound) − Q4 drain | Q3(CPU–GPU) 22.6+2.8·bound 26 / Q5(CPU–NPU) 14.2+0.0·bound 26 / Q4(CPU–GPU) 24(**종료 지연**, n=1)·bound 24 | `gate_f_fig3_attribution.md` | 정본 |
| §IV Q1 오버헤드 | `runtime_overhead.pdf` | `scripts/runtime_overhead_analysis.py` | 미상 | `tests/stale_phantom/steady_state_views_schedule.yaml`(스크립트 기본값) | `runtime_overhead_analysis.md`(scripts/) | 정본 |
| §IV 검출 민감도 | `detection_sensitivity.pdf` | `scripts/detection_sensitivity_analysis.py` | `results/bounded_sweep/run_s*_rep*.csv` — **저장소에 없음** | 미상 | `detection_sensitivity_analysis.md`(scripts/) | 정본 |
| §IV Q1.3 지속 | `q13_failure_persistence.pdf` | `fig/plot_q13_v8.py` — **부재** | `confirmed_values.json` (`q13_*_persist`, `source=b3rep gpu/npu`) | **미보존** — `b3rep` 원시 런이 저장소에 없음 | 미상(논문 Q번호 대응 미확인) | 정본(값) |
| §IV Q1.3 누적 | `q13_cumulative_violation.pdf` | `fig/plot_q13cum_v8.py` — **부재** | `confirmed_values.json` (`q13_*_cumV`) | **미보존**(동일) | 미상 | 정본(값) |
| §IV 유체 검증 | `c3_fluid_validation.pdf` | 없음(저장소 내 미발견) | 미상 | λ=0.70/0.90/0.95×μ\* 스윕 | `c3_stress_report.md` | 정본 |
| §IV 버퍼 스윕 | `b2_buffer_sweep.pdf` | 없음(저장소 내 미발견) | 미상 | 4기법×2플랫폼, λ=0.9μ\*(GPU 123/NPU 93 fps), per-frame deadline 8.13 ms, β_B∈{0.5,1,1.5} | `b2_buffer_sweep_report.md` | 정본 |
| §IV Q2 부하 증가 | `dynamic_load_adaptation.pdf` | `scripts/dynamic_load_validation.py`, `_background_validation.py` | 미상 | `tests/stale_phantom/dynamic_load_views_schedule.yaml`(스크립트 기본값) | 미상 | 정본 |
| §IV Q3 오예측 | `q3_misprediction.pdf` | `fig/plot_q3_v8.py` — **부재** | `confirmed_values.json` (`q3_bg_*`, `source=b2rep_q3`) | 07-25 임시 스케줄(`BoundGuard/Adaptive/Static/Stop-restart.yaml`). 전경 3종+llama1b, λ=80/25, 균일 15 ms, `FRAME_BUFFER=12` | `gate_b_schedule_provenance.md`, `gate_d_config_attribution.md` | 정본 |
| §IV Q4 후보 없음 | `q4_bounded_envelope.pdf` | 없음(저장소 내 미발견) | `q4_bsf_out` — **저장소에 없음** | heavy-4(yolo11x/l/m+resnet50), λ=90, 5×Table~I(70/52/44/12 ms), buffer 300 | `q4_experiment_report.md` | 정본 |
| §IV Q5 NPU | `q5_npu_generalization.pdf` | `fig/plot_q5_v8.py` — **부재** | `confirmed_values.json` (`q5_bg_*`, `source=b2rep_q5`) | 전경 3종+llama1b(NPU), 회복=cand_2. **최종 λ 미상** | `q5_experiment_report.md`, `gate_d_config_attribution.md` | 정본 |
| §IV Q6 ablation | `c2_reactive_comparison.pdf` | 없음(저장소 내 미발견) | 07-25 코너 실행 | `nodwell/reinvoke/hybrid.yaml` + BoundGuard. 2×2, 고정 100 s 창, 5회 | **`c2_integrity_report_v7.md`**(정본) | 정본 |
| §IV 예산×순위 | `d2_budget_rank.pdf` | 없음(저장소 내 미발견) | **analytic** — 상한식 격자 평가(실행 아님) | δ=1 s 단일 인스턴스화, 마커 2개만 hardware | `d2_simulation_report.md`, `figure_fix_report_v14.md` | 정본 |

---

## 2. 표 5개

| 논문 위치 | 표 | 수치 출처 문서 | 상태 |
|---|---|---|---|
| §I/§III | `tab:dependability_mapping` | 미상(개념 표로 추정, 실측 수치 없음) | 미매핑 |
| §IV-A | `tab:gpu_models` (Table~I 모델·격리 지연) | 미상 — 측정 런이 저장소에 없음 | 미매핑 |
| §IV-A | `tab:scenario-config` (신규) | `gate_d_config_attribution.md`, `scenario_configuration_table.md` | 정본 |
| §IV | `tab:b2` | `b2_buffer_sweep_report.md` | 정본 |
| §IV Q6 | `tab:c2-ablation` | **`c2_integrity_report_v2.md` §7 → `_v7.md`** "Q6 완성표" | 정본 |

---

## 3. 폐기본 목록 — **인용 전 반드시 확인**

| 폐기 문서 | 폐기 시점 | 대체 문서 | 폐기 사유 |
|---|---|---|---|
| `c2_reactive_baseline_report.md` | 2026-07-28 배너 | `c2_integrity_report_v2.md` → `_v7.md` | 가변 런길이 집계. 고정 100 s 창 재집계로 **A §4b 1/5→4/5, A Q3 3/5→1/5, A lastV 9.9→19.9±9.9, BG Q4 lastV 65.3→54.8** 변경. **패치 v1이 이 문서를 인용해 표가 논문과 어긋났다** |
| `c2_integrity_report.md` (v1) | 2026-07-28 배너 | `c2_integrity_report_v7.md` | 정정 전 라운드. v2 §7이 **A §4b 1/5→4/5, C Q3 1/5→0/5** 정정 |

### 3.1 부분 폐기 (수치는 그대로이나 **결론이 무효**)

| 문서 | 무효화한 문서 | 내용 |
|---|---|---|
| `q3_experiment_report.md` (07-22, vision-4 + per-model SLO + λ=52) | `hotswap_buffer_bias_fix.md` §6c | *"시나리오 자체가 무효"* — 네 기법 모두 위반 진입조차 안 함. **이 문서의 Q3 수치를 논문 Q3로 인용하면 안 된다.** 논문 Q3는 vision-3 + 균일 15 ms + λ=80(§6b "결론 유지") |

> `c2_integrity_report_v2`~`v6`는 폐기가 아니다. v7이 *"v2~v6 판정 유지"*라 명시하며 **v7이 확정표**다.
> `figure_fix_report{,_v10,_v14,_v15}`·`figure_pipeline_report{,_v16,_v17}`는 누적 라운드이며 상호 모순은
> 확인되지 않았다(각각 다른 작업 번호를 다룬다).

---

## 4. 미참조 산출물 (논문이 쓰지 않는 것)

| 파일 | 무엇인가 | 왜 미참조인가 |
|---|---|---|
| `q3_misprediction_2gen.pdf` | Q3 2-gen(생성 모델 2개) V(t) | 본문이 2-gen을 **문장으로만** 서술(rank 12) |
| `q5_npu_generalization_2gen.pdf` | Q5 2-gen NPU 버전 | 동일 |
| `q3_paperset_q4like_RETIRED.pdf` | vision-4 + per-model SLO Q3 시도 | **시나리오 무효 판정**(§3.1). 파일명에 RETIRED 표기 |
| `b2_metrics.pdf`, `c3_b2_metrics.pdf` | B2 지표(miss rate·drop·p99) | `tab:b2`가 표로 대체 |
| `d2_dwell_response.pdf`, `d2_load_tightness.pdf` | d2 시뮬레이션 부가 평면 | 본문이 `d2_budget_rank`만 인용 |

**임시 산출물**: `_t.pdf` 류는 발견되지 않았다.

---

## 5. 증발한 인프라 (이번 매핑에서 드러난 것)

게이트 B가 스케줄 생성기를, 게이트 F가 확정표 파이프라인을 같은 유형으로 찾았고, 이번에 하나 더 나왔다.

- **`fig/plot_*_v8.py` 재생성 스크립트 5종 전부 부재.** `figure_regeneration_report.md:74`가 생성했다고
  기록하나 `fig/`에는 `check_figures.py`·`confirmed_values.json`·`paper_figure_manifest.json`·`README.md`뿐.
  → **`confirmed_values.json`에서 그림을 다시 그릴 수단이 없다.**
- **`architecture.pdf`가 저장소에 없다**(참조 15개 중 유일한 결번).
- 원시 런 `b2rep_q3`·`b2rep_q5`·`b3rep`·`q4_bsf_out`, `results/bounded_sweep/` — 전부 부재.
- 그림 **8개**는 생성 스크립트가 저장소에 없다(위 표의 "없음"·"부재").

---

## 6. 앞으로의 규칙

1. **모든 신규 리포트는 상단에 헤더를 단다** — 대응 논문 위치, 실행 스케줄 파일과 해시, 작성일.
2. **정정할 때는 새 파일을 만들고 옛 파일에 SUPERSEDED 배너를 단다.** 덮어쓰지 않는다(이력 소멸).
   남기되 배너가 없으면 폐기본 인용 사고가 반복된다.
3. **그림을 재생성하면 생성 스크립트·입력·시각을 기록**하고 이 표의 해당 행을 갱신한다.
   **생성 스크립트를 반드시 커밋한다** — §5가 그러지 않은 결과다.
4. **이 표에 행이 없는 산출물은 논문에 인용하지 않는다.**

---

## 7. 검증 결과

- **`\includegraphics` 15개**: `fig/paper_figure_manifest.json`(main.tex에서 추출, sha256 기록)의 15개와
  §1의 15행이 **정확히 1:1**(차집합 양방향 공집합). ✓
- **표 5개**: §2에 5행 전부 등재. ✓
- **`미상`/`부재`/`미매핑` 칸 수 ≈ 21** (생성 스크립트 8, 입력 데이터 6, 실행 설정 3, 리포트 3,
  표 상태 2 — 개산). **0이 아니다.** 이 숫자가 이 문서의 정직성 지표다.
- **논문이 인용하는 리포트 중 폐기 상태: 0건.** §1·§2의 "리포트" 열은 전부 정본을 가리킨다
  (`tab:c2-ablation` → `c2_integrity_report_v7`). 폐기본 두 건은 §3에만 등장한다.
- **본문과 어긋나는 새 발견: 없음.** 기존 미상(30.0 s, 검출 1.7–2.0 s, $T_{stable}$ 10 s)은
  게이트 F·G에서 이미 보고됐고 이 라운드에서 논문을 수정하지 않았다.

---

## 8. 미매핑 (근거 미확인 — 추정 기입 금지)

| 논문 위치 | 산출물 | 막힌 고리 |
|---|---|---|
| §Q1.5 "single hot-swap" | `b2_buffer_sweep.pdf` | 값 1의 집계 출처 미상 — 리포트에 hot-swap 표가 없다(`transition_unit.md` §3) |
| §Q1.1/Q1.3/Q1.4 ↔ 리포트 | `runtime_overhead`, `c3_fluid_validation`, `detection_sensitivity` | 논문 Q번호 ↔ 저장소 리포트 이름(b2/c3/d2…) 매핑 부재 |
| §Q5 최종 4기법 | `q5_npu_generalization.pdf` | 최종 λ 미상(`q5_experiment_report.md`가 "NPU 전용 λ 튜닝 필요 → 게이트") |
| Q4 2-gen | (본문 서술) | deadline 규약 미상(`gate_d_config_attribution.md`) |
| §Q1 바운드의 $T_{stable}$ | — | 값 10 s의 **원 측정 미상**. 실측 0–4 s(`repetition_report.md`), 유체 스윕 ~90 s — 어느 쪽도 10이 아니다(`gate_g_tstable.md`) |
| §Q1 관측 30.0 s / 검출 1.7–2.0 s | `bounded_recovery_analysis.pdf` | 저장소에서 발견되지 않음(`gate_f_fig3_attribution.md` §1.3) |
| §IV 수치 ↔ 그림 3자 대조 | — | `confirmed_values.json`의 `source`가 가리키는 원시 런이 저장소에 없다 |
| Table~I 격리 지연 | `tab:gpu_models` | 측정 런 미상 |

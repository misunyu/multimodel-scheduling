# 논문 위치 ↔ 저장소 산출물 매핑

**목적**: 그림·수치가 어느 실행에서 나왔는지 되짚는 고리를 **확정된 것부터 쌓는다.** 전체 매핑을 한 번에
만들지 않는다. 근거가 없는 항목은 **"미매핑"으로 명시**하고 추정으로 채우지 않는다(phantom 폐기와 같은 규율).

**작성 규칙**: 한 행을 추가할 때는 `그림 → 생성 스크립트 → 데이터 → 실행 설정 근거`의 네 고리를 모두
확인한 뒤 넣는다. 하나라도 끊기면 그 칸에 "미상"이라 적는다.

마지막 갱신 2026-07-28.

---

## 확정된 매핑

| 논문 위치 | 그림 | 생성 스크립트 | 데이터 | 실행 설정 근거 |
|---|---|---|---|---|
| §II 동기 (Failure Impact Objective) | `qos_score_validation.pdf` | `scripts/qos_recovery_validation.py` (헤더에 명시) | `backup/results_pre_mla100_20260720_185647/results/qos_recovery_*.csv` (2026-04-09/10) — **개별 런 특정은 미상** | `tests/stale_phantom/qos_recovery_schedule.yaml` (3-phase initial/overload/offload). 전경 4종 **mnasnet, squeezenet1.0-12, resnet50, resnext50** (Table~I 밖 3종), 배경 없음, `slo_ms` 미설정 → deadline = `1000/infps` ≈ 333 ms. 네 뷰 전부 실측 산출(phantom 아님). 07-24 **replot**(ε=1.0 정정, 데이터 불변): `PENDING_DECISIONS.md:170`, `figure_regeneration_report.md:24` |
| §Q1 회복 지연 분해 | `bounded_recovery_analysis.pdf` | (값은 `fig/confirmed_values.json` 단일 소스 경유) | **집계 그림** — Q3(`b2rep_q3`), Q5(`b2rep_q5`), Q4(`q4_bsf_out`) 세 시나리오 | Q3·Q5: 전경 3종(yolo11s, resnet50, mobilenet_v2) + 배경 llama1b, 균일 15 ms. Q4: 전경 4종(yolo11x, yolo11l, yolo11m, resnet50), 배경 없음, 5×Table~I. 근거: `gate_b_schedule_provenance.md`, `gate_d_config_attribution.md`, `q4_experiment_report.md` |
| §Q1 Fig.~3 값 귀속 | `bounded_recovery_analysis.pdf` | `fig/plot_bounded_v8.py` (재생성본; 옛 `scripts/bounded_recovery_validation.py`의 `SCENARIOS`는 **placeholder**라 무관) | `fig/confirmed_values.json` 8개 값 = 3시나리오 ×(search+drain+bound) − Q4의 drain | Q3(CPU–GPU) 22.6+2.8, bound 26 · Q5(CPU–NPU) 14.2+0.0, bound 26 · Q4(CPU–GPU) 24(종료 지연, n=1), bound 24. 근거: `gate_f_fig3_attribution.md` |
| §Q3 오예측 | `q3_misprediction.pdf` | — (confirmed_values 경유) | 07-25 임시 스케줄 실행 (`BoundGuard/Adaptive/Static/Stop-restart.yaml`) | 전경 3종 + llama1b, λ=80(burst)/25(stable), 균일 15 ms, `FRAME_BUFFER=12`. 근거: `gate_b_schedule_provenance.md`, `gate_d_config_attribution.md` |
| §Q6 ablation | `c2_reactive_comparison.pdf` | — | 07-25 코너 실행 (`nodwell/reinvoke/hybrid.yaml`) | 2×2 ablation, 고정 100 s 창, 5회. **정본 수치 = `c2_integrity_report_v2.md` §7 → `_v7.md`** (`c2_reactive_baseline_report.md`는 **SUPERSEDED**). 근거: `gate_e_ablation_numbers.md` |
| §Q4 가용 후보 없음 | `q4_bounded_envelope.pdf` | — | `q4_bsf_out` | heavy-4 (yolo11x/l/m + resnet50), 5×Table~I(70/52/44/12 ms), buffer 300. 근거: `q4_experiment_report.md` |

---

## 미매핑 (근거 미확인 — 추정 기입 금지)

| 논문 위치 | 그림 | 막힌 고리 |
|---|---|---|
| §Q1.5 "single hot-swap" | `b2_buffer_sweep.pdf` | 값 1의 **집계 출처 미상** — `b2_buffer_sweep_report.md`에 hot-swap 표가 없다 (`transition_unit.md` §3) |
| §Q1.x 나머지 (Q1.1/Q1.3/Q1.4) | `runtime_overhead.pdf`, `c3_fluid_validation.pdf`, `detection_sensitivity.pdf` 등 | 논문 Q번호 ↔ 저장소 리포트 이름(b2/c3/d2…) 매핑 부재 |
| §Q5 최종 4기법 실험 | `q5_npu_generalization.pdf` | 최종 λ **미상** (`q5_experiment_report.md`가 "NPU 전용 λ 튜닝 필요 → 게이트"로 남김) |
| Q4 2-gen | `q3_misprediction_2gen.pdf`, `q5_npu_generalization_2gen.pdf` | deadline 규약 **미상** (`gate_d_config_attribution.md`) |
| §IV 수치 ↔ 그림 3자 대조 | — | `fig/confirmed_values.json`의 `source` 필드가 가리키는 원시 런(`b2rep_q3`, `b3rep`, `c2_final2` 등)이 **저장소에 없다**(스크래치 증발, `gate_b` §B-3) |

---

## 주의 — 같은 실험의 정정 전/후 리포트가 공존한다

파일명만으로 최신본을 구분할 수 없다. 실제로 `c2_reactive_baseline_report.md`(정정 전)를 인용해
패치 문안이 틀렸던 사례가 있다(`gate_e_ablation_numbers.md`). 인용 전 다음을 확인할 것:

- 문서 상단에 **SUPERSEDED 배너**가 있는가.
- 같은 주제의 `*_v2`, `*_v7` 등 **더 높은 버전**이 있는가 (버전 접미사가 있으면 최고 버전이 정본).
- 그 문서가 **재집계·정정을 자백**하는 절을 담고 있는가(예: "고정 윈도우가 바꾼 두 판정").

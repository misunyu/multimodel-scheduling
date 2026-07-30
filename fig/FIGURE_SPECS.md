# 그림 명세 (재생성용)

**목적**: 재생성 스크립트(`plot_*_v8.py`, `sidecar_util.py`, `build_pipeline_doc.py`)가 **소실 확정**이다.
다시 쓸 때 백지에서 시작하지 않도록, 흩어진 도안 결정과 함정을 한 곳에 모은다.
**리포트가 또 사라져도 이 문서는 남는다 — 그것이 목적이다.**

출처: `fig/README.md`(규약), `figure_fix_report{,_v10,_v14,_v15}.md`(도안 결정),
`gate_f_fig3_attribution.md`(Fig.3 구조), `fig/paper_figure_manifest.json`(그림↔절↔캡션),
`docs/report_q_mapping.md`(그림↔런).

---

## 공통 규약 (`fig/README.md`)

1. **값은 `confirmed_values.json`에서만.** 그리며 다시 계산하지 않는다.
2. **사이드카는 캔버스에서 뽑는다** — `bars[i].get_height()`처럼 실제 아티스트에서. 확정표에서 복사하면
   검사가 자기 자신을 검사하게 되어 아무것도 잡지 못한다(이 실수로 `emit_sidecars.py`를 삭제했다).
3. **`censored`와 `recovered`는 독립**이다.
   - `censored` = 값이 관측 창에 잘림 → **해칭 + 위쪽 화살표 + `≥window`**
   - `recovered` = 그 런이 회복함 → **색/테두리**
   - 반례: Q4 `search=24 s`는 `censored=false, recovered=false`. 해칭을 쓰면 "실제론 더 길 수도"로 읽혀
     `24 = bound 24`의 tight 주장이 흐려진다.
4. **`source_kind`**: `hardware` / `simulation` / `analytic`. 한 그림에 섞이면 마커를 달리하고 범례에 명시.
5. **절단된 값과 측정된 값을 같은 축에 놓지 않는다.**
6. **그림 안 텍스트도 검사 대상** — 제목·축·틱·범례·주석의 숫자는 확정표 값이거나 사유 붙은 allowlist.
7. **`generator` 필드 필수**(v28): 사이드카가 자신을 그린 스크립트의 경로+sha256을 담고,
   `check_figures.py`가 존재·실재·**git 추적**·해시 일치를 검사한다. 실패 시 `exit 1`.

---

## 그림별 명세

### `bounded_recovery_analysis.pdf` — §Q1 회복 지연 분해
| 항목 | 내용 |
|---|---|
| 무엇 | 시나리오별 회복 지연을 **search**(감지원점→후보 커밋)와 **drain**(커밋→$V\le\epsilon$)으로 분해, 바운드와 대비 |
| 축·계열 | x=시나리오(Q3/Q5/Q4 이름 그대로), y=초. 막대는 search+drain **누적**, 바운드는 선/마커 |
| 쓰는 id | `q3_bg_search`(22.6) `q3_bg_drain`(2.8) `q3_bound`(26) / `q5_bg_search`(14.2) `q5_bg_drain`(0.0) `q5_bound`(26) / `q4_search`(24) `q4_bound`(24) — **8개 = 3시나리오×3 − Q4의 drain** |
| 도안 결정 | 기준점을 **detection onset**으로 통일(`figure_fix_report`). Q3는 t0 기준 **22.6**(burst 기준 24.6은 그림 사용 금지) |
| **함정 1** | **Q4는 다른 양이다** — `recovered=false`, "첫 위반→best-so-far 복귀" = **종료 지연**이지 회복 지연이 아니다. 그림이 회복 지연과 종료 지연을 한 축에 섞으므로 **캡션이 밝혀야 한다** |
| **함정 2** | Q4는 `n=1`, 나머지는 `n=5` — 반복 수도 섞인다 |
| **함정 3** | ~~Q3는 `t0` 기준, Q5는 `burst` 기준~~ → **v33에서 전 시나리오 `t0`로 통일**했다. Q5의 값은 원래도 `t0` 기준이었고 `reference` 라벨만 `burst`로 잘못 붙어 있었다. 기준점이 섞이던 원인이 이것이다. Q5가 burst보다 9.6 s 앞서 위반을 시작하는 구간은 `q5_lead`로 따로 기록한다 |
| **함정 5** (v33) | 봉투를 **두 형태로** 그린다 — 런에서 측정한 것과 게재 형태($T$ 대입). 게재 형태는 $k$가 작은 15/25 런에서 깨진다. 성립하는 쪽만 그리면 더 유리하고 덜 정직한 선택이 된다 |
| **함정 4** | Q5 drain=0.0 → 막대 두께 0. 사라지지 않게 표기 |
| 플랫폼 | Q3·Q4 = CPU–GPU, **Q5 = CPU–NPU**(한 그림에 두 플랫폼) |

### `q13_failure_persistence.pdf` / `q13_cumulative_violation.pdf` — §Q1.3
| 항목 | 내용 |
|---|---|
| 무엇 | (좌) 회복 여부, (우) 회복한 기법의 지속시간 / 별도 그림은 누적 위반 $\int V\,dt$ |
| 도안 결정 | **2패널**로 분리 — Static은 회복하지 않아 지속시간이 없다. 절단값을 시간축 막대로 두지 않기 위함 |
| 쓰는 id | `q13_{gpu,npu}_{sr,adaptive,bg}_persist`, `q13_static_lastV`, `q13_*_cumV`, `q13_*_cumV_ratio` |
| 시각 부호 | Static = **censored**(해칭+화살표). 막대 높이를 값으로 쓰지 말 것 |
| 함정 | 창 길이가 계열마다 다르다(58.4–68.6 s). cumV 비율(6.2×/6.3×)은 **격차의 하한** |

### `q3_misprediction.pdf` / `q5_npu_generalization.pdf`
| 항목 | 내용 |
|---|---|
| 무엇 | 기법별 $V(t)$ 궤적 |
| 축 | **x = wall-clock**(샘플 인덱스 아님 — 전환이 압축돼 회복이 ~20으로 보이던 문제) |
| 주석 | Q3: `persist 25.4 = 22.6 + 2.8`. Q5: `converge = T_valid 14.2`(15.2 폐기) |
| 시각 부호 | baseline(미회복) = **open square** |

### `c2_reactive_comparison.pdf` — §Q6
| 항목 | 내용 |
|---|---|
| 무엇 | Q3 조건에서 4변형의 $V(t)$ |
| 정본 수치 | **`c2_integrity_report_v7.md` "Q6 완성표"** — `c2_reactive_baseline_report.md`는 **SUPERSEDED** |
| 함정 | 종료 마커가 **고정 100 s 창**에 정렬돼야 한다(가변 런길이가 회복 판정을 바꿨던 이력) |
| 변형 이름·색 | `A: no-dwell` / `B: re-invoke` / `C: hybrid` / `Adaptive hot-swap` / `BoundGuard` / `Static` / `Stop-restart` (`variant_style`) |

### `q4_bounded_envelope.pdf` · `d2_budget_rank.pdf`
| 항목 | 내용 |
|---|---|
| q4 | $V(t)$ envelope. sample-gap 0 |
| d2 | **`source_kind: analytic`** — 상한식을 격자에서 평가한 평면. 마커 2개만 `hardware`. 평면과 마커의 마커 모양을 달리하고 범례 명시 |

### `b2_buffer_sweep.pdf` — Q1.5 / `tab:b2` (v34 재생성)
| 항목 | 내용 |
|---|---|
| 무엇 | 요청 수준 결과를 $\beta_B$ 축으로. **§IV에서 유일하게 프레임 단위 지표를 보이는 그림** |
| 생성기 | `fig/plot_b2_v11.py` |
| 축·계열 | 좌: p99(log, s) × $\beta_B$, 4기법 × 2플랫폼 / 우: Static의 drop(막대) + miss(2차축) |
| 쓰는 id | `q15_{gpu,npu}_{static,stoprestart,adaptive,boundguard}_b{0.5,1.0,1.5}_p99` (24) + `q15_*_static_b*_{drop,miss}` (12) |
| 반복 | $\beta_B$=0.5는 **n=5**(`tab:b2` 행), 1.0·1.5는 **n=3**. 비대칭이므로 캡션에 명시 |
| **함정 1** | **Static은 전부 검열**이다(회복 없음) → 빈 마커. 측정된 정착값으로 읽히면 안 된다 |
| **함정 2** | miss의 **분모가 기법마다 다르다**(Stop-restart는 재시작 중 프레임을 받지 않아 drop=0). served/late/drop을 표에 함께 실어야 한다 |
| **함정 3** | deadline이 **두 개**다 — 고정한 15 ms(주)와 게재 폴백 8.13 ms(부). p99는 deadline 무관이므로 대비가 그대로 성립한다 |
| **함정 4** | p99는 $\beta_B$에 **거의 평탄**하다. 버퍼가 배치 문제를 고치지 않는다는 것이 요점이며, 축을 좁게 잡으면 없는 추세가 보인다 |

### `runtime_overhead.pdf` — Q1.1 (v34 재생성)
| 항목 | 내용 |
|---|---|
| 무엇 | 정상 상태($V\approx0$, 전이 0회)에서 감시 계층의 비용 |
| 생성기 | `fig/plot_overhead_v11.py` |
| 도안 | **Static 대비 델타** 3패널(CPU / 추론 ms / 처리량). 절대 막대는 산포 안의 차이를 눈대중하게 만든다 |
| **함정 1** | 정상 상태에서 **Adaptive와 BoundGuard는 같은 코드 경로**다 — BoundGuard가 더하는 것은 전부 전이 시점 동작이고 전이가 없다. 두 막대는 두 시스템이 아니라 "검증 무장에 추가 비용 없음"의 증거다. **캡션이 밝혀야 한다** |
| **함정 2** | 원 스크립트의 지표 ①"총 처리율"은 `metrics.csv`의 `total_fps`를 썼는데 그 열은 $\sum 1000/\text{infer\_ms}$ = **용량**이다. ②"뷰당 추론 ms"와 같은 양을 두 번 잰 것이다. v34는 `perframe.csv`로 실측한다 |
| **함정 3** | 원 스크립트는 BoundGuard를 `mode 2`(Reactive)로 본다. v33 정본 매핑은 `mode 1 + validate`이며 mode 2는 별개의 reactive 베이스라인이다 |

### `detection_sensitivity.pdf` — Q1.2 (v34 재생성)
| 항목 | 내용 |
|---|---|
| 무엇 | $(T, \epsilon)$ 3×3 격자 × 4지표 히트맵 |
| 생성기 | `fig/plot_detection_v11.py`, 값은 `scripts/detection_sensitivity_v34.py` |
| 입력 | **하드웨어 실행 아님** — 커밋된 v33 런의 `v_score`를 오프라인 재윈도잉. 게재 입력 `results/bounded_sweep/`는 저장소에 없다 |
| 도안 | 운용점 $(T{=}3, \epsilon{=}1.0)$을 전 패널에서 **박스 표시**. 그림의 목적이 그 선택의 정당화이므로 찾을 수 있어야 한다 |
| **함정** | 회복 지연은 정의되지 않는 셀이 있다(그 $(T,\epsilon)$에서 아무 후보도 회복하지 않음). **0이 아니라 해칭**으로 그린다 — 0은 "즉시 회복"으로 읽힌다 |

### 나머지 (`dynamic_load_adaptation`, `c3_fluid_validation`, `qos_score_validation`, `architecture`)
- 값 변경 없이 **유지** 판정된 그림들(`figure_regeneration_report.md`).
- `qos_score_validation`은 §II 동기 그림이며 **Table~I 밖 모델 4종**을 쓴다(재현 불가, `q1_workload_attribution.md`).
- `architecture.pdf`는 **저장소에 없다**(사용자 보관).

---

## 재생성 시 순서

1. `confirmed_values.json`의 해당 id 확인 → 2. `plot_<fig>_vN.py` 작성(**커밋 필수**) →
3. 사이드카를 **캔버스에서** 추출하고 `generator{path,sha256}` 기록 →
4. `python3 fig/check_figures.py --figdir fig` 통과(0 mismatch) → 5. `docs/report_q_mapping.md` 행 갱신.

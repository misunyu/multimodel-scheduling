# A1 / C2 / E1 — 데이터 인벤토리 (Task 0)

목적: A1/C2/E1 원고 수정의 근거 데이터를 기존 로그에서 확인하기 위한 원자료 위치·스키마 정리.
**새 실험 없음. 기존 로그 재분석/조회만.** 확인 불가 항목은 "missing".

## 0. 표 번호 매핑 (지시문 번호와 논문 번호가 **일치**함)

논문 `paper/main_vision.tex`의 표 순서(=번호): `tab:cotenants`(1) → `tab:single-stream`(2)
→ `tab:decomp`(3) → `tab:main`(4) → `tab:metric-sensitivity`(5) → `tab:reporting`(6)
→ `tab:detector-generality`(7). 지시문의 Table 2/3/4 번호와 일치한다.

> 정정: 본 인벤토리 초판은 `tab:cotenants`(Table 1)를 누락해 번호를 하나씩 당겨 셌음(계수 오류).
> 실제로는 논문 업데이트 전·후 모두 지시문 번호와 일치. 표 순서는 이번 업데이트에서 변경되지 않음
> (본문 문구·캡션만 수정).

| 지시문 명칭 | 논문 실제 라벨 (번호) | 내용 |
|---|---|---|
| — | `tab:cotenants` (**Table 1**, L154) | GPU co-tenant workloads (L1/L2/L3 정의) |
| "Table 2" (single-stream isolated, 24 logs, per-size GPU/NPU) | `tab:single-stream` (**Table 2**, L203) | GPU(FP32)/NPU(INT8) per-size sAP + NPU−GPU diff. **양자화 Q_b(−0.008/−48.4% …)는 이 표의 NPU−GPU 열** |
| "Table 3" (staleness 4 vs 24 thread) | `tab:decomp` (**Table 3**, L250) | 양자화 row + staleness row (staleness: small≈0, medium −0.030/−16.3%, large −0.098/−20.5%) |
| "Table 4" (Oracle N=4) | `tab:main` (**Table 4**, L390) | All-GPU/All-NPU/Oracle worst·mean, N=4 |
| Fig 2/3 (ResNet50 k-sweep) | `fig:persize-sweep`(L278), `fig:sweeps`(L298–364) | per-size loss, worst-stream vs DM |

## 1. 단일 스트림 isolated 평가 (Table 2 = `tab:single-stream`)

- **원자료**: `accv_experiments/results/rev19_table1_threads4.csv`
  - 스키마: `rep,device,n_sids,sap_5095,sap_50,sap_s,sap_m,sap_l,map_5095,map_s,map_m,map_l,infer_ms,skip_pct`
  - device ∈ {GPU, NPU}, 3 reps, `n_sids=24` (24 Argoverse-HD 로그 집계값; **per-log 분해는 이 파일에 없음**).
  - per-size sAP는 3 reps 전부 동일(결정론적): GPU s/m/l = 0.0159/0.1839/0.4768, NPU = 0.0082/0.1477/0.4773.
- **per-log(24로그) 분해**: `accv_experiments/results/rev7_single_stream.csv` (sid×device, 단 rev7 = 이전 버전/구성; threads4 정규 per-log는 `rev19_stdout.log` 내부에만).
- **집계/논문표 재생성본**: `paper/results_data/table1_single_stream/single_stream.csv` (size,gpu,npu,diff,rel_pct — 4행 집계).
- skip≈0 (both devices), infer GPU≈8.3ms / NPU≈10.0ms.

## 2. Staleness probe (Table 3 = `tab:decomp` staleness row)

- **원자료**: `accv_experiments/results/rev18_postproc_levers.csv`
  - 스키마: `threads,rep,npu_lat_ms,npu_skip_pct,gpu_sap_l,npu_sap_l,large_gap,npu_sap_s,npu_sap_m,gpu_sap_s,gpu_sap_m`
  - threads ∈ {4, 24}, 3 reps. NPU 경로 동일, 4-thread(skip≈0) vs 24-thread(skip≈55%).
  - staleness L_b = (24-thread NPU per-size) − (4-thread NPU per-size).
- **집계본**: `paper/results_data/table2_decomp/decomp.csv` (size,quant_rel_pct,staleness_dAP).
- 비트동일성 근거(예측 불변 → 지연만 변함): `paper/results_data/table2_decomp/bitident_summary.json` (20 frames, box/score diff 0.0), `accv_experiments/results/rev21_bitident_check.csv`.

## 3. ResNet50 k-sweep (Fig persize-sweep / sweeps — 지시문 Fig 2/3)

- **정규(canonical) 런**: `accv_experiments/results/rev30_clean_resnet/`
  - `rev30_canonical.csv` (run×group×placement×k×rep, per-size sAP, worst/mean/median/p10, util_mean/p50/p95, mem, power, deadline_margin; 대용량 원값)
  - `rev30_persize.csv`: k, gpu_skip, sap_{s,m,l}, loss_{s,m,l} — **Fig persize-sweep 원자료**
  - `rev30_fig3a_points.csv`: N∈{2,4,8}×k, allgpu_worst(±std), allnpu_worst — **Fig sweeps(a) 원자료** (논문 L318/L350 좌표와 일치)
  - `rev30_oracle_by_contention.csv`: contention_level×k, allgpu/allnpu/oracle worst(±std), oracle_ratio, split{1,2,3}_worst, **pick_dist**, reps — **Fig split / Oracle 원자료**
  - `rev30_tabmain_L1CNN.tex`: 논문 `tab:main`의 **L1_CNN 행**(0.098/0.137/0.083/0.126/0.103/0.137, 24/1) 자동생성본
- **반복/중간지점 per-size(3 reps)**: `accv_experiments/results/rev25_persize_under_contention.csv`
  - 스키마: `point,strategy,rep,gpu_skip,npu_skip,sAP_small,sAP_medium,sAP_large` (point ∈ skip0/skip~42/skip~58/skip~100).
- **재생성본**: `paper/results_data/fig2_sweep/sweep_byN_points.csv`(rev24 기반, **구버전** — 논문 fig 좌표와 상이), `paper/results_data/table4_persize/persize_under_contention.csv`.
- deadline-miss rate: `rev30_oracle_by_contention.csv` gpu_skip_allgpu (k1=24.3 … k8=67.3).

## 4. L3VLM Oracle 후보 평가 로그 (E1, Table 4 = `tab:main`)

- **원자료 (per-run, 5 candidate placements)**: `accv_experiments/results/rev20_5strat_heavybg.csv`
  - 스키마: `rep,bg,N,strategy,placement,worst_sap,mean_sap,gpu_skip,npu_skip`
  - bg ∈ {L1_light, L2_lm, L3_vlm}, N ∈ {2,4,8}, strategy ∈ {All-GPU, Isolated, Cont-aware, All-NPU, Oracle}, 3 reps.
  - placement = per-stream device 문자열 (G/N). L3_vlm N=4 Oracle 선택: rep0 NNNN, rep1 **NGNN**, rep2 NNNN.
  - **주의**: strategy 열은 5종 라벨(All-GPU/Isolated/Cont-aware/All-NPU/Oracle)이며, Oracle 행의 placement가 실제 선택된 후보. Isolated=GGNN, Cont-aware=NNGG가 두 개의 중간(mixed) 후보에 해당.
- **집계본**: `accv_experiments/results/rev21_mean_worst_extract.csv` (bg×N×strategy, worst/mean/mean_minus_worst), `paper/results_data/table3_main/main_worst_mean.csv`.
- **Oracle 후보 순서/선택 규칙**: `phase_rev30_clean.py`(MOVE_ORDER, placement_for_ratio), `phase_rev30_aggregate.py`(max mean-worst 선택), 요약 `accv_experiments/results/FINDINGS_oracle_largefrac.md`.
- **per-stream(스트림별) sAP**: rev20 CSV는 worst/mean만 보유. 스트림별 개별 sAP는 `rev30_canonical.csv`의 `per_stream_json` 열(ResNet 경로) 및 `rev20_stdout.log`(2.2MB, VLM 경로)에 존재. QoS 경로 per-stream은 `exp_qos_results.csv`(stream_id별 행).

## 5. 보조 원자료 (A1 감사용)

| 항목 | 파일 | 비고 |
|---|---|---|
| L1_CNN tab:main 정규값(0.098/0.103, DM24, 9/10 split) | `rev30_clean_resnet/rev30_oracle_by_contention.csv` (RES1), `rev30_tabmain_L1CNN.tex` | pick_dist `{"1":9}` = one-stream split 9/10 |
| metric sensitivity (1.6×) | `accv_experiments/results/rev28_metric_sensitivity.csv` | mean 0.107 / worst 0.066 / p10 0.072 / median 0.0866 |
| QoS (2.5×) | `accv_experiments/results/exp_qos_results.csv` | prio, vlm_sat: 보호 All-GPU worst≈0.033 |
| CPU util 17→93% (LLM) | `accv_experiments/results/cpuload_means.json` | cpu_sys_mean baseline 16.6 → +LM 92.6 |
| GPU util ~68% flat | `rev30_clean_resnet/rev30_canonical.csv` (util_mean), 폐기된 `tab:gpu-util`(주석) | k≥1: 66–70% |

## 6. Missing / 확인 불가

- **per-log(24 로그) threads4 per-size 원표**: 별도 CSV 없음. 집계값(rev19_table1_threads4.csv, n_sids=24)과 rev7(구버전 per-log)만 존재 → Q_b의 **로그별** std는 원자료 부재(단, **run별** std는 rev18/rev19 3 reps로 산출 가능).
- **sid 21(count 최대)을 옮긴 one-stream split의 sAP**: 측정 셀 부재(FINDINGS_oracle_largefrac.md T3 참조) — 0.103의 "top-1 fraction 엄밀성"은 UNCLEAR.
- **재생성 패키지 불일치**: `paper/results_data/table3_main/main_worst_mean.csv`의 L1_CNN 행(0.092/0.131, skip35, Oracle 0.109)은 **rev20 기반(구버전)**으로, 논문 `tab:main`의 L1_CNN 행(0.098/0.137, skip24, Oracle 0.103, **rev30 기반**)과 불일치. → A1 감사 항목.

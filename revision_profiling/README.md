# revision_profiling

CPU–GPU 재profiling용 runtime과 Phase 1(사양 확정·pilot) 산출물. 사양서 사본은 `SPEC_v2.2.md`, 코드 변경 기록은 `CHANGES.md`, 결과 보고는 `REPORT_PHASE1.md`.

**Phase 2(전체 campaign)는 실행하지 않았다.** 사용자 승인 뒤 별도 지시문으로 진행한다.

## 폴더

| 경로 | 내용 |
|---|---|
| `code/` | 새 runtime과 분석 script. 원본 모듈은 import하지 않는다 |
| `offline_profile/` | 모델 × device의 isolated capacity(`offline_profile.json`, run별 JSON) |
| `pilot/smoke/` | §5.1 단일 instance 1×, β=0.5 |
| `pilot/beta_{0.25,0.5,1.0}/` | §5.3 판정표. 0.25·1.0은 1× 단일 GPU run도 여기 있다 |
| `pilot/arrival/` | §5.2 run 72개와 판정표, 개발 중 feeder 시도(`attempt*_dev/`) |
| `pilot/repeat/` | §5.4 반복 편차 |
| `pilot/lm/` | §3 LM 결과(`LM_RESULT.md`) |
| `pilot/drop_metric_5_5_*.csv` | §4·§5.5 세 drop 정의 비교(보고용) |
| `npu_compat/` | §6 CPU–NPU 저장 데이터 호환성 |
| `family_*.csv` | §3.4 family 재구성과 Phase 2 run 수 |
| `schedules/` | Phase 2용(비어 있음) |
| `logs/` | 실행 log, 모델 sha256, 분류 sanity |
| `vendor/` | `tokenizers`(prompt 생성·출력 decode용) |

## 실행

```bash
cd revision_profiling
export MPLCONFIGDIR=/tmp/…          # 임시 폴더
PY="../.venv/bin/python3 -B"
$PY code/offline_profile.py                                   # isolated capacity
$PY code/gen_schedules.py --family g_m_r_t --beta 0.5 --mult 2 --masks all_gpu,all_cpu,half --out x.yaml
$PY code/executor.py --schedule x.yaml --combination combination_015 --out out.json
$PY code/run_campaign.py --out-dir <dir> --repeats 2 <yaml …>   # 재개 가능
bash code/run_phase1_pilot.sh                                 # pilot 1–4단계 전체
```

한 run은 load + warm-up 5 s + 계측 10 s + 정리로 약 16–25 s다.

## 결과 JSON (`schema: revision_profiling/v2.2`)

- `run_key`(schedule | combination | repeat, 고정), `execution_id`(실행마다).
- `instances[].window`: `generated, enqueued, dropped_full, completed, failed, consumer_failed, scheduled_ticks, missed_ticks, realization, pending_start/end, conservation_gen, conservation_flow, latency/service p50·p95, preprocess_ms_mean, q_i`.
- `system`: `throughput_fps = completed/window`, `drop_rate_fps`, `drop_macro`, `drop_max`, `service_rate_sum_legacy`.
- `gpu_samples`: `nvidia-smi` 1 s sampling. `loaded_modules`, `instances[].feeder.modules`: process별 모듈 경로.
- `valid`, `invalid_reasons`: `zero_generated`, `inflight_gt_1`, `module_path`, `conservation`(±1 초과), `consumer_failed`, `total_identity`.
- 같은 이름의 `*.events.npz`: instance별 이벤트 timestamp 원본.

v1 사양서의 §2.4 schema는 저장소에 없어 대조하지 못했다. 위 필드는 v2.2 §2.4–2.5 본문에서 직접 만든 것이다.

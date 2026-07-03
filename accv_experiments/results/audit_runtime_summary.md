# EXP-AUDIT-RUNTIME — Foreground/CPU runtime provenance audit (read-only)

Scope: confirm, from code evidence, the runtime used by each paper measurement for (a) the
foreground GPU detector and (b) the CPU path. No measurement/compile was run. Paper .tex not
modified. Statements below are facts with file:line citations; cells lacking a pinnable producing
script are marked 미상.

## Structural facts (whole codebase)
- The ONLY two foreground GPU detector adapters are:
  - `FGModelGPU` — `_step_d_common.py:70-79`: `self.m = YOLO("yolo11s.pt")`; `self.m.predict(..., device="cuda")`.
  - `FGModelGPUGeneric` — `phase_rev6_sweep.py:121-133`: `self.m = YOLO(pt_name)`; `self.m.predict(..., device="cuda")`.
  Both are Ultralytics **PyTorch**, `device="cuda"`.
- The CPU detector adapters are PyTorch too:
  - `FGModelCPU` — `_step_d_common.py:81-88`: `YOLO("yolo11s.pt")`, `device="cpu"`.
  - `UltralyticsRunner` — `minimal_pipeline/step0_compare_devices.py:58-75`: `YOLO("yolo11s.pt")`, `device=device` ("cpu"/"cuda").
- `onnxruntime.InferenceSession` appears ONLY for background co-tenants — `_step_d_common.py:298` (ResNet50),
  `_step_d_common.py:307` (TinyLLaMA). A grep for `InferenceSession` excluding background terms returns
  **nothing**: there is NO ONNX Runtime session for the foreground detector (GPU or CPU) anywhere.

## Provenance table
| 논문 위치 | 산출 스크립트 (→ CSV) | FG GPU 런타임 | CPU 런타임 | 증거 (코드 라인) | 확정/미상 |
|---|---|---|---|---|---|
| Table 1 / Fig 1 (isolated single-stream, 24 logs, GPU vs NPU) | `phase_rev19_measure.py` → `rev19_table1_threads4.csv` (threads=4) | Ultralytics PyTorch, cuda | n/a (GPU/NPU table; CPU excluded from placement) | `phase_rev19_measure.py:66,123` `FGModelGPUGeneric(...)`; `phase_rev6_sweep.py:128-133` `YOLO(pt_name).predict(device="cuda")`; `:42,72` `THREADS=4` | 확정 (FG GPU runtime); Table-1 GPU figures match |
| Table 2 (thread-lever staleness separation) | `phase_rev18_stage1.py` → `rev18_postproc_levers.csv`; bit-identity `phase_rev22_sweep.py:85-103` → `rev22_bitident.csv` | Ultralytics PyTorch, cuda | n/a | `phase_rev18_stage1.py` imports + uses `FGModelGPUGeneric`; `phase_rev6_sweep.py:128-133` device="cuda" | 확정 |
| Table 3 (All-GPU per-size vs contention, N=4) | `phase_rev25_persize.py` → `rev25_persize_under_contention.csv` | Ultralytics PyTorch, cuda | n/a | `phase_rev25_persize.py` imports `FGModelGPUGeneric` (`phase_rev6_sweep.py:128-133`, device="cuda") | 확정 |
| Table 4 (N=4 × 3 co-tenant L1/L2/L3) | `phase_rev20_heavybg.py` → `rev20_5strat_heavybg.csv` (N=4 extract `rev21_mean_worst_extract.csv`) | Ultralytics PyTorch, cuda | n/a | `phase_rev20_heavybg.py:24` `from phase_rev6_sweep import FGModelGPUGeneric`; uses `measure_multistream`; `phase_rev6_sweep.py:128-133` device="cuda" | 확정 |
| Fig 2/3 (contention sweep N=2/4/8) | `phase_rev22_sweep.py` → `rev22_vlm_sweep.csv` | Ultralytics PyTorch, cuda | n/a | (EXP-QOS 확인 완료) `phase_rev22_sweep.py:27,111` `FGModelGPUGeneric`; `phase_rev6_sweep.py:128-133` | 확정 (사전 확인) |
| Table 5 (metric sensitivity, k8) | `phase_rev22_sweep.py` → `rev22_vlm_sweep.csv` / `rev22_perstream_sap.csv` | Ultralytics PyTorch, cuda | n/a | (EXP-NORM 확인 완료) 동일 rev22 경로 | 확정 (사전 확인) |
| Table 6 (cross-detector v8n/v8s/12s) | `phase_rev26_yolov8n.py`, `phase_rev27_generality.py`, `phase_rev27c_step0.py` | Ultralytics PyTorch, cuda (detector .pt만 교체) | n/a | `phase_rev26_yolov8n.py:150` / `phase_rev27_generality.py:130` `FGModelGPUGeneric(DET["ultralytics_pt"])`; `phase_rev27c_step0.py:29` `FGModelGPUGeneric("yolo12s.pt")`; `phase_rev6_sweep.py:128-133` device="cuda" | 확정 |
| Sec 5 CPU 수치 (31.98 ms, "ORT CPU backend") | `step_a_baseline_sap.py` → `step_a_baseline.csv` / `step_d_device_ap_size_latency.csv` (행 `CPU,31.98,...`) | n/a | Ultralytics **PyTorch**, `device="cpu"` (NOT ONNX Runtime) | `step0_compare_devices.py:66` `YOLO("yolo11s.pt")`, `:70,75` `predict(device="cpu")`; `_step_d_common.py:81-88` `FGModelCPU` device="cpu"; FG용 ORT 세션 부재 | 확정 (백엔드); 산출 스크립트 `step_a_baseline_sap.py` 확정 |

## 종합 판정 (Sec 5 수정 문장이 전 측정에 대해 참인가?)
- "Foreground GPU inference runs FP32 PyTorch (Ultralytics)" — **전 측정에 대해 참** (Table 1–6, Fig 1–3
  모두 `FGModelGPU`/`FGModelGPUGeneric`, `device="cuda"`; 코드베이스에 FG용 ORT 세션 없음).
- "CPU inference uses ONNX Runtime's CPU backend" — **거짓**. 코드 증거상 CPU 검출기 경로는 Ultralytics
  **PyTorch** `device="cpu"`(`step0_compare_devices.py:66-75`, `_step_d_common.py:81-88`)이며, foreground
  검출기에 대한 onnxruntime CPU InferenceSession은 존재하지 않는다. ONNX Runtime은 background co-tenant
  (ResNet50/TinyLLaMA)에만 쓰인다.

## 사실 기술 (거짓 셀의 실제 런타임)
- Sec 5 CPU 수치(31.98 ms)의 실제 CPU 런타임은 **Ultralytics PyTorch (device="cpu")** 이다. 산출 스크립트는
  `step_a_baseline_sap.py`, 디바이스 어댑터는 `step0_compare_devices.py`의 `UltralyticsRunner` 및
  `_step_d_common.py`의 `FGModelCPU` (둘 다 PyTorch CPU). 따라서 "ONNX Runtime CPU backend"는 코드와
  불일치한다.

## 미상/주의
- Table 1의 NPU 측정 출처 세부: GPU 수치는 `rev19_table1_threads4.csv`와 일치 확인. NPU per-size의 정확한
  threads=4 산출 파일이 rev19 단독인지 rev6 baseline 재사용 혼합인지는 본 감사 범위(FG/CPU 런타임)와 무관하며
  별도 확인 대상. (FG GPU 런타임 판정에는 영향 없음.)
- `step_d_device_ap_size_latency.csv`를 파일명으로 직접 기록하는 writer 코드 라인은 grep으로 특정되지 않음
  (집계 산출물). CPU 백엔드 사실 자체는 디바이스 어댑터 코드로 확정.

## 금지 준수
새 측정/컴파일 없음. 논문 .tex 미열람·미수정. LaTeX 수정안 미작성(사실 보고만).

# Table I — Benchmark Models (measured, CPU-GPU eval)

측정(무인 재측정, 워밍업 30 + 300회 중앙값, tight p25-p75). vision = e2e 파이프라인(전처리+추론+후처리), 격리. LLM/VLM = 64-token 생성 e2e(fp16), 격리 프로세스. L_SLO = 5×latency.

| Model | Category | Size (MB) | Latency (ms) | L_SLO (ms) |
|---|---|---:|---:|---:|
| YOLO11n | Detection | 11 | 6.89 | 34.4 |
| YOLO11s | Detection | 37 | 6.19 | 31.0 |
| YOLO11m | Detection | 77 | 8.33 | 41.6 |
| YOLO11l | Detection | 98 | 10.23 | 51.2 |
| YOLO11x | Detection | 218 | 13.16 | 65.8 |
| ResNet50 | Classification | 98 | 2.24 | 11.2 |
| MobileNet-v2 | Classification | 14 | 1.27 | 6.3 |
| LLaMA-1B | LLM | 2357 | 302.00 | 1510.0 |
| Qwen2-VL | VLM | 4213 | 840.00 | 4200.0 |

## 측정 조건
- Size: vision = onnx 파일 크기; LLM/VLM = safetensors 가중치 합.
- Vision latency: model_processors 실제 파이프라인, GPU onnxruntime, 격리, 워밍업 30 + 300회 중앙값. 파이프라인이 CPU-bound라 e2e > 추론시간(Q1.4).
- **YOLO11n ≥ YOLO11s (6.89 ≥ 6.19ms)는 실재**: n의 추론은 더 빠르나(2.41 vs 3.43ms) 정확도가 낮아 저신뢰 박스가 많아 후처리(NMS+draw)가 더 큼. 노이즈 아님(300회 재측정 확인). 테이블 latency는 e2e 기준 일관.
- LLM/VLM latency: 64-token 고정 생성(greedy), GPU fp16, 격리 프로세스. Qwen2-VL은 conv3d cuDNN mismatch로 cuDNN 비활성화(native conv) 하 측정 — cuDNN 정합 시 더 빠를 수 있음(재측정 가치, PENDING).
- L_SLO = 5 × latency (§IV 규칙).

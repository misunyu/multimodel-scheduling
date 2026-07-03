# EXP-QOS v2 — Environment Snapshot

Recorded: 2026-06-12 (session re-run; previous runs cancelled, measurement from scratch)

## Hardware / drivers
| item | observed | expected (§A-4) | drift |
|---|---|---|---|
| GPU | NVIDIA GeForce RTX 5090 (32607 MiB) | RTX 5090 | none |
| Driver | 580.159.03 | 580.159.03 | none |
| CUDA (driver) | 13.0 | 13.0 | none |
| GPU state at snapshot | idle: Xorg 452MiB + gnome-shell 85MiB, **0 compute procs** | — | clean for solo measurement |

## Software
| item | observed | expected (§A-4) | drift |
|---|---|---|---|
| python | 3.10.12 (.venv) | — | — |
| torch | 2.12.0+cu130 | 2.12.0+cu130 | none |
| onnxruntime-gpu | 1.20.1 | 1.20.1 | none |
| ultralytics | 8.4.56 | 8.4.56 | none |
| ORT providers | Tensorrt, CUDA, CPU | — | — |

## QoS mechanism availability
| item | observed | expected (§A-4) | note |
|---|---|---|---|
| stream priority range | greatest=-1, least=0 | greatest=-1 / least=0 | none |
| MPS control binary | /usr/bin/nvidia-cuda-mps-control present | present | — |
| MPS server binary | /usr/bin/nvidia-cuda-mps-server present | present | — |
| MPS daemon running | no | not running | none |

## Model provenance
| file | sha256 |
|---|---|
| yolo11s.pt (FG GPU, Ultralytics PyTorch FP32) | 85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5 |

## Verdict
No drift from expected (§A-4). GPU idle (0 compute processes) → suitable for solo measurement.

# 스케줄 생성 (v27) — 구성 (b) 확정

작성 2026-07-28. **본 재실행 아님.** 산출물은 `schedules/`·`runs/`(추적 경로). 커밋함, 푸시 없음.

## 상태: **Q3 게이트 통과 — 스케줄 확정**

---

## ⚠ 먼저: v26 결론의 근거 정정 (자기 신고)

**v26의 λ 스윕은 배경 LLM을 실제로 기동하지 않은 채 측정됐다.** 실행기는 `--background` 플래그가 있어야
생성 모델 자식 프로세스를 띄우는데(`schedule_executor_main.py:1479`, `BackgroundManager(enabled=bool(background))`),
v26 하네스가 그 플래그를 넘기지 않았다. 로그가 증거다 — v26 런 전부에
`"llama1b is generative/deferred; not started as a vision worker (runs via background manager)"`만 있고
배경 자식 기동 기록이 없다(`grep -c '"--background"'` = 0).

→ v26이 관측한 *"회복 배치 ≈ top-1"*은 **애초에 LLM 경합이 없었으므로 당연한 결과**였다.
생성 모델을 옮기는 것이 무의미한 것이 아니라, **옮길 생성 모델이 돌지 않았다.**

**재측정(배경 LLM 실기동, 구성 (a) vision-4, λ=200)**:

| 배치 | 위반 | 회복 | maxV | lastV |
|---|---|---|---|---|
| top-1 (rank 1) | ✓ | ✗ | 3.365 | 3.235 |
| 회복 (rank 4) | ✓ | ✗ | **5.196** | 4.601 |

**구성 (a)는 여전히 조건 불만족**이고, 회복 배치가 top-1보다 **더 나쁘다**(5.20 vs 3.37).
따라서 **(a)를 배제한 결론 자체는 유지되나, v26 리포트의 근거 서술은 정정되어야 한다.**
또한 **(a)의 λ 스윕 전 구간은 올바른 조건에서 다시 측정되지 않았다** — λ=200 한 점만 재측정했다.

메커니즘(재측정 기준): (a)에서 회복 배치는 LLM을 CPU로 옮기는데, μ\*가 **파이프라인-제한(CPU-bound)**
이므로(`c3_stress_report.md`) 4개 vision 파이프라인의 CPU 단계와 LLM이 경합해 **오히려 악화**된다.
(b)에서는 vision이 3개라 CPU 여유가 남아 그 이전이 이득이 된다.

---

## 작업 119 — 랭킹 확인: **재생성 불요**

| 파일 | 워킹셋 | N | α/β | top-1 | 회복 배치 |
|---|---|---|---|---|---|
| `rankings/ranking_Q3Q6_paper_v3_cpu-gpu.json` | yolo11s, resnet50, mobilenet_v2, **llama1b** | 16 | 0.3 / **1.0** | 전부 GPU(오예측) | **rank 5** |
| `rankings/ranking_Q5_paper_v3_cpu-npu.json` | 동일 | 16 | 0.3 / 1.0 | 전부 NPU | **rank 2** |

확정 구성과 정확히 일치(전경 3 + 생성 1). 회복 순위도 기대값(GPU 5 / NPU 2)과 일치 → 그대로 사용.

---

## 작업 120 — 스케줄 생성

`scripts/generate_schedules.py`가 **랭킹 아티팩트를 읽어** 생성한다. 후보 순서를 손으로 쓰지 않는다.

생성물(`schedules/`):
- `q3_misprediction_cpu-gpu.yaml`
- `q6_ablation_cpu-gpu.yaml` (Q3와 동일 워크로드·동일 랭킹)
- `q5_misprediction_cpu-npu.yaml`

구조: `combination_stable`(전부 가속기, λ=25) → `combination_burst`(랭킹 top-1, λ=80) →
`cand_1..cand_5`(랭킹 rank 1..5, λ=80). 전경 3종은 `view1..view3`, 생성 모델은 `display: none`(headless).

**provenance(헤더 주석)**: 시나리오·플랫폼·워킹셋(전경/배경 구분)·**랭킹 경로와 sha256**·α/β·λ·
활성 background·"Q6 코너는 이 파일을 그대로 쓰고 실행 모드로만 구분".

생성된 후보 순서(= 랭킹 순서):

| | cand_1 | cand_2 | cand_3 | cand_4 | **cand_5** |
|---|---|---|---|---|---|
| GPU (yolo11s,resnet50,mobilenet,llama1b) | `aaaa` | `aaca` | `acaa` | `caaa` | **`aaac` = llama1b→CPU (회복)** |
| NPU | `aaaa` | **`aaac` (회복)** | `acaa` | `caaa` | `aaca` |

→ **GPU 회복 = cand_5, NPU 회복 = cand_2.** 게시 런의 라벨과 정확히 일치한다(게이트 B).

---

## 작업 121 — 조건 확인 (전면 탐색 아님)

배치 고정(`--adaptive-mode 3`), 배경 LLM 기동(`--background`), 문서화된 Q3 설정
(λ=80, `FSRR_FRAME_BUFFER=12`, 균일 `L_SLO`=15 ms, 생성 2 req/s), 20–25 s.

### Q3 (cpu-gpu) — **조건 만족**

| 배치 | rank | 위반 | 회복(strict $t_r$) | maxV | lastV |
|---|---|---|---|---|---|
| top-1 | 1 | **✓** | ✗ | 3.251 | 2.781 |
| **회복(llama1b→CPU)** | 5 | ✓ | **✓** | 2.120 | **0.271** |

→ top-1은 위반 상태로 남고, 회복 배치는 $V\le\epsilon$로 내려가 유지된다. **게시 Q3 거동 재현.**
확정 λ: **stable 25 / burst 80**.

> 참고: 같은 설정에서 `--background` 없이 돌리면 top-1 maxV가 **0.163**으로 위반조차 하지 않는다.
> 배경 LLM이 이 시나리오의 경합 원천임을 보여주는 대조다.

### 미확인 (이번 라운드에서 측정하지 않음)
**Q5(cpu-npu), Q2/§4b, Q4, Q1.3.** Q3가 게이트였고 통과했다. 나머지는 각자의 워킹셋·λ가 달라
(§4b는 vision-3만·λ=45·buffer=2, Q4는 heavy-4·λ=90·buffer=300) **별도 랭킹과 확인이 필요**하다.
추정으로 채우지 않았다.

---

## 작업 122 — 옛↔새 대응표

| 시나리오 | 옛 스케줄(보존) | 새 스케줄 | 워크로드 차이 | λ (옛/새) | 후보 공간 |
|---|---|---|---|---|---|
| **Q3** | `tests/stale_phantom/ml_misprediction_runtime_bg.yaml` | `schedules/q3_misprediction_cpu-gpu.yaml` | 옛: 7개 명명 중 **5개가 phantom으로 미기동**(mnasnet·resnext50·shufflenet·squeezenet·vgg19), 실제 실행은 resnet50+yolov4→yolo11s. **대응 없음**(그 파일은 게시 수치의 출처가 아님) | — / 25·80 | — / 16 |
| **Q3(실측 계보)** | 07-25 임시 `BoundGuard/Adaptive/Static/Stop-restart.yaml`(**저장소에 없음**) | 동일 | 동일 워킹셋(vision3+llama1b) 재현 | 25·80 / 25·80 | 16 / 16 |
| **Q6** | 07-25 `nodwell/reinvoke/hybrid.yaml`(**저장소에 없음**) | `schedules/q6_ablation_cpu-gpu.yaml` | 동일. 코너는 실행 모드로 구분 | 25·80 / 동일 | 16 / 16 |
| **Q5** | 07-25 임시(**저장소에 없음**) | `schedules/q5_misprediction_cpu-npu.yaml` | 동일 구조, NPU | 80 / 80 | 16 / 16 |
| §4b·Q4·Q1.3 | `tests/stale_phantom/*` | **미생성** | 워킹셋·λ가 달라 별도 라운드 | — | — |

**억지로 짝짓지 않았다** — 옛 phantom 파일은 게시 수치의 출처가 아니므로 "대응 없음"으로 표시했다.

---

## 작업 123 — 검증 (5항목 전부 통과)

```
[123-4] 후보 순서 = 랭킹 순서 (기계적 대조)
  q3_misprediction_cpu-gpu.yaml      cand_1..5 == ranking ranks 1..5 : OK
  q5_misprediction_cpu-npu.yaml      cand_1..5 == ranking ranks 1..5 : OK
  q6_ablation_cpu-gpu.yaml           cand_1..5 == ranking ranks 1..5 : OK

[123-1/2/5] 생성 스케줄 실행 (q3, --background)
  ALIAS 표시:      0 건   (정확 이름만 통과)
  UNRESOLVED:      0 건   (fail-fast 통과)
  headless 기동:   1 건   (생성 모델 경로 통과 = v26 회귀 재발 없음)
  NameError/Trace: 0 건
  vision 워커 기동: 3 건   (전경 3종)

  자기점검 로그 발췌:
    combination_stable: yolo11s -> yolo11s
    combination_stable: resnet50 -> resnet50
    combination_stable: mobilenet_v2 -> mobilenet_v2
    combination_stable: llama1b -> llama1b
    ...   (ALIAS 표시 없음)

[123-3] 음성 대조 — 보존된 옛 어휘 스케줄
  exit=1
  ScheduleValidationError: Schedule 'ml_misprediction_runtime_bg.yaml' names models the
  runtime cannot resolve: ['mnasnet','resnext50','shufflenet-v2-12','squeezenet1.0-12','vgg19','yolov4']
```

---

## 작업 124 — 커밋

`schedules/`(3종), `scripts/generate_schedules.py`, `scripts/lambda_search.py`(--background 배선),
조건 확인 런(`runs/`), 본 리포트. **푸시하지 않음.**

## 사람 판단이 필요한 것
1. **v26 리포트 정정 범위** — (a) 배제 결론은 유지되나 근거가 바뀌었다. (a)의 λ 전 구간을 올바른 조건에서
   다시 훑을지 여부(현재 λ=200 한 점만 재측정).
2. **나머지 시나리오**(Q5·§4b·Q4·Q1.3)의 랭킹·스케줄 생성과 조건 확인 — 별도 라운드.

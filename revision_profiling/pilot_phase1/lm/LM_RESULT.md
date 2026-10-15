# LM 시도 결과 (사양 v2.2 §3)

2026-10-13. 요청 정의: 고정 prompt 32 tokens + greedy 16 steps, KV cache 없이 매 step 전체 forward. 요청 1건 = 16 token 생성 완료. K와 Δ는 바꾸지 않았다. 기존 "LM 실행 기록"(이미지 분류로 돈 것)은 쓰지 않았다.

근거 파일: `offline_profile/runs/{gpt2,tiny-llama-chat-onnx}_{gpu,cpu}.json`(closed loop), `pilot/smoke/runs/single_{gpt2,tiny-llama-chat-onnx}_b0.5_x1__*.json`(1×, β=0.5), `pilot/beta_{0.25,1.0}/single_gpu/runs/`.

## 판정 요약

| | GPT-2 | TinyLLaMA |
|---|---|---|
| 실행 가능(양쪽 device, 예외 0) | 예 [확인] | 예 [확인] |
| 출력 token 비어 있지 않음, CPU·GPU 출력 동일 | 예 [확인] | 예 [확인] |
| latency 안정 (p95/p50 < 2) | GPU 1.01, CPU 1.22 | GPU 1.20, CPU 1.10 |
| 계측 구간 조건 `completed ≥ 20` (1×, β=0.5) | **GPU 100, CPU 56 → 통과** | **GPU 17, CPU 12 → 판정 보류("계측 구간 부족")** |
| 파일이 원고 서술과 일치 | 확인 대상 아님 | **불일치**(아래) |
| 권장 | 포함 | 제외 후보. 포함하려면 K·Δ·파일 중 하나를 사용자가 정해야 한다 |

포함/제외는 모델별로 사용자가 결정한다.

## GPT-2

- 파일: `models_onnx/gpt2.onnx`, 653.7 MB, sha256 `c9de47f8…`. 입력 `input_ids`, `attention_mask`(int64, `[batch, sequence]`). Past-KV 입력 없음. 출력 이름은 `last_hidden_state`지만 shape이 `[batch, sequence, 50257]`이라 logits다 [확인: onnx graph].
- Tokenizer: 저장소와 로컬 HF cache에 없었다. `huggingface.co/openai-community/gpt2`에서 `tokenizer.json`을 받아 `code/lm_assets/gpt2/`에 두었다(네트워크 접근 가능했음). `tokenizers` package는 `.venv`에 없어 `revision_profiling/vendor/`에 따로 설치했다. Runtime은 미리 만든 token id만 쓰므로 tokenizer를 import하지 않는다.
- 측정:

| device | isolated capacity (req/s) | 요청당 service p50 (ms) | 1× β=0.5: 요청 rate / generated / completed / dropped |
|---|---|---|---|
| GPU | 20.0 | 47.6 | 10.0 req/s / 100 / 100 / 0 |
| CPU | 5.5 | 178.2 | 10.0 req/s / 100 / 56 / 45 |

- 생성 예: `' existing bus lines.\n\nThe city council also voted to approve a $1'`.
- 구현 비용: `code/lm_runner.py` 약 60줄. 추가 작업 없음.
- 남은 문제: 매 step마다 logits 전체(`seq × 50257` float32, 약 9.6 MB)가 host로 복사된다. 요청 정의의 일부로 두었다. CPU smoke run 1건에서 경계 읽기 conservation이 −1이었다(허용 범위, §5.1 표 참조).

## TinyLLaMA

- 파일: `models_onnx/tiny-llama-chat-onnx/model.onnx`(143 KB) + `model.onnx.data`(909.5 MB). sha256이 폴더의 `hash_record_sha256.json`과 일치한다 [확인].
- **(a) Variant**: llmware가 배포한 **int4 양자화본**이다(`README.md`: "ONNX int4 quantized version", producer `onnxruntime-genai`). Graph의 253 node 중 111개가 `MatMulNBits`다. 원고의 2,488 MB와 맞지 않는 이유가 이것으로 설명된다 [확인: 파일 / 추정: 원고 수치는 다른 정밀도의 export].
- **(b) 원고 서술에 맞는 파일**: 저장소에 없다. 로컬 HF cache에 원본 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`의 `model.safetensors`가 있으나 ONNX가 아니다. 2,488 MB짜리 ONNX를 만들려면 export를 새로 해야 하고(`optimum` 또는 `onnxruntime-genai` builder, 둘 다 `.venv`에 없음), 원고가 어떤 정밀도였는지는 저장소로 알 수 없다 [확인 불가].
- Graph가 `past_key_values.*` 입력 44개를 요구한다. "KV cache 없음" 정의에 맞추려고 매 step 길이 0인 past를 넣었다.
- **GPU 배치가 순수 GPU 실행이 아니다** [확인: ORT profiling]. CUDA EP session에서 forward 1회당 node 24개(주로 `GroupQueryAttention` 22개)가 CPU EP로 간다. 그 결과 TinyLLaMA를 "GPU"에 두어도 process CPU 사용이 계측 10 s 동안 226 core·s다(다른 모델의 GPU 단일 run은 5–7 core·s). 이 instance 하나가 24개 CPU thread를 계속 쓴다.
- 측정:

| device | isolated capacity (req/s) | 요청당 service p50 (ms) | 1× β=0.5: 요청 rate / generated / completed / dropped |
|---|---|---|---|
| GPU | 3.3 | 299.9 | 1.65 req/s / 16 / 17 / 0 |
| CPU | 1.2 | 830.9 | 1.65 req/s / 16 / 12 / 4 |

- 계측 구간 조건: 1×에서 completed가 GPU 17, CPU 12로 20 미만이다. β별 1× 단일 GPU는 β=0.25에서 9건, β=1.0에서 33건이다. CPU는 capacity가 1.2 req/s라 10 s에 최대 12건이므로 **어떤 β에서도 20건에 닿지 못한다.** 사양에 따라 통과 판정을 내리지 않았다.
- 생성 예: `'port, and add a new bus rapid transit line. The council also discussed'`.
- 남은 문제 세 가지: (1) 파일이 원고 서술과 다르다, (2) 10 s 구간의 완료 건수가 통계적으로 부족하다, (3) GPU 배치에서도 CPU를 전부 쓴다 — 이 모델이 들어간 workload는 배치와 무관하게 CPU가 포화된다(pilot의 모든 run에서 process CPU 약 238 core·s / 10 s).
- 포함하려면 필요한 결정: K를 줄이거나(예: 4) Δ를 늘리거나, 다른 export로 교체. 어느 것도 사용자 승인 사항이다.

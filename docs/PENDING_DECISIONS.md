# 사람 판단 대기 목록 (20260721_205708)

돌아오셨을 때 이 목록만 보면 다음 결정이 가능하도록 정리합니다.
각 항목: 무엇을 / 왜 멈췄나 / 관측 데이터 / 선택지.

## [C-1] 용어: "drop rate" vs "deadline miss" (논문 판단)
- 무엇: active 코드에 `drop_rate_fps` 메트릭 필드 다수. drop rate(큐-full 드롭)와 deadline miss는 별개 지표.
- 왜 멈춤: 이름을 바꾸면 필드명·다운스트림 파싱·기존 CSV 호환이 깨질 수 있고, 논문 용어 정합은 사람 판단.
- 데이터: drop_rate_fps는 unified_viewer 메트릭 CSV/results JSON의 필드. B2에서 drop은 miss의 성분으로 이미 집계됨(정합).
- 선택지: (a) 그대로 유지(권장 — drop은 miss의 부분집합, 별개 지표로 유효) / (b) 필드명 리팩터(호환성 위험, 비권장).

## [C-2] 휴면 legacy 데이터 (수정 불필요, 참고용)
- utils.py:175 `_LEGACY_TO_BASE`(resnet50_small/big, yolov3): 신규 경로는 reg.get 사용, 이 매핑은 dormant·무해.
- reactive_deploy.py:60 `_MODEL_GPU_MEM_MB`: 신규 9모델 추가됨 + 구 항목 보존(이미 문서화). 무해.
- scripts/q13·generate_comparison: legacy 재현 경로(신규 coverage 밖, 의도적 미이전).
- 조치: 없음. 제거는 legacy 재현 스크립트 영향 있어 사람 판단.

## [C-3] methods_and_runbook.html 부재
- 지시문 작업 C의 runbook 갱신 대상 파일이 저장소에 없음. 새로 만들지 여부는 사람 판단(SKIP).
## [B] Q5 CPU-NPU 재실험 — 시나리오 정의 필요 (SKIP)
- 무엇: Q5 그림을 단일 MLA100 신규 데이터로 재측정.
- 왜 멈춤: Q5의 정확한 시나리오(워킹셋·λ·페이즈 구조·infeasible 조건)를 모름. 임의 시나리오를 Q5로 라벨링하면 오염.
- 이미 확보된 CPU-NPU 4기법 데이터(재사용 가능):
  - `docs/phaseA_report.md` + phaseA CSV: 4기법 × cpu_npu V(t) 궤적(Static 18.4 / StopRestart 0 / Adaptive 2.5 / BoundGuard).
  - `docs/phaseA_boundguard_report.md`: BoundGuard cpu_npu bounded persistence(mode 1 + validate, 확정).
  - `docs/figures/c3_b2_metrics.pdf`(NPU 패널): 4기법 miss rate·drop/late·p99/p999.
- 선택지: (a) Q5 시나리오 정의를 주면 그 파라미터로 재측정+그림 / (b) 위 기존 데이터로 Q5 그림 구성(시나리오가 phaseA/B2와 부합하면). 사람이 Q5 정의를 확인해야 진행 가능.
## [A] Background LLM/VLM 격리 — 통합 결정 (SKIP, 조사·측정만 완료)
### 자동 완료(참고 데이터)
- mobilint background 코드: `runtime/llm_engine.py`(LLMEngine, transformers), `run_concurrency_study.py`, `model_processors.py`(run_llm/vlm_process). GPU=transformers, NPU=mobilint W8 체크포인트(mobilint/Llama 2.2G, mobilint/Qwen 3.2G 캐시됨).
- **GPU DRAM 예산**: GPU 32GB. llama1b 풋프린트 **2866 MiB**, vision ~1-2GB. 별 프로세스 공유 시 여유(qwen 추정 ~5-6GB 포함해도 32GB 내). → GPU background는 DRAM 제약 아님.
- **NPU DRAM**: smi 툴 부재로 직접 측정 불가(/dev/aries0만). maccel/mblt API로 조회해야 하나 조사 필요.
### SKIP한 판단 (사람 필요)
1. **Qwen2-VL conv3d cuDNN 우회**: 호환 cuDNN 설치 vs cuDNN 비활성화(현재 840ms). 방향 판단.
2. **vision(onnxruntime GPU) + LLM(torch GPU) 공존 = 프로세스 분리 필수**(cuDNN 충돌, 확정). background를 실제 실험 파이프라인에 통합하려면 워커 프로세스 분리가 선행 — 이건 실행모델 변경(되돌리기 어려움)이라 무인 금지. 통합 설계는 사람 확인.
3. **NPU DRAM 초과 시 모델 수 조정**: NPU DRAM 측정 방법(maccel API) 확보 후 vision+LLM mxq 동시 적재 가능 여부 판정 필요.
### 무회귀
- 격리 러너를 실제 파이프라인에 붙이지 않았으므로 vision GPU/NPU/BoundGuard 무영향. background 통합은 SKIP.

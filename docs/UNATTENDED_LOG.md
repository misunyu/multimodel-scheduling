# 무인 실행 로그 (20260721_205708 시작)

## 무결성 베이스라인 (시작 시점)
- vision GPU: speedup **4.8x** (gpu 34.7ms vs cpu 166.5ms), torch-free 중 GPU 정상 (cuDNN 폴백 0)
- NPU (.mxq): yolo11s 4.1ms 정상
- 브랜치: ubuntu_gpu_fsrr
- 기존 성과: C3 fluid 검증(누적 0.3%/drain/발산), B2 4기법, benchmark Table I, GPU torch-free 복구 — 모두 보존 대상

## 작업 로그
| 작업 | 시각 | 결과 | 근거 | 산출물 |
|---|---|---|---|---|
| C 문서/감사 | (읽기) | PASS | 감사 완료: stale/구모델명 잔재는 휴면 legacy(이미 문서화)·scripts(legacy재현). methods_and_runbook.html 부재(N/A). 안전 자동수정 대상 없음 | 아래 PENDING |
| D 재측정 | (측정) | PASS | vision latency 300회 재측정 안정화. YOLO11n>s는 실재(후처리 지배, 노이즈 아님). 테이블 갱신 | docs/benchmark_model_table.{md,csv} |
| B Q5 CPU-NPU | (판단) | SKIP | Q5 정확한 시나리오 파라미터 미상. 새 시나리오를 Q5로 라벨링=추측 금지. 기존 CPU-NPU 4기법 데이터 존재(phaseA/B2). PENDING 기록 | - |
| A background LLM (조사·측정) | (측정) | PARTIAL | 코드 조사·GPU DRAM 예산 측정 완료. NPU DRAM 측정 불가(smi 부재). 통합 결정은 SKIP(PENDING) | 아래 |

## 마감 요약 (21:03:42)
- **무결성**: 시작 gpu 4.8x/npu 4.1ms → 종료 gpu 4.7x/npu 4.7ms (동일, 회귀 없음). cuDNN 폴백 0 유지.
- **코드 로직 무변경**: 이번 무인 세션은 docs/측정만. unified_viewer/view_handlers 등 로직 파일은 이전 작업 상태 그대로(미커밋, 사용자 몫).
- **main.tex 불변**: 저장소에 .tex 없음.
- **커밋**: 9d47cf1 (docs만, 병합·푸시 안 함).
- **결과 요약**: A=PARTIAL(조사·DRAM측정, 통합 SKIP), B=SKIP(Q5 시나리오 미상), C=PASS(감사, 자동수정 대상 없음), D=PASS(latency 재측정 안정화).
- **다음 결정**: docs/PENDING_DECISIONS.md 참조 (A 통합·cuDNN우회·NPU DRAM, B Q5 정의, C 용어).
- **오염 없음**: SKIP-우선으로 추측 실행 안 함. 생성 데이터는 전부 무결성 검증 통과분.

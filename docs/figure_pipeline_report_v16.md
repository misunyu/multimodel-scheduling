# figure_pipeline.html 재생성 (v16) — 요약

날짜: 2026-07-26 · 재실험·값변경·그림 재생성·main.tex 수정 없음 · 문서는 `fig/build_pipeline_doc.py`로 생성
산출물: **`docs/figure_pipeline.html`**(단일 파일, 외부 의존 0, 오프라인) · 갱신 생성기 `fig/build_pipeline_doc.py`

## 변경 요약 (v11 → v16)
- **작업 58-1 그림**: 참조 그림 14 → **15**(`d2_budget_rank` 신규). 검증됨 5 → **6**, 미감사 9. 갱신 그림
  6종(q13_failure 2패널·q13_cumulative·q3·q5·bounded·d2_budget)의 캡션·계보·미리보기(base64) 전부 재추출.
- **작업 58-2 스키마**: 규약에 **`recovered`**(회복 여부, `censored`와 독립; Q4 search=24 반례) +
  **`source_kind`**(hardware/simulation/analytic, 각 뜻·예) 절 추가. **d2_budget_rank이 한 그림에 analytic
  평면+hardware 마커가 섞인 유일 사례**임을 카드·규약에 명시. 값표에 `source` 열 추가.
- **작업 58-3 프로그램**: `sidecar_util.py`(캔버스 추출) 등재, `emit_sidecars.py` **삭제 이유** 기록(confirmed
  복사 → checker 자기검사 결함), `check_figures.py` 확장(recovered·source_kind·모순검사·**텍스트 스캔**) 반영,
  구세대 plot_v8/v9 폐기·현행 표시.
- **작업 58-4 재현 절차**: 6종 재생성 + checker 명령·기대출력으로 갱신.
- **작업 59 D2 계보(§3b, 본문)**: `d2_budget_rank`만 논문 등재, `d2_dwell/load` **의도적 제외**와 이유(하드웨어
  2점 회복률 순서 뒤집힘, v13 작업47) 기록. "실패도 결과"로 §9 공백이 아니라 계보 본문에.
- **작업 60 미사용 PDF(§9)**: docs/figures **21개 중 논문 15개, 미사용 7개** 실측 목록화(성격 분류). `*_2gen`은
  Qwen-1 실증(삭제 금지), `d2_dwell/load`는 리포트 전용, `_RETIRED`는 폐기표시. **접미사 오집 위험** 명시.
  파일 이동·삭제 없음(목록만).
- **작업 61 이력·폐기값**: 감사 이력 v11–v15 추가. 폐기값에 **D2 회복률 곡선(미등재·미검증)**·**v13 budget
  평면(sim persist 오라벨)** 추가.

## 작업 62 자체 검증 (문서에 포함)
- ① main.tex \includegraphics(**14**, 파서 보유본) ⊆ 카드(**15**): 모두 포함. 주입 1개=`d2_budget_rank`.
- ② 카드 참조 confirmed id 전부 실재: 예(bad_ids 없음).
- ③ `check_figures.py --figdir fig` 출력 임베드: **0 mismatches**.
- ④ 하드코딩 없음(값은 confirmed에서 렌더, 사이드카는 캔버스).
- ⑤ 캡션 정합: 파싱 14개는 main.tex 직접 추출(정의상 일치), 주입 1개는 지시문 근거 캡션.

## 추적 실패 / 한계 (정직 기록)
- **main.tex 갱신본 미보유**: 내 파서 보유본(`f90965c7-main.tex`)은 14개(d2_budget_rank 없음). 지시문 v14/v15가
  "d2_budget_rank 논문 등재·캡션 반영 완료"라 명시하므로 15번째로 **주입**하고 그 사실을 카드·§8·§9에 표기.
  갱신 main.tex를 받으면 주입 없이 15개가 파싱된다.
- **미사용 PDF 목록 차이**: 지시문 작업60의 예시 파일들(bounded_recovery_analysis_accumulation,
  dynamic_load_adaptation_antara, ml_misprediction_alternative, latency_over_time 등)은 **이 docs/figures에
  존재하지 않는다.** 문서는 실제 존재하는 21개 기준으로 7개 미사용을 목록화(§0.1 저장소에서 읽어 쓴다).
  예시 파일들은 다른 위치(논문 소스 트리 등)에 있을 수 있음.
- 생성 스크립트 미상: `architecture`·`c3_fluid_validation`(fig/·scratchpad에 생성 스크립트 미발견) — §9 유지.

## 무결성
- 재실험·값변경·그림 재생성·main.tex 수정 없음(문서화만). 저장소 상태에서 읽어 생성, 미상은 미상으로 표기.

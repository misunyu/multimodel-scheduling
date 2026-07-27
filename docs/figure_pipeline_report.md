# 그림 파이프라인 문서화 (v11) — 요약

날짜: 2026-07-26 · 재실험·값변경·main.tex 수정 없음
산출물: **`docs/figure_pipeline.html`**(단일 파일, 외부 의존 0, 오프라인 개봉) · 생성기 `fig/build_pipeline_doc.py`

## 무엇을 문서화했나
- **작업 38**: `main.tex`의 `\includegraphics`를 파싱해 실제 참조되는 **그림 14개** 목록화(파일시스템 훑기 아님).
  각 그림의 `\label`·캡션 전문·절을 figure 환경에서 추출.
- **작업 39**: 그림별 계보를 **저장소에서 확인**해 채움 — 생성 스크립트(현행 세대 명시, 구세대 병기), 입력
  데이터(플롯 스크립트 grep + confirmed `source` 필드), 사이드카 존재·항목 수, 쓰는 confirmed id 표.
  **검증 5개**(bounded/q13_failure/q13_cumulative/q3/q5)와 **미감사 9개**를 뱃지로 구분.
- **작업 40**: 단일 HTML — 개요 · 파이프라인 도식(CSS, 이미지 파일 없음) · 규약(confirmed `_meta`에서 렌더) ·
  그림 카드 14(base64 PNG 미리보기 인라인) · 데이터 출처 · 프로그램 · 재현 절차 · 감사 이력 · 알려진 공백.
- **작업 41 (자체 검증, 문서에 포함)**:
  - main.tex 목록 == 카드 목록: **일치(14)**.
  - 카드가 쓰는 confirmed id 전부 실재: **예**(bad_ids 없음).
  - `check_figures.py --figdir fig` 출력 임베드: **0 mismatches**.
  - 하드코딩 없음: 값은 confirmed에서 읽어 렌더, 사이드카는 캔버스에서 추출.

## 핵심 설계 (지시문 원칙 반영)
- **하드코딩 금지**: 문서의 모든 수치는 `confirmed_values.json`에서 읽어 렌더. 값이 바뀌면 생성기 재실행.
- **검증/미감사 뱃지 구분**: 14개 중 5개만 확정표+사이드카+checker 대상. 나머지 9개는 계보만.
- **폐기값 목록**(§7): 옛 문서에서 보면 stale인 값 — 20.0(→25.4)·59.4(censored)·15.2(→14.2)·114.0(censored)·
  105–106(censored)·21.4/23.6(비정본 배치/하네스)·"24.6 as persist"(=T_valid). 옛 리포트 독자용.
- **삭제 파일 기록**: `emit_sidecars.py`(왜 삭제했는지 = 자기검사 결함) 포함.
- **자체 생성**: 손으로 쓰지 않고 `build_pipeline_doc.py`로 생성.

## 알려진 공백 (§9, 문서에 명시)
- **미감사 9개**: c2_reactive_comparison, q4_bounded_envelope, dynamic_load_adaptation, c3_fluid_validation,
  b2_buffer_sweep, detection_sensitivity, runtime_overhead, qos_score_validation, architecture. 계보만.
- **생성 스크립트 미상 2개**: `architecture`, `c3_fluid_validation`(fig/·scratchpad에서 생성 스크립트 미발견).
- **미결 판단**: censored 축 전환(v8 작업27, PENDING). **미착수**: c2 2패널화, D2 시뮬레이션.

## 무결성
- 재실험·값변경·그림 재생성 없음(문서화만). main.tex·알고리즘·파라미터·메인앱 코드 불변.
- 문서는 저장소 상태에서 읽어 생성 — 추측으로 빈칸 채우지 않고 미상은 미상으로 표기.

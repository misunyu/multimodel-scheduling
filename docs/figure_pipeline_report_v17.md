# main.tex 저장소 제거 · 매니페스트 인터페이스 전환 (v17) — 요약

날짜: 2026-07-26 · 재실험·값변경·그림 재생성 없음 · 문서는 `fig/build_pipeline_doc.py`로 생성
산출물: `docs/figure_pipeline.html`(재생성, 매니페스트 기반) · `fig/paper_figure_manifest.json`(신규 인터페이스) · 갱신 생성기

## 작업 63 — main.tex 제거
- **제거 위치**: 저장소 루트 `main.tex`(직전 v17 시도에서 배치했다가 사용자 지시로 삭제). 현재 저장소에 `main.tex`
  **0개**(`find` 확인). 저장소는 애초에 정본을 보유한 적 없음(코드 저장소) → **별도 백업 불요**(원본은 업로드로 존재).
- **같은 위치 자산**: `architecture.pdf`(저장소 루트, 35 KB) — 논문 그림 자산. **제거하지 않음**.
- **잔여 참조(게이트)**: `main.tex`를 읽던 코드는 `fig/build_pipeline_doc.py` **하나뿐** → 작업 64에서 매니페스트
  읽기로 전면 교체(파싱·주입 코드 삭제). 다른 잔여 참조 없음.
- **무결성 문구 교체**: "main.tex 불변 확인" → **"main.tex 부재(설계) — 정본은 저장소 밖, 인터페이스는
  `fig/paper_figure_manifest.json`"**. (아래 PENDING·이 리포트에 반영)

## 작업 64 — 생성기 매니페스트 기반 전환
- `fig/paper_figure_manifest.json` 배치(15그림, `_meta.main_tex_sha256=6626c5f6…`, 124655 bytes).
- 생성기에서 **main.tex 경로·`\includegraphics`/`\label`/`\caption` 파싱·절 추정·주입(FORCE_FIGS) 코드 전량 삭제.**
  그림 카드의 유일 원천 = 매니페스트 `figures` 배열.
- 문서 상단에 **매니페스트 sha16 + 추출일** 표기(= "이 문서가 어느 논문 판본인가"의 유일 근거).
- **정합성 검사 3종**(생성기 내): (1) 매니페스트 file 전부 그림 디렉터리 존재(부재 시 실패) — 결과 0건,
  (2) 디렉터리에 있으나 매니페스트에 없는 파일 = 미사용 후보(오류 아님), (3) 매니페스트 file이 confirmed 참조
  여부 = 뱃지 분류 입력(오류 아님).
- **한계 명시(§8)**: 저장소에 main.tex가 없어 **매니페스트 staleness 자동 검증 불가** — 해시는 기록만, 대조는 사람.

## 작업 65 — 재생성 (검증표)
| 항목 | v16 | v17 |
|---|---|---|
| 그림 카드 | 15 (14 파싱 + 1 주입) | **15 (전부 매니페스트)** |
| 주입 | 1 | **0** |
| base64 미리보기 | 14 | 14 (architecture.pdf는 PNG 없음 — 루트 자산, 카드에 "미리보기 없음" 표기) |
| 캡션 출처 | 14 main.tex + 1 지시문 | **15 전부 매니페스트** |
| §8 자체검증 ① | ⊆ | **정확히 일치 (15 = 15), 주입 0, PDF 부재 0** |
- v16의 "주입"·"main.tex 갱신본 미보유" 서술 제거. **§7 감사 이력에 v16 주입을 한 줄로 보존**("stale main.tex(14)
  위에서 15번째 주입 → v17 해소").

## 작업 66 — 그림 디렉터리 선언 · 미사용 목록
- **그림 디렉터리(저장소 전체, .venv·backup 제외)**: `docs/figures` **21개(정본 출력 선언)**, 저장소 루트 `.` 1개
  (architecture.pdf, 논문 자산). **지시문이 언급한 "논문 소스 트리 ~33 PDF"는 이 저장소에 존재하지 않는다**
  (정본 트리는 저장소 밖) — §0.3대로 경로 명시하고 실상 보고.
- **미사용 7개(경로 포함)**: `docs/figures/{b2_metrics, c3_b2_metrics, d2_dwell_response, d2_load_tightness,
  q3_misprediction_2gen, q3_paperset_q4like_RETIRED, q5_npu_generalization_2gen}.pdf`. 성격 분류:
  `*_2gen`=Qwen-1 2-gen 실증(**삭제 금지**), `d2_dwell/load`=D2 리포트 전용(§3b), `_RETIRED`=폐기표시, 나머지=옛세대.
  접미사 오집 위험(`_epsilon_1`·`_5cand`) 명시. **파일 이동·삭제 없음.**
- 지시문 66-2 예시 중 `*_epsilon_1`·`*_5cand`·`*_accumulation`·`ml_misprediction_alternative`·`latency_over_time`·
  `vscore_over_time*`는 **이 docs/figures에 부재** → "해당 없음"으로 처리(실존만 목록화).

## 작업 67 — 정본 관리 규약 (§6b에 기재)
1. 정본 main.tex는 저장소 밖. 2. 논문 변경 시 새 매니페스트 수령, sha 다르면 재생성. 3. 투고 패키지는 저장소
밖에서(저장소는 그림·검증 산출물 내보내기). 4. 그림 변경 시 그림+사이드카 함께 내보냄.

## 자체 검증 (문서 포함)
- 매니페스트 그림(15)==카드(15), 주입 0, PDF 부재 0: **일치**.
- confirmed id 전부 실재(bad_ids 없음). checker `0 mismatches`(불변, 텍스트 스캔 유지).
- **빌드 한계(정직)**: `pdflatex` 미설치·`.bib`/`.bst` 부재 → 이 환경에서 4패스 빌드 **실행 불가**. 문서 생성은
  빌드가 아니라 매니페스트 파싱이라 무관. 정적 확인: 그림 15개 PDF 전부 존재(architecture는 루트), ref/label
  균형(refs 26 ⊆ labels 30). 완전 LaTeX 빌드는 저장소 밖에서 수행.

## 무결성
- 재실험·값변경·그림재생성·main.tex 재도입 없음. checker 0 mismatch. main.tex 부재는 설계(정본 외부).

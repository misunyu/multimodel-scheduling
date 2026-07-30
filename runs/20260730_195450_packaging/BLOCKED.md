# 제출 패키지 구성 — 단계 1에서 중단

- 작성: 2026-07-30
- 스크립트: `build_package.py` (0b → 1 → 2 → 3, 첫 하드 실패에서 중단)
- **`manuscript/` 미수정** (읽기·복사만). zip·`dist/` 미생성.

## 0. 사전 확인

### 0a. CLAUDE.md 정책 — 추가함

`CLAUDE.md`가 저장소에 없었으므로 신규 생성하고 지정된 한 줄을 넣었다
(`manuscript/` 밖이므로 허용 범위):

> manuscript/ 이하는 채팅 세션에서만 편집되는 읽기 전용 미러다. 수정 금지, 대조·참조만 허용.

### 0b. 미러 최신성 — **4/4 통과, 진행**

| 검사 | 결과 |
|---|---|
| `REVISION: 2026-07-30 r2` 주석 존재 | **통과** (`:5`) |
| `11 of 45` 존재 | **통과** (3회: abstract, `:128`, Fig.2 캡션) |
| `29 of 45` 부재 | **통과** (0회) |
| `architecture_overview` 참조 부재 | **통과** (0회) |

미러 sha256 `ae0a1ab4…`, 353행.

## 1. 의존 파일 수집 — **2건 부재로 중단**

tex 파싱 결과(하드코딩 없음): `\bibliography{mlforsys_main}` → 1건,
로컬 `.sty`(스톡 패키지 제외) → 1건, `\includegraphics` → 2건.

| 종류 | 파일 | 상태 | 경로 / sha256 |
|---|---|---|---|
| figure | `fig_divergence.pdf` | **확보** | `analysis/figures/fig_divergence.pdf` — sha256 `3c13b1df…f1fd5` **요구 해시와 일치** (사본 1개, 구판 혼입 없음) |
| figure | `fig_transfer.pdf` | **확보** | `analysis/figures/fig_transfer.pdf` — sha256 `d9899184…c10754` |
| bib | `mlforsys_main.bib` | **부재** | `manuscript/` → `analysis/figures/` → 저장소 전체 rglob 전부 미발견 |
| sty | `neurips_2025.sty` | **부재** | 동일하게 전부 미발견 |

그림은 지시대로 정확히 2개이고(`figure_count_is_2: true`), `fig_divergence.pdf`는 해시
일치본이 유일하게 존재한다. 반면 `.bib`과 `.sty`는 **시스템 어디에도 없다** — 저장소 전체는
물론 `/` 기준 depth 6 검색에서도 `mlforsys_main.bib`·`neurips_2025.sty`·`neurips_2026.sty`가
0건이고, 저장소에 `.bib`/`.sty`/`.bbl`/`.cls` 파일 자체가 하나도 없다. `manuscript/` 디렉토리
내용은 `mlforsys_main.tex` 단 하나다.

→ 지시된 정상 복구 경로: **사용자가 `mlforsys_main.bib`과 `neurips_2025.sty`를
`manuscript/`에 복사한 뒤 이 스크립트를 재실행**한다. 스크립트는 `manuscript/`를 최우선으로
탐색하므로 그곳에 두면 바로 채택된다.

## 2. 격리 컴파일 — 미실행 (단계 1 차단 + 툴체인 부재)

단계 1에서 중단되어 실행하지 않았다. 다만 **단계 1이 해결되어도 현 환경에서는 단계 2를
실행할 수 없다**:

| 도구 | 상태 |
|---|---|
| `pdflatex` | **미설치** |
| `bibtex` | **미설치** |
| `latexmk`, `xelatex`, `lualatex`, `tex`, `kpsewhich` | 전부 미설치 |
| `pdftotext` | 설치됨 (`/usr/bin/pdftotext`) |
| texlive dpkg 패키지 | 0건 |

즉 `.bib`/`.sty`를 채워 넣더라도 TeX 배포판(texlive 등) 설치가 선행되어야 한다. 검증 기준
자체는 스크립트에 이미 구현되어 있다 — error 0, undefined reference/citation 0,
`pdftotext -f 5 -l 5` 첫 줄 == `References`, 금지 문자열
(`29 of 45`, `29/45`, `22/30`, `7 of 15`) 0건, `[TODO:` 0건.

참고로 원고 텍스트 수준에서는 금지 문자열 4종이 이미 0건이고 `\RES`/`\TODO`
**사용처도 0건**(정의와 주석만 존재)이므로, 렌더 후 기준도 통과할 전망이다 — 단 이는
소스 grep 결과이고 PDF 렌더 검증을 대체하지 않는다.

## 3. 패키지 생성 — 미실행

단계 2 통과가 전제 조건이므로 `dist/`와 zip을 만들지 않았다. `PACKAGING_NOTES.md`도
패키징 시점에 생성되도록 스크립트에 구현되어 있으며(기준 리비전 주석 원문, neurips_2026
보류 문구, 파일별 sha256 목록), 이번 실행에서는 생성되지 않았다.

## 보고 대상 — 원고 쪽 문제 (수정하지 않음)

`manuscript/`는 수정 권한이 없으므로 목록만 남긴다.

1. **`.bib`·`.sty` 미러 누락** — 원고 미러가 `.tex` 단일 파일로만 복사되어 있어 컴파일
   가능한 상태가 아니다. 채팅 세션에서 두 파일도 함께 내려받아 `manuscript/`에 두는 것이
   필요하다.
2. **`neurips_2025.sty` 사용** — 원고 주석(`:8`, `:12-13`)이 2026 CFP에 따른 교체를
   PENDING으로 표시하고 있다. 스타일 교체 후 패키징 재실행이 필요하다(작업 지시의 보류
   사항과 동일).

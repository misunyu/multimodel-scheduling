# `paper/` — 논문 배포본 (읽기 전용)

> **이 디렉터리를 편집하지 마십시오.** 정본은 저장소 밖에 있습니다.
> 여기 있는 `main.tex`는 사본이 아니라 **배포본**입니다.

---

## 왜 읽기 전용인가

한 자산이 두 곳에서 편집되면 반드시 갈라집니다. 이 저장소에서 실제로 일어난 일입니다.

- **v16**: 저장소의 낡은 `main.tex`(그림 14개)로 파이프라인 문서를 생성해 15번째 그림을 **주입**했다.
- 최근: 누적된 업로드 디렉터리에서 **옛 그림이 재실행본을 덮어썼다.** 파일 크기를 대조해서야 발견했다.

그래서 v17에서 논문 정본을 저장소 밖에 두기로 했고, 이 디렉터리는 그 결정을 **되돌리는 것이 아니라
읽기용 배포본을 추가**하는 것입니다.

**관례가 아니라 해시가 강제합니다.** `paper.lock`이 `main.tex`와 그림 13종의 sha256을 담고
있으며, `scripts/check_paper.py`가 매번 대조합니다. 여기서 한 글자라도 고치면 검사가 실패합니다.

---

## 구성

```
paper/
  main.tex            배포본 (편집 금지)
  IEEEtran.cls        학회 템플릿
  IEEEtran.bst        참고문헌 스타일
  failsafe_main.bib   참고문헌
  paper.lock          해시 대장
  README.md           이 문서
```

**그림은 이 디렉터리에 없습니다.** `main.tex`가 `\graphicspath{{../figures/}}`로 저장소의
`figures/`를 직접 읽습니다. 복사본을 두지 않는 이유는 위와 같습니다 — 복사하면 갈라집니다.

저장소 루트의 `figures/`는 **`docs/figures/`를 가리키는 심볼릭 링크**입니다. 그림 생성 스크립트
(`fig/plot_*_v11.py`)가 쓰는 곳이 `docs/figures/`이고, 논문이 읽는 경로가 `../figures/`이므로,
링크가 둘을 **복사 없이** 잇습니다. 링크를 지우면 빌드가 그림을 찾지 못합니다.

`figures/`에는 미사용 산출물도 함께 있습니다. **논문이 실제로 쓰는 것은 `paper.lock`의
`figures` 목록이 정의합니다**(현재 13종).

---

## 빌드

```bash
make paper          # = paper/build.sh
```

또는 직접:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

**4패스가 필요합니다.** 절·그림·표·식·문헌 번호가 서로를 참조하므로 한 번만 돌리면 참조가
한 판 뒤처집니다.

기대 결과: **22페이지**, `LaTeX Warning: Reference` 및 `Citation` **0건**.

> **이 기계에는 TeX가 설치돼 있지 않습니다**(`pdflatex`·`bibtex` 부재). 그래서 v36에서는 빌드를
> 실행해 확인하지 못했고, 대신 빌드가 **필요로 하는 것**을 검사로 확인했습니다: `main.tex`가
> `\includegraphics` 하는 13종이 전부 `figures/`에 있고 해시가 `paper.lock`과 일치합니다
> (`make check`). 페이지 수와 참조 0건은 TeX가 있는 환경에서 확인해야 합니다.

---

## 정본이 갱신되면

논문은 저장소 밖에서 수정됩니다. 갱신본이 오면:

1. **`main.tex`와 새 `paper.lock`을 함께** 받아 이 디렉터리에 배치합니다. 둘 중 하나만 받지 않습니다.
2. 4패스 빌드로 22페이지·참조 0건을 확인합니다.
3. `make check`로 논문↔확정표 대조를 돌립니다.

`paper.lock` 없이 `main.tex`만 갱신하면 해시 검사가 실패합니다. **그것이 의도된 동작입니다** —
어느 판본인지 모르는 배포본을 허용하지 않습니다.

---

## 새 실험이 수행되면

```
실험 → runs/ 에 산출물 → fig/confirmed_values.json 갱신
     → make figures      그림·사이드카 재생성
     → make check        그림↔확정표, 논문↔확정표 대조
     → 어긋난 줄 목록을 저장소 밖으로 전달
     → 갱신된 main.tex + paper.lock 을 받아 배치
```

**`check_paper.py`가 짚는 줄 목록이 논문에서 고쳐야 할 곳입니다.** 지금까지 사람이 하던 대조입니다.

수치에 `%!cv <id>` 주석이 달린 줄만 자동 대조됩니다. 주석이 없는 수치는 **미추적**으로 목록에
나오며(`make check-full`), 검사가 무엇을 덮지 못하는지 보여줍니다.

### `%!cv` 주석을 붙이는 방법

주석은 정본에 들어가야 하므로 저장소가 붙일 수 없습니다. 대신 **목록을 냅니다**:

```bash
python3 scripts/gen_cv_worklist.py                       # 목록 생성(저장소에서)
python3 scripts/apply_cv_annotations.py --tex <정본 경로>  # 적용(저장소 밖에서)
```

`docs/paper_cv_annotation_worklist.md`가 사람이 읽는 형태,
`docs/paper_cv_annotations.json`이 위 스크립트가 먹는 형태입니다. 각 항목은 **대상 줄의
sha256**을 함께 담습니다 — 줄 번호는 편집으로 밀리고, 엉뚱한 줄에 붙은 주석은 *그 문장이
하지도 않은 주장에 값을 매다는* 조용한 오염이 되기 때문입니다. 해시가 어긋나면 붙이지 않고 멈춥니다.

주석 형식은 줄 **끝**에 붙는 `%!cv id1,id2`이며, 자기가 끝맺는 **문단** 전체를 범위로 삼습니다.

### 검사가 덮지 못하는 것

- **글자로 쓴 수**(`seven times`, `fifteen transitions`)는 수치가 아니어서 보이지 않습니다.
- **확정표에 없는 값**은 주석을 달 대상이 아예 없습니다. 이쪽이 더 중요한 결함이며,
  `docs/paper_distribution_report.md` §4에 목록이 있습니다.
- **표 셀**은 의도적으로 달지 않았습니다. 표는 갱신할 때 통째로 교체되므로 조용히 낡을 위험이
  산문보다 낮습니다.

---

## 관련 문서

| 문서 | 내용 |
|---|---|
| `fig/README.md` | 그림 값 파이프라인 규약 |
| `fig/FIGURE_SPECS.md` | 그림별 명세와 함정 |
| `runs/README.md` | 원시 런 영속화 규약 |
| `fig/confirmed_values.json` | 모든 수치의 단일 소스 |
| `docs/paper_cv_annotation_worklist.md` | 정본에 붙일 `%!cv` 주석 목록 |
| `docs/paper_distribution_report.md` | 이 배치의 근거와 검사 출력 |
| 루트 `Makefile` | `figures` / `check` / `paper` / `lock` 진입점 |

**확정표에 없는 값은 그림에도 논문에도 넣지 않습니다.**

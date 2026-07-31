# 논문 배포본 배치 · 수치 주석 · 논문↔확정표 검사 (v36)

작성 2026-07-31. **커밋함, 푸시 없음.** 논문 문안은 쓰지 않았다 — 검사 결과만 적는다.

산출물: `paper/`(배포본), 루트 `figures/` 링크, `Makefile`, `scripts/{check_paper,gen_paper_lock,
gen_cv_worklist,apply_cv_annotations,propose_cv_annotations}.py`,
`docs/paper_cv_annotation{s.json,_worklist.md}`, 픽스처 `scripts/testdata/paper_fixture/`.

---

## 0. 한 줄 요약

논문 배포본이 저장소에 놓였고, **편집을 해시로 막고**, 새 실험이 확정표를 갱신하면 **논문의 어느 줄이
낡았는지 자동으로 나오게** 했다. 주석 16개(추적 값 64건)를 정본에 붙이는 목록을 냈고, 그 목록이
실제로 통과함을 저장소 밖 사본에서 **64/64 확인**했다. 그 과정에서 논문이 **확정표에 없는 값을 쓰고
있고**, Q4 한 문단이 **재실행 전후 값을 섞어 쓰고 있다**는 것이 드러났다(§4).

---

## 1. 작업 188 — 배치

```
paper/                  main.tex · IEEEtran.{cls,bst} · failsafe_main.bib · paper.lock · README.md · build.sh
figures/  ->  docs/figures/        (심볼릭 링크)
Makefile                figures / check / paper / lock 진입점
```

`paper/`는 **최상위**에 둔다 — `docs/`에는 리포트가 85개라 규칙이 묻힌다.

### 그림을 중복 배치하지 않기 위해 링크를 썼다

`main.tex`의 `\graphicspath{{../figures/}}`는 저장소 루트의 `figures/`를 가리키는데, 그림을 실제로
생성하는 곳은 `docs/figures/`다(`fig/plot_*_v11.py`의 출력 경로). 복사본을 만들면 갈라진다 —
그것이 v16과 최근 사고의 원인이다. 그래서 **`figures` → `docs/figures` 심볼릭 링크**를 두었다.
링크는 사본이 아니므로 갈라질 수 없고, 배포본은 편집 없이 그대로 쓸 수 있다.

### `architecture.pdf` — 문서 셋의 오기를 정정했다

`paper.lock`이 요구하는 13종 중 `architecture.pdf`만 `figures/`에서 보이지 않았다.
찾아보니 **저장소 루트에 있었고**(`figure_pipeline_report_v17.md`가 그렇게 기록하고 있다),
해시도 `paper.lock`과 일치했다. 반면 **다른 세 문서는 "저장소에 없다"고 적고 있었다** —
`docs/report_q_mapping.md`(2곳), `fig/FIGURE_SPECS.md`, `runs/README.md`. 셋 다 정정했고,
파일은 `git mv`로 `docs/figures/`에 옮겨 논문이 읽는 경로에 놓았다.

이 오기는 무해하지 않았다: `runs/README.md`는 이것을 *"의도적으로 저장소 밖에 둔 자산"*으로
분류하고 있었고, 그 분류가 맞다면 **논문은 저장소만으로는 빌드될 수 없다**는 뜻이 된다. 실제로는
빌드된다.

> **작업 중 사용자가 `paper/architecture.pdf`를 업로드했다.** `docs/figures/architecture.pdf`와
> **바이트 동일**(sha256 `3e1f7fd3…`, 35,883 B)이어서 커밋하지 않고 지웠다. 이유는 이 작업의 원칙
> 그대로다 — 그림을 두 곳에 두면 갈라진다. 더구나 LaTeX는 `\graphicspath`보다 **현재 디렉터리를
> 먼저** 찾으므로 `paper/architecture.pdf`가 `figures/`의 것을 **가린다**. 둘이 언젠가 달라지면
> 논문은 조용히 옛 그림으로 빌드된다 — `paper/README.md`가 적어 둔 "누적된 업로드 디렉터리에서
> 옛 그림이 재실행본을 덮어썼다"와 같은 사고다. 바이트가 같으므로 잃은 것은 없다.

### 빌드 — **이 기계에서는 확인하지 못했다**

`pdflatex`·`bibtex`가 설치돼 있지 않다. 빌드를 돌리지 못했으므로 **22페이지·참조 0건은 확인하지
않았다.** 대신 빌드가 필요로 하는 전제를 검사로 확인했다: `\includegraphics` 13종이 전부
`figures/`에 있고 해시가 `paper.lock`과 일치한다(§5 출력). 페이지 수 확인은 TeX가 있는 환경의 몫이다.

---

## 2. 작업 189 — `paper.lock`

`figures` 목록은 **손으로 적지 않는다.** `scripts/gen_paper_lock.py`가 `main.tex`의
`\includegraphics`를 파싱해 만든다(주석 줄은 제외). 기본 동작은 `--check`이고 쓰기는 `--write`를
요구한다 — 편집된 `main.tex`에 대고 무심코 재생성하면 **편집을 lock에 세탁해 넣는 것**이 되어,
lock이 막으려던 바로 그 일이 된다.

받아 둔 `paper.lock`이 파싱으로 재현되는지부터 확인했다:

```
$ python3 scripts/gen_paper_lock.py
parsed 13 figures from main.tex
[OK] committed paper.lock reproduces from main.tex + figures/
```

**13종 = `main.tex`가 실제로 쓰는 것 전부.** `figures/`에는 42개가 있으므로 나머지 29개는 미사용이다
(미참조 산출물 목록은 `docs/report_q_mapping.md` §4).

---

## 3. 작업 190 — 수치 주석 목록

**배포본을 편집하지 않았다.** `%!cv` 주석은 저장소 밖 정본에 들어가야 하므로 목록으로 냈다.

| 산출물 | 내용 |
|---|---|
| `docs/paper_cv_annotation_worklist.md` | 사람이 읽는 형태 — 줄 번호, 줄 발췌, 붙일 주석, 각 id의 확정표 값 |
| `docs/paper_cv_annotations.json` | `scripts/apply_cv_annotations.py`가 먹는 형태 |
| `scripts/gen_cv_worklist.py` | 위 둘의 생성기. 값은 확정표에서 읽으므로 목록이 낡은 사본을 들고 있을 수 없다 |
| `scripts/apply_cv_annotations.py` | **정본에** 적용. `paper/` 아래 경로는 거부한다 |

**주석 16개 · 추적 값 64건(고유 id 55건).** 지시문의 목표는 25~35개였다 — 세는 단위를 밝힌다:
**유지 부담은 주석 수(16)에 붙고**, 덮는 범위는 값 수(64)다. 한 문장에 여러 값이 들어가는 산문이
많아 이 비율이 나왔다. 주석을 25개로 늘리려면 표 셀로 내려가야 하는데, 표는 갱신 시 통째로 교체되어
조용히 낡을 위험이 산문보다 낮다(우선순위 4).

### 붙는 자리 (§IV 산문 우선)

| 줄 | 무엇 | 값 수 |
|---|---|---|
| 557 | Q1 회복 총시간·lead, 양 플랫폼 | 4 |
| 579 | Q1.1 정상상태 처리량·추론 지연·CPU 점유 | 9 |
| 600 / 602 | Q1.2 검출 지연·회복 지연 스윕, T=1 대 T=3 | 5 / 4 |
| 625 / 627 | Q1.3 Static 종단 V·cumV·지속 범위·배율 / lead뿐인 search | 9 / 1 |
| 644 / 646 | Q1.4 스윕 회귀 기울기 / 세 서비스율 | 2 / 6 |
| 684 / 686 / 688 | Q1.5 버퍼 스윕·재배치 대 정지·정지재시작 대 in-place | 4 / 4 / 3 |
| 700 | Q2 구분 불가능한 회복 시간 | 2 |
| 714 | Q3 persist·search·drain·lead·봉투·baseline 종단 V | 6 |
| 743 | Q4 종단 위반 | 1 |
| 775 | Q5 persist·lead·baseline 종단 V | 3 |
| 817 | Q6 변형 A 종단 위반 | 1 |

### 목록이 실제로 통과하는지 — 저장소 밖 사본에서 확인

정본이 아직 없으므로 배포본을 **스크래치로 복사**해 적용하고 검사했다(저장소에는 남기지 않았다).

```
16 applied, 0 skipped
paper.lock:      OK (main.tex + 13/13 figures)
tracked values:  64 checked, 0 mismatch
untracked:       198 numerals on 59 lines of section 'Experimental Evaluation' without %!cv
figures:         13 referenced, 13 present
[OK] PAPER MATCHES CONFIRMED TABLE (0 mismatches).
```

**64건 전부 통과한다.** 미추적은 327 → 198로 줄었다.

### 줄 해시로 고정한 이유

각 항목이 **대상 줄의 sha256**을 함께 담고, `apply_cv_annotations.py`는 일치할 때만 붙인다.
줄 번호는 편집 한 번에 밀리고, 엉뚱한 줄에 붙은 주석은 **그 문장이 하지도 않은 주장에 값을 매다는**
조용한 오염이 된다 — 검사기는 통과시키고 사람은 알아채지 못한다.

---

## 4. **검사가 드러낸 것** — 논문이 확정표와 어긋나는 지점

주석을 다는 것보다 이쪽이 이 작업의 실제 소득이다. 세 종류가 나왔다.

### 4.1 Q4 한 문단이 **재실행 전후 값을 섞어 쓴다** ← 가장 심각

| 줄 | 문장 | 값 | 확정표 |
|---|---|---|---|
| 741 | "reverts to the best placement it observed and holds there, **ending at $V(t)\approx55$**, the lowest of the four" | 55 | — |
| 743 | "the **terminal violation reaches $V(t)=362\pm112$**" | 362±112 | `q4_bg_lastV` = **362.5** ✓ |
| 756 | "**$V(t)\approx55$** is far above $\epsilon=1$" | 55 | — |

`≈55`는 **게재본의 옛 값 54.84**다. v33 재실행이 이것을 **362.5±112.2**로 바꿨고
(`docs/main_rerun_report.md` §2, 6.6배 차, phantom 오염 가설), L743은 새 값으로 갱신됐으나
**L741과 L756은 옛 값으로 남았다.** 같은 절 안에서 같은 양이 두 값으로 서술된다.

같은 문단의 baseline 종단 값(L741 `74`--`76`, L743 peak `75`--`78`)도 확정표에 대응 항목이 없고,
v33 재실행값(B 294.2 / C 292.6 / Adaptive 315.1)과 자릿수가 다르다 — 같은 계열의 미갱신으로 보인다.
**가설이며 검증하지 않았다.**

→ **본문 수정이 필요하다.** 수정 전에는 `%!cv q4_bg_lastV`를 L741·L756에 달 수 없다 —
달면 즉시 불일치로 잡히는데, 그것이 옳은 동작이지만 지금 붙일 주석은 아니다. L743에만 달았다.

### 4.2 논문이 쓰는데 **확정표에 없는 값**

`fig/README.md`의 규약은 *"확정표에 없는 값은 그림에 넣지 않는다"*이고, `paper/README.md`는 이를
논문까지 확장한다. 아래는 그 규약에 걸린다 — 주석을 달 대상 자체가 없다.

| 줄 | 값 | 무엇 |
|---|---|---|
| 462 | 5.2 / 3.4 | vision-4 구성 비교의 $V(t)$ |
| 585 | 0.15–0.21 / 1.0 / 1.5–2.2 | Q1.1 전환 시 서비스 중단 구간 |
| 712 | 3.7–4.1 | Q3 baseline 종단 $V$ (같은 절 L714는 **3.5** = `q3_baseline_lastV`) |
| 739·741·743 | 72 / 74–76 / 55 / 45 / 237 / 186 / 146 / 75–78 | Q4 용량·종단·서비스율·피크 |
| 745·747 | 51–59 / 25 / 54–60 / 26 / 40 | Q4 생성 모델 2개 사례 |
| 752 | 24–26 / 23 | Q3 봉투의 두 형태 |
| 773 | 7 / 34–49 / 4.4 | Q5 NPU 경합 |
| 775 | 9.6 | Q5 lead 중 컨트롤러 비활성 구간 |
| 817 | 2226±34 / 264±4 | Q6 변형 A와 baseline의 누적 위반 |
| 821 | 8 | 유체 시뮬레이션 오차 |

`docs/main_rerun_report.md`가 이 중 다수(Q6 cumV 2226/264, Q5의 9.6 s)를 담고 있으므로 **값이
없는 것이 아니라 확정표에 올라오지 않은 것**이다. 확정표에 추가하면 그대로 추적 대상이 된다.
**이번에 추가하지 않았다** — v36은 배치와 도구이며 확정표 편집은 범위 밖이다.

### 4.3 검사가 **원리적으로** 덮지 못하는 것

- **글자로 쓴 수.** L627 *"recovering in only **two** runs of five, and making **fifteen**
  transitions where the others make **six**"* — `q4b_a_recoveries`=2, `q6_a_hotswaps`=7,
  `q6_bg_hotswaps`=6이 확정표에 있는데도 수치가 아니어서 보이지 않는다. L817도 같다(*"seven
  times against BoundGuard's six"*).
- **파생값.** Q1.1의 ΔCPU(−0.18 / +0.12 pp)는 확정표의 두 항목의 **차**이며 항목이 아니다.
  L579의 원값 9건에 주석을 달아 우회했다. 차 자체를 추적하려면 `q13_*_cumV_ratio` 선례처럼
  파생 항목을 확정표에 넣어야 한다 — **권고이며 이번에 하지 않았다.**
- **표 셀.** 의도적으로 달지 않았다(우선순위 4). `tab:b2` 8행 48값이 미추적의 큰 몫이다.

### 4.4 미추적 327건의 구성 (주석 적용 전 기준)

| 구분 | 대략 | 성격 |
|---|---|---|
| 설정·파라미터 ($T$, $\epsilon$, $N_{cand}$, $\beta_B$, λ, 반복 수) | ~90 | 측정이 아니다. 낡을 수 없다 |
| Table~I 모델 지연 (L441–449) | 27 | 별도 출처(`docs/benchmark_model_table.csv`) |
| `tab:b2` 셀 (L664–672) | 40 | 표 — 우선순위 4 |
| `tab:c2-ablation` 셀 (L791–796) | 30 | 표 — 회복 횟수 |
| **주석으로 덮은 산문** | 64 | ← 이번 작업 |
| **확정표에 없는 산문 수치** | ~40 | ← §4.2, 실제 결함 |
| 기타(백분율 비교, 봉투 산술 등) | 나머지 | |

---

## 5. 작업 191 — `scripts/check_paper.py`

### 검사 다섯

1. **배포본 해시** — `main.tex` + 참조 그림 13종 대 `paper.lock`
2. **수치 대조** — `%!cv` 주석의 각 id가 그 줄(문단)에 실제로 나타나는가
3. **미추적 목록** — §IV의 `%!cv` 없는 수치 (**경고, 실패 아님**)
4. **그림 참조** — `\includegraphics` 대상이 `figures/`에 있는가
5. **삭제된 그림** — 미사용 표시된 것을 아직 참조하는가

### 설계 판단 둘 (스크립트 docstring에 사유 기록)

**단위 환산을 검사기가 한다.** 확정표는 계측 단위(`p99`는 ms, `miss`는 비율), 논문은 읽기 좋은
단위(33.7 s, 63 %)를 쓴다. 주석에 배율을 손으로 적게 하면 그 배율이 낡으므로, 검사기가 항목의
`unit`을 보고 ms↔s·비율↔%를 환산한다.

**허용오차 대신 자릿수 반올림.** 백분율 허용오차는 33.7과 33.9를 함께 통과시킨다. 대신 확정표
값을 **그 줄이 인쇄한 자릿수로 반올림**해 정확히 일치를 요구한다. 실패 메시지가 "논문은 X, 확정표는
반올림하면 Y"가 되어 고칠 곳이 바로 보인다.

### 구현 중 픽스처가 잡은 결함 둘

1. `%!cv` 정규식에 `$` 고정이 없어, *"주석이 없는 수치는…"* 같은 **산문 속 언급까지 주석으로 읽었다.**
   줄 끝 고정으로 고쳤다.
2. 주석을 **그 줄에만** 걸었더니, 문장이 여러 줄로 접힌 원고에서 값이 앞 줄에 있어 전부 실패했다.
   현재 `main.tex`는 한 문단이 한 줄이라 드러나지 않는 문제다. 주석이 **자기가 끝맺는 문단**을
   범위로 삼도록 고쳤다(빈 줄·환경 경계·다른 주석에서 멈춤).

> 두 번째는 정밀도를 조금 내준다 — 문단이 넓으면 무관한 값이 우연히 맞을 수 있다. 현재 원고에서는
> 문단=한 줄이라 손실이 없고, 접힌 원고를 받았을 때 **전부 거짓 실패**하는 쪽이 더 나쁘다고 봤다.

### 음성 테스트 — 출력 전문

픽스처는 `scripts/testdata/paper_fixture/`(실 확정표 id를 쓰는 소형 `.tex`). 배포본에 아직
주석이 없어 수치 검사를 실 파일로는 돌릴 수 없기 때문이며, **`main.tex`의 사본이 아니다** —
사본으로 자라면 이 설계가 금지하는 바로 그 두 번째 편집본이 된다.

**(a) 배포본을 저장소에서 한 글자 고침 → 해시 실패** *(실 `paper/main.tex`)*

```
paper.lock:      FAIL (main.tex + 13/13 figures)
tracked values:  0 checked, 0 mismatch
untracked:       327 numerals on 75 lines of section 'Experimental Evaluation' without %!cv
figures:         13 referenced, 13 present

[FAIL] 1 problems:
  - main.tex 해시 불일치 — 0c32aedfa9044d46… != lock 98f947155c30ed76…
    저장소에서 편집했거나 정본이 갱신됐다. 배포본은 편집하지 않는다; 정본이 갱신됐다면
    새 main.tex 와 새 paper.lock 을 함께 받는다 (paper/README.md).
rc=1
```

**(b) 주석 단 줄의 수치를 바꿈 → 수치 불일치**

```
tracked values:  7 checked, 1 mismatch

[FAIL] 1 problems:
  - main.tex:17: q3_bg_search = 26.6 s 가 이 줄에 없다
    줄의 수치: 3.8, 0.7, 30.4, 0.5, 24.1, 1.0
    확정표를 줄의 자릿수로 반올림하면: 26.6 s, 26600 ms
rc=1
```

**(c) 확정표 값이 바뀜(= 새 실험) → 논문의 그 줄이 낡았다고 잡힘** ← **이 도구의 목적**

```
tracked values:  7 checked, 2 mismatch

[FAIL] 2 problems:
  - main.tex:17: q3_bg_search = 24.1 s 가 이 줄에 없다
    줄의 수치: 3.8, 0.7, 30.4, 0.5, 26.6, 1.0
    확정표를 줄의 자릿수로 반올림하면: 24.1 s, 24100 ms
  - main.tex:20: q15_gpu_static_b0.5_p99 = 31200.0 ms 가 이 줄에 없다
    줄의 수치: 7.4, 33.7, 99, 63
    확정표를 줄의 자릿수로 반올림하면: 31 s, 31.2 s, 31200 ms
rc=1
```

두 번째 항목이 **단위 환산과 반올림을 동시에** 거쳐 잡힌 것이다(ms 확정표 ↔ s 논문).

**(d) 확정표에 없는 id를 주석에 적음 → 실패**

```
tracked values:  8 checked, 1 mismatch

[FAIL] 1 problems:
  - main.tex:22: 확정표에 없는 id — q6_a_cumV
rc=1
```

`q6_a_cumV`는 실제로 §4.2가 지적한 **논문은 쓰는데 확정표에 없는 값**이다.

**복원 후 재확인**: 픽스처 `7 checked, 0 mismatch`, 실 배포본 `paper.lock: OK`.

---

## 6. 작업 192 — 재생성 진입점

루트 `Makefile`. **새로 만든 것은 없다** — 이미 있던 스크립트를 잇는다.

```
make figures   fig/plot_*_v11.py 8종 → figures/*.pdf + 사이드카
make check     check_figures(그림↔확정표) + gen_paper_lock(재현) + check_paper(논문↔확정표)
make check-full  위 + 미추적 수치 줄 목록
make paper     paper/build.sh (4패스)
make lock      새 main.tex 수령 후 paper.lock 재동기화
```

현재 `make check` 출력:

```
checked 208 values across 11 sidecars
[OK] ALL FIGURE VALUES MATCH CONFIRMED TABLE (0 mismatches).

parsed 13 figures from main.tex
[OK] committed paper.lock reproduces from main.tex + figures/

paper.lock:      OK (main.tex + 13/13 figures)
tracked values:  0 checked, 0 mismatch
untracked:       327 numerals on 75 lines of section 'Experimental Evaluation' without %!cv
figures:         13 referenced, 13 present
[OK] PAPER MATCHES CONFIRMED TABLE (0 mismatches).
```

### 그림 재생성이 `paper.lock`을 매번 깨뜨리던 문제

matplotlib은 PDF에 `/CreationDate`를 찍는다. **내용이 완전히 같아도 재생성할 때마다 sha256이
바뀐다** — 실측으로 확인했다(같은 35,984 바이트, `/CreationDate`만 다름, 해시 상이). 그대로
두면 `make figures`가 돌 때마다 `paper.lock`이 거짓으로 깨진다.

`SOURCE_DATE_EPOCH`를 고정하면 **바이트까지 동일**해진다(두 번 생성해 동일 해시 확인).
`Makefile`이 이를 export한다.

> **주의**: 현재 커밋된 PDF들은 이 고정 **이전에** 생성됐다. 이 변경 이후 `make figures`를 처음
> 돌리면 그림 해시가 한 번 바뀌고 `make lock` 재동기화가 필요하다. 그 다음부터는 안정적이다.
> **이번에 그림을 재생성하지 않았다** — 범위 밖이다.

---

## 7. 작업 193 — 새 실험 후 절차

`paper/README.md`에 절차로 적었다.

```
실험 → runs/ 에 산출물 → fig/confirmed_values.json 갱신
     → make figures            그림·사이드카 재생성
     → make lock               그림 해시가 바뀌었으므로 재동기화
     → make check              그림↔확정표, 논문↔확정표
     → check_paper.py 가 짚은 줄 목록을 저장소 밖으로 전달   ← 이 작업의 산출물
     → 정본 수정 → 새 main.tex + 새 paper.lock 배치 → make check 재실행
```

주석을 새로 붙일 때는 `gen_cv_worklist.py` → `apply_cv_annotations.py`(정본에서).

---

## 8. 커밋

| 커밋 | 내용 |
|---|---|
| `03e8832` | `paper/` 배포본 · `figures/` 링크 · `Makefile` · 검사기/생성기 5종 · 주석 목록 · 픽스처 · 문서 정정 4곳 · 본 보고서 |

**푸시하지 않았다.** 미푸시 커밋은 `git log origin/ubuntu_gpu_fsrr..HEAD`.

---

## 9. 남은 것 — **사람 판단**

1. **Q4 문단의 값 혼재(§4.1).** L741·L756의 `≈55`는 재실행 전 값이다. 본문 수정이 필요하며,
   수정 후 `%!cv q4_bg_lastV`를 붙이면 그 자리가 이후로는 자동 추적된다. **내가 고치지 않았다** —
   배포본은 편집하지 않는다.
2. **확정표에 없는 산문 수치 ~40건(§4.2).** 값이 없는 것이 아니라 확정표에 올라오지 않은 것이다.
   올릴지, 올린다면 어느 것부터인지는 판단이 필요하다. Q4 계열과 Q6 cumV가 가장 크다.
3. **파생값 항목화(§4.3).** Q1.1의 ΔCPU처럼 "두 항목의 차"를 확정표 항목으로 둘지.
   `q13_*_cumV_ratio` 선례가 있다.
4. **표 셀 주석(§4.4).** `tab:b2`·`tab:c2-ablation`의 70여 값. 위험은 낮지만 미추적의 큰 몫이다.
5. **빌드 확인(§1).** 이 기계에 TeX가 없다. 22페이지·참조 0건은 다른 환경에서 확인해야 한다.
6. **그림 재생성 1회(§6).** `SOURCE_DATE_EPOCH` 고정을 실제로 반영하려면 한 번 재생성하고
   `make lock`을 돌려야 한다. 범위 밖이라 하지 않았다.

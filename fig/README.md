# `fig/` — 그림 값 파이프라인

> **이 디렉터리는 반드시 git에 추적되어야 한다.** 이전에는 세션 스크래치에만 존재하다가 사라졌고,
> 그 결과 게시된 그림의 원시 런과 값 대장을 되짚을 수 없게 됐다
> (`docs/phantom_model_audit.md` 작업 79). `.gitignore`가 이 디렉터리를 제외하지 않는지 확인할 것.

---

## ⚠ 현재 상태: 값은 잠정이다

`confirmed_values.json`의 런타임 실행 유래 값은 **phantom model 오염**의 영향을 받는다 — 스케줄이
이전 세대 모델 어휘를 써서, 명명된 4~7개 중 실제로 실행된 것은 1~2개뿐이었다
(`docs/phantom_model_audit.md`). 재실행 후 값을 갱신하고 `_meta.status`를 제거하기 전까지
**논문에 인용하지 않는다.**

`source_kind: "analytic"`(상한식 평가)과 시뮬레이션 유래 값은 이 오염과 무관하다.

---

## 파이프라인

```
런 산출물(csv/log)  →  confirmed_values.json  →  plot_*.py  →  figure.pdf
                              │                      │
                              │                      └→ figure.values.json   (사이드카)
                              └──────── check_figures.py ◄──────┘
```

- `confirmed_values.json` — **모든 그림 값의 단일 소스.** 여기에 없는 값은 그림에 넣지 않는다.
- `paper_figure_manifest.json` — 논문(`main.tex`)에서 추출한 그림→라벨·절·캡션 매핑.
  **논문 정본은 저장소 밖에 있다**(중복 사본이 갈라지는 것을 막기 위한 의도적 설계).
  `_meta.main_tex_sha256`이 어느 판본을 기술하는지 알려준다. 논문이 바뀌면 새 매니페스트를 받는다.
- `check_figures.py` — 사이드카를 확정표와 대조한다. 불일치 시 `exit 1`.

## 규약

**값은 확정표에서만 온다.** 그림을 그리며 다시 계산하지 않는다. 문서에도 하드코딩하지 않는다.

**사이드카는 캔버스에서 뽑는다.** `bars[i].get_height()`처럼 실제로 그려진 아티스트에서 추출한다.
확정표에서 **복사하면 검사가 자기 자신을 검사하게 되어** 아무것도 잡지 못한다. (이 실수로
`emit_sidecars.py`를 삭제했다.)

**`censored`와 `recovered`는 독립이다.**
- `censored` — 값이 관측 창에 잘렸는가. 시각 부호: 해칭 + 위쪽 화살표 + `≥window`.
- `recovered` — 그 런이 회복했는가. 시각 부호: 색·테두리.
- 반례: Q4의 `search=24`s는 `censored=false, recovered=false`(탐색이 완료됐으나 회복은 없음).
  해칭을 쓰면 "실제론 더 길 수도"로 읽혀 `24 = bound 24`의 tight 주장이 흐려진다.

**`source_kind`**
- `hardware` — 실측 런
- `simulation` — 시뮬레이터 출력
- `analytic` — 측정 상수를 상한식에 넣어 평가한 값

한 그림에 두 종류가 섞이면 마커를 달리하고 범례에 명시한다
(예: `d2_budget_rank`는 평면이 `analytic`, 마커 두 점이 `hardware`).

**절단된 값과 측정된 값을 같은 축에 놓지 않는다.** 방향이 유리하든 불리하든 종류가 다르면 비교가 아니다.

**그림 안 텍스트도 검사 대상이다.** 제목·축 라벨·범주형 틱·범례·주석·하단 문구의 숫자는 전부
확정표 값이거나 **사유가 붙은 allowlist** 항목이어야 한다. 수치 축 눈금은 제외.

## 사용

```bash
python3 fig/check_figures.py --figdir fig
# 기대: checked N values across M sidecars
#       [OK] ALL FIGURE VALUES MATCH CONFIRMED TABLE (0 mismatches).
```

값을 바꾸면 `confirmed_values.json`을 먼저 고치고 그림을 다시 생성한 뒤 checker를 돌린다.
그림만 고치면 checker가 잡는다 — 그게 이 장치의 목적이다.

## 검증 장치를 고칠 때

**음성 테스트를 통과하지 못한 검증 장치는 검증 장치가 아니다.** 고의로 틀린 값·잘못된 기준점·
누락 필드를 넣어 checker가 실패하는지 확인하고, 그 출력을 리포트에 남긴다.

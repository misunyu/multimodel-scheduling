# working set 명명 체계 조사 — base1–base5 vs S1–S10

- 작성: 2026-08-04
- **읽기 전용.** 코드·데이터·원고 미수정. 산출물은 `naming.md` + `facts.json`.
- 대조 원고 리비전: `manuscript/mlforsys_main.tex` **r4** (sha256 `d31d17a0…`)
- 1차 근거: `working_sets.yaml` (커밋 `0c251a4`, 2026-07-15)

---

## 1. 세트 구성 표 (15개)

`rate_factors`는 `working_sets.yaml`의 `rate_factors_by_size` 규칙을 세트 크기에 적용한
결과다 (≤2모델 → small, 3–5 → medium, ≥6 → large). `stratified`는 `stratified_sets` 목록.

### base 계열 (5개)

| set_id | N | 모델 목록(정렬) | VLM | LLM | rate_factors | stratified |
|---|---|---|---|---|---|---|
| base1 | 4 | llama1b, qwen2_vl, resnet50, **yolo11n** | ✓ | ✓ | 1.0 / 2.0 / 3.0 | — |
| base2 | 4 | llama1b, qwen2_vl, resnet50, **yolo11s** | ✓ | ✓ | 1.0 / 2.0 / 3.0 | — |
| base3 | 4 | llama1b, qwen2_vl, resnet50, **yolo11m** | ✓ | ✓ | 1.0 / 2.0 / 3.0 | — |
| base4 | 4 | llama1b, qwen2_vl, resnet50, **yolo11l** | ✓ | ✓ | 1.0 / 2.0 / 3.0 | — |
| base5 | 4 | llama1b, qwen2_vl, resnet50, **yolo11x** | ✓ | ✓ | 1.0 / 2.0 / 3.0 | — |

### S 계열 (10개)

| set_id | N | 모델 목록(정렬) | VLM | LLM | rate_factors | stratified |
|---|---|---|---|---|---|---|
| S1 | 2 | resnet50, yolo11s | — | — | 2.0 / 3.0 / 4.0 | — |
| S2 | 3 | mobilenet_v2, resnet50, yolo11s | — | — | 1.0 / 2.0 / 3.0 | — |
| S3 | 3 | llama1b, resnet50, yolo11s | — | ✓ | 1.0 / 2.0 / 3.0 | — |
| S4 | 3 | qwen2_vl, resnet50, yolo11s | ✓ | — | 1.0 / 2.0 / 3.0 | — |
| S5 | 5 | llama1b, mobilenet_v2, qwen2_vl, resnet50, yolo11s | ✓ | ✓ | 1.0 / 2.0 / 3.0 | — |
| S6 | 6 | llama1b, mobilenet_v2, qwen2_vl, resnet50, yolo11m, yolo11s | ✓ | ✓ | 1.0 / 1.5 / 2.0 | ✓ |
| S7 | 7 | llama1b, mobilenet_v2, qwen2_vl, resnet50, yolo11l, yolo11m, yolo11s | ✓ | ✓ | 1.0 / 1.5 / 2.0 | ✓ |
| S8 | 8 | llama1b, mobilenet_v2, qwen2_vl, resnet50, yolo11l, yolo11m, yolo11s, yolo11x | ✓ | ✓ | 1.0 / 1.5 / 2.0 | ✓ |
| S9 | 5 | mobilenet_v2, resnet50, yolo11m, yolo11n, yolo11s | — | — | 1.0 / 2.0 / 3.0 | ✓ |
| S10 | 7 | mobilenet_v2, resnet50, yolo11l, yolo11m, yolo11n, yolo11s, yolo11x | — | — | 1.0 / 1.5 / 2.0 | ✓ |

**집계**: 15세트 / N 범위 2–8 / 모델 합집합 9종 / stratified 5 · exhaustive 10 /
VLM 포함 10세트 · LLM 포함 10세트 · vision-only 4세트(S1, S2, S9, S10).

## 2. base1–base5의 변주 축 — **관찰된 구성 사실**

다섯 세트의 모델 목록을 교집합·차집합으로 대조한 결과:

- **공통 부분 (5세트 전부에 존재)**: `{llama1b, qwen2_vl, resnet50}` — LLM 1 + VLM 1 + 분류기 1.
- **세트별 고유 모델 (공통을 뺀 나머지)**: 정확히 **1개씩**이며 전부 YOLO11 검출기다.

  | 세트 | 고유 모델 |
  |---|---|
  | base1 | yolo11n |
  | base2 | yolo11s |
  | base3 | yolo11m |
  | base4 | yolo11l |
  | base5 | yolo11x |

- 다섯 고유 모델은 YOLO11의 **n/s/m/l/x 다섯 변형 전부**이고, 변형 하나가 정확히 한 세트에
  대응한다(중복·누락 없음). 세트 번호 순서 base1→base5는 n→s→m→l→x 순이다.
- 그 결과 **다섯 세트는 N=4, VLM·LLM 포함 여부, rate_factors(1/2/3×), 비층화(exhaustive)가
  모두 동일**하고, **검출기 변형 하나만 다르다.** (facts.json의
  `base_all_N_equal_4`, `base_all_have_vlm_and_llm`, `base_all_same_rates`,
  `base_all_exhaustive` 전부 true.)

즉 base 계열은 **다른 축을 고정한 채 검출기 크기 하나만 움직이는 5점 계열**이다. 이는
구성에서 직접 읽히는 사실이며, 그 구성을 택한 의도는 아래 §3에서 확인한 범위를 넘어
문서화되어 있지 않으므로 여기서 추정하지 않는다.

## 3. "base" 명명의 유래

### 3-1. 이름이 처음 등장한 지점

`base1`–`base5`라는 식별자는 **커밋 `0c251a4`** (2026-07-15 14:50, "Pilot collection: the
approach works, GPU drift is gone, and S8 needs the full data")가 `working_sets.yaml`을
**신규 생성**하면서 처음 등장한다.

- 그 커밋의 부모(`0c251a4^` = `49b02f6`) 트리에서 `base1`을 검색하면 **0건**이다
  (`git grep -l "base1" 0c251a4^ -- '*.yaml' '*.md' '*.py'` → NONE). 즉 이름 자체가 이
  커밋에서 만들어졌다.
- 커밋 메시지 말미: *"working_sets.yaml is committed as the canonical 15-set definition for
  schedule generation."* — 파일의 역할만 밝히고 `base`라는 단어는 설명하지 않는다.

### 3-2. 파일 안의 유일한 근거 주석

`working_sets.yaml:1-2, 15`:

```yaml
# Working set definitions -- the canonical source for schedule generation and
# re-collection. 15 sets: 5 existing 4-model sets + 10 new.
...
  # ---- existing 5 (4-model) ----
```

→ **`base` = "이미 존재하던(existing) 5개 4-모델 세트"**를 가리킨다. 이것이 저장소에 있는
유일한 직접 근거다.

### 3-3. "existing 5"의 실체 — 재수집 이전 스케줄

주석이 말하는 "기존 5개"는 재수집 이전 스케줄 파일에 실재한다.

`schedules_cpu_gpu.yaml` / `schedules_cpu_npu.yaml` (각 160 combos, 커밋 `81cae87`
"data: static profile, contention datasets, and trained placement predictors")를 파싱하면
**정확히 다음 5개 4-모델 세트**가 각각 32 combos씩 들어 있다:

```
llama1b, qwen2_vl, resnet50, yolo11n     (32 combos)
llama1b, qwen2_vl, resnet50, yolo11s     (32)
llama1b, qwen2_vl, resnet50, yolo11m     (32)
llama1b, qwen2_vl, resnet50, yolo11l     (32)
llama1b, qwen2_vl, resnet50, yolo11x     (32)
```

이는 base1–base5의 모델 목록과 완전히 일치한다. 다만 그 파일들은 세트를
`combination_N`으로만 열거하고 **세트 id를 부여하지 않는다** — 즉 이 5개 구성은
`working_sets.yaml` 이전부터 존재했으나 **이름은 없었다.**

(`model_schedules.yaml`은 이 중 한 세트(`…yolo11x`)만 8 combos로 담고 있다.)

### 3-4. 관련 서술

`PILOT_REPORT.md:86`:

> 파일럿 예측기(**기존 5×4모델 학습**)로는 8모델·mobilenet_v2 세트를 예측할 수 없다.

파일럿 예측기가 이 5개 세트로만 학습됐음을 확인해 준다(= 후속 확장의 출발점이었다는
사실 근거).

### 3-5. 판정

- **"기존 5개 4-모델 세트"라는 지시 대상은 문서화되어 있고 실물로 확인된다**
  (`working_sets.yaml:2,15` 주석 + `schedules_cpu_{gpu,npu}.yaml`의 실제 구성 + `PILOT_REPORT.md:86`).
- 그러나 **하필 `base`라는 단어를 고른 이유를 설명한 서술은 저장소 어디에도 없다.**
  커밋 메시지·주석·보고서 모두 단어 선택을 해설하지 않는다.
- 요약: **지시 대상은 확정, 어원 서술은 없음.**

## 4. S1–S10 대비 요약

base 계열은 N=4·VLM+LLM 포함·1/2/3× rate·비층화를 **전부 고정**한 채 YOLO11 검출기 변형
하나만 n→x로 바꾸는 계열이고, S 계열은 그 고정된 축들을 각각 변화시키는 쪽이다. S 계열에서
N이 2–8로 퍼지면서(base는 4 고정) rate 규칙도 함께 갈리는데, S1만 small 규칙(2/3/4×)을,
S6–S8·S10이 large 규칙(1/1.5/2×)을 받는 반면 base 5개는 전부 medium(1/2/3×)이다. 생성 모델
구성도 base는 5세트 모두 LLM·VLM 동시 포함으로 균일한 데 반해, S 계열은 vision-only 4개
(S1, S2, S9, S10), LLM 단독 1개(S3), VLM 단독 1개(S4), 둘 다 포함 4개(S5–S8)로 갈라진다.
층화 샘플링도 S 계열에만 적용된다(S6–S10 5개, base는 0개). 검출기 변형 축에서 보면 base는
다섯 변형을 세트 간에 하나씩 나눠 갖는 반면, S6–S8·S10은 한 세트 안에 여러 변형을 동시에
넣어 동시성을 키우는 방식이다.

## 5. 일관성 확인 — **모순 0건**

| 확인 대상 | 값 | 표 1과 대조 |
|---|---|---|
| 원고 `:115` "**15 workload sets**" | 15 | ✓ |
| 원고 `:115` "**two to eight** concurrent instances" | 세트 N 범위 **2 (S1) – 8 (S8)** | ✓ **일치** |
| 원고 `:115` "**Nine models**" + 목록 | 합집합 9종(llama1b, mobilenet_v2, qwen2_vl, resnet50, yolo11 n/s/m/l/x) | ✓ |
| 원고 `:115` "vary … the **detector variant**" | base1–5가 검출기 변형만 바꾼 계열(§2) | ✓ |
| 원고 `:115` "include **vision-only controls**" | S1, S2, S9, S10 | ✓ |
| 원고 `:118` "**Ten** sets exhaustively / remaining **five** stratified" | exhaustive 10 / stratified 5 | ✓ |
| 원고 부록 **D** 커버리지 표의 Set·N 열 | 15세트 N 전부 일치 (S4=3, base1–5=4, S9=5, S10=7 …) | ✓ |
| 원고 부록 **A** per-set 표의 Set·N 열 | 15세트 전부 일치 | ✓ |
| `fig_divergence` y축 라벨 (`make_figures.py`의 `SET_N`) | 15세트 N 전부 `working_sets.yaml`과 일치 | ✓ |
| 동 `SET_ORDER` | 15세트를 빠짐없이 포함 | ✓ |

**부수 확인**: r4에서 부록 소절 순서가 **A, B, C, D, E**로 정렬되어 있고
(coverage = D, sweep = E), 본문 참조도 `:118` → Appendix D, `:177` → Appendix E로 맞는다.
r2에서 지적했던 "A, B, C, **E, D**" 역순 문제는 해소되었다.

**"two to eight" 검산 상세**: N=2는 S1 단 하나, N=8도 S8 단 하나이므로 양 끝값이 모두
실재한다. 중간값 3, 4, 5, 6, 7도 각각 존재해(3: S2·S3·S4 / 4: base1–5 / 5: S5·S9 /
6: S6 / 7: S7·S10) 2–8 구간에 빈 값이 없다.

---

## UNKNOWN

- **`base`라는 단어 선택의 어원** — §3-5. 지시 대상("기존 5개 4-모델 세트")은 확정되었으나,
  단어 자체를 설명하는 서술이 저장소에 없다. 부록 문안에서 어원을 주장하려면 이 저장소
  밖의 근거가 필요하다.

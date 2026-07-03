# FINDINGS — Oracle / large-object-fraction 구현 확인 (read-only)

대상 문구 (현행 `paper/main_vision.tex`):
- L194: "Starting from All-GPU, it moves one additional stream to the NPU at each
  step, **in descending order of each stream's large-object fraction.**"
- L196: "moves only the single stream with the highest large-object fraction …
  realizable online, since each stream's large-object fraction **can be
  estimated from the detector's own recent outputs.**"
- (task가 인용한 옛 문구) "large-object fraction, **which we compute offline from
  the dataset.**"

정규 런 = `accv_experiments/results/rev30_clean_resnet/` (`phase_rev30_clean.py`,
NPU mode="single", threads=4, 3–10 reps). 코드/`.tex` 미수정.

---

## T1 — large-object fraction의 정의·계산

```
정의 = GT annotation (오프라인) / 정적(로그당 1값) / large = COCO large bin area ≥ 96²(=9216px²)
       / 분모 = 전체 instance 수 (n_large / n_total)
소스 = accv_experiments/scripts/step_e_size_classification.py
```
- (a) **무엇으로**: ground-truth annotation. detector 출력 아님.
  `step_e_size_classification.py:69-70` `if "area" in a … ar = float(a["area"])`
  (COCO GT의 area 필드; 없으면 bbox w*h).
- (b) **정적/동적**: **정적**. 로그(=stream)당 1개 고정값. 전체 GT로 1회 계산하여
  `results/step_e_size_classification.csv`에 저장. window/run마다 재계산 없음.
- (c) **large 정의**: COCO large bin. `step_e:38-39` `SMALL_MAX=32**2; MEDIUM_MAX=96**2`,
  `:82` `large_mask = areas >= MEDIUM_MAX` → area ≥ 96². 다른 임계 아님.
- (d) **분모**: large instance 수 / 전체 instance 수.
  `step_e:96` `"pct_large_count": round(n_l / n_obj, 4)`. (frame 비율 아님.
  면적기반 `pct_large_area = a_l/a_tot`(:99)도 계산되나 정렬엔 미사용 — 아래 T2 참조.)

**그러나 — oracle가 실제로 쓰는 순서는 이 fraction을 런타임 정렬한 게 아니라
사전선언된 하드코딩 리스트다** (T2에서 상술). T1의 fraction은 그 하드코딩 라벨이
GT 크기분류와 일치함을 보장하는 근거로만 쓰임 (`results/audit_sizegroup_summary.md`).

## T2 — Oracle candidate 정렬·선택

```
정렬 키   = 사전선언 하드코딩 MOVE_ORDER = [3, 21, 22, 2] (large-dominant 먼저, 그다음 small)
            → fraction의 "런타임 내림차순 정렬"이 아니며, 큰-그룹 내부 순서는 내림차순도 아님
선택 기준 = 측정된 mean worst-stream sAP 최대 (오프라인 진단)
온라인 관측 사용 = 아니오
```
- (a) **정렬 키**: `phase_rev30_clean.py:64` `MOVE_ORDER = [3, 21, 22, 2]
  # NPU-first order: large-object streams first (rev29-identical)`.
  ratio r 후보 배치는 `:106-109`
  ```python
  def placement_for_ratio(r):
      npu_set = set(MOVE_ORDER[:r])
      return ["NPU" if sid in npu_set else "GPU" for sid in PANEL4]
  ```
  → r=0 All-GPU, r=1 {3}→NPU, r=2 {3,21}, r=3 {3,21,22}, r=4 All-NPU.
  이는 **2-tier 그룹핑**(large-dominant {3,21} 먼저, 그다음 small-dominant {22,2})이며,
  연속값 per-stream large-fraction의 **엄격한 내림차순 정렬이 아님**. (런타임 argsort/
  정렬 코드 없음 — 하드코딩 상수.)

  ▲ **내림차순 불성립 증거** (`results/step_e_size_classification.csv`, `pct_large_count`):
  | sid | log[:8] | pct_large_count |
  |---|---|---|
  | 21 | e9a96218 | **0.2689** (최대) |
  | 3  | 2d12da1d | 0.2559 |
  | 22 | f1008c18 | 0.1066 |
  | 2  | 1d676737 | 0.0571 |

  엄격 내림차순이라면 **[21, 3, 22, 2]**여야 하나, MOVE_ORDER는 **[3, 21, 22, 2]**
  (3과 21이 뒤바뀜; 21이 더 큰데 3이 먼저). 면적기반(`pct_large_area`)으로 정렬하면
  [22,3,2,21]로 이 역시 불일치.

- (b) **선택 기준**: 측정된 worst-stream sAP 최대. `phase_rev30_aggregate.py:66-68`
  ```python
  mw = {r: sub[sub["ratio"] == r]["worst_sap"].mean() for r in range(5)}
  o_r = max(mw, key=mw.get)        # oracle ratio = max mean worst-stream sAP
  ```
  worst_sap는 per_stream_sap의 실제 측정 COCO sAP(`phase_rev30_clean.py:141,152`).
- (c) **온라인 관측 사용**: 아니오. 후보군 순서는 오프라인 GT 크기분류(step_e)에서 옴,
  선택은 런에서 실측한 oracle sAP. 런타임 detector 관측으로 순서를 정하는 코드 없음.

## T3 — one-stream split의 정체 + 0.103 출처

```
candidate   = MOVE_ORDER[:1] = {sid 3} → NPU.  (large-dominant 스트림이지만,
              pct_large_count 최대 스트림은 sid 21이며 sid 3 아님 → "highest" 문구와 불일치)
0.103 출처  = results/rev30_clean_resnet/rev30_oracle_by_contention.csv : 행 RES1
              split1_worst = oracle_worst = 0.1026 (≈0.103), gpu_skip=24.3, ratio=1
              tab:main 행은 phase_rev30_aggregate.py:tabmain_row (resnet_k==1)에서 생성
9/10 runs   = 동일 candidate (ratio=1 = one-stream split). 예
```
- (a) one-stream split = `placement_for_ratio(1)` = {3}→NPU (GGNN, sid3만 NPU).
  large-dominant 스트림은 맞지만 **패널 내 pct_large_count 최대는 sid 21(0.2689)**,
  sid 3(0.2559)이 아님. 즉 "the single stream with the **highest** large-object
  fraction"을 count 기준으로 엄밀히 보면 sid 21이어야 하나 코드는 sid 3을 옮김.
  (set 특성상 r=2의 {3,21}은 순서 무관 → 3-vs-21 차이는 **오직 one-stream split에서만**
  드러나며, 그게 바로 논문이 강조한 0.103 지점.)
- (b) **0.103 출처**: `results/rev30_clean_resnet/rev30_oracle_by_contention.csv` 행 `RES1`
  (resnet_k=1, gpu_skip_allgpu=24.3): `allgpu_worst=0.0979`(≈0.098),
  `allnpu_worst=0.083`, `oracle_worst=0.1026`, `oracle_ratio=1`, `split1_worst=0.1026`,
  `oracle_worst_std=0.0031`(≈0.003). 0.1026→반올림 0.103. tab:main(L320 ResNet행)·
  L422·L524의 0.103과 정합.
- (c) **9/10 runs**: 같은 행 RES1의 `pick_dist = {"0": 1, "1": 9, "2":0,"3":0,"4":0}`
  → ratio=1(one-stream split, sid3→NPU)이 10 reps 중 9회 선택. L524 "selected in
  nine of ten runs"와 정합. 동일 candidate.

---

## 판정

```
원고 L194 "in descending order of each stream's large-object fraction"
  → 불일치 (부분).
    실제: 오프라인 GT 크기분류(size_label: large-dominant/small-dominant)에 따른
    사전선언 2-tier 하드코딩 순서 MOVE_ORDER=[3,21,22,2] (large 그룹 먼저).
    연속값 large-fraction을 런타임 내림차순 정렬하지 않으며, large 그룹 내부
    순서 [3,21]은 pct_large_count 내림차순([21,3])과 반대.

원고 (옛) "which we compute offline from the dataset"
  → 일치. fraction/라벨은 step_e가 GT annotation으로 오프라인 1회 계산
    (step_e_size_classification.py → step_e_size_classification.csv).

원고 L196 (현행) "estimated from the detector's own recent outputs … realizable online"
  → 불일치. 그런 온라인/detector기반 추정 코드는 없음. 구현은 전적으로 오프라인
    GT 하드코딩 순서. (이 문장은 '실현 가능성' 주장이지 구현 서술이 아님.)
```

**코드가 실제로 하는 일 (한 줄):** Oracle은 N+1개 후보를, 오프라인 GT 크기분류로
미리 정한 large-dominant-우선 고정 순서(MOVE_ORDER=[3,21,22,2])대로 스트림을 하나씩
NPU로 옮겨 만들고, 그중 **실측 worst-stream sAP가 최대**인 배치를 고른다. one-stream
split은 그 순서의 첫 스트림(sid 3)을 옮긴 것이다 (단, 패널 내 large-fraction 최대는
sid 21이라 "highest fraction" 문구와 엄밀히는 어긋남).

### UNCLEAR / 미발견
- sid 3 대신 (count-최대인) sid 21을 옮긴 one-stream split의 sAP는 **측정되지 않음**
  (해당 GNGN 셀 부재) → 0.103이 "엄밀한 top-1 fraction"이었으면 달라졌을지는 UNCLEAR.
- "descending order" 문구가 (i) 큰/작은 2-tier 의미인지 (ii) 엄밀 연속 정렬 의미인지는
  원고상 모호. 코드는 (i)만 보장하고 (ii)는 보장하지 않음.

### 권고 (수정은 사용자 몫, 코드 일치 위주)
- L194: "in descending order of each stream's large-object fraction" →
  "moving large-object-dominant streams first (stream order pre-declared offline
  from each log's ground-truth large-object fraction)" 류로, 2-tier·오프라인·GT임을
  드러내는 표현 권장.
- L196의 온라인/detector-output 실현가능성 문구는 구현과 무관하므로, 구현을 기술하려면
  "computed offline from the dataset's ground-truth annotations"가 정확. 온라인 실현
  가능성은 별도 '가능성' 주장으로 유지하되 구현 서술과 구분 권장.
```
```

## Compliance
Read-only. 모든 주장에 파일:라인 + step_e/rev30 저장 CSV 근거. 추론 항목은 UNCLEAR로
명시. `.tex`·코드·결과 미수정. 산출물 = 본 파일.

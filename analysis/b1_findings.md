# B1 Findings — 크기 의존성의 메커니즘 근거 (temporal IoU decay + 양자화 오류 분해)

리뷰어 B1("현상만 보고, 근거 없음")에 대한 두 개의 메커니즘 프로브 결과.
**정직성 규칙 준수**: 가설과 반대로 나온 부분은 그대로 보고함.

산출물:
- Task 1: `analysis/b1_temporal_iou_decay.csv`, `analysis/b1_staleness_sap_proxy.csv`,
  `analysis/b1_staleness_pred.csv`, `analysis/b1_iou_decay.pdf`
- Task 2: `analysis/b1_quant_decomposition.csv`, `analysis/b1_quant_conf_shift.csv`,
  `analysis/b1_quant_conf_shift.pdf` (덤프: `analysis/b1_dets/{gpu_fp32,npu_int8}.npz`)
- 스크립트: `analysis/b1_temporal_iou_decay.py`, `analysis/b1_staleness_delay_model.py`,
  `analysis/b1_dump_detections.py`, `analysis/b1_quant_decompose.py`

측정 대조 기준 (Table 3 / `c2_per_size_losses.csv`):
- 양자화 Q_b: small −0.008(−48.4%), medium −0.036(−19.7%), large +0.001(≈0)
- staleness L_b: small −0.001(−8.0%), medium −0.030(−16.0%), large −0.098(−20.5%)
- baseline GPU sAP A_b: small 0.0159, medium 0.1839, large 0.4768

---

## 1. 핵심 수치와 판정

### Task 1 — Temporal IoU decay (GT-only, 검출기 무관, 24 로그, 동일 track IoU)

**(1a) 소박한 기하 직관은 반박됨 (가설 불지지, 오히려 반대).**
지시문/상식 가설 = "가까운(큰) 객체는 ego-motion 픽셀 변위가 커서 self-IoU가
더 빨리 감소". 실측은 반대:

| δ(ms) | small self-IoU | medium | large |
|---|---|---|---|
| 33.3 | 0.827 | 0.867 | 0.920 |
| 166.7 | 0.561 | 0.640 | 0.734 |

- **큰 객체가 모든 δ에서 self-IoU 최고·가장 느리게 감쇠**. IoU는 박스 크기에
  상대적이라 큰 박스가 같은 픽셀 변위를 더 잘 견딤.
- COCO 매칭 임계(0.5) 아래로 떨어지는 비율(δ=167ms): small **0.409**, medium 0.293,
  large 0.150 — 소형이 대형의 ~2.7배.
- **판정: 원시 self-IoU 기준 large 최속 감쇠 = 불지지 (small이 최속).**

**(1b) 그러나 측정 L_b 순서는 데이터셋 기하로 설명됨 (정제 메커니즘, 지지).**
sAP는 IoU 임계 10개(.50:.05:.95)를 평균한다. stale 검출을 box_t 재사용으로 보고
임계별 보존율 ret(δ)=mean_τ P(self-IoU≥τ)를 계산, 프록시 A_b·(1−ret(δ)):

- **측정 L_b 순서(large>medium>small)를 모든 δ에서 재현 (5/5).** (`b1_staleness_sap_proxy.csv`)
- 뒤집힘의 원인: 큰 객체는 (i) 회복 가능 sAP 질량의 대부분 보유(A_large 0.48 vs
  A_small 0.016), (ii) 높은 IoU 영역(fresh 0.92)에 살아 0.75/0.80/0.85/0.90 임계가
  촘촘 → 완만한 감쇠라도 여러 임계를 가로질러 sAP를 크게 잃음. 소형은 이미
  검출기 바닥·저-IoU라 staleness가 제거할 sAP가 거의 없음.

**(1c) miss-rate + 지연분포 가중 프록시 (재구성, GPU 미사용).**
`per_stream_sap` 페어링 루프를 24-시퀀스 실제 프레임 레이아웃에 대해 재현하고,
24-thread thrash 상태(총지연 36.5ms, skip 55%, core 10.7 + postproc 25.8±14.2ms)에
캘리브레이션한 지연 트레이스로 staleness δ 분포를 뽑음.
L_b_pred = P(miss)·E_δ[A_b(1−ret(δ))]:

| 시나리오 | large | medium | small | 순서 | 크기(large,med 2× 이내) |
|---|---|---|---|---|---|
| S0: δ=1 점질량 (P=0.55) | −0.030 (0.31×) | −0.021 (0.73×) | −0.0024 (1.87×) | ✅ | ❌ |
| S1: 캘리브 지연모델 | −0.108 (1.10×) | −0.062 (2.09×) | −0.0068 (5.19×) | ✅ | ❌ |
| **측정 L_b** | −0.098 | −0.030 | −0.0013 | | |

- **순서는 두 시나리오 모두 재현.** large 절대값은 S1에서 근접(−0.108 vs −0.098).
- **small/medium는 과대예측(5.2×, 2.1×) → 전 크기 크기재현은 실패.**
- 과대예측 원인(그 자체가 통찰): small/medium sAP는 이미 recall-제한이라
  localization 감쇠가 이미 놓친 검출과 겹쳐 한계 손실이 작음 → Task 2의
  "소형 손실 = score/recall 경로"와 정합.
- **결론: 본문은 "순서 재현(robust)"만 주장.** 무가중 δ2 우연일치(−0.108 vs −0.098)는
  본문에서 제외(캘리브 P(miss)=1.0이 측정 skip=55% 정의와 달라 신뢰 불가).

### Task 2 — 양자화 per-size 오류 분해 (FP32 GPU vs INT8 NPU, isolated offline, 24 로그)

덤프 결정론 확인: NPU 2-rep, max_box_diff=0, max_score_diff=0 (**PASS**, rev18 std=0 재확인).
FP32 127,796 dets / INT8 108,338 dets (INT8이 ~15% 적음 = 양자화 recall 손실).
분모 = FP32가 IoU≥0.5로 맞춘 GT 객체. 4분류 (`b1_quant_decomposition.csv`):

| bin | (i) match+score유지 | (ii) match+score하락 | (iii) 미검출(recall/score) | (iv) localization | score경로(ii+iii) | loc경로(iv) |
|---|---|---|---|---|---|---|
| small | 14.2% | 30.1% | **51.3%** | 4.4% | **81.5%** | 4.4% |
| medium | 29.3% | 48.0% | 17.0% | 5.7% | 65.0% | 5.7% |
| large | **67.2%** | 27.7% | 3.9% | 1.2% | 31.7% | 1.2% |

confidence shift (매칭쌍): small **−0.088**, medium −0.086, large **−0.043**.

**판정 (가설 지지):**
- **소형 손실은 압도적으로 score/recall 경로**(81.5%), 그중 절반이 완전 미검출(iii 51.3%).
  **localization(iv)은 4.4%로 무시할 수준.** → "소형은 score가 INT8 노이즈에 취약해
  매칭 실패가 지배적" 가설 지지.
- **미검출(iii) 순서 small 51.3% > medium 17.0% > large 3.9%** = Q_b 크기순서
  (−48%/−20%/≈0)와 정확히 일치.
- **Q_large≈0의 이유 = (i)의 지배**: 대형은 67% 박스+score 모두 유지, 미검출 3.9%·
  localization 1.2%로 양자화가 제거하는 대형 AP가 거의 없음.
- **메커니즘**: INT8은 confidence를 일률적으로 낮춤(대형에서도 −0.043). 소형은 baseline
  confidence가 CONF=0.25 문턱 근처(FP32 mean 0.476)라 −0.088 하락이 다수를 문턱 아래로
  밀어냄 → recall 붕괴. 대형은 0.80에서 시작해 여유가 커 같은 하락이 미검출을 거의
  유발 안 함. **localization(박스 IoU) 저하는 전 크기에서 미미(1–6%)** — 양자화는 박스를
  거의 안 움직이고 점수를 움직인다.

**두 메커니즘의 상보성**: 양자화 = confidence/recall 축(소형 집중), staleness =
localization/temporal 축(대형 집중). 크기 의존성의 방향이 반대인 것이 자연스럽게 설명됨.

---

## 2. Sec 4.3 삽입 후보 문장 (영문 LaTeX)

### (A) 양자화 — 지지, 단일 버전
```latex
To move beyond phenomenology, we match FP32 (GPU) and INT8 (NPU) detections
per frame against the ground truth and decompose, for every object the FP32
model localizes, what INT8 changes (Fig.~\ref{fig:quant_conf}). The
size-dependence of the quantization penalty $Q_b$ is a \emph{confidence}, not a
localization, effect: INT8 uniformly depresses detection scores (mean shift
$-0.088$ small, $-0.086$ medium, $-0.043$ large), and localization degradation
(IoU falling below $0.5$ with the box otherwise intact) accounts for only
$1$--$6\%$ of degraded objects at every size. Small objects sit near the
$0.25$ confidence floor (FP32 mean $0.48$), so the score drop pushes $51\%$ of
them below threshold into missed detections; large objects start at $0.80$ with
ample margin, so $67\%$ survive with box \emph{and} score intact and
$Q_{\text{large}}\!\approx\!0$. The fraction lost to this recall path
($51\%/17\%/4\%$ for small/medium/large) orders exactly as $Q_b$
($-48\%/-20\%/\approx\!0$).
```

### (B) Staleness — **버전 1 (순서만; 권장, 현재 결과 기준)**
```latex
The staleness penalty $L_b$ has the opposite size profile, and a
detector-independent geometric probe explains its \emph{ordering}. Computing the
self-IoU of each tracked ground-truth box across a temporal offset $\delta$
(Fig.~\ref{fig:iou_decay}) shows that raw localization actually decays
\emph{fastest} for small objects (self-IoU $0.83\!\to\!0.56$ over $167$\,ms
vs.\ $0.92\!\to\!0.73$ for large), because IoU is scale-relative---so the naive
"large objects move more'' intuition is reversed. What matches the measured
ordering is how sAP aggregates over the dense high-IoU thresholds: weighting the
per-$\delta$ retention $\mathrm{ret}(\delta)=\mathbb{E}_\tau\,P(\text{self-IoU}\!\ge\!\tau)$
by the baseline mass $A_b$ yields a loss proxy $A_b\,(1-\mathrm{ret})$ whose
size ordering (large $>$ medium $>$ small) reproduces $L_b$ at every offset.
Large objects dominate $L_b$ because they carry most of the recoverable sAP and
live in the high-IoU regime where even mild temporal decay crosses several
COCO thresholds.
```

### (B') Staleness — **버전 2 (크기까지; 조건부, 미채택)**
> 사용 조건: 지연분포 재구성이 전 크기 크기재현에 성공했을 때만. **현재 미충족**
> (small/medium 과대예측). 참고용으로만 남김; 본문 채택 금지.
```latex
Reconstructing the served-detection staleness distribution of the 24-thread
regime and weighting the retention proxy by the measured miss rate predicts
$L_b^{\text{pred}}=P(\text{miss})\,\mathbb{E}_\delta[A_b(1-\mathrm{ret}(\delta))]$
of $-0.11/-0.06/-0.007$ for large/medium/small, in agreement with the measured
$-0.098/-0.030/-0.001$.
```
*(注: 이 문장은 medium 2.1×·small 5.2× 과대예측을 숨기므로 정직성 규칙상 부적합.)*

### (C) 소박한 기하 직관 선제 반박 — 한 문장 (지시대로 반드시 포함)
```latex
Notably, this ordering is \emph{not} explained by larger objects displacing more
in pixels: because IoU is scale-relative, the per-object self-IoU of small
ground-truth boxes decays fastest of all size bins, and the large-object
penalty arises only once the metric's high-IoU-threshold weighting is accounted
for.
```

---

## 3. 본문 vs 부록 배치 제안

- **본문 (Sec 4.3)**: 양자화 conf-shift 그림 `b1_quant_conf_shift.pdf` 1개 + 문장 (A),(C).
  리뷰어 B1의 "양자화 근거" 요구에 가장 직접적이고, 3-패널 히스토그램이 압축적이며
  결정적(소형 INT8 점수가 0.25 문턱에 쌓임이 육안으로 보임). 4분류 표는 인라인 소형 표 또는
  캡션에 핵심 수치(51/17/4% 미검출)만.
- **부록**: temporal decay 3-패널 그림 `b1_iou_decay.pdf` 전체 + 지연가중 프록시 표
  (`b1_staleness_pred.csv`). 본문에는 문장 (B)만 두고 "see Appendix, Fig X"로 포인터.
- 근거: 그림 2개를 모두 본문에 넣으면 과함. 양자화 쪽이 리뷰어 지적의 무게중심이고,
  staleness는 순서-설명이라 문장+부록그림으로 충분.

---

## 4. 한계 (원고에 명시할 것)

1. **GT decay는 검출기 무관 기하 분석** → 측정 L_b의 절대값 예측이 아니라
   **순서/상대 규모**의 설명. stale 검출을 "이전 프레임 박스 완벽 재사용"으로 이상화
   (검출기 노이즈·NMS 재배열 제외).
2. **지연가중 프록시는 small/medium 과대예측** (recall-제한 baseline 미반영). large 절대값
   근접(1.10×)은 부분적으로 우연일 수 있어 본문에서 크기 주장 안 함.
3. **지연분포 재구성의 모델 의존성**: iid 정규 postproc 지연 가정(thrashing 자기상관 무시),
   캘리브 P(miss)=1.0이 측정 skip=55%(추론-초과-주기 비율)와 정의가 달라 괴리 존재.
4. **conf-shift 그림은 매칭 생존자만 표시** → 문턱 아래로 사라진 소형 51%가 빠져 실제
   점수 하락을 과소표현. 캡션에 명시 필요.
5. **양자화 분해 분모 = FP32-매칭 객체**: 소형 n=8,818 (소형 GT 92,476의 9.5%만 FP32가
   맞춤). 소형 통계는 "검출 가능한 소형"이라는 작고 쉬운 부분집합에서 나온 값.
6. 두 프로브 모두 YOLOv11s·640·CONF 0.25·IoU 0.45·Argoverse-HD val 조건에 한정.

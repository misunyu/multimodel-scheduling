# 패치 문안 v3 — §Q1 · §II 두 문장 (적용은 사람)

작성 2026-07-28. 근거: `q1_workload_attribution.md`. **`main_patched.tex` 현행 문장을 인용해 대조 가능하게.**
v2(`maintex_patch_text_v2.md`)는 그대로 유효하며, 이 문서는 v2가 **원문 유지**로 남겨둔 두 문장만 다룬다.

---

## 1. §Q1 — 분기 **B** (4종이 아님)

**현행** (`main_patched.tex` §Q1):
> We evaluate this behavior across diverse contention scenarios using **four** foreground models.

**교체안** (개수 단언 제거 + 시나리오별 구성 명시):
> We evaluate this behavior across three contention scenarios that differ in workload: the two
> misprediction scenarios use three foreground vision models (YOLO11s, ResNet50, MobileNet-v2) with a
> background LLaMA-1B, and the infeasible-candidate scenario uses four heavier foreground models
> (YOLO11x, YOLO11l, YOLO11m, ResNet50) with no background model. Table~\ref{tab:scenario-config} states
> both configurations.

**근거**: 이 절의 그림 `bounded_recovery_analysis.pdf`는 단일 실험이 아니라 **Q3·Q5·Q4 집계**다
(`fig/confirmed_values.json`의 8개 값 출처 = `b2rep_q3`, `b2rep_q5`, `q4_bsf_out`). 전경은 각각 3·3·4종.

### 1.1 파급 — 바운드 귀속 (수치는 바꾸지 않는다)

현행이 33 s 바운드와 관측 21.0–30.0 s를 보고한다면, per-scenario 바운드가 **서로 다르다**는 점을 밝혀야
한다(δ가 시나리오마다 다르기 때문):

| 시나리오 | 바운드 | δ 근거 |
|---|---|---|
| Q3 | **26 s** | $T+k(T_v+\delta)$, δ = LLM 이전 2.9 s |
| Q5 | **26 s** | 동일 |
| Q4 | **24 s** | $T+N_{cand}(T_v+\delta)+\delta$, δ = vision in-place 1 s |

**캡션 보강안** (그림 캡션에 한 문장):
> The bound is instantiated per scenario, since the transition cost $\delta$ differs by what a candidate
> relocates: 26\,s for the misprediction scenarios, where recovery moves the generative model
> ($\delta\approx2.9$\,s), and 24\,s for the infeasible-candidate scenario, where every transition is an
> in-place vision swap ($\delta\approx1$\,s).

> **주의**: 관측 21.0–30.0 s가 어느 시나리오 조합의 범위인지는 `confirmed_values.json`에 **그 범위값
> 자체가 없어 미상**이다. 범위를 그대로 둘지 per-scenario로 분해할지는 사람 판단.

### 1.2 `tab:scenario-config` 추가 행

Q3·Q5·Q4 세 행은 **이미 표에 있다**(v2 A-2). §Q1은 그 세 실험을 묶어 보는 절이므로 **추가 행 불필요**.

---

## 2. §II — 개수는 맞고 **구성이 Table~I 밖**

**현행** (`main_patched.tex` §II):
> ...measurements from our CPU--GPU testbed with **four** concurrent DNN applications.

개수 4는 **참**이다(4월 실행 CSV에서 view1–4 전부 실측 산출, phantom 아님). 그러나 그 네 모델은
**mnasnet, squeezenet1.0-12, resnet50, resnext50**로, **Table~I에 있는 것은 resnet50 하나뿐**이다.

### 2-A. 권장안 — 모델 이름을 걸지 않고 동기 그림임을 명시

> ...measurements from our CPU--GPU testbed with four concurrent DNN applications. This motivating
> measurement predates the model set of Table~\ref{tab:models} and uses four CPU-class vision models; it
> illustrates the failure modes the objective must address, and none of the later results depend on it.

**이유**: §II는 **문제 제기용 동기 그림**이고 성능 주장을 뒷받침하지 않는다. 사실을 밝히되 §IV 결과와
분리하면 독자가 Table~I 대조를 시도하지 않는다.

### 2-B. 대안 — 모델을 명시

> ...measurements from our CPU--GPU testbed with four concurrent vision applications (MnasNet,
> SqueezeNet, ResNet50, ResNeXt50) running on CPU.

**이유**: 가장 정확하다. 다만 Table~I에 없는 이름 셋이 §II에 등장해 독자가 표를 찾게 된다 → 그 경우
Table~I 각주나 §II 본문에 "이 그림은 다른 모델 집합을 쓴다"는 단서가 함께 필요하다.

### 2-C. 그림 캡션

현행 캡션(*"Comparison of failure impact under increased input rates..."*)은 모델을 언급하지 않으므로
**수정 불필요**. 2-B를 택하면 캡션에도 같은 이름을 넣을지 결정할 것.

### 2.1 `tab:scenario-config` 추가 행 (§3.1 각주 전제 유지용)

이 실험은 deadline이 **`1000/infps` 폴백(≈333 ms)** 이라 per-model 규약이 아니다. Table~I 각주
(*"applies except where Table~III states otherwise"*)를 그대로 두려면 **이 행을 추가**해야 한다:

```latex
Motivating measurement & MnasNet, SqueezeNet, R50, ResNeXt50 & --- & $1000/\lambda_i$ \\
```

- 표 캡션이 "공통 기본값에서 벗어나는 실험"으로 범위를 한정한다면, 이 행은 **벗어나는 사례이므로 추가가
  맞다**(§Q1의 세 시나리오와 달리).
- 약어를 쓰면 캡션의 약어 목록에 MnasNet/SqueezeNet/ResNeXt50를 더할 것.

---

## 3. 분기 C를 쓰지 않은 이유

두 문장 모두 **확인에 성공**했으므로 개수 단언을 정성 서술로 낮추는 분기 C는 해당하지 않는다.
다만 §Q1은 "확인 결과 4종이 아님"이라 개수 단언 자체가 제거되고(§1), §II는 "4종은 맞으나 구성이 다름"이라
개수는 유지된다(§2).

**Threats 문단 추가 불필요**: 이번 두 항목은 미보존이 아니라 **확인됨**이다. 기존 미보존 목록
(NPU 미스프리딕션 λ, 정상예측·2-gen deadline 규약)은 그대로 두고, 여기에 **더하지 않는다.**
단, §1.1의 "관측 21.0–30.0 s 범위의 시나리오 귀속 미상"은 남으므로, 그것까지 공개하려면 Threats에
한 항목을 더할 수 있다(선택).

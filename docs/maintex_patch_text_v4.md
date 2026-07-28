# 패치 문안 **v4** — §Q1 · Fig.~3 · §II 종결 (적용은 사람)

작성 2026-07-28. v3(`maintex_patch_text_v3_q1.md`) 보존. 근거: `gate_f_fig3_attribution.md`.
**v3 대비**: F-2/F-3 반영(범위 재산출·검출 수치 제거), 플랫폼 명시, Q4가 다른 양임을 캡션에 명시,
탐색 바운드 26/24를 **캡션에서 제외**, §II는 2-A + Threats 별도 문장 + Table~II 각주 한정.

---

## 1. §Q1 — 교체 문장

**현행** (`main_patched.tex` §Q1):
> We evaluate this behavior across diverse contention scenarios using **four** foreground models.

**교체안**:
> We evaluate this behavior across three scenarios that differ in workload and platform. The two
> misprediction scenarios run three foreground vision models (YOLO11s, ResNet50, MobileNet-v2) with a
> background LLaMA-1B---one on the CPU--GPU platform, one on CPU--NPU---and the infeasible-candidate
> scenario runs four heavier foreground models (YOLO11x, YOLO11l, YOLO11m, ResNet50) with no background
> model on CPU--GPU. Table~\ref{tab:scenario-config} states all three configurations.

---

## 2. §Q1 — 관측 범위 문장 (**F-2 / F-3 반영**)

**현행** (지시문 인용):
> ...observed recovery of $21.0$–$30.0$\,s ... detection ... $1.7$–$2.0$\,s ...

### 2.1 회복 범위 — 재산출 (**F-2**)

확정표의 값에서 직접 계산한 회복 지연(= search + drain)은 **Q3 25.4\,s, Q5 14.2\,s**이며,
**범위는 14.2–25.4\,s**다. 본문의 21.0–30.0\,s는 이 값들과 일치하지 않는다. 특히 하한 **21.0은
확정표가 "사용 금지"로 표시한 값**(Q5의 `t0` 기준 대체값)이고, **30.0은 저장소에서 발견되지 않는다**.

**교체안** (범위를 쓰되 기준점 혼재를 밝히는 형태):
> Recovery completes in $25.4$\,s on CPU--GPU and $14.2$\,s on CPU--NPU, in both cases within the
> instantiated envelope. The two are measured from different onsets---the GPU scenario from the first
> violation, the NPU scenario from the rate increase, because there the system is already in violation
> before the load rises---so we report them separately rather than as a single interval.

**대안** (범위 형태를 유지하려면):
> Recovery completes within $14.2$–$25.4$\,s across the two scenarios that recover.

> **권고**: 첫 번째 안. Q3는 `t0` 기준, Q5는 `burst` 기준이라 **한 구간으로 묶으면 기준점이 섞인다**
> (`gate_f_fig3_attribution.md` §1.4).

### 2.2 검출 범위 — **수치 제거** (**F-3**)

Fig.~3의 데이터에는 `T_detect`에 해당하는 양이 **없다**(quantity는 search/drain/bound뿐). 따라서
$1.7$–$2.0$\,s는 이 그림에서 도출되지 않으며 출처가 확인되지 않는다.

**교체안** (정성 서술로 낮춤):
> Detection is bounded by the monitoring window by construction, and was stable across scenarios; we
> therefore report the post-detection phases, which are what the placement policy influences.

*(수치를 넣으려면 검출 지연을 산출한 실행을 특정해야 한다 — 현재 미상.)*

### 2.3 Threats 추가 항목 (F-3에 따른 필수)

기존 미보존 목록과 **같은 문단**에 넣는다(§II와 달리 이것은 미보존 유형이다):
> the detection-delay figures quoted for the recovery-latency breakdown, whose originating run we could
> not identify.

---

## 3. Fig.~3 캡션 보강

**현행 캡션**:
> Breakdown of recovery latency into detection delay and post-detection phases. Detection remains
> stable, while total recovery remains below the analytical upper bound (33 s) instantiated from
> $T + N_{cand}(T_v + \delta) + T_{stable}$, despite increasing interference.

**교체안**:
> Recovery latency per scenario, split into the search phase (detection onset until a candidate is
> committed) and the drain phase (commit until $V(t)\le\epsilon$), against the per-scenario bound. The
> first two bars are scenarios that recover---CPU--GPU and CPU--NPU misprediction; the drain phase is
> zero in the latter, where the committed candidate is immediately feasible. The third bar is the
> infeasible-candidate scenario, which does not recover: its bar is the interval from the first
> violation to the best-so-far revert, i.e.\ a termination latency rather than a recovery latency, and
> it is a single run where the others are five. Because $\delta$ is not uniform---relocating the
> generative model costs $\approx2.9$\,s against $\approx1$\,s for an in-place vision swap---the
> envelope $T + N_{cand}(T_v+\delta) + T_{stable}$ widens from $33$\,s to about $42$\,s in the scenarios
> that move the generative model; every observation stays below it.

**반영한 것**:
- **(a)** Q4가 다른 양(종료 지연)이고 `n=1`임을 명시.
- **(b)** search/drain 2구간 + 바운드 구성.
- **(c)** 축 라벨이 시나리오 이름이므로 캡션이 같은 이름을 쓴다.
- **§2** δ 비균일 → **같은 양(33 → 42.5 s)** 으로만 서술. **탐색 바운드 26/24는 끌어오지 않는다**
  (§Q3·§Q4가 각자 보고).
- 플랫폼 명시.

---

## 4. §II — 2-A 채택

**현행** (`main_patched.tex` §II):
> ...measurements from our CPU--GPU testbed with **four** concurrent DNN applications.

**교체안 (2-A)**:
> ...measurements from our CPU--GPU testbed with four concurrent DNN applications. This motivating
> measurement predates the model set of Table~\ref{tab:models} and uses four CPU-class vision models; it
> illustrates the failure modes the objective must address, and no result in Section~IV depends on it.

*(개수 4는 참 — 4월 실행에서 네 뷰 모두 실측 산출. 모델 이름은 노출하지 않는다(2-B 미채택);
이름은 `q1_workload_attribution.md`에 남는다.)*

### 4.1 Threats — **별도 문장** (미보존 목록에 섞지 않는다)

> The motivating measurement of Section~II was taken with a model set that the runtime no longer
> supports; its configuration is recorded but the measurement cannot be reproduced with the current
> implementation. No result in Section~IV depends on it.

### 4.2 Table~II 각주 교체 (**Table~III에 §II 행을 추가하지 않는다**)

**현행**: *"This column applies except where Table~III states otherwise."*
**교체안**:
> This column applies to the experiments of Section~IV except where Table~\ref{tab:scenario-config}
> states otherwise.

*(이렇게 하면 §II 실험을 §IV 구성 표에 넣지 않아도 전제가 성립하고, 약어 3개를 더할 필요도 없다.)*

### 4.3 deadline 표기

어디에 적든 `$1000/\lambda_i$` 대신 **"one inter-arrival interval"** 로 풀어 쓴다.
(§II 실험은 `slo_ms` 미설정 → 뷰별 deadline이 도착 간격 하나와 같다.)

---

## 5. 적용 순서 권고

1. §1(전경·플랫폼) → 2.1(회복 범위) → 2.2(검출 수치 제거) → 3(캡션)을 **한 묶음**으로. 나눠 적용하면
   절 안에서 수치가 어긋난다.
2. §4(§II) + 4.1 Threats + 4.2 각주를 **한 묶음**으로.
3. 2.3 Threats 항목은 2.2와 함께.

## 6. 반영하지 않은 것
- **그림 재생성** — 값 귀속만 밝혔고 그림의 데이터는 바뀌지 않았다.
- **수치 변경** — 25.4 / 14.2 / 24 / 26 / 33은 전부 기존 값이다. 범위 문장만 실제 값에 맞췄다.
- **탐색 바운드 26/24의 §Q1 유입**, **§II 모델명 노출**, **Table~III의 §II 행** — 전부 지시문 §6대로 배제.

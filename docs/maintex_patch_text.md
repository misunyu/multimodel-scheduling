# main.tex 패치 문안 (§2 · §4 · §5) — **적용은 사람**

작성 2026-07-28. **저장소에 `main.tex`가 없다**(`git ls-files | grep .tex` 공집합). 지시문 §6.4의
"main.tex diff"는 산출 불가이므로, **적용 가능한 문안**으로 낸다. 현재 문장을 볼 수 없으므로 인용된
현행 표현은 지시문이 제공한 것에 한한다.

근거: `gate_d_config_attribution.md`, `scenario_configuration_table.md`, `revisit_posthoc.md`,
`gate_c_rerun_decision.md`.

---

## A. §IV-A — 실험별 구성 표 도입 (지시문 §2)

### A-1. 교체할 문장

**현행**: *"We designate **four** vision models (YOLO11s, YOLO11m, ResNet50, and MobileNet-v2) as
foreground workloads"*
→ **교체안**:

> We do not fix a single foreground set across experiments. Each experiment activates a subset of the
> models in Table~I, and Table~\ref{tab:scenario-config} states, per experiment, the foreground models,
> the active background models, the input rate $\lambda$, the deadline convention, and the buffer depth.
> The misprediction experiments (Q3, Q5, Q6) use three foreground vision models
> (YOLO11s, ResNet50, MobileNet-v2) together with a background LLaMA-1B; the infeasible-candidate
> experiment uses four heavier vision models and no background model.

**현행**: *"We set $L_{SLO,i} = 5 \cdot \text{latency}_i$ to provide sufficient headroom"*
→ **교체안** (무조건문 → 규약 중 하나):

> Deadlines follow one of two conventions, stated per experiment in
> Table~\ref{tab:scenario-config}: a per-model deadline $L_{SLO,i} = 5\cdot\text{latency}_i$ derived from
> the isolated latencies in Table~\ref{tab:models}, or a uniform deadline applied to every foreground
> view. The misprediction experiments use a uniform 15\,ms deadline; the infeasible-candidate experiment
> uses the per-model convention.

**Table~I의 $L_{SLO}$ 열 각주**:

> This column applies only to the experiments that use the per-model convention
> (Table~\ref{tab:scenario-config}); the misprediction experiments use a uniform deadline.

### A-2. 새 표 (LaTeX 초안)

값은 `scenario_configuration_table.md`에서 왔다. **"미상" 칸은 사람이 채우거나 행을 빼야 한다.**

```latex
\begin{table}[t]
\caption{Per-experiment workload configuration. Foreground models are the ones whose latency is
monitored by $V(t)$; background models generate contention but are not monitored.}
\label{tab:scenario-config}
\centering\small
\begin{tabular}{@{}llllll@{}}
\toprule
Experiment & Foreground & Background & $\lambda$ & Deadline & Buffer \\
\midrule
Misprediction (GPU)  & 11s, R50, MNv2 & LLaMA-1B & 80 / 25 & 15\,ms unif. & 12 \\
Ablation (GPU)       & 11s, R50, MNv2 & LLaMA-1B & 80 / 25 & 15\,ms unif. & 12 \\
Misprediction (NPU)  & 11s, R50, MNv2 & LLaMA-1B & 80      & 15\,ms unif. & 12 \\
Correct prediction   & 11s, R50, MNv2 & ---      & 45      & ---          & 2   \\
No feasible cand.    & 11x, 11l, 11m, R50 & --- & 90      & $5\times$    & 300 \\
Two generative       & 11s, R50, MNv2 & LLaMA-1B, Qwen2-VL & 90 & ---     & --- \\
\bottomrule
\end{tabular}
\end{table}
```

> `---` 칸은 **로그·리포트에 값이 없는 항목**이다(추정 기입하지 않았다). 논문에 넣기 전에 사람이
> 확정하거나, 해당 열을 그 행에서 생략해야 한다. 특히 **Q5 최종 λ**와 **2-gen SLO 규약**이 미상이다.

### A-3. $V(t)$ 비대칭 한 문장 (지시문 §2.3, 신규)

> Because the misprediction experiments apply a single 15\,ms deadline to all three foreground views,
> the resulting $V(t)$ is dominated by YOLO11s: the uniform deadline is roughly twice as strict as the
> per-model value for YOLO11s (31\,ms) and about twice as loose for ResNet50 (11\,ms) and MobileNet-v2
> (6\,ms). This does not affect the recovery conclusion, since the recovery placement moves only the
> background LLM off the accelerator and leaves all three vision placements unchanged, so no foreground
> model is made worse by it.

### A-4. Q4 2-gen 워킹셋 명시 (지시문 §2.4)

> This two-generative case uses three foreground vision models rather than the four of the
> infeasible-candidate case, because the second generative model already saturates the accelerator;
> keeping four vision models would have made every placement infeasible for a reason unrelated to the
> point being made. Its recovery placement---both generative models on the CPU---sits at rank twelve of
> the thirty-two placements this working set admits.

---

## B. Q6 구조 패치 (지시문 §5)

### B-1. 2×2 격자표 (지시문 §5.1)

```latex
\begin{table}[t]
\caption{Factorial ablation of the two mechanisms. Rows remove dwell (the validation window $T_v$);
columns remove progress (descent through the ranked candidate list). Neighbouring cells isolate one
mechanism: A$\leftrightarrow$BoundGuard is the effect of dwell, C$\leftrightarrow$BoundGuard the effect
of progress. Each cell reports recoveries out of five runs, terminal $V$, and hot-swaps.}
\label{tab:c2-ablation}
\centering\small
\begin{tabular}{@{}llcc@{}}
\toprule
& & \multicolumn{2}{c}{progress} \\
\cmidrule(l){3-4}
Regime & dwell & kept & removed \\
\midrule
\multirow{2}{*}{Correct pred.}
 & kept    & \textbf{5/5}, 0.0, 1.0 & 5/5, 0.0, 0 \\
 & removed & 1/5, 3.3, 10.6         & 5/5, 0.0, 0 \\
\midrule
\multirow{2}{*}{Misprediction}
 & kept    & \textbf{5/5}, 0.0, 7.0 & 0/5, 1.2, 0 \\
 & removed & 3/5, 9.9, 8.8          & 0/5, 1.2, 0 \\
\midrule
\multirow{2}{*}{No feasible cand.}
 & kept    & 0/5 bnd, 65.3, 9.0 & 0/5 bnd, 74.5, 0 \\
 & removed & 0/5 bnd, 66.6, 9.0 & 0/5 bnd, 74.4, 0 \\
\bottomrule
\end{tabular}
\end{table}
```

- 좌상 = BoundGuard, 우상 = **C (hybrid)**, 좌하 = **A (no-dwell)**, 우하 = **B (re-invoke)**.
- 수치 출처: `c2_reactive_baseline_report.md` §3.1–3.3 (5회, 런 길이 120 s 고정).
- **B 칸에 각주**: *"This corner is the adaptive hot-swap baseline: with neither dwell nor progress,
  re-invoking the predictor returns the same top-ranked placement, so the two are the same system."*
  (Q3 표에서 별개 경쟁자로 읽히지 않게 하는 장치.)
- **재방문 열은 넣지 않았다** — `revisit_posthoc.md`대로 A는 산출 불가이고, B/C만 넣으면 설계의 확인을
  발견처럼 보이게 한다. 넣으려면 §C-2 참조.

### B-2. bounded ↔ recovery 정정 (지시문 §5.2)

**Intro 기여 항목** — 현행 *"Only their combination preserves **bounded persistence** across every regime"*
→ 교체안:

> Only their combination *recovers* in every regime where recovery is possible, and terminates with the
> best observed placement where it is not. Boundedness alone does not separate the corners: every corner
> terminates, because termination comes from the finite candidate budget rather than from validation.

**Q6 결론부** — 현행 *"BoundGuard is the only corner that holds in all three regimes"*
→ 교체안 ("holds"의 정의를 명시):

> BoundGuard is the only corner that both recovers whenever a feasible placement exists within the
> budget (5/5 under correct prediction and 5/5 under misprediction) and terminates on the best observed
> placement when none does. Removing either mechanism breaks the first property while leaving the second
> intact.

**Abstract·Introduction 훑기**: 같은 "bounded persistence" 표현이 있으면 위 기준으로 통일. (저장소에
본문이 없어 확인 불가 — 사람이 확인.)

### B-3. A의 best-so-far 명시 (지시문 §5.3)

> All four corners retain the same candidate budget ($N_{cand}=5$) and the same best-so-far revert on
> exhaustion; the corners differ only in dwell and progress.

**현행** *"the variance across runs reflects which one it happens to stop on"* → 교체안:

> The variance across runs reflects which placement it selects on exhaustion. Removing dwell does not
> remove the best-so-far rule; it removes the observation the rule depends on, so the selection is made
> from service rates measured during the transient right after each swap. The rule still fires, but on
> meaningless input---which is why progress alone does not rescue the corner.

**로그 뒷받침**: A는 실행마다 서로 다른 배치에 정착한다(`gggg`/`gggc`/`gcgg`/`cggg` 4종), BoundGuard는
31/31이 회복 배치 `cggg`로 정착.

### B-4. 한계 절 (지시문 §5.4)

> We did not impose an explicit cap on the number of transitions; termination comes from the candidate
> budget, and the fixed observation window acted as a de-facto wall-clock limit. No run reached that
> limit in the infeasible-candidate experiment. The no-dwell corner's single non-converging run under
> correct prediction ended by window expiry, not by hitting a transition cap.

추가로 리포트에는 남기되 논문 반영은 선택:
- 재방문 수치는 **기존 로그의 사후 산출**이며 새 계측이 아니다. A에 대해서는 **산출 불가**.
- 재실행을 하지 않은 이유: 정본 β로 랭킹을 재산출한 결과 이 시나리오의 후보 순서가 **바뀌지 않았다**
  (회복 배치 rank 5, β=0.5·1.0 동일) — `gate_c_rerun_decision.md`.

---

## C. 반영하지 않은 항목

1. **수치 갱신 (`q3q6` §6.1)** — 게이트 C-1로 **철회**. "rank five", "margin is zero",
   `d2_budget_rank` $k^\ast$ 전부 **원문 유지**. 그림 `d2_budget_rank.pdf`·`q3_misprediction.pdf`
   **재생성 불필요**.
2. **재방문 열** — A 산출 불가로 표에서 제외(§B-1). 넣으려면 런 내부 시퀀스 계측 + 재실행이 필요하며,
   이는 C-1로 근거가 사라진 새 측정이다(사람 판단).
3. **`c2_reactive_comparison.pdf` 재생성** — 표 구조만 바뀌고 그림의 데이터·축은 그대로이므로
   **불필요**. (지시문 §6.5 "필요한 경우만".)

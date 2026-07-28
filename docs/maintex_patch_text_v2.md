# main.tex 패치 문안 **v2** — 적용은 사람

작성 2026-07-28. v1(`maintex_patch_text.md`)은 보존. **저장소에 `main.tex`가 없어 diff가 아닌 문안이다.**
근거: `gate_e_ablation_numbers.md`(E-1), `transition_unit.md`, `gate_d_config_attribution.md`,
`scenario_configuration_table.md`, `revisit_posthoc.md`, `gate_c_rerun_decision.md`.

**v1 대비 변경**: B-1 표를 정본 수치로 교체(게이트 E) · 전환 단위 명시(§2) · A-3 배수 정정 ·
A-4 사후합리화 삭제 · B-3 코드 근거 확정 · line 85 앞문장/line 479/551/162 보완 · 표 표기 정의 ·
Threats 문장 · 2×2 칸에 변형 이름 병기.

---

## A. §IV-A — 실험별 구성 (v1 §0 채택분 유지)

### A-1. 교체 문장 (v1 그대로)

**"four vision models ... as foreground"** → 실험별 위임 + 표 참조:

> We do not fix a single foreground set across experiments. Each experiment activates a subset of the
> models in Table~I, and Table~\ref{tab:scenario-config} states, per experiment, the foreground models,
> the active background models, the deadline convention, and the buffer depth. The misprediction
> experiments (Q3, Q5, Q6) use three foreground vision models (YOLO11s, ResNet50, MobileNet-v2) together
> with a background LLaMA-1B; the infeasible-candidate experiment uses four heavier vision models and no
> background model.

**"We set $L_{SLO,i}=5\cdot\text{latency}_i$"** → 규약 중 하나로 격하:

> Deadlines follow one of two conventions, stated per experiment in Table~\ref{tab:scenario-config}: a
> per-model deadline $L_{SLO,i} = 5\cdot\text{latency}_i$ derived from the isolated latencies in
> Table~\ref{tab:models}, or a uniform deadline applied to every foreground view. The misprediction
> experiments use a uniform 15\,ms deadline; the infeasible-candidate experiment uses the per-model
> convention.

### A-2. 표 (§4.4 표기 정의 반영, $\lambda$ 열 제외)

```latex
\begin{table}[t]
\caption{Per-experiment workload configuration. Foreground models are the ones whose latency $V(t)$
monitors; background models generate contention but are not monitored. Abbreviations: 11s/11m/11l/11x =
YOLO11s/m/l/x, R50 = ResNet50, MNv2 = MobileNet-v2.}
\label{tab:scenario-config}
\centering\small
\begin{tabular}{@{}llll@{}}
\toprule
Experiment & Foreground & Background & Deadline \\
\midrule
Misprediction (GPU)   & 11s, R50, MNv2     & LLaMA-1B & 15\,ms uniform \\
Ablation (GPU)        & 11s, R50, MNv2     & LLaMA-1B & 15\,ms uniform \\
Misprediction (NPU)   & 11s, R50, MNv2     & LLaMA-1B & 15\,ms uniform \\
Correct prediction    & 11s, R50, MNv2     & ---      & unrecorded \\
No feasible candidate & 11x, 11l, 11m, R50 & ---      & $5\times$ per-model \\
Two generative        & 11s, R50, MNv2     & LLaMA-1B, Qwen2-VL & unrecorded \\
\bottomrule
\end{tabular}
\end{table}
```

- **$\lambda$·버퍼 열은 뺐다** — 값이 실험마다 있고 없어(§4.6 Threats) 열을 두면 빈칸이 남는다. 필요한
  경우 본문에서 정성적으로 언급한다.
- `unrecorded` = 로그·리포트에 남아 있지 않음(**추정 기입 없음**).

### A-3. $V(t)$ 비대칭 (배수 정정)

> Because the misprediction experiments apply a single 15\,ms deadline to all three foreground views,
> the resulting $V(t)$ is dominated by YOLO11s: relative to the per-model convention the uniform deadline
> is about $2.1\times$ stricter for YOLO11s (31\,ms), about $1.3\times$ looser for ResNet50 (11\,ms), and
> about $2.3\times$ looser for MobileNet-v2 (6\,ms). This does not affect the recovery conclusion: the
> recovery placement moves only the background LLM off the accelerator and leaves all three vision
> placements unchanged, so no foreground model is made worse by it.

### A-4. Q4 2-gen (사후 합리화 삭제 — 사실만)

> This two-generative case uses three foreground vision models. Its recovery placement---both generative
> models on the CPU---sits at rank twelve of the thirty-two placements this working set admits.

*(v1의 "because the second generative model already saturates the accelerator..."는 **삭제**. 3-vision은
미스프리딕션 계열 전체가 공유하는 워킹셋이고, 2-gen을 위해 골랐다는 문서 근거가 없다.)*

---

## B. Q6 구조 패치

### B-1. 2×2 격자표 — **정본 수치**(게이트 E §4), 변형 이름 병기

```latex
\begin{table}[t]
\caption{Factorial ablation of the two mechanisms, five runs per cell over a fixed 100\,s window. Rows
remove dwell (the validation window $T_v$); columns remove progress (descent through the ranked
candidate list). Neighbouring cells isolate one mechanism: A$\leftrightarrow$BoundGuard is the effect of
dwell, C$\leftrightarrow$BoundGuard the effect of progress. Cells report recoveries out of five,
terminal $V$, and per-view hot-swaps.}
\label{tab:c2-ablation}
\centering\small
\begin{tabular}{@{}llll@{}}
\toprule
Regime & dwell & progress kept & progress removed \\
\midrule
\multirow{2}{*}{Correct pred.}
 & kept    & \textbf{BoundGuard} 5/5, 0.0, 3   & C 5/5, 0.0, 0 \\
 & removed & A 4/5, ---, 9.4                   & B 5/5, 0.0, 0 \\
\midrule
\multirow{2}{*}{Misprediction}
 & kept    & \textbf{BoundGuard} 5/5, 0.0, 7   & C 0/5, 1.19, 0 \\
 & removed & A 1/5, 19.9$\pm$9.9, 8.8          & B 0/5, 1.25, 0 \\
\midrule
\multirow{2}{*}{No feasible cand.}
 & kept    & \textbf{BoundGuard} 0/5 bnd, 54.8 & C 0/5 bnd, 74.1, 0 \\
 & removed & A 0/5 bnd, 97.6$\pm$52.6          & B 0/5 bnd, 74.5, 0 \\
\bottomrule
\end{tabular}
\end{table}
```

- **B 칸 각주**: *"This corner is the adaptive hot-swap baseline: with neither dwell nor progress,
  re-invoking the predictor returns the same top-ranked placement, so the two are the same system."*
- **단위 각주**: *"Hot-swap counts are per view: one placement change can swap several views."*
- `---` = 정본 문서가 그 칸 값을 싣지 않음(추정 없음).
- **재방문 열 없음** — A는 산출 불가, B/C만 넣으면 설계를 발견처럼 보이게 한다(`revisit_posthoc.md`).

### B-2. bounded → recovery (§4.1 범위 확대: line 85의 **두 문장 모두**)

**앞 문장** — 현행 *"...show empirically that they are not interchangeable: removing either breaks the
property, but in different regimes and with different signatures... one leaving the system stalled on a
placement it has already found wanting"* → 교체안:

> We show empirically that the two mechanisms are not interchangeable: removing either breaks
> \emph{recovery}, but in different regimes and with different signatures---removing progress leaves the
> system pinned on the top-ranked placement it keeps re-selecting, while removing dwell leaves it
> switching on measurements taken before any placement has settled.

*(변경점 둘: "the property"→"recovery"; "a placement it has already found wanting"→"the top-ranked
placement it keeps re-selecting" — dwell 없는 코너 B는 그 배치를 **판정한 적이 없다**.)*

**마지막 문장** — *"Only their combination preserves bounded persistence across every regime"* → v1대로:

> Only their combination \emph{recovers} in every regime where recovery is possible, and terminates with
> the best observed placement where it is not. Boundedness alone does not separate the corners: every
> corner terminates, because termination comes from the finite candidate budget rather than from
> validation.

**§Q6 결론부(line 826)** — v1대로:

> BoundGuard is the only corner that both recovers whenever a feasible placement exists within the
> budget (5/5 under correct prediction and 5/5 under misprediction) and terminates on the best observed
> placement when none does. Removing either mechanism breaks the first property while leaving the second
> intact.

**범위 한정**: Abstract(38·42)·Introduction(64)의 "bounded"는 **탐색 종료**에 관한 서술이라 정확하다.
**수정 대상은 line 85와 826 두 곳뿐**이다.

### B-3. A의 best-so-far (**코드로 확정**)

> All four corners retain the same candidate budget ($N_{cand}=5$) and the same best-so-far revert on
> exhaustion; the corners differ only in dwell and progress.

**현행** *"the variance across runs reflects which one it happens to stop on"* → 교체안:

> The variance across runs reflects which placement it selects on exhaustion. Removing dwell does not
> remove the best-so-far rule; it removes the observation the rule depends on. With dwell, a candidate's
> service rate is measured over the tail of its validation window; without dwell, the controller advances
> the instant $V(t)$ exceeds $\epsilon$ and the same rule is applied to a rate measured over the brief
> interval since the swap, while the placement is still in transient. The rule still fires, but on
> input that no longer distinguishes the candidates---which is why progress alone does not rescue the
> corner.

**코드 근거 (확정)**: `schedule_executor_main.py:1008-1022`(no-dwell 경로) — *"advance the INSTANT
V>eps, with NO T_v observation. Record a service-rate observation over the (tiny) elapsed window so
best-so-far is kept -- the whole point is that with no dwell this measurement is near-meaningless"*;
`svc = (ic_now - entry_count)/elapsed_nd`. 대비 `:1084-1093`(validate 경로)는 `tail_counts`로 **T_v 창의
tail**에서 산출. → v1의 서술을 낮출 필요 없이 **그대로 쓸 수 있다.**

**로그 뒷받침**: A는 실행마다 다른 배치에 정착(`gggg`/`gggc`/`gcgg`/`cggg` 4종), BoundGuard는 31/31이
회복 배치로 정착.

### B-4. 한계 절 (v1 방식 + 수치 확정)

> We did not impose an explicit cap on the number of transitions; termination comes from the candidate
> budget, and the fixed 100\,s observation window acted as a de-facto wall-clock limit. No run reached
> that limit in the infeasible-candidate experiment. The no-dwell corner's single non-recovering run
> under correct prediction ended by window expiry, not by hitting a transition cap.

*(코드 확인: 명시적 전환 상한 없음. 종료는 후보 예산 소진 → best-so-far 복귀.)*

---

## C. 전환 단위 명시 (신규, §2)

- 표 열: **`hot-swaps (per view)`** + 위 단위 각주.
- 본문 — 현행 *"A makes $9.4\pm4.1$ transitions where BoundGuard makes three"* → 교체안:

> Under correct prediction A performs $9.4\pm4.1$ per-view hot-swaps where BoundGuard performs three;
> under misprediction the counts are $8.8$ and seven.

- **§Q1.5의 "reduces to a single hot-swap"**: §Q6와 **다른 실험**(버퍼 스윕: vision-only,
  $\lambda=0.9\mu^\ast$, per-frame 8.13 ms deadline)이다. 문장에 **어느 실험인지 명시**할 것. 값 1의
  집계 출처는 저장소에서 확인되지 않아(**미상**) 값 자체는 손대지 않았다.
- 권고(선택): thrashing 지표로 $\Sigma\delta_{vis}$ 병기 — 정상예측 BoundGuard 680 ms vs A 3919 ms
  ($5.8\times$), 미스프리딕션 1719 vs 3496 ms ($2.4\times$). 뷰 수에 좌우되지 않는다.

---

## D. 누락 보완

### D-1. line 479 — CPU–NPU "all treated as foreground"

현행 *"four concurrent application instances, all treated as foreground workloads"* → 교체안:

> we construct workloads with four concurrent application instances: three foreground vision models
> whose latency $V(t)$ monitors, and a background LLaMA-1B that contends for the accelerator without
> being monitored.

*(인스턴스 수 4는 맞고 "all foreground"만 틀렸다 — §Q5 서술 및 시나리오 표와 정합.)*

### D-2. line 551 (§Q1 "four foreground models") · line 162 (§II "four concurrent DNN applications")

- **line 551**: Q1 계열의 전경 구성이 저장소 로그·리포트에서 **확인되지 않는다**(리포트↔논문 Q번호
  매핑 부재 — `scenario_configuration_table.md` 마지막 행). → **사람 확인 필요.** 확인 불가면 §D-4의
  배제형 각주 + §D-3 Threats로 처리한다.
- **line 162**: §II 동기 그림은 별개 실험이며 **본 라운드에서 확인하지 않았다.** 사람 확인 대상.
- **손대지 않을 것**: line 746("four heavy vision applications" — 4종 확정), 492·531·811(기법·코너 수).
  **"four" 일괄 치환 금지.**

### D-3. Threats to Validity (신규)

> Some configuration parameters of the earlier experiments were not persisted alongside their logs.
> Where that is so we state the qualitative setting in the text rather than a number, and mark the entry
> as unrecorded in Table~\ref{tab:scenario-config}.

미보존 구체 항목(**Q5 최종 $\lambda$**, **2-gen deadline 규약**, **정상예측 deadline 규약**)을 열거할지는
사람 판단이나, **미보존이라는 사실 자체는 공개**한다.

### D-4. Table~I $L_{SLO}$ 열 각주 — 배제형

Q1 계열 행이 시나리오 표에 없으므로 포함형 각주는 독자를 막다른 곳으로 보낸다. **배제형**으로:

> This column applies except to the misprediction experiments, which use a uniform deadline
> (Table~\ref{tab:scenario-config}).

---

## E. 반영하지 않은 항목 (v1 C절 유지 + 갱신)

1. **수치 갱신 (`q3q6` §6.1)** — 게이트 C-1로 철회. "rank five", "margin is zero",
   `d2_budget_rank` $k^\ast$ 전부 **원문 유지**. `d2_budget_rank.pdf`·`q3_misprediction.pdf` **재생성 불필요**.
2. **A의 재방문 열** — 산출 불가(`revisit_posthoc.md`). 넣으려면 런 내부 시퀀스 계측 + 새 실행 필요 —
   게이트 C-1로 근거가 사라진 새 측정이므로 **사람 판단**.
3. **`c2_reactive_comparison.pdf` 재생성** — **불필요.** 게이트 E는 *내가 인용한 출처*를 바로잡았을 뿐
   그림의 원 데이터를 바꾸지 않았고, 그림은 정본 계열(고정 윈도우, `c2_integrity_report_v2` §7이
   *"종료 마커 4개 전부 100.0 s 정렬"*로 확인)에서 생성됐다.
4. **§Q4·§Q6 비가용 종단값 조정** — 불필요. 정본에서 54.8 ≈ 논문 ≈55로 **이미 일치**(게이트 E §3).

# 게이트 G — $T_{stable}\approx10$\,s의 출처

작성 2026-07-28. **조회·이력 확인만. 새 실행·재계측·그림 재생성 없음.**

판정: **G-3 — 출처 미확인. $33$\,s를 유지하고 $10$\,s가 측정값이 아님을 명시한다.**

**대조 기준(사전 고정)**: 발견값이 $10\pm2$\,s 안이면 출처로 인정, 벗어나면 불일치로 보고.
→ **어느 후보도 이 구간에 들지 않았다.** 근접값을 사후에 채택하지 않았다.

---

## 1. 확인 결과 (우선순위 순)

### 1-1. Q1.4 유체 검증 스윕 — **불일치 (너무 크다)**

`docs/c3_stress_report.md`가 $\lambda=0.70/0.90/0.95\times\mu^\ast$ 스윕의 **drain 기울기**를 남긴다:

| config | λ (fps) | drain 기울기 (fps/s) | 예측 $-(\mu^\ast-\lambda)$ |
|---|---|---|---|
| gpu 0.70/0.90/0.95× | 96/123/130 | −41.7 / −14.2 / **−8.0** | −41/−14/−7 |
| npu 0.70/0.90/0.95× | 72/93/98 | −33.5 / −11.9 / **−7.1** | −31/−10/−5 |

기울기는 남았으나 **drain 시간(초)은 직접 기록되지 않았다.** 유일하게 환산 가능한 지점
(`c3_stress_report.md:30`, *"gpu_l0.90: backlog_total 1284→0을 μ\*−λ 속도로 배수"*):

$$T_{stable}(\text{gpu},0.90\times) = 1284 / 14.2 \approx \mathbf{90\ s}$$

$0.95\times$는 기울기가 더 작으므로($8.0$) 더 길다. → **$10\pm2$\,s 밖(훨씬 크다).**
이 스윕은 오히려 $T_{stable}=Q/(\mu^\ast-\lambda)$의 **발산**을 실증하는 데이터다.

### 1-2. 산식의 이력 — **저장소에 없음**

`3 + 5(3+1) + 10 = 33` 형태의 인스턴스화는 **저장소 문서 어디에도 나타나지 않는다**
(내가 쓴 `gate_f_fig3_attribution.md`가 `main.tex`를 인용한 것이 유일). `T_stable`을 $10$\,s와 잇는
문서 줄은 **0건**. 저장소 문서가 인용하는 바운드는 전부 **$T_{stable}$ 항이 없는** 형태다:

- `q4_experiment_report.md:84-85`: $N_{cand}(T_v+\delta)=5\times(3+1)=20$\,s, $T+\cdot=23$\,s
- `vt_definition_fix_report.md:62`: $T+N_{cand}(T_v+\delta)\approx25$\,s
- `confirmed_values.json`: Q3/Q5 `bound` = $T+k(T_v+\delta)$, Q4 = $T+N_{cand}(T_v+\delta)+\delta$

→ $+T_{stable}$ 항은 **논문에만** 존재하며, 저장소 산출물과 이력에 근거가 남아 있지 않다.

### 1-3. 실측 $T_{stable}$ — **불일치 (너무 작다)**

정작 §Q1의 두 시나리오에서 이 양이 **직접 측정돼 있다**(`docs/repetition_report.md` §런별 원자료):

**Q3 (CPU–GPU, BoundGuard, 5회)**

| rep | 1 | 2 | 3 | 4 | 5 | 평균±SD |
|---|---|---|---|---|---|---|
| $T_{stable}$ (s) | 0.0 | 3.0 | 3.0 | 4.0 | 0.0 | **2.0 ± 1.7** (최대 4.0) |

**Q5 (CPU–NPU, BoundGuard, 5회)**: $T_{stable}$ = **0.0 ± 0.0**
(`q5_experiment_report.md:93`도 *"Q3 ~4 s / Q5 ~0 s (T-창), 둘 다 소형 버퍼"*로 교차 확인.)

→ **0–4\,s**로 $10\pm2$\,s 밖(작다).

### 1-4. b2 버퍼 스윕 — **해당 데이터 없음**
`b2_buffer_sweep_report.md`에 drain 시간 항목이 없다(지표가 p99 지연·드롭).

---

## 2. 판정 — **G-3**

$10$\,s는 세 후보 어디에도 귀속되지 않는다. 그리고 그 위치가 시사적이다:

```
실측 T_stable (§Q1 시나리오)     0 – 4 s
                    ↓
              [ 10 s ]  ← 본문 인스턴스화: 측정 어디에도 없음
                    ↓
유체 스윕 고부하 drain (λ=0.90μ*)   ~90 s   (λ→μ*에서 발산)
```

즉 $10$\,s는 **관측된 모든 값보다 크고, 이론이 예고하는 발산 구간보다는 훨씬 작은 중간값**이다.
"측정값"으로 방어할 수 없지만 "보수적 placeholder"로는 정합적이다 — $T_{stable}$은
$\lambda\to\mu^\ast$에서 발산하므로 **어떤 단일 측정값도 이를 상한하지 못한다**. 이것이 G-2(실측
최댓값으로 재인스턴스화)를 택하지 않는 이론적 이유이기도 하다.

**G-2를 택하지 않은 이유(기록)**: 실측 최댓값($2.8$\,s 또는 $4.0$\,s)으로 바꾸면 봉투가 $33\to26$\,s로
좁아지고 관측 $25.4$\,s와의 여유가 $0.6$\,s로 얇아진다. 더 중요한 것은 **발산하는 항을 관측 최댓값으로
고정하는 것이 이론상 부당**하다는 점이다(지시문 §2 권고와 일치).

---

## 3. 문안 (G-3, 적용은 사람)

**현행** (`main_patched.tex` §Q1):
> With $T=3$\,s, $T_v=3$\,s, $N_{cand}=5$, $\delta \approx 1$\,s, and $T_{stable} \approx 10$\,s:
> $T_{recovery} \le 3 + 5(3+1) + 10 = 33$\,s.

**교체안** (수치 불변, 성격만 명시):
> With $T=3$\,s, $T_v=3$\,s, $N_{cand}=5$, $\delta \approx 1$\,s, and $T_{stable} \approx 10$\,s:
> $T_{recovery} \le 3 + 5(3+1) + 10 = 33$\,s. The last term is a conservative placeholder rather than a
> measurement: Eq.~(\ref{eq:tstable}) makes $T_{stable}$ diverge as $\lambda\to\mu^{\ast}$, so no single
> measured value bounds it. We instantiate it well above the stabilisation times observed in the
> scenarios of this section ($0$--$4$\,s), while remaining far below the drain times the fluid sweep
> exhibits at high load.

*(`0–4 s`는 §1-3의 실측 범위다. 지시문 §2의 예시 문장은 drain $0$–$2.8$\,s를 인용했으나, 같은 양의
직접 측정치는 `repetition_report.md`의 $T_{stable}$ 열이며 **Q3 최대 4.0\,s**다. 더 보수적인 쪽을 썼다.)*

### 3.1 Threats — 미보존 목록에 추가

§II의 재현 불가 문장과 달리 **이것은 미보존 유형**이므로 기존 목록에 넣는다:

> the value at which $T_{stable}$ is instantiated in the recovery-latency bound, whose originating
> measurement we could not identify.

### 3.2 파급 없음
$33$\,s가 유지되므로 Fig.~3 캡션의 *"from $33$\,s to about $42$\,s"*(패치 v4 §3)도 **그대로**다.
$\delta=2.9$ 봉투 $3+5(3+2.9)+10=42.5$\,s 역시 불변. 관측 $25.4 < 33 < 42.5$ 유지.

---

## 4. `report_q_mapping.md` 행 추가 — **하지 않음**
G-1(출처 확인)이 아니므로 확정 매핑에 넣을 고리가 없다. 대신 **미매핑 표**에 한 행을 더한다:

| §Q1 바운드의 $T_{stable}$ | — | 값 $10$\,s의 **원 측정 미상**. 실측치는 0–4 s(`repetition_report.md`), 유체 스윕은 ~90 s — 어느 쪽도 10이 아님 (`gate_g_tstable.md`) |

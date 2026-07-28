# INVALID — λ 설정 결함으로 이 런의 수치는 사용하지 않는다

**무효 사유**: λ를 **뷰당** 적용했으나 μ*는 **공유 파이프라인 처리율**이다.
- GPU: 123 fps/뷰 × 3뷰 = **369 fps 요구** vs 실효 **μ*≈137 fps** → **2.7배 초과**
- NPU: 93 × 3 = **279 fps** vs **μ*≈103** → **2.7배 초과**

`FSRR_FRAME_BUFFER=2`에서 초과분이 지연이 아니라 **드롭**으로 빠지므로 V(t)는 낮게 유지되지만
**드롭이 영구히 멈추지 않는다.** 그 결과 실행기의 포화 가드가 상시 발동한다:

```
QoS-validate combo=cand_1 elapsed=3.1s V_postswap=0.190 V_cumulative=1.057 eps=1.0
             backlog_slope=0.00/s tail_drops={view1:40} service_rate=327.8fps
QoS-triggered advance: saturated: views ['view1'] still dropping in the last 1.0s
             (queue clipped at cap -> arrival>service); slope/V unreliable, advancing
```

**BoundGuard가 후보를 검증한 적이 없다.** 5런 × 5후보 = **25/25 기각이 전부 `saturated`**이며,
cand_1·cand_2는 V_postswap≈0.21로 이미 feasible이었는데도 기각됐다. 따라서 persist 29.6 s는
회복 시간이 아니라 **스킵 5회의 소요**이고, 이 조건에서는 Q1.3이 측정하려는 회복 거동이 발생하지 않는다.

**v29 λ 판정의 결함**: 회복을 `V ≤ ε`로만 판정하고 **드롭 여부를 조건에 넣지 않았다.** 그래서
"GPU 30–400 전 범위 만족"이 나왔고 상한을 찾지 못했다.

## 이 런들이 여전히 유효한 것 (그래서 삭제하지 않는다)

- **포화 가드의 거동과 발동 조건**에 대한 유일한 실측
- 실행기가 남기는 사유 문구와 그 분류 방법론
- λ 재판정(v32 작업 154)의 근거 데이터
- 논문 Threats의 재료 — "드롭으로 낮아진 V와 처리해서 낮아진 V는 다르다"

상세: `docs/lambda_redetermination_report.md`

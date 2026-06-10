# rev12 재측정 지시문 — Tables 1·5·6를 단일 SDK normal-state로 통일

## 0. 목적 (P0-D 해소)

현재 논문의 세 테이블이 **서로 다른 런타임 상태**에서 측정되어 모순됨:

| 테이블 | 출처 상태 | yolo11s large 양자화 손실 |
|---|---|---|
| Table 1 (single-stream) | rev7 | **+0.2% (n.s.)** ← T-D0가 outlier로 확정 |
| Table 5 (gen-decomp) | rev6 | **−21%** |
| Table 6 (gen-gain) | rev10 | (multi-stream gain) |

**T-D0 (8-run, results/repro_driverstate.csv, 2026-06-04)**가 단일-스트림 yolo11s NPU AP_large @ L0를 cold/warm/loaded 섞어 8회 측정 → **gap = −0.096 ± 0.002** (run-to-run std 극소). 결론: **rev7(+0.2%)가 outlier이고, normal state의 large 손실은 −0.096 absolute(≈ −20% relative)**. 즉 rev6(−21%)가 오히려 normal state에 가까움.

**따라서 통일 기준은 rev7 Table 1이 아니라 T-D0 normal state다.** Table 1·5·6를 모두 이 상태에서 다시 측정해 한 상태로 맞춘다.

> ⚠️ 이 결과는 Table 1의 "large unaffected" 행과 본문의 three-size 서술(−47 / −19.5 / +0.2)이 outlier 위에 세워졌음을 뜻함. 이건 SEMANTIC-LOCK / HUMAN_QUEUE 사안이므로 **자동 수정 금지**. T1(아래)이 권위 있는 단일-스트림 값을 산출하면, 서술 변경은 사람이 결정.

---

## 1. 고정 상태(pinned state) 정의 — 모든 측정 공통

- **NPU 바이너리**: 레거시 mxq `b2441f9d` (`models/mobilint_backup/yolo11s.mxq`). T-D0가 검증한 normal state.
- **SDK / model-zoo / 드라이버**: T-D0(`repro_driverstate.csv`, 2026-06-04 10:47–11:45 KST) 실행과 **동일 버전**. cold/warm/loaded에 robust함이 이미 확인됨.
- **앵커 디텍터**: YOLOv11s. 데이터셋: Argoverse-HD val 24 logs, forward ring camera.
- **스크립트 수정 금지**: `accv_experiments/scripts/` 의 기존 측정 엔트리포인트만 사용. 새 플래그가 필요하면 호출 인자로만 처리.
- **각 출력 CSV 헤더(또는 sidecar .json)에 기록**: git commit, SDK 버전, 드라이버 버전, mxq sha(`b2441f9d`), 측정 타임스탬프. (상태 추적용)

---

## 2. 측정 작업

### T1 — 단일-스트림 재확인 (Table 1 권위값 확정)
- 24-log, single-stream, per-size(small/med/large) NPU−GPU sAP gap을 **고정 상태**에서 재측정.
- **수용 게이트**: large gap = **−0.096 ± 0.002** 재현 (T-D0와 일치). 만약 +0.2% n.s.가 다시 나오면 상태가 고정 안 된 것 → 중단하고 바이너리/SDK 핀 재확인.
- 출력: `results/rev12_single_stream.csv` (size별 GPU/NPU/gap, 24-log mean±std, Wilcoxon p).

### T2 — gen-decomp 재측정 (Table 5)
- YOLO11 **s/m/l/x** 각각: per-size 양자화 손실(L0, 무경합) + single-stream staleness(L2_LM), 고정 상태.
- **수용 게이트**: yolo11s 행이 **T1과 일치**(특히 large = −0.096±0.002 ≈ −20% rel). 불일치 시 상태 불일치.
- 각 디텍터의 INT8 export가 이 상태에 존재/일관한지 **함께 보고**. 없으면 그 디텍터 행은 "no INT8 export"로 제외(현재 논문도 그렇게 처리됨).
- 출력: `results/rev12_gen_decomp.csv`.

### T3 — gen-gain 재측정 (Table 6)
- YOLO11 s/m/l/x, **N=4 / L1_light / Comp.A**, Contention-aware − Isolated의 worst & mean gain.
- **3-run** (rev10 A-2와 동일 프로토콜). per-run 값 + mean/std 모두 기록.
- 출력: `results/rev12_gen_gain.csv` (det × {worst_gain, mean_gain} × 3 reps + mean/std + inverts? bool).

### T4 — (선택, ~45분) N=8 8-run 재현성
- 현재 N=8 재현성은 **3-run**(std 0.0001–0.0021). 논문 Limitations를 "8-run"으로 쓰려면 이 작업 필요.
- rev11 N=8 worst-stream(1 det × 4 strat × 1 bg)을 **8회** 반복, strategy별 mean±std 보고.
- **안 돌리면** 논문은 "3-run, std ≤ 0.0021"로 정확히 기술(아래 §3). 권장: 시간 없으면 T4 생략하고 3-run 표현 사용 — 통계적으로 충분.
- 출력(돌릴 경우): `results/rev12_n8_repro8.csv`.

---

## 3. 재현성 표현 — 정확한 워딩 (논문/주석에 그대로 사용)

| 측정 | 반복 | 실측 | 허용 표현 |
|---|---|---|---|
| 단일-스트림 yolo11s L0 large | **8-run** (T-D0) | −0.096 ± 0.002 | "8-run repeatability; large NPU−GPU gap −0.096 ± 0.002 across cold/warm/loaded" |
| N=4 multi-stream (rev10 A-2) | **3-run** | — | "three independent runs" |
| N=8 worst-stream (rev11) | **3-run** | std 0.0001–0.0021 (max spread 0.0051) | "across three runs, per-strategy std ≤ 0.0021" |

**금지 표현**:
- ✗ "N=8을 8회 반복으로 확인" — N=8엔 한 적 없음 (T4 돌리기 전까지).
- ✗ "rev9 ±0.01 이내"를 *재현성 검증*으로 서술 — 그건 rev9 1-run 값 vs rev11 3-run 평균의 단순 차이일 뿐. 정확히는 "rev11 3-run std ≤ 0.0021, 그리고 rev9 1-run 값과 모두 ±0.01 이내" (참고치로만).

### 논문 Limitations 재현성 문장 (drop-in, 현재 정확값)
> We verify the comparison is stable rather than noisy: across three independent runs the $N=8$ worst-stream sAP of each strategy reproduces with per-strategy standard deviation at most $0.0021$, and an 8-run single-stream test (cold/warm/loaded) pins the normal-state quantization loss to within $\pm0.002$, so the reversal and its magnitude are not run-to-run artifacts.

(지난 턴 제가 넣은 "$N=8$ ... within $\pm0.003$ / 8-run"은 **오류** — 위 문장으로 교체.)

---

## 4. 완료 후 (CSV 업로드하면 내가 처리)

1. Table 5·6를 rev12 통일값으로 교체.
2. Table 1 large 행 + three-size 서술 재조정 — **SEMANTIC-LOCK 사안이므로 변경안을 명시적으로 제시하고 승인받은 뒤** 반영(−0.096/≈−20%가 normal이면 "large unaffected"는 폐기, three-size 서술 수정).
3. Limitations 재현성 문장을 §3 drop-in으로 확정.
4. 세 테이블이 한 상태(yolo11s large 셀 일치)임을 교차검증.

## 5. 수용 기준 (acceptance)
- T1·T2의 yolo11s large 셀이 −0.096 ± 0.002로 일치.
- Q(small)은 전 디텍터에서 여전히 가파른 음수(−46~−56% 수준).
- T3의 worst-gain 부호가 4개 디텍터 모두 양(역전 유지).
- 모든 CSV에 상태 메타데이터(§1) 기록.
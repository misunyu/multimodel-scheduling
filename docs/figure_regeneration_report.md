# 그림 재생성 — 확정표 정합 · censored 시각 규약 (v8)

날짜: 2026-07-26 · 재실험 없음(기존 로그에서 재플롯) · main.tex 불변 · 파라미터 불변
값 출처: **[c2_integrity_report_v7.md](c2_integrity_report_v7.md) §25 확정표** → `scratchpad/fig/confirmed_values.json`(단일 소스)

> **요약**: 확정표 수치와 어긋난 그림 5종을 재생성하고, **모든 값을 `confirmed_values.json` 단일 소스에서**
> 읽게 했다. 그림마다 **사이드카(`.values.json`)**를 덤프하고, **자동 대조 스크립트(`check_figures.py`)**로
> 확정표와 맞는지 검사한다 — **불일치 0건.** censored(미회복) 변형은 전 그림 공통 규약(해칭+↑+"no recovery"
> +창 길이)으로 그려 회복 시간처럼 오독되지 않게 했다.

## 작업 26 — 영향받는 그림 분류

| 파일 | 스크립트 | 사용 지표 | 확정표 불일치 | 조치 |
|---|---|---|---|---|
| **q13_failure_persistence** | plot_q13_err→**plot_q13_v8** | `persist`(샘플수, 편향) | Static 59.4 censored를 실측막대로·회복 9–12(샘플) | **재생성**(wall-clock 11–13 + censored 규약) |
| **q3_misprediction** | plot_q3→**plot_q3_v8** | V(t) 궤적, **x축=샘플인덱스** | 전환 압축(회복 ~20 대신 25.4) | **재생성**(x축 wall-clock) |
| **q5_npu_generalization** | (신규)**plot_q5_v8** | V(t) 궤적, x축=샘플인덱스 | 15.2 폐기·x축 압축 | **재생성**(wall-clock, 14.2 주석) |
| **bounded_recovery_analysis** | plot_bounded→**plot_bounded_v8** | search/drain+bound | Q5 search 16→14.2·Q3 drain 5→2.8 | **재생성**(확정값) |
| **q13_cumulative_violation** | plot_q13_err→**plot_q13cum_v8** | `cumV`(시간가중, 무편향) | 값 불변 | 재생성(사이드카·규약 일관, 값 동일) |
| q4_bounded_envelope | plot_q4 | V(t) envelope | sample-gap 0(작업11) | 유지(값 무관)·사이드카 권장 |
| runtime_overhead | plot_overhead_err | e2e ms(persist 무관) | 없음 | 유지 |
| detection_sensitivity | plot_sensitivity | 검출 파라미터 스윕 | 없음 | 유지 |
| dynamic_load_adaptation | plot_q2 | V(t) 궤적 | 없음(축 수치 확인) | 유지 |
| qos_score_validation | plot_fig1 | V(t) 정의 검증 | 없음 | 유지 |
| c3_fluid_validation, c3_b2_metrics, b2_metrics, b2_buffer_sweep | 기타 | 처리량/버퍼 | 없음 | 유지 |
| q3_misprediction_2gen, q5_..._2gen | 2-gen 한계실증 | V(t) | 별도(qwen 보고서) | 이번 범위 밖 |
| architecture 등 | — | 도식 | 없음 | 유지 |

- **샘플 기반 편향(persist) 그림 = q13_failure_persistence 하나**(v6 작업19와 일치). 궤적 그림은 x축이 샘플
  인덱스면 같은 압축을 겪으므로 q3/q5를 wall-clock으로 재생성.

## 작업 27 — censored 시각 규약 (전 그림 동일 적용)

미회복(`lastV>ε`) 변형은 **회복한 것과 시각적으로 구분**하고 값이 창에서 끊겼음을 명시:
1. **해칭 open 막대**(속 빈, `////`) — 채운 막대(회복)와 구분.
2. 막대 끝 **↑ 화살표** — 창에서 절단(`≥window`) 표시.
3. 막대 안 세로 텍스트 **"no recovery (lastV=X)"**.
4. 범례에 **"censored (no recovery, ≥window)"**, 축/각주에 **관측 창 길이** 명시(창은 실험별로 다름).
- 시범 적용: `q13_failure_persistence`(Static 양 플랫폼). bounded/q3/q5에도 동일(막대는 해칭, 궤적은 open square).
- **대안(사람 판단)**: 축을 회복시간 → **"회복 여부 + 종점 V"**로 전환(v3 작업8 권고). 미회복이 시간축에
  놓이지 않아 오독 원천 제거. 단점: 회복 변형 간 시간 비교가 사라짐. **장단 병기, 현재는 규약안 채택.**

## 작업 28 — 재생성 결과

| 그림 | 확정 반영 |
|---|---|
| q13_failure_persistence | 회복 3기법 wall-clock(SR 12.6/10.8·Ad 11.0/11.8·BG 11.2/12.0), Static censored 해칭, 창 58.4s |
| bounded_recovery_analysis | Q3 search 24.6+drain 2.8·Q5 14.2·Q4 24(censored, revert), bound 26/26/24 |
| q3_misprediction | x축 wall-clock, BoundGuard 회복 ~28s, persist 25.4=22.6+2.8 주석, baseline open square |
| q5_npu_generalization | x축 wall-clock, converge=T_valid 14.2 주석, baseline(V≈33) open square |
| q13_cumulative_violation | ∫V 6.2×/6.3×(무편향, 보수적 주기) |
- 변형 이름/색 통일: `A: no-dwell`·`B: re-invoke`·`C: hybrid`·`Adaptive hot-swap`·`BoundGuard`·`Static`·`Stop-restart`
  (confirmed_values.json `variant_style`). 오차막대=확정 SD, 반복 수(5회) 각주 명시.

## 작업 29 — 사이드카 + 자동 대조 (핵심 산출물)

- 각 재생성 그림에 **`<figure>.values.json`** 동봉(계열 라벨·값·SD·censored 플래그·주석 수치).
- **`fig/check_figures.py`**: 전 사이드카를 `confirmed_values.json`과 대조. 불일치를 목록 출력, 있으면 exit 1.
- **이번 대조 결과: 5개 사이드카 13개 값, 불일치 0건.**
  ```
  checked 13 values across 5 sidecars ... ALL FIGURE VALUES MATCH CONFIRMED TABLE (0 mismatches).
  ```
- **효과(재발방지)**: 확정표(=본문 수치)가 바뀌면 `check_figures.py`가 즉시 어긋난 그림을 지목한다. 본문·그림
  불일치가 이번 감사의 반복 실패모드였고, 이제 그림 갱신 누락이 자동으로 드러난다.

## 작업 30 — c2_reactive_comparison 2패널화 — **후순위(미착수)**
- 현재 단일 패널(Q3). §4b 패널 병치는 Q6 대비(A가 정합조건에서도 폭주)를 그림으로 보이나 **선택 사항**이라
  작업 26~29 우선 완료 후로 미룸. 필요 시 c2_final2/s4b + q3 데이터로 좌우 2패널(y축 공유, 100s 정렬) 생성 가능.

## 교체된 파일 목록
- PDF+PNG: `q13_failure_persistence`, `q13_cumulative_violation`, `q3_misprediction`, `q5_npu_generalization`,
  `bounded_recovery_analysis`.
- 신규: 5× `.values.json`(사이드카), `fig/confirmed_values.json`(단일 소스), `fig/check_figures.py`(대조),
  `fig/plot_*_v8.py`(재생성 스크립트).

## 무결성
- 재실험 없음. 값은 확정표(v7 §25) 단일 소스에서만. 알고리즘·파라미터·main.tex·legacy/backup 불변. 로그 무변경.
- P1~P9: 원자료 로그 불변, 재플롯만. 반복 수·SD·창 길이 그림에 명시(P6 분산 표기).

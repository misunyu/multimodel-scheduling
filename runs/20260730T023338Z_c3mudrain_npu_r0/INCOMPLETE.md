# INCOMPLETE — 중단된 파일럿 런 (v35 C3)

`INVALID`(설정 결함으로 측정이 성립하지 않음)도 `SUPERSEDED`(더 나은 값으로 대체됨)도 아니다.
**측정 자체는 성립했으나 런이 끝나기 전에 중단**되어 산출물이 일부 없다.

## 무엇이 없는가
- `run_manifest.json` **없음** — 해시·provenance·전환 요약이 기록되지 않았다.
- `aggregate.json` **없음**.
- 남은 것: `metrics.csv`(53샘플), `perframe.csv`, `executor.log`, `schedule_snapshot.yaml`.

## 사유 (로그 원문)
```
[Main] Received signal 15; initiating graceful shutdown...
[Shutdown] Throughput data saved to .../results/performance_20260730_113338.json
[Executor] Execution stopped by user.
```
SIGTERM으로 중단됐다. 배출 국면 진입 직후이며, 중단 시점까지의 샘플은 정상이다.

## 그럼에도 유효한 것
`scripts/analyze_c3.py`의 `anchor_from_mudrain()`이 이 런에서도 앵커를 재산출한다:
**μ\* = 83.7 fps**(재실행분 `20260730T023450Z_c3mudrain_npu_r0`은 86.9). 두 값의 차 3.2 fps는
**앵커가 배출 3~5샘플에서 추정된 저해상도 값**임을 보여주는 증거이며, v35 리포트 §5의
NPU 앵커 오차(+12%) 논의의 근거 중 하나다. 그래서 삭제하지 않는다.

**이 런의 값을 확정표에 쓰지 않는다.** 확정표의 `c3_npu_mu_star_anchor`는 완주한 재실행분을
`source`로 가리킨다.

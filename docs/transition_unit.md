# 전환 횟수의 단위 — 확인과 통일 결정

작성 2026-07-28. **코드·로그 확인만. 새 실행 없음.**

---

## 1. `hotswaps` 단위 = **뷰 단위 (per view)** — 코드로 확정

`adaptive_deploy.py`의 전환 루프는 **뷰마다** 판정하고 뷰마다 스왑한다:

```python
for vname in ...:                       # 뷰 단위 순회
    if _same_device(old_cfg, new_cfg):
        print(f"[AdaptiveDeploy] {vname}: same device ..., keeping worker")
        kept_views.add(vname)
    else:
        print(f"[AdaptiveDeploy] {vname}: device changed (... -> ...), hot-swapping")
        self._hot_swap_view(vname, old_cfg, new_cfg)      # :206
        swapped_views.add(vname)
```

`_hot_swap_view`는 **뷰 하나**를 교체하며 완료 시 `"[AdaptiveDeploy] {vname}: hot-swap complete
(delta=...ms)"`를 출력한다. 리포트의 `hotswaps`는 이 사건 수다.

**교차 검증**: 미스프리딕션에서 BoundGuard는 후보 **5개**를 순회하는데 hot-swaps는 **7**이다.
배치 단위라면 5를 넘을 수 없다 → **뷰 단위 확정**. (한 배치 전환이 여러 뷰를 동시에 옮길 수 있다.)

---

## 2. 논문의 "$9.4\pm4.1$ transitions where BoundGuard makes three"

- 값의 출처는 `c2_integrity_report_v7.md`의 **§4b(정상예측)** 열이다: *"A. no-dwell 4/5(**hs 9.4**)"*,
  그리고 v2가 *"thrashing 5.8배(Σδ_vis 3919 vs 680 ms, **hs 9.4 vs 3**)"*라 적는다.
  → **A 9.4 / BoundGuard 3은 정상예측 체제의 뷰 단위 hot-swap 수**다.
- 미스프리딕션 체제의 대응 값은 **BoundGuard 7 / A 8.8**로 다르다.
- 따라서 논문 문장은 **뷰 단위**이며 **정상예측 실험**을 가리킨다. 단위·체제 모두 현재 본문에 없다.

> 즉 "transitions"라는 낱말이 배치 전환으로 읽히는 것이 문제이지, 값 자체는 정본과 일치한다.

---

## 3. §Q1.5의 "BoundGuard reduces to a single hot-swap" — **다른 실험 (별도 확인 필요)**

- §Q1.5는 **버퍼 스윕**(`b2_buffer_sweep_report.md`: 4기법 × 2플랫폼, $\lambda=0.9\mu^\ast$
  (GPU 123 / NPU 93 fps), per-frame deadline 8.13 ms, $\beta_B\in\{0.5,1,1.5\}$)이고,
  §Q6 정상예측은 **§4b 워크로드**(vision 3, $\lambda=45$, buffer 2)다. → **서로 다른 실험**이다.
- 다만 `b2_buffer_sweep_report.md`에 **hot-swap 횟수 표가 없다**(backlog 보존 로깅만 언급). 따라서
  "single hot-swap"이 어느 집계에서 나왔는지 **저장소에서 확인되지 않는다** → **미상**.
- 결론: 두 문장은 **자기모순이 아니다**(다른 실험). 단, 현재 본문이 그 구분을 밝히지 않아 모순처럼
  읽히므로 **각 문장에 실험을 명시**해야 한다. 값 1의 출처는 미상이므로 **건드리지 않고** 실험만 명시한다.

---

## 4. 통일 결정

**단위를 "hot-swaps (per view)"로 통일하고, 표 캡션과 본문에 명시한다.**

근거: (a) 저장소에 남은 집계가 전부 뷰 단위다. (b) 배치 단위 수치는 어디에도 없고 로그로 재산출할 수
없다(`results/*.json`에 전환 기록 없음 — 게이트 E §1). (c) 열을 빼는 선택지도 있으나, 뷰 단위로
일관되게 표기하면 정보가 보존되고 오독만 제거된다.

- 표 열 이름: **`hot-swaps (per view)`**
- 본문: *"A performs $9.4\pm4.1$ per-view hot-swaps under correct prediction, where BoundGuard performs
  three"* 처럼 **단위와 체제를 함께** 적는다.
- 보조 지표 권고: **$\Sigma\delta_{vis}$**(전환에 쓴 총 시간)를 함께 제시하면 thrashing이 더
  직접적으로 드러난다 — §4b BoundGuard 680 ms vs A **3919 ms**(5.8×), Q3 1719 vs **3496 ms**(2.4×).
  이 값은 뷰 수에 좌우되지 않아 단위 논쟁에서 자유롭다.

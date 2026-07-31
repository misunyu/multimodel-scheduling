# Hot-swap 전환 자원 재측정 (ISSRE R4) — v38

작성 2026-07-31. **커밋함, 푸시 없음.** 논문 문안은 쓰지 않았다 — 사실과 주석 위치까지.

산출물: `runs/*_mem{BG,A}_r*` 9런(캠페인 6 + 파일럿 1 + 샘플러 대조 2),
`fig/confirmed_values.json` **+22항목**(472 → 494),
`scripts/{mem_sampler,run_hotswap_mem_campaign,analyze_hotswap_mem,check_transition_structure,
emit_hotswap_confirmed}.py`, `scripts/run_scenario.py --mem`.

---

## 0. 한 줄 요약

**v18의 구조적 결론 셋은 전부 그대로 성립한다**(코드·스케줄 재판독, §4). **수치는 그대로가 아니다** —
정본 구성에서 다시 재니 호스트 RSS 전환 피크가 **+201 MiB가 아니라 평균 +1988 / 최대 +3099 MiB**이고,
여유는 **308배가 아니라 18배**다. 그리고 **v18이 "전 전환에서 0"이라 한 디바이스 메모리 증분이
0이 아니다** — vision in-place swap에서 최대 **+574 MiB**의 VRAM 오버슈트가 실측된다(§3.3).
피크는 **0.1 s**(10 Hz 표본 1개) 지속하므로 **1 Hz 샘플링으로는 대부분 놓친다.**

---

## 1. 작업 199 — 워크로드와 배경 기동 확인

확정표 v2의 **Q3 구성**: 전경 YOLO11s / ResNet50 / MobileNet-v2, 배경 **LLaMA-1B**, CPU–GPU,
λ 25/80, buffer 12. `combination_burst` 이후 후보를 순회하므로 **과부하 중에 시작하는 전환**이며,
이것이 R4가 지목한 시나리오다.

| 변형 | 스케줄 | 트리거 | 전환 |
|---|---|---|---|
| **BoundGuard** | `q3_misprediction_cpu-gpu.yaml`, mode 1 | `validate` | 6회 |
| **A (no dwell)** | `q6_ablation_cpu-gpu.yaml`, mode 1 | `v-above` | 7회 — 전 변형 중 최다 |

각 3반복. A를 넣은 이유는 **churn이 심할 때 피크가 누적되는지**가 R4의 우려에 가장 가깝기 때문이다.

### 배경 기동 — **판정 전에 확인했다** (v26이 빠뜨린 단계)

6런 전수:

```
20260731T044450Z_memBG_r0: bg-llm loaded=2 starts=2 'generating continuously'=2
... (6런 모두 동일)
```

로그 원문: `[bg-llm llama1b/gpu] loaded (3450 ms); generating continuously`,
`[bg-llm] start llama1b on gpu (pid 77112)`. `starts=2`는 초기 기동 1 + **GPU→CPU 이전 1**이다.

### phantom 대조 — 4모델이 전부 살아 있다

v18이 무효인 이유는 명명된 4뷰 중 2뷰만 돌았기 때문이다. 이번 런의 집계:

| 런 | view1 | view2 | view3 | 배경 |
|---|---|---|---|---|
| memBG_r0 | yolo11s@GPU 5086회 | resnet50@GPU 5770회 | mobilenet_v2@GPU 5926회 | llama1b 기동·생성 확인 |
| memA_r0 | yolo11s@GPU 4484회 | resnet50@GPU 5033회 | mobilenet_v2@**CPU** 4996회 | 같음 |

vision 3뷰가 각각 4천~6천 회 추론했고 생성 모델은 별도 프로세스로 기동·생성했다.
(집계 JSON은 vision 워커만 세므로 배경은 `inference_count`가 없다 — 그래서 로그로 확인한다.)

---

## 2. 작업 200 — 계측

- **관찰자 사이드카** `scripts/mem_sampler.py`: 실행기 PID 트리의 host RSS(psutil) +
  GPU VRAM(NVML, 우리 PID 귀속분과 디바이스 총량)을 **10 Hz**로 샘플링.
  실측 **9.4 Hz, 최대 간격 0.116 s**(요구 ≥5 Hz 충족). 실행기 *바깥*에서 재므로 측정이
  측정 대상의 일부가 되지 않는다.
- `run_scenario.py --mem`로 통합 — 산출물은 `runs/<run>/mem.csv`에 **직접** 쓴다.
  v18의 샘플러는 `scratchpad/v18/`에 있었고 **지금 저장소에 없다**(네 번째 증발).

### 두 가지 계측상의 정직한 한계

1. **`rss_mib`는 프로세스 트리의 RSS 합이므로 공유 페이지를 중복 계산한다** — 절대값은 과대
   추정이다. 그러나 **증분**(피크 − 인접 정상 상태)에서는 공유분이 상쇄되므로 인용할 값은 증분이고
   절대값이 아니다. PSS가 정확하나 `smaps_rollup`을 10 Hz로 십여 프로세스에 돌리는 것은 잡으려는
   과도 현상보다 느리다.
2. **`vram_ours`는 NVML의 PID 귀속분 합**이고 `vram_total`은 타 프로세스를 포함한 디바이스 총량이다.
   "우리가 얼마를 더했나"와 "기계가 한계에 얼마나 가까웠나"는 다른 질문이라 둘 다 싣는다.

### 샘플러 부하 대조 — **결론 없음(n=1)**

동일 시나리오를 샘플러 유·무로 각 1회:

| | total_fps | view3_fps |
|---|---|---|
| 샘플러 있음 (r90) | 687.5 | 328.2 |
| 샘플러 없음 (r91) | 734.7 | 371.9 |
| 차 | **−6.4 %** | **−11.7 %** |

**그러나 같은 조건 3런(r0–r2, 전부 샘플러 있음)의 런간 산포가 total 3.9 %, view3 11.0 %다.**
즉 위 차이는 산포 안에 있고 **n=1로는 샘플러 비용을 분리할 수 없다.** v18은 −0.5~−5.9 %로
적었으나 같은 한계를 안고 있었을 것이다. 메모리 풋프린트는 수 % 의 CPU 변동에 둔감하므로 이
불확실성이 §3의 값을 위협하지는 않는다 — 다만 **비용을 측정했다고 주장하지 않는다.**

---

## 3. 작업 201 — 전환 종류별 자원

정상 상태 = 창 경계 앞뒤 2 s의 중앙값. 피크 = 전환 창의 최대.
증분 = 피크 − **인접 정상 상태의 큰 쪽**(새 정상 상태로 올라간 것을 겹침 비용으로 오인하지 않기 위함).
전환 창 = `metrics.csv`의 **구멍**(§3.4).

### 3.1 종합 (캠페인 6런, 파일럿·대조 제외)

| 전환 종류 | n | 지표 | 증분 평균 | 증분 최대 | 피크 유지 | 피크 최대 | 최악 전환의 여유 |
|---|---|---|---|---|---|---|---|
| **생성 모델 이전** | 6 | host RSS | **+1988** | **+3099** | **0.10 s** | 10513 MiB | **18×** |
| | 6 | GPU VRAM | −1110 | **0** | 1.23 s | 4896 MiB | **오버슈트 없음** |
| **vision in-place** | 16 | host RSS | +171 | +649 | 1.71 s | 8206 MiB | 94× |
| | 16 | GPU VRAM | −98 | **+574** | 1.61 s | 5628 MiB | 49× |

가용량: 호스트 RAM **63562 MiB**, GPU VRAM **32607 MiB**(RTX 5090).
음수 증분은 **피크가 정상 상태를 넘지 않았다**(오버슈트 없음)는 뜻이다.

### 3.2 생성 모델 이전 — 런별 (평균만 내지 않는다)

| 런 | 전환 | 창 | RSS 정상→피크→정상 | 증분 | 유지 | VRAM 증분 |
|---|---|---|---|---|---|---|
| memBG_r0 | cand_4→cand_5 | 7 s | 4041 → **10453** → 7354 | **+3099** | 0.1 s | 0 |
| memBG_r1 | cand_4→cand_5 | 6 s | 4160 → 10387 → 7750 | +2637 | 0.1 s | 0 |
| memBG_r2 | cand_4→cand_5 | 6 s | 4139 → 8914 → 7999 | +915 | 0.1 s | 0 |
| memA_r0 | cand_3→cand_5 | 7 s | 4263 → 9943 → 7979 | +1964 | 0.1 s | −2348 |
| memA_r1 | cand_4→cand_5 | 7 s | 4455 → **10514** → 8035 | +2478 | 0.1 s | −2092 |
| memA_r2 | cand_4→cand_5 | 7 s | 4211 → 9038 → 8206 | +832 | 0.1 s | −2220 |

10 Hz 원 계열이 보여주는 순서(memBG_r99 파일럿, t는 런 시작 기준):

```
43.30  rss 4204.9  vram 2526      ← 옛 GPU child 살아 있음
43.41  rss 1845.3  vram 2526      ← kill: RSS 가 2360 MiB 떨어진다
...    (새 CPU child 가 4 s 동안 가중치를 적재)
47.33  rss 3432.8
47.44  rss 4443.7
47.65  rss 8700.8
47.76  rss 10153.3 ← 피크
47.86  rss 8230.0  ← 이미 내려왔다
48.62  rss 8111.2  vram 2540      ← 새 정상 상태
```

**피크는 겹침이 아니다.** 옛 child가 먼저 죽고(43.41) 새 child가 적재되는 동안 나타나므로,
**적재 자체의 과도 현상**이다 — 가중치 파일 버퍼와 텐서가 잠시 함께 상주한다. R4가 상정한
"옛+새 동시 실행"과는 다른 원인이며, **크기는 오히려 더 크다.**

### 3.3 **v18과 갈리는 지점 — VRAM 증분이 0이 아니다**

v18은 *"디바이스 메모리 증분은 전부 0"*이라 적었다. vision in-place swap에서 **+574 MiB**가 나온다
(`memBG_r1`, cand_2→cand_3). 10 Hz 원 계열:

```
34.14  vram 4952   ← 정상
34.35  vram 5222
34.45  vram 5494
34.89  vram 5526   ← 피크, 정상보다 +574
35.31  vram 4760   ← 새 정상
```

원인은 구조적 결론과 모순되지 않지만 **그 결론이 함의하지 않는 것**이다:

- 구조적 결론은 **한 뷰의 모델이** 같은 디바이스에서 중복되지 않는다는 진술이다(§4-1). 참이다.
- 그러나 **한 전환에서 여러 뷰가 동시에 움직인다.** 뷰 X가 CPU→GPU로 들어오는 동안 뷰 Y의
  GPU 세션이 아직 살아 있으면, **디바이스는 둘의 합을 본다.** `_hot_swap_view`가 새 워커를
  `start()`한 뒤 **ready 후에야** 옛 워커를 정지하기 때문이다.

→ **"교차이동이므로 디바이스 메모리 피크가 없다"는 추론은 성립하지 않는다.** 피크는 작지만
(32607 MiB의 **1.8 %**) 0이 아니고, 여유 49배다. v18이 0을 본 것은 2뷰만 살아 있어 동시 이동이
없었기 때문으로 보인다 — **가설이며 검증하지 않았다**(그 런들은 남아 있지 않다).

### 3.4 부수 발견 — 1 Hz `metrics.csv`에 전환만큼의 구멍이 있다

`metrics.csv`는 생성 모델 이전 구간에서 **6–7 s 동안 표본을 남기지 않는다**(예: t=41.0 다음 행이
t=48.0). 배치를 적용하는 동안 실행기의 초당 샘플러가 멈추기 때문이다. 즉 **논문의 V(t) 계열은
전환 구간을 관측하지 못한다.** 이 구멍이 곧 전환 창이며, 그 안을 보는 것이 10 Hz 관찰자의 존재
이유다. 분석기는 이 구멍을 전환 창의 정의로 쓴다 — 자원 계열에서 전환을 찾으면 순환이 된다.

### 3.5 A 변형에서 피크가 누적되는가 — **아니다**

| 변형 | 전환 | 마지막 전환 후 정상 RSS | 런별 |
|---|---|---|---|
| BoundGuard | 6회 | **7701 ± 266 MiB** | 7354 / 7750 / 7999 |
| A (no dwell) | 7회 | **8073 ± 97 MiB** | 7979 / 8035 / 8206 |

차 **+372 MiB**이며 **BoundGuard 자체의 런간 산포(±266, 범위 645) 안에** 있다. 두 범위는 겹친다
(BG 7354–7999, A 7979–8206). 게다가 A의 중간 정상값들은 오히려 **초기값보다 낮게 내려간다**
(2642–2951 MiB) — 후보를 오가며 모델이 디바이스를 옮기기 때문이고, 단조 증가의 흔적이 없다.

마지막 정상 RSS가 초기(≈3600)보다 높은 것은 누수가 아니라 **배치의 성질**이다: 회복 배치는 LLM을
CPU에 두므로 가중치가 호스트 RAM에 상주한다. **n=3이므로 372 MiB 차를 부정할 근거도 확정할
근거도 없다** — 없다고 말하는 것이 아니라 이 표본으로는 보이지 않는다고 말한다.

---

## 4. 작업 202 — 구조적 결론 재확인 (`scripts/check_transition_structure.py`)

v19 이후 코드가 바뀌었으므로 세 결론을 **기계적으로** 다시 확인했다.

```
[1] cross-device only : 63 schedule files scanned, 0 same-device model swaps
[2] kill-then-start   : sync(): _stop_one at 829, _start_one at 1015 → kill-then-start
[3] no pre-flight check: 9 pattern hits, 0 unaccounted-for
[OK] 세 구조적 결론 전부 v18 그대로 성립한다.
```

1. **교차이동만 존재한다.** 스케줄 63종에서 한 뷰가 디바이스를 유지한 채 모델만 바꾸는 전환 **0건**.
2. **kill-then-start 유지.** `BackgroundManager.sync`가 옮겨지는 child를 먼저 `_stop_one`하고
   그 뒤에 `_start_one`한다(`runtime/background_llm.py:70-78`).
3. **사전 자원 점검은 여전히 없다.** v18과 **같은 grep 패턴**으로 9건이 걸리나 전부 무관하다
   (버퍼 계수 주석, "insufficient score history" 로그, 축 여백 주석 등 — 스크립트가 사유와 함께
   열거한다). 전환 전 잔여 VRAM/DRAM을 확인하는 코드는 없다.
   `backup/`은 실행되지 않는 스냅샷이라 경로로 제외했다(사유를 코드에 적었다).
   두 건은 **v38 자신의 계측 장치**가 "headroom"이라는 이름을 쓰기 때문에 걸린 것이다 —
   검사기가 새 언급을 사람에게 보이려 한 것이므로 감추지 않고 사유와 함께 열거한다.

> **셋 중 하나도 바뀌지 않았다.** 따라서 v18의 강한 답 — R4의 *"assumes sufficient residual
> resources"*가 코드에 대한 정확한 진술이라는 것 — 은 유지된다. 흔들린 것은 **수치와, 수치에서
> 끌어낸 추론 하나**(§3.3)다.

---

## 5. 작업 203 — 확정표 등재

**22항목**(472 → 494). 전 항목 `source`가 런 디렉터리를 가리킨다.
값은 `scripts/emit_hotswap_confirmed.py`가 `analyze_hotswap_mem.py`를 **호출해** 계산하므로
리포트·확정표·콘솔 표가 갈라질 수 없다.

| 그룹 | id |
|---|---|
| 생성 이전 | `hs_gen_{rss,vram}_{incr,incr_max,hold}`, `hs_gen_rss_headroom` (7) |
| vision in-place | `hs_vision_{rss,vram}_{incr,incr_max,hold,headroom}` (8) |
| 전 구간 피크 | `hs_peak_{rss,vram_ours,vram_total}` (3) |
| 가용량 | `hs_host_ram`, `hs_vram_total` (2) |
| churn | `hs_{bg,a}_final_rss` (2) |

스키마: `censored=false`(창에 잘린 값이 아니다), `recovered=null`(회복에 대한 진술이 아니다),
`source_kind=hardware`, `reference=commit`(배치가 적용된 시각 기준; 가용량과 churn은 `window`),
`n`은 전환 수(생성 6, vision 16) 또는 런 수.

검사기 어휘에 **`rss`·`vram`·`hold`·`headroom`** 넷을 추가했다. host RSS와 GPU VRAM을 한 어휘로
묶으면 *"디바이스 증분 0"*과 *"호스트 증분 2 GB"*가 같은 축에 놓이는데, **이 라운드의 결론이
정확히 그 구분 위에 있다.**

**v18의 옛 값(+201 MiB, 308×)은 확정표에 없었다.** 등재하지 않았고, STALE 표시할 대상도 없다.
`docs/hotswap_resource_report.md`(v18) 자체는 **감사 이력이므로 지우지 않는다** — 이 문서가
그 수치를 대체한다.

---

## 6. 작업 204 — 주석 위치 목록

### 6.1 v38 값은 **아직 붙일 곳이 없다**

지시문은 *"현재 §III과 Threats에 v18 수치를 인용하는 문장이 있다"*고 전제하나, **그런 문장은 없다.**
v18 스스로 *"`main.tex` 미수정(문안 초안만)"*이라 적었고, 실제로 §III Placement Manager와 Threats에
전환 자원 수치는 **하나도 없다.** `%!cv`는 값이 그 줄에 인쇄돼 있어야 붙으므로 **지금은 부착 불가**다.

붙일 자리는 정해 둔다(본문이 쓰인 뒤):

| 자리 | 현재 문장 | 들어갈 id |
|---|---|---|
| L356 | *"must be performed in-place without relying on redundant resources or parallel execution capacity"* | `hs_gen_vram_incr_max`, `hs_vision_vram_incr_max` |
| **L364** | *"This overlapping execution avoids any gap in processing capacity during the transition."* — **R4가 겨눈 문장** | `hs_vision_vram_incr_max`, `hs_vision_vram_headroom`, `hs_vision_rss_incr` |
| L374 부근 | NPU는 동시 실행 불가 서술 | (§7 미계측 사실) |
| Threats External | 32 GB/62 GiB 환경의 일반화 한계 | `hs_host_ram`, `hs_vram_total`, `hs_gen_rss_headroom` |

### 6.2 메커니즘 검증 — 픽스처에서 통과 확인

논문에 문장이 없으므로 **`scripts/testdata/paper_fixture`**(논문 사본이 아닌 시험 하네스)에
v38 값을 쓰는 문장을 넣어 네 새 어휘가 실제로 검사되는지 확인했다:

```
$ python3 scripts/check_paper.py --paper scripts/testdata/paper_fixture
tracked values:  11 checked, 0 mismatch
[OK] PAPER MATCHES CONFIRMED TABLE (0 mismatches).
```

### 6.3 **검사기가 잡은 것 — 받은 주석 하나가 틀렸다**

이번 라운드에 새 `main.tex`가 도착했고 v37에서 등재한 Q4 값으로 문장이 복원돼 있었다.
`check_paper.py`가 즉시 불일치를 냈다:

```
main.tex:741: q4_b_lastV = 294.2 V 가 이 줄에 없다
    줄의 수치: 362, 112, 293, 315, 480, 46, 298, 323
    확정표를 줄의 자릿수로 반올림하면: 294
```

**본문이 아니라 주석이 틀렸다.** 문장은 *"end lower, at $293$--$315$"*로 범위를 적는데, 그 범위의
양끝은 `q4_c_lastV`(292.6→293)와 `q4_adaptive_lastV`(315.1→315)이며 **`q4_b_lastV`(294.2)는
범위 안쪽이라 인쇄되지 않는다.** 같은 줄의 peak 범위 298–323도 `q4_b_maxV`·`q4_adaptive_maxV`로
설명된다. 정정된 주석:

```latex
%!cv q4_bg_lastV,q4_c_lastV,q4_adaptive_lastV,q4_bg_maxV,q4_b_maxV,q4_adaptive_maxV
```

이것으로 그 줄의 수치 8개가 전부 설명된다. **정본에서 교체해야 한다** — `apply_cv_annotations.py`는
이미 주석이 있는 줄을 건너뛰므로 교체를 수행하지 못한다(도구의 한계, §8).

### 6.4 목록 갱신과 저장소 밖 검증

새 `main.tex`에서 v36/v37 목록의 3줄(627·743·817) 해시가 어긋나 있었다 — 본문이 바뀐 자리다.
`gen_cv_worklist.py`가 현재 `main.tex`에서 해시를 다시 뽑는다. 또한 **글자로 쓴 수가 아라비아
숫자로 바뀌어**(v37 §4.3이 지적한 항목) L627·L817에 `q4b_a_recoveries`·`q4b_a_hotswaps`·
`q4b_bg_hotswaps`·`q6_a_hotswaps`·`q6_bg_hotswaps`를 붙일 수 있게 됐다. 목록에 넣었다.

```
$ python3 scripts/apply_cv_annotations.py --tex <저장소 밖 사본>/main.tex
17 applied, 0 skipped

$ python3 scripts/check_paper.py --paper <저장소 밖 사본>
paper.lock:      OK (main.tex + 13/13 figures)
tracked values:  85 checked, 0 mismatch
untracked:       225 numerals (190 on 56 lines without %!cv, 35 on 14 annotated lines that no id explains)
figures:         13 referenced, 13 present

[OK] PAPER MATCHES CONFIRMED TABLE (0 mismatches).
```

**85건 전부 통과**(목록 79 + 배포본이 이미 달고 온 6, L741은 §6.3의 정정을 적용한 상태).
v37의 73 → **85**.

---

## 7. **NPU는 계측하지 않았다**

이번 시나리오는 CPU–GPU다. NPU DRAM의 전환 피크는 **직접 재지 않았다.** 알고 있는 것은:

- NPU DRAM 16 GB가 구속 조건이라는 것은 §IV-A와 Threats에 이미 있고, v33 Q5에서 vision 3 + 양자화
  LLaMA-1B **4개가 동시 상주**함이 확인돼 있다(`main_rerun_report.md` Q5).
- 코드 경로는 같다 — 교차이동, kill-then-start, 사전 점검 없음(§4는 플랫폼 무관한 판독이다).
- **그러나 전환 중 DRAM 피크의 크기는 미상이다.** GPU에서 vision swap이 +574 MiB를 만들었으므로
  (§3.3) NPU에서도 0이라고 가정할 근거가 없고, 여유는 32 GB가 아니라 16 GB 기준이 된다.

**별도 라운드로 남긴다.** 이 사실을 논문이 밝혀야 한다.

---

## 8. 커밋

| 커밋 | 내용 |
|---|---|
| `573b4ac` | 재측정 9런 · 확정표 +22항목 · 계측/분석/구조검사 스크립트 5종 · `run_scenario.py --mem` · 주석 목록 갱신 · 본 보고서 |

**푸시하지 않았다.**

---

## 9. 남은 것 — **사람 판단**

1. **§III L364의 문장(§6.1).** *"This overlapping execution avoids any gap in processing
   capacity"*는 R4가 겨눈 바로 그 주장이고, 이제 그 겹침의 비용이 측정돼 있다
   (VRAM +574 MiB / 여유 49×, host RSS +171 MiB). **문안은 쓰지 않았다.**
2. **v18의 "VRAM 증분 0" 서술(§3.3).** 초안 문안이 그렇게 적혀 있었다면 정정이 필요하다.
   교차이동은 **한 뷰의** 중복을 막을 뿐 **디바이스 합의** 오버슈트를 막지 않는다.
3. **L741 주석 교체(§6.3).** 도구가 교체를 못 하므로 정본에서 손으로 바꿔야 한다.
4. **L743의 낡은 peak(146 / 75–78).** 같은 절 L741은 이미 480±46 / 298–323으로 갱신됐는데
   L743은 옛 값을 유지한다 — 한 문단이 같은 양을 두 값으로 서술한다(v37 §4.1과 같은 형태).
5. **NPU 전환 피크(§7).** 계측할지, 미계측으로 명시하고 넘어갈지.
6. **사전 자원 점검 부재(§4-3).** v18이 "실질 잔여 위험"이라 부른 것이 그대로다. v19 작업 73의
   fail-safe는 여전히 보류다.

# 언어모델 deadline miss(y2) 측정·집계 방식 조사

> **성격**: 조사(read-only). 코드·데이터 수정 없음. 수치는 전부 **본수집 540창 × 2플랫폼**
> (`xgboost_model/full_collection_540/`, 파일럿 160창 아님)에서 실측.

## 0. 결론 (요약)

| 질문 | 판정 |
|---|---|
| 언어 deadline이 y2에 반영되나? | **반영됨** — 측정 y2가 CPU-llama를 페널티(S3 Δy2 **+0.149**) |
| 요청 수(0.03%)로 희석되나? | **아니오** — y2는 요청-pooled가 아니라 **per-view 평균** |
| 그럼 문제는 없나? | **부분 과소평가 있음** — per-view 평균이라 LM 발언권이 **1/N views**로 희석 (S8은 +0.030뿐) |
| y3가 보완하나? | **예, 지배적으로** — CPU-llama y3 손실 −6.7~−12.6, 세트 크기에 안정적 |
| B.5 "NPU는 llama를 CPU에" 결론은 y2 무반영 탓? | **아니오** — `compare_platforms.py`가 **raw fps로 점수**(y1 지배)낸 산물. 정규화 점수론 llama=NPU가 최적 |

---

## 1. 모델 유형별 deadline 정의 (코드 근거)

**요청 하나의 deadline은 모델 종류와 무관하게 동일 공식** — `model_registry.py:23-27`:
```
deadline_ms = (1000 / infps) * DEADLINE_FACTOR       # DEADLINE_FACTOR = 3.0
infps = rate_factor * baseline_rate                  # 모델별 baseline
```
- 종류(vision/llm/vlm)별 분기 없음. 언어모델도 **자기 infps 기준 period deadline**을 받는다.
- e2e latency는 언어모델도 계산됨 — `model_processors.py:516` `latency_ms = wait_ms + gen_ms`
  (대기 + **생성 전체 시간**). 즉 생성 1건 통째의 지연이 deadline과 비교된다.
  (TTFT/TPOT가 아니라 **생성 1건 e2e 기준**.)

**실제 deadline 값** (같은 공식, infps만 다름):
| 모델 | baseline | deadline @1.0× | @2.0× | @3.0× |
|---|---:|---:|---:|---:|
| llama1b | 0.138 | **21.7 s** | 10.9 s | 7.2 s |
| qwen2_vl | 0.217 | 13.8 s | 6.9 s | 4.6 s |
| resnet50 | 157.3 | 19.1 ms | 9.5 ms | 6.4 ms |
| yolo11s | 44.9 | 66.8 ms | 33.4 ms | 22.3 ms |

→ 언어 deadline은 **초 단위**(관대), 비전은 **밀리초**. CPU llama 생성(~198 s) ≫ 21.7 s → 전건 late.

## 2. y2 집계 단위와 분모 (코드 근거)

**`unified_viewer.py:1298-1326`** — per-view 집계 후 **단순 평균**:
```
offered_v = round(infps_v * window_sec)      # 뷰별 demand (1316)
on_time_v = completed_v - late_v             # e2e latency <= deadline (1311)
rate_v    = (offered_v - on_time_v) / offered_v          # 뷰별 miss rate (1319)
y2 = mean(rate_v for v in views)             # 뷰 동일 가중 평균 (1326)
```
- **분모는 뷰별로 각자**(offered_v). 세트 전체 요청을 pooling하지 않는다.
- 주석(1322-1324)이 명시: *"each application weighted equally, so it is not dominated by
  the highest-rate model"* — **의도적으로 요청 수 격차를 무시**.
- **비전 miss와 언어 miss는 분리 가능**(뷰별 `deadline_miss_rate` 계산됨). 단 수집 시
  window에는 **총합 y2만 저장**되어(재구성한 540창도 동일), 저장 데이터에서 뷰별 miss는
  직접 못 꺼낸다. → offered는 스케줄에서 결정적 재구성, 완료건수는 저장됨(아래).

## 3. 실측: 언어 vs 비전 창당 요청 수 · y2 분모 비율

본수집 540창, 창당(180 s):

| | 완료건수/창 (중앙값) | offered/창 (중앙값) | 완료율(중앙) |
|---|---:|---:|---:|
| llama1b (NPU) | 8 | 50 | 0.17 |
| qwen2_vl (NPU) | 37 | 68 | 0.56 |
| vision (전체) | **3,067** | (수천~2.8만) | — |

- 언어 완료건수(수~수십)는 비전(수천)보다 **100~1000배** 적다.
- **언어 요청이 전체 offered에서 차지하는 비율: 중앙값 0.03%, 최대 0.22%.**
  → **만약 y2가 요청-pooled였다면 언어 deadline은 사실상 안 잡힌다(0.03%).**
  → 그러나 y2는 **per-view 평균**이라 언어도 **1/N views 가중**을 받는다. 이 집계 방식이
    가설(요청 수 희석)을 **막고 있다.**

## 4. llama1b-CPU 조합의 y2 실제 기여 (핵심)

S3(llama,resnet,yolo11s)에서 **vision 배치·rate가 같고 llama만 CPU↔NPU**인 12쌍 비교(측정):

| llama | 측정 y2 (평균) | 측정 y3 (평균) |
|---|---|---|
| CPU | 높음 | ~0 |
| NPU | 낮음 | 높음 |
| **차이 Δ(cpu−npu)** | **y2 +0.149** (12쌍 전부 +) | **y3 −12.57** |

→ **CPU-llama는 y2를 실제로 끌어올린다(+0.149).** 느린 CPU 생성이 offered(≈25-50) 대비
거의 미완(완료 0-1) → 그 뷰 miss≈1.0 → 평균에 1/3 기여. **y2는 무영향이 아니다.**

### 세트 크기별 1/N 희석 (과소평가의 실체)
| 세트 | views | Δy2(cpu−npu) | Δy3 | 이론 1/N |
|---|---:|---:|---:|---:|
| S3 | 3 | **+0.149** | −12.6 | 0.33 |
| base1 | 4 | +0.109 | −9.0 | 0.25 |
| S5 | 5 | +0.081 | −8.5 | 0.20 |
| S6 | 6 | +0.066 | −8.1 | 0.17 |
| S7 | 7 | +0.047 | −6.9 | 0.14 |
| S8 | 8 | **+0.030** | −6.7 | 0.12 |

→ Δy2가 **~1/N을 정확히 추종**. 언어 deadline 신호가 **세트가 클수록 희석**되어 S8에선
+0.03뿐. **이것이 과소평가의 실체** — 요청 수 때문이 아니라 **per-view 평균의 1/N 가중** 때문.

## 5. y2 / y3 역할 분담

- **y2**: 언어 deadline miss를 잡되 **1/N로 희석**(위).
- **y3**: CPU-llama 토큰 손실 **−6.7~−12.6** — 세트 크기에 **훨씬 안정적**(3→8 views에서 2배만 감소,
  y2는 5배 감소). y3는 토큰 **합**이라 뷰 수로 평균되지 않기 때문.
- → **언어를 CPU에 둔 페널티는 y2보다 y3가 지배적으로 본다**, 특히 대형 세트에서.
- **단 y3 마스킹 주의**(`SCORE_FORMULA_AUDIT.md`): CPU-llama의 y3는 학습에서 마스킹
  (`y3_valid=False`)되고 스코어링은 세트단위라 **외삽된 y3**로 페널티를 준다. 측정 y3는 0이지만
  학습 라벨로는 안 쓰이므로, 예측기의 CPU-llama 페널티는 외삽 정확도에 의존.

## 6. B.5 "NPU는 llama를 CPU에 배치가 최적" 재검토

- B.5의 `compare_platforms.py:33`은 **raw 측정값**으로 점수:
  `score_combo(total_throughput_fps, deadline_miss_rate, total_tokens_per_s, ...)`.
  raw fps(≈100-200)가 α·y2(0.3×~0.9)·β·y3(0.5×~16)를 **압도** → 사실상 "최대 throughput 선택".
  CPU-llama가 NPU를 vision에 양보해 최대 fps → **raw 점수로만 CPU-llama가 최적**으로 보였다.
- **정규화 점수(예측기 실제 방식, y1/y3 [0,1])로 재계산하면 S3 최적은 llama=NPU**
  (combination_164/172/180, 전 rate). y2·y3가 제 몫을 하기 때문.
- **결론: B.5의 CPU-배치는 y2 무반영이 아니라 raw-throughput 지배의 산물.** y2는 CPU-llama를
  올바르게 페널티(+0.149)하고, 정규화 스코어링에선 NPU-llama가 이긴다.

## 7. 판정 + 수정 제안 (제안만, 미변경)

**판정: 언어 deadline은 y2에 반영되나, per-view 평균의 1/N 가중으로 대형 세트에서 과소평가된다
(요청 수 희석은 아님). y3가 그 손실을 지배적으로 보완하며, B.5의 CPU-배치 결론은 y2가 아니라
비교 스크립트의 raw 스코어링 탓이다.**

과소평가 정도: 소형(3뷰) 세트는 Δy2 +0.15로 충분, 대형(8뷰)은 +0.03로 미미. y3(−6.7~−12.6)가
전 크기에서 보완하므로 **점수 레벨의 실질 왜곡은 제한적**. 단 y3 마스킹 외삽 의존이 잔여 리스크.

수정 제안(우선순위 순, 본 조사 범위 밖):
1. **가장 실효**: 오프라인 비교(`compare_platforms.py` 등)를 **정규화 점수로 통일**. raw fps 스코어링이
   B.5류 왜곡의 실제 원인. (예측기 GUI 경로는 이미 정규화되어 있음.)
2. y2 집계를 **요청 가중과 per-view 평균의 혼합**으로 두어 대형 세트에서 언어 발언권을 유지할지 검토.
   (현행 per-view 평균은 "애플리케이션 동등" 설계 의도가 있으므로 트레이드오프.)
3. 언어 deadline을 **별도 지표**(TTFT/TPOT 또는 생성 완료율)로 분리해 y2와 함께 리포트,
   score에는 y3가 이미 토큰 비용을 반영하므로 이중계산 주의.
4. y3 마스킹-스코어링 불일치(조합단위↔세트단위) 정합 — `SCORE_FORMULA_AUDIT.md` 제안과 연동.

> **계수(α/β) 재검토와의 선후**: 요청대로 이 조사가 선행했다. 결과 — y2는 언어를 (제한적이나마)
> 반영하므로 **계수 재검토는 유효**하되, 대형 세트의 1/N 희석과 y3 의존을 감안해 진행 권장.

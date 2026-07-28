# β 정합성 확정 및 랭킹 재생성 (v23)

작성 2026-07-28. 성격: 확정 + 재생성. 스윕·스케줄 생성·재실행 없음. **커밋(푸시 없음).**

---

## 작업 96 — β 정본 확정 (게이트) → **정본 β = 1.0**

결과가 아니라 **근거**로 정했다. 근거를 강한 순서로:

### 96-1-2 (가장 강한 근거) 예측기 평가가 전제한 β = **1.0**
형제 저장소 `../multimodel-scheduling-mobilint/evaluate_model.py` — 예측기 **품질 평가**(Spearman
rank corr, top-1/top-5 hit, score ratio)를 산출하는 코드:
- `--beta` 기본값 = **1.0**, 주석 *"LM y3 weighted equally with vision y1"* (line 190).
- oracle 랭킹과 예측 랭킹을 **동일 β로** 합성(`both S = T + beta*T_tok - alpha*miss`, line 109/125/131).

→ 배포 β가 0.5이면 **보고된 품질 지표가 실제 배포된 랭커를 서술하지 않게 된다.** 게이트 96-2 첫 칸
("평가가 β=1.0으로 산출됨 → 1.0, 배포는 평가와 같아야 한다")에 정확히 해당.

### 96-1-1 git 이력
`DEFAULT_BETA=0.5`는 커밋 **f02331b (2026-07-20) "swap in MLA100-retrained xgboost predictor"** 에서
설정됐다. `XGBOOST_MLA100_SWAP.md §6 (미결 항목 — β값 충돌, 사람 판단 필요)`가 그 시점에 이미
명시했다: *"작업 지시문은 논문 공식으로 β=0.5를 명시했고, 이식해 온 MOBILINT 코드의 canonical default는
β=1.0이다."* — 즉 0.5는 **미결 상태로 남아 있던 값**이지 확정이 아니었다.

### 96-1-1 (근거의 반증) "논문 공식 β=0.5"는 실재하지 않음
`deploy_predictor_logic.py`의 옛 주석은 *"the FSRR/BoundGuard paper defines S = y1 + 0.5*y3 - 0.3*y2"*
라 적었으나, **논문은 β를 언급하지 않는다**(§IV의 `β_B`는 버퍼 계수로 별개). 0.5의 유일한 정당화였던
"논문 공식"이 실재하지 않으므로, 0.5는 **드리프트**다(기본값과 결정의 어긋남 — phantom 사고와 동류).

### 96-1-3 / 96-1-4 번들·학습 반영 여부
- 번들(`deploy_*_{y1,y2,y3,features,coverage}.json`)에 **α·β 저장 없음**(스캔 확인).
- β는 **점수 합성 시점 정책 가중치**이며 **학습 밖**이다(양 저장소 suite 주석·`score_combo` 확인).
  → 게이트 96-2 "학습에 들어가 있음 → 정지" **아님**. 모델 불변, LM 포함 셋의 랭킹만 바뀜, vision-only는
  임의 β에서 동일.

### 판정
| 근거 | 값 |
|---|---|
| 예측기 품질 평가 전제 β (mobilint evaluate_model.py) | **1.0** ← 결정적 |
| 양 저장소 suite `DEFAULT_BETA` | 1.0 (설계 의도 "동등 가중") |
| fsrr `deploy_predictor_logic` 옛 값 | 0.5 (드리프트, 실재하지 않는 "논문 공식"에 근거) |
| 번들 저장 / 학습 반영 | 없음 / 아님 (STOP 아님) |

**정본 β = 1.0.** (α = 0.3 불변.)

---

## 작업 97 — 드리프트 지점 봉쇄

1. **`rankings/generate_rankings.py`가 α·β를 명시**한다. `DEFAULT_*` 암묵 상속 제거 →
   파일 상단에 `CANONICAL_ALPHA=0.3`, `CANONICAL_BETA=1.0`를 **직접 선언**하고 그 값으로 생성.
2. **`deploy_predictor_logic.DEFAULT_BETA` 0.5 → 1.0** 정합화. 주석을 정본 근거(평가 전제 β=1.0,
   0.5는 드리프트)로 교체. 이 값을 기본으로 쓰던 유일한 신예측기 호출부(`best_deploy_finder_executor`)는
   이제 자동으로 정본을 쓴다(검증 스크립트들은 legacy 예측기라 무관).
3. **비정본 α·β 거부**: 생성기는 정본과 다르면 `--allow-noncanonical` 없이는 `SystemExit`.
   - 확인: `--beta 0.5` → 거부(`Refusing to generate ... non-canonical`); `--beta 0.5 --allow-noncanonical`
     → `[NON-CANONICAL]` 표시 후 실행.

---

## 작업 98 — 랭킹 재생성 (정본 β=1.0)

v22와 동일 시나리오×플랫폼 8종 전부 재생성. 어휘 검증(정확 이름, 별칭 없음) 통과.
`_provenance.targets.beta = 1.0` 기록 확인.

- vision-only(Q1.3, Q2)는 β 무관 → 점수 **불변**(0.985465 / 0.92888 그대로). β 격리 검증.
- LM 포함(Q3/Q5/Q6/2-gen)만 점수·순위 변동.

### 98-1. 파생 수치 대조 (값은 사람이 정함)

| 항목 | legacy(옛 예측기) | v22 β=0.5 | **정본 β=1.0** |
|---|---|---|---|
| Q3/Q6 cpu-gpu 회복 배치 순위 | 5 | 3 / 32 | **4 / 32** |
| Q5 cpu-npu 회복 배치 순위 | 2 | 2 / 32 | **2 / 32** |
| Q3 2-gen 회복(llama1b만; qwen2_vl CPU 금지) | — | 해당없음 | **6 / 32** |

- **회복 위 순위**: GPU rank 1–3, NPU rank 1 모두 **생성 모델을 가속기에 유지**하는 배치(옛 워킹셋-크기
  의존 구조 재확인). β=1.0이 y3를 더 보상해 GPU 회복이 3→4로 한 칸 내려감(생성 모델을 가속기에 두는
  배치의 점수가 상대적으로 오름).
- **결정론(β=1.0 재확인)**: 회복 순위 3회 동일 `[4,4,4]`(GPU) / `[2,2,2]`(NPU). 피처 벡터에 런타임 상태
  없음은 β와 무관하므로 v22 판정 유지.
- **rate 민감도(β=1.0)**: foreground rate 0.25×~4×(16배)에서 회복 순위 **불변**(GPU 4, NPU 2).
- **`qwen2_vl` CPU 금지**(DEVICE_CONSTRAINTS gpu/npu)는 β 무관 — 2-gen 회복이 llama1b만 이전 가능인
  구조 불변.

### 유지되는 v22 결론
결정론, 생성 시점 어휘 검증, 아티팩트 인터페이스((b), fsrr 내 커밋 번들), legacy 격리 — 전부 유지.

---

## 작업 99 — 커밋 (푸시 없음)

- 재생성 랭킹 8종(β=1.0) + 옛 β=0.5 랭킹 `rankings/stale_beta0.5/`(STALE.md 동봉, 보존).
- `rankings/generate_rankings.py`(명시 α·β + 비정본 거부), `deploy_predictor_logic.py`(DEFAULT_BETA=1.0
  + 주석), 본 리포트.
- `git ls-files rankings/` 출력은 커밋 절에 기재. **푸시하지 않음.**

---

## 하지 않은 것

- **β 스윕** — 범위 밖(정책 품질 축은 예측기 논문의 몫).
- **N_cand 값 결정 · 스케줄 생성 · 재실행** — 랭킹 확정 후, 사람.
- **논문 문안** — 본문은 β 미언급, 불변.
- **재학습** — β는 학습 밖이라 불필요(96 판정).
- **푸시.**

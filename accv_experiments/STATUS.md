# STATUS — ACCV 논문 프로젝트 진행 상황 (Claude Code 세션 복원용)

_이 파일을 읽으면 새 세션에서 맥락이 복원됩니다. 디스크의 모든 산출물은 세션과 무관하게 보존되어 있습니다._

## 프로젝트
ACCV 2026 더블블라인드 논문. 주제: **streaming-perception 평가의 blind spot** — single-camera isolated 평가가 multi-camera 배치(GPU/NPU)를 잘못 고른다.
- 플랫폼: RTX 5090 GPU + Mobilint MLA100 NPU(ARIES, INT8, global8, mxq `b2441f9d`)
- 검출기: YOLOv11s(main), 데이터셋: Argoverse-HD(forward camera, 24 val logs)
- 양쪽 device **threads=4**(올바른 operating point)
- 논문 파일: `accv_experiments/paper/main_vision.tex` (+ `main.bib`)

## 현재 상태 (논문은 사실상 완성 단계)
- 그림 3개(정성 실패 / contention sweep 3곡선 / regret 곡선), 표 6개
- 기여 C1(평가 실패) · C2(두 손실 인과분리) · C3(worst-stream) 모두 다중 증거
- detector 일반성: **4 검출기 × 3 아키텍처** 재현 완료 → 아래 Table 6
- 평가-실패 프레이밍 일관(제목·abstract·contribution·Discussion). 페이지 ~15(나중에 압축 예정).

## 완료된 측정 (results/ 에 CSV·로그 보존됨 — 절대 삭제 금지)
- rev21: N=4 worst/mean (main 표)
- rev22: per-stream sAP, bit-identical
- rev24: N=2/4/8 sweep (Fig 2)
- rev25: per-size under contention (Table 4)
- rev26: YOLOv8n cross-detector (Table 6)
- rev27: YOLOv8s(재현✓), YOLOv10s(export 손상, 부적합)
- rev27b: YOLOv10s 진단 → H2(vendor export 양자화 손상, 우리가 못 고침) 확정
- rev27c: YOLO12s STEP0 게이트 PASS(small-biased) + STEP1-3 측정 **완료, 재현✓**
- rev28: systematic-failure 재분석(regret/metric/dominance) — A2/A3/A5 논문 반영됨, A1 미사용

## Table 6 (cross-detector, 현재 논문에 반영된 값)
| Detector | isolated small | isolated large | VLM | worst NPU/GPU | crossover |
|---|---|---|---|---|---|
| YOLOv11s | GPU (−48%) | tie (+0.1%) | NPU | 0.083/0.016 | ~48% |
| YOLOv8n | GPU (−36%) | GPU (−8.1%) | NPU | 0.060/0.009 | ~36% |
| YOLOv8s | GPU (−44%) | tie (−0.3%) | NPU | 0.080/0.015 | ~31% |
| YOLO12s | GPU (−36%) | tie (−1.2%) | NPU | 0.080/0.011 | ~45% (sweep 간격상 근사) |
- YOLOv10s는 export 손상(large −70%, large-biased)으로 제외 → Limitations에 정직 note.

## 진행 중 막힌 것 (중요하지 않음)
- `results/rev27c_summary.md` 파일 생성이 **출력 형식 오류(도구 호출 앞에 잘못된 토큰 누출)**로 중단됨.
- **이 파일은 없어도 논문 작업에 지장 없음** — rev27c 측정 데이터(CSV·stdout)는 이미 보존·반영 완료.
- 새 세션에서 다시 만들려면: 아래 "다음 할 일" 참고. 만들 때 **bash heredoc**(`cat > path << 'EOF'`)으로 쓸 것(파서 민감도 회피).

## 다음 할 일 (우선순위)
1. (선택) `rev27c_summary.md` 재생성 — `cat > ... << 'EOF'` heredoc으로. 내용: 위 Table 6의 YOLO12s 행 + STEP0 게이트 PASS + 재현 판정.
2. **실제 llncs+accv 클래스로 최종 빌드** — `paper/`에서 `pdflatex → bibtex → pdflatex×2`. 정확한 페이지 수·overfull 확정(현재 article 하니스 15쪽; llncs는 더 조밀).
3. 페이지 한도(보통 14쪽) 초과 시 압축(Intro↔Sec2↔Sec3 중복 제거 등).
4. (낮은 우선순위) 재현 패키지 + paper 폴더 정리 + 설명 HTML — 지시문 `accv_paper_results_data_request.md` 참고.

## ★ 핵심 가드레일 (항상 준수)
- **날조 금지.** 측정 안 된 값 보고 금지. 안 나오면 안 나온 대로.
- 모든 device threads=4. core 측정 스크립트 수정 금지(detector만 교체).
- 측정 작업 **동시 실행 금지**(threads=4 operating point·frame-skip 오염). 단독 실행.
- 새 export는 STEP 0 게이트(로드 + small-biased sanity)부터. large-biased면 부적합(v10s 교훈).
- 논문 수정 후 항상 2-pass pdflatex + 미해결 참조 0 + cite↔bib 정합 확인.
- **출력 규율**: 도구 호출 블록은 함수 태그로 바로 시작. 그 앞에 어떤 텍스트/토큰도 두지 말 것. 파일 쓰기는 가능하면 bash heredoc 사용.

## EXP-AUDIT-V10QGAP (2026-06-18)
YOLOv10s FP32(GPU) vs INT8(NPU) per-size sAP, single-stream tab:single-stream 경로(threads=4, isolated N=1,
L0, 24 logs×3reps, skip 0%/0%). INT8 = 올바른 vendor export `~/.mblt_model_zoo/vision/aries/global8/yolov10s.mxq`
(rev27 generality는 mxq를 yolov8s.mxq로 잘못 로드했던 정황 → STEP0a에서 확인, 본 측정은 정정).
- 결과(상대 %): small −45.0% / medium −19.8% / large **−4.6%** → **small-biased (v11s와 동일 구조)**.
- large 3수치: **FP32 large=0.4525, INT8 large=0.4319, gap_large=+0.0206**.
- VERDICT: **large-object collapse 재현 안 됨**(A/B/C 밖). FP32 large healthy(≈v11s 0.477), INT8 small-biased.
  → YOLOv10s를 large-collapse negative case로 쓰는 서술은 이 측정으로 미지지(원 negative는 export-identity
  mix-up 가능성). 산출물: v10qgap_raw.csv, v10qgap_means.json, v10qgap_stdout.log.

## EXP-GEN-V10 (2026-06-18)
YOLOv10s detector-generality 행 (correct export). mxq sha256=1aae03e4...(yolov10s.mxq, audit와 동일).
generality harness(rev27) 재사용, threads=4, N=4, 3reps. 기존 오염 rev27_yolov10s_*.csv는
*.OLD_yolov8smxq.bak로 백업. 측정 4필드: isolated small −44.8% / isolated large −4.6% /
worst_NPU/worst_GPU(VLM)=0.082/0.016 / crossover ≈30%(k=1, grid k∈{0,1,2,4,8}→skip{0,30,45,58,72}%).
Table7 행: YOLOv10s & GPU(−45%) & (−4.6%) & NPU & 0.082/0.016 & ≈30%. positive case(이전 negative는
yolov8s.mxq mix-up). 산출물 rev27_yolov10s_{single_stream,contention,sweep}.csv, _CORRECTED_stdout.log.

## EXP-GEOM-STALENESS (2026-06-19)
GT-only 기하 staleness probe (검출기/GPU 없음). Argoverse-HD val 24 logs, association=track IDs(coverage
d=1/2/3=99.2/98.5/97.8%), 크기=delivery frame(t+Δ) COCO bins. frozen IoU(box_t,box_{t+Δ}) per size.
- 결과: frac_below_0.5:0.95 순서 = small>medium>large (모든 Δ). raw disp는 large가 큼(13.5 vs 3.85@d2)
  이나 정규화 disp는 small이 큼(0.186 vs 0.075) → 큰 박스 IoU 견고성이 지배.
- VERDICT: **UNSUPPORTED** — d=2 frac_below_0.5:0.95: small=0.4297 / large=0.2229 (large<small; 가설은
  large>small). 측정 staleness(large>medium>small)와 정반대 → 단순 기하 설명 논문 반영 금지, 대안 원인
  재검토 필요. 산출물: geom_staleness_raw.csv, geom_staleness_means.json, phase_geom_staleness.py.

## EXP-SIZE-MASS (2026-06-19)
복구 가능 정확도(TP) 크기 분포 @ clean point. prediction source = REGENERATED clean single-stream GPU FP32
(yolo11s.pt, threads=4, conf=0.25, skip~0). TP = COCO-style greedy match IoU0.5(+0.5:0.95). honest TP counts
(share-weighted AP 미사용).
- GT share: small 38.5% / medium 40.8% / large 20.7% (총 253,941).
- TP share: small 8.8% / medium 48.4% / large 42.9% ; recall small 0.092 / medium 0.479 / large 0.837.
- VERDICT: **SUPPORTED** — small TPshare 8.8%(recall 0.092) vs large 42.9%(recall 0.837); 복구 가능 정확도
  medium+large에 91% 집중. → EXP-GEOM(small 개별 fragile)와 Table2(small staleness≈0) 화해: small은 탐지가
  안 돼 잃을 정확도가 없음. heuristic TPshare×fracBelow=medium>large>small로 large>medium 정확순서까지는
  미재현(고-skip 대형Δ 추정, 미측정). 산출물: size_mass_raw.csv, size_mass_means.json, phase_size_mass.py.

## EXP-GEOM-BIGDELTA (2026-06-19)
GT-only frozen-IoU probe 확장 (Δ∈{1,2,3,5,8,12}, 2 cohort, translation/scale 분해). coverage 전 Δ ≥90%
(survivorship flag 없음). Δ=2 regression 재현 OK(0.430/0.342/0.223).
- frac-below: 모든 Δ에서 small>medium>large (large 항상 least fragile). large−medium gap Δ2 −0.119→Δ12 −0.037
  (닫히나 crossover 없음).
- translation-only=small>med>large(병진은 small-biased); scale-only(looming)=large>med>small(Δ12 0.405/0.288/0.233,
  large-biased!) 그러나 translation이 full IoU 지배 → 순효과 small>medium>large 유지.
- VERDICT: **NOT CLOSED** — Δ≤12 내 large가 medium 추월 없음(Δ12 large 0.643 < medium 0.680). 현실 Δ2–5에서
  gap 최대. 기하는 large>medium 미설명 → 보수적 서술(small≈0=detectability, large vs medium=측정값 그대로).
  산출물: geom_bigdelta_raw.csv, geom_bigdelta_means.json, phase_geom_bigdelta.py.

## EXP-CROSSOVER-COLLAPSE (Path A, 기존 데이터만 — 신규 측정 없음)
- 질문: detector별 worst-stream 교차점(raw skip≈29/31/36/44/48%)이 정규화 L(C*)/Q≈1로 collapse하는가?
- 정의(tautology-safe): Q=isolated N=1 all-size NPU deficit(single_stream; sweep 아님), L(C*)=GPU^worst(0)−NPU_flat=D_sweep. R=D_sweep/Q_iso.
- Q_iso는 step1 stdout area=all 블록(streaming-sAP phase0=GPU/phase2=NPU)에서 추출, 저장된 per-size CSV로 검증(오차<0.003).
- 결과 R: v8n 1.04 / v8s 1.72 / v10s 1.25 / v12s 2.66 / v11s 0.81 (range 0.81–2.66, CV 44%).
- VERDICT: **NOT SUPPORTED** — independent Q로는 collapse 안 함(raw skip CV 19%보다 분산 더 큼). L/Q≡1은 Q:=D_sweep일 때만 성립(tautology, 금지). 함의: isolated 단일스트림 deficit로 다중카메라 교차점 예측 불가(worst-stream이 deficit를 detector별 1.0–2.7× 증폭) → 논문 핵심 주장 강화.
- 산출물: crossover_collapse.csv, crossover_collapse_summary.md. core/results/.tex 불변.

## rev29 — 조건 의존적 최적 배치 (단독 실행, 신규 측정)
- 설계: v11s, N=4, threads=4, NPU global8, ResNet50 압력 sweep(k=0/1/2/4/8) + VLM 포화점. split ratio r∈{0..4}, large-first 이동 [3,21,22,2]. 각 셀 3 reps. sanity pre/post: npu_infer 10.2/10.1ms skip 0% large_gap -0.0223(안정). RES0 All-GPU 0.1148=rev22 재현.
- worst-stream(All-GPU/split/All-NPU/Oracle@ratio):
  RES0(skip0.2%): 0.1148 / - / 0.0835 → **All-GPU**(split1 0.1149=노이즈 tie)
  RES1(skip38%): 0.0887 / **split1 0.0910** / 0.0834 → **split이 둘 다 이김**(3/3 rep, vs GPU +0.0023, vs NPU +0.0076; split1 gpu_skip~19% 안정, 33ms경계 아님)
  RES2(skip51%): 0.0813 / 0.0795(r1) / 0.0834 → All-NPU(split2 0.0836 tie)
  RES4(skip59%)/RES8(skip72%): All-GPU 0.077/0.069 << All-NPU 0.0835 → **All-NPU**(split2 tie)
  VLM(skip97%): All-GPU 0.0187 / split≤0.024 / **All-NPU 0.0832** 지배
- VERDICT: **3-regime 확인(뉘앙스 포함)** — 저경합 All-GPU / 중간(skip~38%) split 엄밀 우위(안정점) / 포화 All-NPU. 389행 강한 주장("mixed never beats") **반증**. 단 split 엄밀 우위 대역 좁음(RES1 1개), 고경합서 split는 All-NPU로 수렴(이기지 않음). isolated 평가는 전구간 All-GPU 처방 → 중간·포화 mis-rank(C1 강화). Oracle=진단 상한(스케줄러 아님).
- 산출물: rev29_split_sweep.csv, rev29_oracle_by_contention.csv, rev29_report.md, rev29_sanity.csv, phase_rev29_split.py. core/기존 results/논문 불변.

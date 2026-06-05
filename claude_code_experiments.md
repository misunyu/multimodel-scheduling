# 다음 작업 리스트 + 바로 다음 Claude Code 지시문 (rev 4)

대상 파일: **`paper/main_vision.tex`** (방금 narrative·HW·정합성 수정 완료본).

---

## A. 무엇이 방금 수정됐나 (참고)

- HW: "MLA100"은 **ARIES 칩 기반 PCIe 카드**임이 확인됨(80 TOPS/25W/8코어 일치). 명칭 정밀화 완료,
  메모리 수치는 `% [VERIFY-HW]`로 표시. (별도 "ARIES100" 제품은 없음 — ARIES=칩, MLA100=카드.)
- finding 4(ordering): abstract/§3.3/§7 intro/conclusion의 "small-rich first" **무조건 단정**을
  "argmax(L−Q)가 먼저 — anchor(v11s/v8s)에서는 small-rich, 극단 capacity에서는 large-rich"로 한정.
  Proposition 1(L>Q)·C1(역전 존재)은 universal로 유지.
- finding 5(L≈0): contention C가 **background + 동시 스트림(concurrency)** 을 모두 포함함을 모델에 명시.
  single-stream L≈0는 반례가 아니라 "single-camera는 L을 못 본다"는 thesis의 **확인 증거**로 재배치.
  gen_decomp의 L≈0와 gen_gain의 multi-stream 역전을 본문에서 연결.
- gen_decomp 절대단위 Q가 large에 큰 것 + global8/single 모드 차이를 본문·Limitations에 정직 명시.

---

## B. 다음 작업 리스트 (우선순위순)

### B1 — [데이터 재집계, 개입 불필요] L 슬라이스를 의미 있는 레벨로 재추출  ← **바로 다음 작업**
gen_decomp의 `L(s,L1)`이 전부 ≈0이라 표가 약함. 148-cell sweep 데이터는 이미 있으므로
**재측정 없이** L을 staleness가 실제로 나타나는 레벨에서 재집계한다. 상세 §C.

### B2 — [데이터 재집계] gen_gain의 noisy ratio 정리
v8m의 `worst/mean ratio = −7.9×` 등 음수 비율이 표에 노출됨. mean-gain을 별도 열로 분리해
독자가 분모가 작고 noisy함을 직접 보게 하거나, ratio 열을 제거하고 worst-gain만 헤드라인.
기존 `gen_gain.csv`로 재집계(날조 없음).

### B3 — [측정, toolchain 필요] global8/single 모드 정합
gen_decomp의 v8(single) vs v11s(global8) 절대단위 비교 불가 문제. 둘 중 하나로 통일:
(a) Table 1(v11s)을 single 모드로도 재측정해 같은 모드 축 확보, 또는
(b) v8 family를 global8로도 재측정. qb toolchain/모드 토글 가능 여부에 따라 결정.
→ toolchain 부재면 현 상태(within-mode 한정 + Limitations 명시)로 제출 가능. **차단 아님.**

### B4 — [측정, toolchain 필요] YOLOv11n/m INT8 컴파일
qb compiler가 PATH에 없어 skip됨. 권한·calibration set 확보되면 v11 capacity 축도 채움.
없으면 v8 family(3–68M)로 capacity-scaling은 이미 충분 — **선택 사항**.

### B5 — [측정, 비-YOLO] RT-DETR/PicoDet
INT8 export 부재 → future work로 이미 명시. toolchain 생기면 1행 추가. **선택.**

### B6 — [데이터 준비 필요] 2nd dataset (BDD/nuScenes)
로컬 사본 없음. 다운로드는 무인 위험 → 사람이 준비한 뒤 별도 실행. **선택.**

### B7 — [사람 결정] 인간 검토 큐
- `% [VERIFY-HW]` MLA100 카드 on-board DRAM 용량 확정.
- C2 헤드라인 비율(abstract 1.75–3.5× vs gen_gain v11s 4.9×)을 어느 셀 기준으로 통일할지.
- §3 C* 정의로 **sAP-gap**을 정본 채택(절대-mAP은 diagnostic) — 이미 본문 반영됨, 확인만.

> 핵심: **B1·B2만 끝내면 제출 가능한 일관 버전**이 된다. B3–B6은 강화용/선택.

---

## C. 바로 다음 작업 지시문 (B1 + B2, 무인 실행)

> 1~3차 지시문의 작업 원칙 그대로: 수치 날조 금지, idempotent + manifest 체크포인트,
> 결정 로깅, 검증된 셀만 논문 반영, 표/그림은 CSV에서 스크립트로 자동 생성, 원본 보존.
> 막혀도 멈추지 말고 TBD + 사유 기록 후 다음으로. 모든 변경의 before/after diff를
> `RESULTS_STATUS.md`에 남길 것.

### TASK B1 — L 슬라이스 재집계 (재측정 금지, 기존 sweep 데이터만)
1. 148-cell sweep 결과(`results/p1_*.csv`, `cstar_v2_sap.csv` 등)에서 각 검출기·size group별
   **single-stream** `L(s,C)`를 **모든 ladder level**(L0, L1_light, L1_heavy, L2_lm, L3_VLM)에 대해 추출.
2. 각 size group에서 staleness가 **처음으로 유의미**(예: |L| ≥ 0.005 sAP)해지는 최저 level을 찾아
   `results/L_onset.csv`로 기록.
3. gen_decomp 표의 staleness 블록을 다음 중 **데이터가 뒷받침하는** 형태로 교체:
   - 우선안: `L(s, L2_lm)`(staleness가 실제 나타나는 대표 level) 열로 교체, 또는
   - 대안: `L(s, C_max)` = ladder 상 최대 staleness 열 + 각주에 "L1_light에서는 ≈0" 명시.
   single-stream임을 캡션에 유지(이미 명시됨)하고, multi-stream 증거는 gen_gain으로 연결(본문 이미 반영).
4. 만약 L2_lm 이상에서도 single-stream L이 거의 0이면(budget 내):
   그 사실을 그대로 보고하고, gen_decomp의 L 열을 **single-stream 기준 그대로 두되**
   "single-stream으로는 staleness가 거의 안 잡힌다 → multi-stream(gen_gain)이 유일한 L 증거"라는
   결론을 `RESULTS_STATUS.md`에 적는다. (이 경우 표 구조는 그대로, 본문 정합 이미 완료.)
5. (가능하면) **multi-stream L 직접 추정**: N=4 size-diverse 구성에서 GPU에 남은 stream의
   per-group sAP를 N=1 대비 차감해 "concurrency-induced L"을 `results/L_multistream.csv`로 산출.
   이게 양수로 크게 나오면 thesis의 핵심 정량 근거가 되므로 gen_decomp에 보조 열/표로 추가 검토.
- 출력: `paper/tables/gen_decomp.tex` 갱신, `results/L_onset.csv`, `results/L_multistream.csv`(가능 시).

### TASK B2 — gen_gain ratio 정리
1. `gen_gain.csv`에서 각 검출기의 mean-gain(절대값)을 별도 열로 산출.
2. `paper/tables/gen_gain.tex`를 **worst gain | mean gain | worst/mean** 3열로 재구성
   (음수/noisy ratio는 값 그대로 두되, mean-gain을 함께 보여 분모가 작음을 독자가 직접 판단하게).
   worst-gain을 굵게(operative claim). 캡션의 noisy-denominator 설명은 유지.
3. main_vision.tex inline `tab:gen-gain`도 동일 구조로 갱신.
- 출력: `paper/tables/gen_gain.tex`, main_vision.tex inline 표 갱신.

### TASK B-검증 (필수)
- B1/B2 갱신 후 `pdflatex`(또는 레포 빌드)로 **컴파일 통과** 확인. 실패 시 로그와 함께 보고.
- 표 수치가 소스 CSV와 일치하는지 재대조(자동 assert).
- `AUTONOMOUS_RUN_REPORT.md`에: 바뀐 표 before/after, 컴파일 결과, 남은 TBD(B3–B6),
  인간 결정 큐(B7) 갱신.

### 코드 작성 전 1회 보고 후 즉시 진행 (멈추지 말 것)
1. 기존 sweep CSV에 ladder별 per-group single-stream sAP가 다 들어있는지(B1 재집계 가능 여부).
2. multi-stream L 직접 추정(B1.5)이 기존 N=4 데이터로 가능한지.
3. 불가하면 어느 폴백(§C TASK B1 step 4)을 택했는지 로깅 후 진행.
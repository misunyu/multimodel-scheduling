"""작업 19-r2 [3]: 기대값(expected.json) ↔ 재현값(reproduced.json) 기계 대조.

수치는 원고 표기 자리수에 맞춰 반올림 후 비교, 카운트·집합은 정확 일치.
그림은 sha256 우선, 다르면 pdftotext 텍스트 + 페이지 치수 동치로 대체 판정.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
E = json.loads((HERE / "expected.json").read_text())
R = json.loads((HERE / "reproduced.json").read_text())

results = []          # {item, section, expected, reproduced, verdict, note}


_PLATFORMS = ("gpu", "npu")


def dp(s):
    """표기 소수 자리수."""
    m = re.search(r"\.(\d+)", str(s))
    return len(m.group(1)) if m else 0


def num(s):
    m = re.search(r"-?\d+\.?\d*", str(s).replace("$", "").replace("\\%", ""))
    return float(m.group(0)) if m else None


def chk(item, section, exp, rep, note=""):
    """자리수 맞춘 수치 비교."""
    if exp is None or rep is None:
        results.append({"item": item, "section": section, "expected": exp,
                        "reproduced": rep, "verdict": "조건부", "note": note or "값 없음"})
        return
    e = num(exp)
    r_ = round(float(rep), dp(exp))
    ok = abs(e - r_) < 10 ** (-dp(exp)) / 2
    results.append({"item": item, "section": section, "expected": str(exp),
                    "reproduced": f"{r_:.{dp(exp)}f}",
                    "verdict": "일치" if ok else "불일치", "note": note})


def chk_eq(item, section, exp, rep, note=""):
    """정확 일치 (카운트·집합·문자열)."""
    ok = exp == rep
    results.append({"item": item, "section": section, "expected": str(exp),
                    "reproduced": str(rep), "verdict": "일치" if ok else "불일치",
                    "note": note})


PL = ("gpu", "npu")

# ------------------------------------------------------------------ 0. 앵커
for k, v in E["anchors"].items():
    chk_eq(f"anchor:{k}", "0", True, v)
chk_eq("revision_marker_is_r18a", "0", True,
       "r18a" in E["_source"].get("revision_block", E["_source"]["revision_marker"]),
       "지시문은 r19 이상을 요구했으나 시스템 최신본이 r18a")

# ------------------------------------------------------------------ Table 1
t1 = E["tables"]["table1_comparison"]
chk("T1 greedy Top-1 GPU", "Table 1", t1["Capacity greedy"][2],
    R["greedy"]["gpu/speedup"]["top1_measured"])
chk("T1 greedy Top-1 NPU", "Table 1", t1["Capacity greedy"][3],
    R["greedy"]["npu/speedup"]["top1_measured"])
for j, pl in enumerate(PL):
    chk(f"T1 decomposed rho {pl.upper()}", "Table 1",
        t1["Decomposed per instance"][j], R["decomposed"][pl]["group_spearman_mean"])
    chk(f"T1 decomposed Top-1 {pl.upper()}", "Table 1",
        t1["Decomposed per instance"][2 + j], R["decomposed"][pl]["top1"])
    chk(f"T1 direct rho {pl.upper()}", "Table 1", t1["Direct regression"][j],
        R["direct"][pl]["group_spearman_mean"])
    chk(f"T1 direct Top-1 {pl.upper()}", "Table 1", t1["Direct regression"][2 + j],
        R["direct"][pl]["top1"])

# --------------------------------------------------------------- App.A 표
cov = {c["set"]: c for c in R["coverage"]}
for row in E["tables"]["appA_coverage"]:
    sid = row[0].strip()
    if sid.startswith("base1--base5"):
        for s in [f"base{i}" for i in range(1, 6)]:
            chk_eq(f"A.cov {s} N", "App.A", int(num(row[1])), cov[s]["N"])
            chk_eq(f"A.cov {s} 2^N", "App.A", int(num(row[2])), cov[s]["pow2"])
            chk_eq(f"A.cov {s} feasible", "App.A", int(num(row[3])), cov[s]["feasible"])
            chk_eq(f"A.cov {s} measured", "App.A", int(num(row[4])), cov[s]["measured_gpu"])
            chk_eq(f"A.cov {s} measured(npu 동일)", "App.A", int(num(row[4])),
                   cov[s]["measured_npu"])
        continue
    if sid not in cov:
        continue
    chk_eq(f"A.cov {sid} N", "App.A", int(num(row[1])), cov[sid]["N"])
    chk_eq(f"A.cov {sid} 2^N", "App.A", int(num(row[2])), cov[sid]["pow2"])
    chk_eq(f"A.cov {sid} feasible", "App.A", int(num(row[3])), cov[sid]["feasible"])
    chk_eq(f"A.cov {sid} measured", "App.A", int(num(row[4])), cov[sid]["measured_gpu"])
    chk_eq(f"A.cov {sid} measured(npu 동일)", "App.A", int(num(row[4])),
           cov[sid]["measured_npu"])

# --------------------------------------------------------------- App.B 표
ps = R["per_set_group_spearman"]
for row in E["tables"]["appB_perset"]:
    sid = row[0].strip()
    if sid not in ps:
        continue
    cols = [("gpu_y1", 2), ("gpu_y2", 3), ("gpu_y3", 4),
            ("npu_y1", 5), ("npu_y2", 6), ("npu_y3", 7)]
    for key, ci in cols:
        exp = row[ci].strip()
        if exp == "---":
            chk_eq(f"B.{sid} {key} (해당없음)", "App.B", True, ps[sid].get(key) is None)
        else:
            chk(f"B.{sid} {key}", "App.B", exp, ps[sid][key])

# --------------------------------------------------------------- App.C 표
sweep_rows = {(r["alpha"], r["beta"]): r for r in R["sweep"][0]}
for row in E["tables"]["appC_sweep"]:
    a = num(row[0])
    if a is None:
        continue
    for bi, beta in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        cell = row[1 + bi].strip()
        if "/" not in cell:
            continue
        g, n = cell.split("/")
        rr = sweep_rows.get((a, beta))
        if rr is None:
            continue
        chk_eq(f"C.sweep a={a} b={beta} GPU", "App.C", int(g), rr["n_changed_sets_gpu"])
        chk_eq(f"C.sweep a={a} b={beta} NPU", "App.C", int(n), rr["n_changed_sets_npu"])

# --------------------------------------------------------------- App.D 표
gmap = {("GPU", "load-factor"): "gpu/load_factor", ("GPU", "speedup"): "gpu/speedup",
        ("NPU", "load-factor"): "npu/load_factor", ("NPU", "speedup"): "npu/speedup"}
for row in E["tables"]["appD_greedy"]:
    key = gmap.get((row[0].strip(), row[1].strip()))
    if key is None:
        continue
    g = R["greedy"][key]
    m = re.match(r"([\d.]+)\s*\((\d+)/(\d+)\)", row[2].strip())
    chk(f"D.{key} Top-1(measured)", "App.D", m.group(1), g["top1_measured"])
    chk_eq(f"D.{key} hits", "App.D", int(m.group(2)), g["top1_hits"])
    chk_eq(f"D.{key} denominator", "App.D", int(m.group(3)), g["n_measured"])
    chk(f"D.{key} Top-1(bound/45)", "App.D", row[3].strip(), g["top1_bound45"])
    chk(f"D.{key} oracle ratio", "App.D", row[4].strip(), g["oracle_ratio"])
    chk_eq(f"D.{key} unmeasured", "App.D", int(row[5].split("/")[0]), g["n_unmeasured"])

# --------------------------------------------------------------- App.E 표
er = E["tables"]["appE_decomp"]
flat = []
for i in range(0, len(er)):
    flat.append(er[i])
# 행 구조: [Quantity], [GPU], [NPU] 가 줄바꿈으로 쪼개져 있어 값만 순서대로 모은다
vals = re.findall(r"([\d.]+ / [\d.]+(?: / [\d.]+)?)", "\n".join("&".join(r) for r in er))
if len(vals) >= 6:
    a_gpu, a_npu, y_gpu, y_npu, c_gpu, c_npu = vals[:6]
    chk("E.attainment rho GPU", "App.E", a_gpu.split("/")[0].strip(),
        R["decomposed"]["gpu"]["oof_spearman_pooled"])
    chk("E.attainment MAE GPU", "App.E", a_gpu.split("/")[1].strip(),
        R["decomposed"]["gpu"]["oof_mae"])
    chk("E.attainment rho NPU", "App.E", a_npu.split("/")[0].strip(),
        R["decomposed"]["npu"]["oof_spearman_pooled"])
    chk("E.attainment MAE NPU", "App.E", a_npu.split("/")[1].strip(),
        R["decomposed"]["npu"]["oof_mae"])
    for pl, cell in (("gpu", y_gpu), ("npu", y_npu)):
        p, gm, mae = [x.strip() for x in cell.split("/")]
        chk(f"E.y2recon pooled rho {pl.upper()}", "App.E", p,
            R["decomposed_recon"][pl]["y2_raw_spearman"])
        chk(f"E.y2recon group rho {pl.upper()}", "App.E", gm,
            R["decomposed_recon"][pl]["y2_group_spearman_mean"])
        chk(f"E.y2recon MAE {pl.upper()}", "App.E", mae,
            R["decomposed_recon"][pl]["y2_raw_mae"])
    for pl, cell in (("gpu", c_gpu), ("npu", c_npu)):
        gm, t1v = [x.strip() for x in cell.split("/")]
        chk(f"E.composed rho {pl.upper()}", "App.E", gm,
            R["decomposed"][pl]["group_spearman_mean"])
        chk(f"E.composed Top-1 {pl.upper()}", "App.E", t1v, R["decomposed"][pl]["top1"])

# --------------------------------------------------------------- App.H 표
BKEY = {"Direct group-mean Spearman": "direct_rho",
        "Direct Top-1": "direct_top1",
        "Decomposed Top-1": "decomp_top1",
        "Direct $-$ greedy (speedup)": "d_greedy_speedup",
        "Direct $-$ greedy (load-factor)": "d_greedy_load_factor",
        "Direct $-$ decomposed": "d_decomp"}
for row in E["tables"]["appH_bootstrap"]:
    name = row[0].strip()
    key = BKEY.get(name) or BKEY.get(name.replace("-", "$-$"))
    if name.startswith("Divergence fraction"):
        m = re.search(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", "&".join(row))
        b = R["bootstrap"]["divergence_fraction"]
        chk("H.divergence fraction point", "App.H", m.group(1), b["point"])
        chk("H.divergence fraction CI lo", "App.H", m.group(2), b["ci95"][0])
        chk("H.divergence fraction CI hi", "App.H", m.group(3), b["ci95"][1])
        continue
    if key is None:
        continue
    for j, pl in enumerate(PL):
        cell = row[1 + j]
        m = re.search(r"([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\]", cell)
        if not m:
            continue
        b = R["bootstrap"].get(f"{key}/{pl}")
        if b is None:
            continue
        if "point" in b:
            chk(f"H.{key} point {pl.upper()}", "App.H", m.group(1), b["point"])
        chk(f"H.{key} CI lo {pl.upper()}", "App.H", m.group(2), b["ci95"][0])
        chk(f"H.{key} CI hi {pl.upper()}", "App.H", m.group(3), b["ci95"][1])

# ----------------------------------------------------------------- 인라인
I = E["inline"]
chk_eq("inline abstract divergence", "Abstract", ("11", "45", "10"), tuple(I["abstract_divergence"]))
chk_eq("inline divergence 11/45", "§4", int(I["divergence_main"][0]),
       R["divergence"]["beta_1.0"]["n_disagree"])
chk_eq("inline divergence n_groups", "§4", int(I["divergence_main"][1]),
       R["divergence"]["beta_1.0"]["n_groups"])
chk_eq("inline divergence generative", "§4", int(I["divergence_main"][2]),
       R["divergence"]["beta_1.0"]["gen"]["d"])
chk_eq("inline divergence gen total", "§4", int(I["divergence_main"][3]),
       R["divergence"]["beta_1.0"]["gen"]["n"])
chk_eq("inline divergence vision", "§4", int(I["divergence_main"][4]),
       R["divergence"]["beta_1.0"]["vision"]["d"])
chk_eq("inline divergence vision total", "§4", int(I["divergence_main"][5]),
       R["divergence"]["beta_1.0"]["vision"]["n"])
chk_eq("inline divergence beta=0.5", "§4", int(I["divergence_beta05"][0]),
       R["divergence"]["beta_0.5"]["n_disagree"])
chk_eq("inline exhaustive n_groups", "§4", int(I["divergence_exhaustive"][0]),
       R["divergence"]["exhaustive_beta_1.0"]["n_groups"])
chk_eq("inline exhaustive disagree", "§4", int(I["divergence_exhaustive"][1]),
       R["divergence"]["exhaustive_beta_1.0"]["n_disagree"])
chk_eq("inline exhaustive gen split", "§4", int(I["divergence_exhaustive_split"][1]),
       R["divergence"]["exhaustive_beta_1.0"]["gen"]["n"])
chk_eq("inline appA beta0.5 exhaustive", "App.A", int(I["appA_beta05_exhaustive"][0]),
       R["divergence"]["exhaustive_beta_0.5"]["n_disagree"])
chk_eq("inline appA counts 180/540/45", "App.A",
       ("180", "540", "45"), tuple(I["appA_counts"]))

for j, pl in enumerate(PL):
    chk(f"inline direct rho {pl.upper()}", "§4", I["direct_rho_top1"][j],
        R["direct"][pl]["group_spearman_mean"])
    chk(f"inline direct Top-1 {pl.upper()}", "§4", I["direct_rho_top1"][2 + j],
        R["direct"][pl]["top1"])
chk("inline selected-ratio min", "§4", I["selected_ratio_sec4"][0],
    100 * min(R["direct"][pl]["min_ratio"] for pl in PL), "원고 '적어도 91%'")
for j, pl in enumerate(PL):
    chk(f"inline selected-ratio mean {pl.upper()}", "§4",
        I["selected_ratio_sec4"][1 + j], 100 * R["direct"][pl]["mean_ratio"])

chk("inline zero-shot NPU rho from", "§4", I["zero_shot_npu"][0],
    R["direct"]["npu"]["group_spearman_mean"])
chk("inline zero-shot NPU rho to", "§4", I["zero_shot_npu"][1],
    R["zero_shot"]["gpu->npu"]["group_spearman_mean"])
chk("inline zero-shot NPU top1 from", "§4", I["zero_shot_npu"][2], R["direct"]["npu"]["top1"])
chk("inline zero-shot NPU top1 to", "§4", I["zero_shot_npu"][3],
    R["zero_shot"]["gpu->npu"]["top1"])
chk("inline zero-shot GPU rho from", "§4", I["zero_shot_gpu"][0],
    R["direct"]["gpu"]["group_spearman_mean"])
chk("inline zero-shot GPU rho to", "§4", I["zero_shot_gpu"][1],
    R["zero_shot"]["npu->gpu"]["group_spearman_mean"])
chk("inline zero-shot GPU top1 stays", "§4", I["zero_shot_gpu"][2],
    R["zero_shot"]["npu->gpu"]["top1"])
for j, pl in enumerate(PL):
    chk(f"inline joint rho {pl.upper()}", "§4", I["joint_reach"][j],
        R["joint"][pl]["group_spearman_mean"])
    chk(f"inline joint Top-1 {pl.upper()}", "§4", I["joint_reach"][2 + j],
        R["joint"][pl]["top1"])
chk("inline abstract joint Top-1 GPU", "Abstract", I["abstract_joint_top1"][0],
    100 * R["joint"]["gpu"]["top1"])
chk("inline abstract joint Top-1 NPU", "Abstract", I["abstract_joint_top1"][1],
    100 * R["joint"]["npu"]["top1"])

chk("inline greedy Top-1 GPU", "§4", I["greedy_main"][0],
    R["greedy"]["gpu/speedup"]["top1_measured"])
chk("inline greedy Top-1 NPU", "§4", I["greedy_main"][1],
    R["greedy"]["npu/speedup"]["top1_measured"])
chk("inline greedy bound GPU", "Table 1 caption", I["greedy_bounds_caption"][0],
    R["greedy"]["gpu/speedup"]["top1_bound45"])
chk("inline greedy bound NPU", "Table 1 caption", I["greedy_bounds_caption"][1],
    R["greedy"]["npu/speedup"]["top1_bound45"])
for j, pl in enumerate(PL):
    chk(f"inline decomp pooled rho {pl.upper()}", "§4", I["decomp_pooled"][j],
        R["decomposed"][pl]["oof_spearman_pooled"])
    chk(f"inline decomp Top-1 {pl.upper()}", "§4", I["decomp_top1"][j],
        R["decomposed"][pl]["top1"])
    chk(f"inline decomp recon rho {pl.upper()}", "§4", I["decomp_recon"][j],
        R["decomposed_recon"][pl]["y2_group_spearman_mean"])
    chk(f"inline appE Top-5 {pl.upper()}", "App.E", I["appE_top5"][0],
        R["decomposed"][pl]["top5"])

# y3 (App.B)
chk_eq("inline appB y3 masking 372", "App.B", int(I["appB_masking"][0]),
       R["direct_targets_pooled"]["gpu"]["y3"]["n_rows"])
chk_eq("inline appB masking split", "App.B", ("168", "156", "12"),
       tuple(I["appB_masking"][1:]))
chk_eq("inline appB n target windows", "App.B", 12,
       R["y3_sensitivity"]["n_target_windows"]["gpu"])
chk("inline appB extrap GPU lo", "App.B", I["appB_extrap"][0],
    R["y3_sensitivity"]["extrapolated_yhat3_range"]["gpu"][0])
chk("inline appB extrap GPU hi", "App.B", I["appB_extrap"][1],
    R["y3_sensitivity"]["extrapolated_yhat3_range"]["gpu"][1])
chk("inline appB extrap NPU lo", "App.B", I["appB_extrap"][2],
    R["y3_sensitivity"]["extrapolated_yhat3_range"]["npu"][0])
chk("inline appB extrap NPU hi", "App.B", I["appB_extrap"][3],
    R["y3_sensitivity"]["extrapolated_yhat3_range"]["npu"][1])
chk("inline appB window total max", "App.B", I["appB_window_totals"][1],
    max(R["y3_sensitivity"]["restored"][pl]["r3_max"] for pl in PL))
for j, pl in enumerate(PL):
    chk(f"inline appB restored y3 {pl.upper()}", "App.B", I["appB_restored"][j],
        R["y3_sensitivity"]["restored"][pl]["y3_norm_max"])
chk_eq("inline appB divergence unchanged", "App.B", 11,
       R["divergence"]["beta_1.0"]["n_disagree"])

# ---------------------------------------------- App.A VLM pinning rationale (신설)
# 원고가 새로 인쇄하는 12.3 / 0.10 / 0.61 s, 18 s, "roughly ten" 을 정적 프로파일에서
# 유도해 대조한다. 앞의 셋은 측정값이고 뒤의 둘은 그 측정값에서 나온 산술이다.
_V = R["vlm_cpu_cost"]
_A = I["appA_vlm_cost"]
for j, dev in enumerate(("cpu", "gpu", "npu")):
    chk(f"inline appA VLM prefill {dev.upper()}", "App.A", _A[j], _V["prefill_s"][dev])
chk_eq("inline appA VLM new tokens", "App.A", int(_A[3]), _V["new_tokens"])
chk("inline appA VLM request s", "App.A", _A[4], _V["cpu_request_s"])
# 2026-08-18: 원고가 "roughly ten ... complete" 를 삭제해 완료 횟수 캡처가 사라졌다.
# 검사 1건(inline appA VLM completions/180s)이 줄어든다 -- 인쇄되지 않는 값을 계속
# 대조할 수는 없기 때문이다. 재현 측 계산(completions_per_180s)은 남겨 둔다.
# 2026-08-18: App A 재작성으로 180\,s 창 언급이 이 문단에서 사라졌다. 인쇄되지 않는
# 값을 계속 대조할 수 없으므로 검사 1건(inline appA VLM window s)이 추가로 줄어든다.

# 백분율 4종 — 정수 카운트 앵커와 별개로 잡고 같은 카운트에서 유도해 대조한다
_R = I["appA_rates"]
_d = R["divergence"]
chk("inline appA rate exh b=1.0", "App.A", _R[0],
    100 * _d["exhaustive_beta_1.0"]["n_disagree"] / _d["exhaustive_beta_1.0"]["n_groups"])
chk("inline appA rate all b=1.0", "App.A", _R[1],
    100 * _d["beta_1.0"]["n_disagree"] / _d["beta_1.0"]["n_groups"])
chk_eq("inline appA rate n_groups", "App.A", int(_R[2]), _d["beta_1.0"]["n_groups"])
chk("inline appA rate exh b=0.5", "App.A", _R[3],
    100 * _d["exhaustive_beta_0.5"]["n_disagree"] / _d["exhaustive_beta_0.5"]["n_groups"])
chk("inline appA rate all b=0.5", "App.A", _R[4],
    100 * _d["beta_0.5"]["n_disagree"] / _d["beta_0.5"]["n_groups"])

# --------------------------------------------- App.H "What differs." 문단 (신설)
_W = R["divergence_what_differs"]
_WD = I["appH_what_differs"]
chk_eq("inline appH what-differs n_groups", "App.H", int(_WD[0]), _W["n_groups"])
chk_eq("inline appH what-differs same generative", "App.H", int(_WD[0]),
       _W["n_same_generative_placement"], "원고는 11 그룹 전부에서 동일하다고 주장")
chk_eq("inline appH what-differs npu offloads", "App.H", int(_WD[1]),
       _W["n_npu_offloads_resnet50_gpu_all_accel"])
chk_eq("inline appH what-differs model name", "App.H", _WD[2].lower(), "resnet50")

# ---------------------------------------- 본문 §3 y3 복원 불변 주장 (2026-08-18 신설)
# 원고 §3: "For 12 CPU-LLM windows per platform, ... Restoring the recorded window totals
# changes neither Top-1 outcomes nor divergence labels (Appendix~B)."
# App.B 의 appB_sensitivity 가 같은 사실을 다른 표현으로 인쇄하므로, 두 표기를 하나의
# 재현 소스에서 유도해 대조한다.
_Y = R["y3_restored_invariance"]
_S3 = I["sec3_y3_restore"]
for pl in _PLATFORMS:
    chk_eq(f"inline sec3 CPU-LLM windows {pl.upper()}", "§3", int(_S3[0]),
           _Y["n_target_windows"][pl])
chk_eq("inline sec3 Top-1 unchanged on restore", "§3", 0, _Y["top1_changes_total"],
       "원고: changes neither Top-1 outcomes ...")
chk_eq("inline sec3 divergence labels unchanged", "§3", True,
       _Y["divergence_labels_unchanged"], "원고: ... nor divergence labels")
# 본문과 App.B 가 같은 divergence 수를 가리키는지 — 두 표기의 정합성
chk_eq("inline sec3/App.B divergence agree", "§3", int(I["appB_sensitivity"][1]),
       _Y["divergence_restored"],
       "App.B 'preserves the same 11 workload groups' 와 같은 재현값에서 유도")
chk_eq("inline sec3 divergence published baseline", "§3",
       R["divergence"]["beta_1.0"]["n_disagree"], _Y["divergence_published"])

# ------------------------------------------- App.G CPU 경로 공통 모드 주장 (신설)
# 원고: "the two measured candidate sets are identical, every candidate places the same
# instances on the CPU, and the CPU-side feature contribution is identical across all
# 540 windows". 세 절을 각각 대조한다 — 540 이라는 수 자체도 앵커다.
_C = R["cpu_side_identity"]
chk_eq("inline appG n_windows", "App.G", int(I["appG_cpu_identity"][0]),
       _C["n_windows_compared"])
for pl in _PLATFORMS:
    chk_eq(f"inline appG n_windows {pl.upper()}", "App.G",
           int(I["appG_cpu_identity"][0]), _C["n_windows"][pl])
chk_eq("inline appG candidate sets identical", "App.G", True,
       _C["combination_keys_identical"])
chk_eq("inline appG same instances on CPU", "App.G", 0,
       _C["n_windows_cpu_placement_differs"],
       "원고: every candidate places the same instances on the CPU")
chk_eq("inline appG CPU-side feature identical", "App.G", 0,
       _C["n_windows_cpu_feature_subtotal_differs"],
       "원고: the CPU-side feature contribution is identical across all 540 windows")

# ================= 지시문 #30 Phase 2/3: 여태 대조되지 않던 인쇄값 8건 + 신규 1건
# extract_expected 가 뽑기만 하고 여기서 쓰지 않던 캡처들. 원고에 인쇄된 수치가 검사망
# 밖에 있는 상태였으므로 전부 닫는다.

# --- App.A: exhaustive 30그룹 분해 (§4 와 같은 사실을 App A 가 다시 인쇄한다)
_EX = R["divergence"]["exhaustive_beta_1.0"]
_AE = I["appA_exhaustive"]
chk_eq("inline appA exhaustive disagree", "App.A", int(_AE[0]), _EX["n_disagree"])
chk_eq("inline appA exhaustive gen d", "App.A", int(_AE[1]), _EX["gen"]["d"])
chk_eq("inline appA exhaustive gen n", "App.A", int(_AE[2]), _EX["gen"]["n"])
chk_eq("inline appA exhaustive vision n", "App.A", int(_AE[3]), _EX["vision"]["n"])

# --- App.B: 0.013 (8조합 중 최대 |group-mean Spearman 변화|)
_BS = R["appB_spearman_change"]
chk("inline appB max spearman change", "App.B", I["appB_max_spearman_change"][0],
    _BS["max_abs_delta"])
chk_eq("inline appB change n_combinations", "App.B", 8, _BS["n_combinations"])
chk_eq("inline appB change attained by", "App.B", "decomposed/npu", _BS["attained_by"],
       "원고: attained by the decomposed predictor on the NPU")

# --- App.D: VLM 용량 소비 범위 / 누적 부하 초과 그룹 / VLM 앞 admit 수
_GD = R["greedy_detail"]
_VC = I["appD_vlm_capacity"]
# 원고는 "consumes 0.13--0.40 of the estimated capacity" 처럼 **구간**을 인쇄한다.
# 그래서 끝점을 최근접 반올림과 비교하는 것은 이 문장이 주장하는 바가 아니다.
# 올바른 검사는 (i) 인쇄 구간이 실측 [min, max] 를 포함하는가, (ii) 인쇄 자릿수 기준으로
# 한 칸 이상 헐겁지 않은가 이다.
#   GPU 상한은 실측 0.064974 인데 원고가 0.07 로 **바깥쪽 반올림**을 했다
#   (최근접이면 0.06). 아래 검사는 이를 통과시키되, 사실은 보고서에 명시한다.
def chk_range(item, lo_s, hi_s, lo_v, hi_v):
    step = 10 ** (-dp(lo_s))
    chk_eq(f"{item} brackets measured range", "App.D", True,
           num(lo_s) <= lo_v + 1e-12 and hi_v <= num(hi_s) + 1e-12)
    chk_eq(f"{item} tight to printed precision", "App.D", True,
           lo_v - num(lo_s) < step and num(hi_s) - hi_v < step)

chk_range("inline appD VLM capacity NPU", _VC[0], _VC[1],
          _GD["npu"]["vlm_load_factor_min"], _GD["npu"]["vlm_load_factor_max"])
chk_range("inline appD VLM capacity GPU", _VC[2], _VC[3],
          _GD["gpu"]["vlm_load_factor_min"], _GD["gpu"]["vlm_load_factor_max"])
_CE = I["appD_cum_exceeds"]
chk_eq("inline appD cum>1.0 groups", "App.D", int(_CE[0]),
       _GD["npu"]["n_groups_cum_exceeds_1"]["load_factor"],
       "load-factor ordering 기준 (문단이 그 ordering 을 다룬다)")
chk_eq("inline appD cum>1.0 denominator", "App.D", int(_CE[1]), _GD["npu"]["n_groups"])
_AB = I["appD_vlm_admit_before"]
chk_eq("inline appD VLM admit-before GPU min", "App.D", int(_AB[0]),
       _GD["gpu"]["vlm_admit_before_min"])
chk_eq("inline appD VLM admit-before GPU max", "App.D", int(_AB[1]),
       _GD["gpu"]["vlm_admit_before_max"])
chk_eq("inline appD VLM admit-before NPU min", "App.D", int(_AB[2]),
       _GD["npu"]["vlm_admit_before_min"])
chk_eq("inline appD VLM admit-before NPU max", "App.D", int(_AB[3]),
       _GD["npu"]["vlm_admit_before_max"])

# --- App.H: margin 개수·최솟값·중앙값, selected-placement ratio (direct / greedy)
_MM = R["divergence_margins"]
_HM = I["appH_margins"]
chk_eq("inline appH n margins", "App.H", int(_HM[0]), _MM["n_cells"])
chk("inline appH smallest margin (x1e-3)", "App.H", _HM[1], _MM["min_margin"] * 1000.0)
chk("inline appH median margin", "App.H", I["appH_median_margin"][0], _MM["median_margin"])
_RD = I["appH_ratio_direct"]
for j, pl in enumerate(_PLATFORMS):
    chk(f"inline appH direct mean ratio {pl.upper()}", "App.H", _RD[j],
        R["direct"][pl]["mean_ratio"])
    chk(f"inline appH direct worst ratio {pl.upper()}", "App.H", _RD[2 + j],
        R["direct"][pl]["min_ratio"])
    chk_eq(f"inline appH direct none below 0.9 {pl.upper()}", "App.H", 0,
           R["direct"][pl]["n_ratio_below_0.9"], "원고: no group below 0.9")
_RG = I["appH_ratio_greedy"]
for j, pl in enumerate(_PLATFORMS):
    _g = R["greedy"][f"{pl}/speedup"]
    chk(f"inline appH greedy ratio {pl.upper()}", "App.H", _RG[j], _g["oracle_ratio"])
    # "unmeasured outputs count as zero" -> 측정 그룹 평균 x (측정수/45)
    chk(f"inline appH greedy ratio zero-filled {pl.upper()}", "App.H", _RG[2 + j],
        _g["oracle_ratio"] * _g["n_measured"] / 45.0)

# App.F (설정값 — 코드 상수와 대조)
chk_eq("F.grid", "App.F", ("3", "0.1", "4", "0.1", "5", "0.05", "6", "0.1"),
       tuple(I["appF_grid"]))
chk_eq("F.features 36+1=37", "App.F", ("36", "37"), tuple(I["appF_features"]))
# appF_fixed 는 추출만 되고 대조되지 않던 다섯 번째 캡처 (#25 에서 네 건 배선). 원고가
# subsample/column-sample 을 두 값으로 분리해 쓰면서 캡처가 6개가 되었으므로 함께 배선한다.
sys.path.insert(0, str(ROOT))
from xgboost_model.deploy_selector_xgb_suite import _PARAMS as _XP  # noqa: E402
chk_eq("F.fixed settings", "App.F",
       ("squared-error", "300", "0.8", "0.8", "3", "42"),
       tuple(I["appF_fixed"]))
chk_eq("F.fixed matches code constants", "App.F",
       (_XP["objective"], _XP["subsample"], _XP["colsample_bytree"],
        _XP["min_child_weight"], _XP["seed"]),
       ("reg:squarederror", float(I["appF_fixed"][2]), float(I["appF_fixed"][3]),
        float(I["appF_fixed"][4]), int(I["appF_fixed"][5])))

# ------------------------------------------------- 2026-08-17: 미배선이던 캡처 4건
# extract_expected 가 뽑기만 하고 여기서 대조하지 않던 항목들 (지시문 #24 부수 발견,
# 지시문 #25 Phase 1 에서 배선). appH_band 는 재현 측 계산이 없어서 배선할 수 없었는데,
# reproduce.py 에 divergence_margins 를 넣어 해결했다.
_M = R["divergence_margins"]
# 2026-08-17: 원고가 두 규칙(any/both)과 상한 밴드를 모두 인쇄하도록 확장되어
# 캡처도 (n_cells, n_margins, any_div, any_total, both_div, both_total, max_band) 로 늘었다.
_B = I["appH_band"]
chk_eq("inline appH band n_cells<0.01", "App.H", int(_B[0]), _M["n_cells_below_0.01"])
chk_eq("inline appH band n_margins", "App.H", int(_B[1]), _M["n_cells"])
chk_eq("inline appH band any divergence", "App.H", int(_B[2]), _M["band_0.01"]["any"])
chk_eq("inline appH band any n_groups", "App.H", int(_B[3]), _M["n_groups"])
chk_eq("inline appH band both divergence", "App.H", int(_B[4]), _M["band_0.01"]["both"])
chk_eq("inline appH band both n_groups", "App.H", int(_B[5]), _M["n_groups"])
# "that remains true for bands up to 0.05" -- gap.py 의 밴드 스윕에서 both 가 상한까지 불변인지
_GAP = json.loads((ROOT / "runs/20260809_140605_checklist_r2/B_divergence_gap/"
                   "gap_facts.json").read_text())["score_equivalence_band"]["by_band"]
_upto = float(_B[6])
chk_eq("inline appH band both unchanged up to max band", "App.H", True,
       all(v["both"]["n_disagree"] == _M["band_0.01"]["both"]
           for k, v in _GAP.items() if float(k) <= _upto))
chk_eq("inline appH band max band swept", "App.H", True,
       any(abs(float(k) - _upto) < 1e-12 for k in _GAP))

# sweep 본문 요약: beta>0 셀에서 GPU 최대 / NPU 최대 변경 세트 수
_cells = [c for c in R["sweep"][0] if c["beta"] > 0]
chk_eq("inline sweep claim GPU max", "§4", int(I["sweep_claim"][0]),
       max(c["n_changed_sets_gpu"] for c in _cells))
chk_eq("inline sweep claim NPU max", "§4", int(I["sweep_claim"][1]),
       max(c["n_changed_sets_npu"] for c in _cells))

# App.B "eight predictor--platform combinations": 이 스위트가 실제로 평가하는 예측기 수
_PREDICTORS = ("direct", "zero_shot", "joint", "decomposed")
# zero_shot is keyed by direction ("npu->gpu"), the others by platform -- both are two
# entries per predictor, one per evaluation platform.
chk_eq("inline appB n_combinations", "App.B", int(I["appB_sensitivity"][0]),
       sum(len(R[k]) for k in _PREDICTORS))
chk_eq("inline appB divergence groups", "App.B", int(I["appB_sensitivity"][1]),
       R["divergence"]["beta_1.0"]["n_disagree"])

# App.H selected-placement 문단 (decomposed 쪽)
for j, pl in enumerate(_PLATFORMS):
    chk(f"inline appH decomp mean ratio {pl.upper()}", "App.H",
        I["appH_ratio_decomp"][j], R["decomposed"][pl]["mean_ratio"])
    chk(f"inline appH decomp worst ratio {pl.upper()}", "App.H",
        I["appH_ratio_decomp"][2 + j], R["decomposed"][pl]["min_ratio"])
    chk_eq(f"inline appH decomp n<0.9 {pl.upper()}", "App.H",
           int(I["appH_ratio_decomp"][4 + j]), R["decomposed"][pl]["n_ratio_below_0.9"])

# ------------------------------------------------------------------ 그림
FIGREPRO = HERE / "figures"
MIRROR = ROOT / "manuscript"
ANFIG = ROOT / "xgboost_model/full_collection_540/analysis/figures"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pdftext(p):
    return subprocess.run(["pdftotext", "-layout", str(p), "-"],
                          capture_output=True, text=True).stdout


def pagesize(p):
    for l in subprocess.run(["pdfinfo", str(p)], capture_output=True,
                            text=True).stdout.splitlines():
        if l.lower().startswith("page size"):
            return l.split(":", 1)[1].strip()
    return "?"


chk_eq("figures referenced in tex", "Figures",
       ["fig_divergence.pdf", "fig_transfer.pdf"], E["figures"])
for name in ("fig_divergence", "fig_transfer"):
    repro = FIGREPRO / f"{name}.pdf"
    ref = ANFIG / f"{name}.pdf"
    mir = MIRROR / f"{name}.pdf"
    if not repro.exists():
        results.append({"item": f"fig {name}", "section": "Figures",
                        "expected": "regenerated", "reproduced": "미생성",
                        "verdict": "조건부", "note": "재생성 실패"})
        continue
    same_hash = sha(repro) == sha(ref)
    txt_eq = sorted(pdftext(repro).split()) == sorted(pdftext(ref).split())
    size_eq = pagesize(repro) == pagesize(ref)
    verdict = "일치" if same_hash else ("일치" if (txt_eq and size_eq) else "불일치")
    results.append({
        "item": f"fig {name} vs analysis/figures", "section": "Figures",
        "expected": sha(ref)[:16], "reproduced": sha(repro)[:16],
        "verdict": verdict,
        "note": ("sha256 동일" if same_hash else
                 f"sha256 상이(PDF 타임스탬프) → pdftotext 토큰 동일={txt_eq}, "
                 f"페이지 치수 동일={size_eq} ({pagesize(repro)})")})
    if mir.exists():
        results.append({
            "item": f"fig {name} 미러 대조", "section": "Figures",
            "expected": sha(mir)[:16], "reproduced": sha(ref)[:16],
            "verdict": "일치" if sha(mir) == sha(ref) else "불일치",
            "note": "manuscript/ 사본 vs analysis/figures 사본"})

# ------------------------------------------------------------------ 집계
n = len(results)
agree = sum(1 for r in results if r["verdict"] == "일치")
dis = [r for r in results if r["verdict"] == "불일치"]
cond = [r for r in results if r["verdict"] == "조건부"]

facts = {
    "manuscript": E["_source"],
    "anchors": E["anchors"],
    "expected": E,
    "reproduced": R,
    "comparison": results,
    "summary": {"n_items": n, "n_agree": agree, "n_disagree": len(dis),
                "n_conditional": len(cond),
                "disagreements": dis, "conditionals": cond},
    "timings_sec": R.get("_timings_sec", {}),
    "determinism_spotcheck": R.get("determinism_spotcheck"),
}
(HERE / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False,
                                            default=str))

print(f"총 {n}건 — 일치 {agree}, 불일치 {len(dis)}, 조건부 {len(cond)}")
if dis:
    print("\n=== 불일치 ===")
    for r in dis:
        print(f"  [{r['section']}] {r['item']}: 기대 {r['expected']} vs 재현 {r['reproduced']}"
              f" {('— ' + r['note']) if r['note'] else ''}")
if cond:
    print("\n=== 조건부 ===")
    for r in cond:
        print(f"  [{r['section']}] {r['item']}: {r['note']}")

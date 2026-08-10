"""작업 19-r2 [1]: 원고 tex 에서 검증 대상 기대값을 직접 파싱한다.

원고가 정본이므로 값을 손으로 옮겨 적지 않고 tex 를 파싱한다. 결과는 expected.json.
읽기 전용 — manuscript/ 는 읽기만 한다.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
TEX = ROOT / "manuscript" / "mlforsys_main.tex"
src = TEX.read_text()
lines = src.splitlines()

E = {"_source": {"path": str(TEX.relative_to(ROOT)),
                 "sha256": hashlib.sha256(TEX.read_bytes()).hexdigest(),
                 "n_lines": len(lines),
                 "revision_marker": lines[0]}}

# ------------------------------------------------------------------ 0. 앵커
E["anchors"] = {
    "appB_extrapolation_0.90_to_0.98": bool(re.search(r"\$0\.90\$ to \$0\.98\$", src)),
    "appD_Unmeasured_and_bound_column": bool(
        re.search(r"Top-1 \(bound, /45\)", src) and re.search(r"Unmeasured", src)),
    "sec2_imputed_as_zero": bool(re.search(r"imputed as zero", src)),
}

# ------------------------------------------------------------------ 그림
E["figures"] = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", src)


# ------------------------------------------------------------------ 표 파서
def table_body(label):
    i = src.index("\\label{" + label + "}")
    seg = src[i:]
    b = seg.index("\\begin{tabular}")
    e = seg.index("\\end{tabular}")
    return seg[b:e]


def num_rows(label, strip_math=True):
    """tabular 본문에서 & 로 나뉜 셀을 행 단위로 반환 (규칙 줄 제외)."""
    out = []
    for raw in table_body(label).splitlines():
        ln = raw.strip()
        if (not ln or ln.startswith("\\begin") or ln.startswith("\\toprule")
                or ln.startswith("\\midrule") or ln.startswith("\\bottomrule")
                or ln.startswith("\\cmidrule") or ln.startswith("\\multicolumn")
                or "&" not in ln):
            continue
        cells = [c.strip().rstrip("\\\\").strip() for c in ln.rstrip("\\\\").split("&")]
        if strip_math:
            cells = [c.replace("$", "").strip() for c in cells]
        out.append(cells)
    return out


E["tables"] = {}

# --- Table 1 (tab:comparison) ---
t1 = {}
for cells in num_rows("tab:comparison"):
    if cells[0].startswith("Method") or not cells[0]:
        continue
    t1[cells[0]] = cells[1:]
E["tables"]["table1_comparison"] = t1

# --- App.A coverage ---
E["tables"]["appA_coverage"] = [c for c in num_rows("tab:app_coverage")
                                if c[0] not in ("Set",)]
# --- App.B per-set ---
E["tables"]["appB_perset"] = [c for c in num_rows("tab:app_perset")
                              if c[0] not in ("Set",)]
# --- App.C sweep ---
E["tables"]["appC_sweep"] = [c for c in num_rows("tab:app_sweep")
                             if not c[0].startswith("\\alpha") and c[0] != "\\alpha"]
# --- App.D greedy ---
E["tables"]["appD_greedy"] = [c for c in num_rows("tab:app_greedy")
                              if c[0] not in ("Platform",)]
# --- App.E decomposed ---
E["tables"]["appE_decomp"] = [c for c in num_rows("tab:app_decomp")
                              if c[0] not in ("Quantity",)]
# --- App.H bootstrap ---
E["tables"]["appH_bootstrap"] = [c for c in num_rows("tab:app_boot")
                                 if c[0] not in ("Quantity",)]

# ------------------------------------------------------------- 인라인 수치
def grab(pattern, name, group=None):
    m = re.search(pattern, src)
    if not m:
        return None
    return m.groups() if group is None else m.group(group)


E["inline"] = {
    # abstract / results
    "abstract_divergence": grab(r"differs between the GPU and NPU in (\d+) of (\d+) workload groups, including (\d+) involving generative", None),
    "abstract_joint_top1": grab(r"Top-1 accuracies of ([\d.]+)\\% on the GPU and ([\d.]+)\\% on the NPU", None),
    "direct_rho_top1": grab(r"group-mean Spearman \$([\d.]+)/([\d.]+)\$ and Top-1 \$([\d.]+)/([\d.]+)\$", None),
    "selected_ratio_sec4": grab(r"at least \$(\d+)\\%\$ of the best measured score in every group \(mean \$([\d.]+)\\%/([\d.]+)\\%\$\)", None),
    "divergence_main": grab(r"disagree on the best measured placement in (\d+) of (\d+) groups, including (\d+) of (\d+) generative groups and (\d+) of (\d+) vision-only", None),
    "divergence_beta05": grab(r"Reducing the generative weight to \$\\beta\{=\}0\.5\$ increases the number of disagreements to (\d+) of (\d+)", None),
    "divergence_exhaustive": grab(r"Among the (\d+) exhaustively measured groups, (\d+) disagree", None),
    "divergence_exhaustive_split": grab(r"corresponding to (\d+) of (\d+) generative groups and none of the (\d+) vision-only", None),
    "sweep_claim": grab(r"changes in at most (\d+) of (\d+) workload sets across the sweep in Appendix~C, whereas the NPU best measured placement changes in up to (\d+)", None),
    "zero_shot_npu": grab(r"reduces Spearman correlation from \$([\d.]+)\$ to \$([\d.]+)\$ and Top-1 accuracy from \$([\d.]+)\$ to \$([\d.]+)\$", None),
    "zero_shot_gpu": grab(r"Spearman correlation decreases from \$([\d.]+)\$ to \$([\d.]+)\$, while Top-1 accuracy remains \$([\d.]+)\$", None),
    "joint_reach": grab(r"reaching Spearman \$([\d.]+)/([\d.]+)\$ and Top-1 \$([\d.]+)/([\d.]+)\$", None),
    "greedy_main": grab(r"Top-1 accuracies of \$([\d.]+)\$ on the GPU and \$([\d.]+)\$ on the NPU among outputs present", None),
    "greedy_bounds_caption": grab(r"the corresponding /45 lower bounds are ([\d.]+) and ([\d.]+)", None),
    "decomp_pooled": grab(r"pooled Spearman correlations of \$([\d.]+)\$ on the GPU and \$([\d.]+)\$ on the NPU for individual attainment", None),
    "decomp_top1": grab(r"Top-1 accuracies fall to \$([\d.]+)\$ and \$([\d.]+)\$", None),
    "decomp_recon": grab(r"group-mean Spearman reaches only \$([\d.]+)\$ on the GPU and \$([\d.]+)\$ on the NPU", None),
    # appendix A
    "appA_exhaustive": grab(r"the GPU and NPU disagree on the best measured placement in (\d+) groups: (\d+) of (\d+) generative groups and none of the (\d+) vision-only", None),
    "appA_beta05_exhaustive": grab(r"At \$\\beta\{=\}0\.5\$, (\d+) of the (\d+) exhaustive groups diverge", None),
    "appA_counts": grab(r"this procedure produces (\d+) placement configurations. Evaluating each configuration at three request-rate levels yields (\d+) measurement windows and (\d+) workload groups", None),
    # appendix B (y3)
    "appB_masking": grab(r"trained only on the (\d+) windows per platform .*? the excluded (\d+) comprise (\d+) windows from vision-only workloads and (\d+) in which the LLM", None),
    "appB_window_totals": grab(r"recorded window totals range from \$([\d.]+)\$ to \$([\d.]+)\$ tok/s", None),
    "appB_extrap": grab(r"range from \$([\d.]+)\$ to \$([\d.]+)\$ on the GPU and from \$([\d.]+)\$ to \$([\d.]+)\$ on the NPU", None),
    "appB_restored": grab(r"remain at most \$([\d.]+)\$ on the GPU and \$([\d.]+)\$ on the NPU", None),
    "appB_sensitivity": grab(r"changes no Top-1 outcome across the (\w+) evaluated predictor--platform combinations and leaves the same (\d+) workload groups", None),
    "appB_max_spearman_change": grab(r"maximum absolute change in group-mean Spearman correlation is \$([\d.]+)\$", None),
    # appendix D
    "appD_vlm_capacity": grab(r"VLM consumes ([\d.]+)--([\d.]+) of the estimated capacity, compared with ([\d.]+)--([\d.]+) on the GPU", None),
    "appD_cum_exceeds": grab(r"cumulative estimated load exceeds 1\.0 in (\d+) of the (\d+) NPU workload groups", None),
    # appendix E
    "appE_top5": grab(r"Top-5 accuracy remains \$([\d.]+)\$ on both platforms", None),
    # appendix H
    "appH_margins": grab(r"smallest of the (\d+) margins is \$([\d.]+)\\times10\^\{-3\}\$", None),
    "appH_median_margin": grab(r"the median is \$([\d.]+)\$ \((\d+)\\% of the median", None),
    "appH_band": grab(r"Only (\w+) cells fall below \$?0\.01\$?.*?divergence count would be (\d+) of (\d+)", None),
    "appH_ratio_direct": grab(r"averages \$([\d.]+)\$/\$([\d.]+)\$ \(GPU/NPU\) for direct regression, with a worst case of \$([\d.]+)\$/\$([\d.]+)\$", None),
    "appH_ratio_decomp": grab(r"decomposed baseline averages \$([\d.]+)\$/\$([\d.]+)\$ but with heavier tails \(worst \$([\d.]+)\$/\$([\d.]+)\$; (\w+) and (\w+) groups below", None),
    "appH_ratio_greedy": grab(r"Greedy averages \$([\d.]+)\$/\$([\d.]+)\$ over measured outputs under speedup ordering, and \$([\d.]+)\$/\$([\d.]+)\$ when unmeasured", None),
    # appendix F
    "appF_grid": grab(r"four-point grid---\$\((\d), ([\d.]+)\)\$, \$\((\d), ([\d.]+)\)\$, \$\((\d), ([\d.]+)\)\$, \$\((\d), ([\d.]+)\)\$", None),
    "appF_fixed": grab(r"\((squared-error) objective, (\d+) boosting rounds, subsample and column-sample \$([\d.]+)\$, minimum child weight \$(\d)\$, seed \$(\d+)\$\)", None),
    "appF_features": grab(r"yields (\d+) features; the instance count \$N\$ completes the (\d+)-dimensional", None),
}

missing = [k for k, v in E["inline"].items() if v is None]
E["_inline_unmatched"] = missing

(HERE / "expected.json").write_text(json.dumps(E, indent=2, ensure_ascii=False))
print("revision:", E["_source"]["revision_marker"][:70])
print("anchors:", E["anchors"])
print("figures:", E["figures"])
for k, v in E["tables"].items():
    print(f"  table {k}: {len(v)} rows")
print(f"inline captured: {len(E['inline']) - len(missing)}/{len(E['inline'])}")
if missing:
    print("  UNMATCHED:", missing)

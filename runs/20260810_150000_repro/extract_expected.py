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
                 "revision_marker": lines[0],
                 # 2026-08-17: a newer REVISION line was prepended, so the r18a marker is
                 # no longer lines[0]. Keep the whole leading comment block -- the marker
                 # check reads this, not just the first line.
                 "revision_block": "\n".join(
                     l for l in lines[:20] if l.startswith("%"))}}

# ------------------------------------------------------------------ 0. 앵커
E["anchors"] = {
    "appB_extrapolation_0.90_to_0.98": bool(re.search(r"\$0\.90\$ to \$0\.98\$", src)),
    # App.D's bound column was renamed "Top-1 (bound, /45)" -> "Top-1 (LB)" when the
    # caption started defining LB; accept either spelling.
    "appD_Unmeasured_and_bound_column": bool(
        re.search(r"Top-1 \((?:bound, /45|LB)\)", src) and re.search(r"Unmeasured", src)),
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
# 2026-08-17: the body prose was rewritten (Sec. 3 consolidation), so several anchors
# now have to span a sentence boundary -- hence `flags`. Anchors stay pinned to the
# NUMBERS plus one identifying noun phrase; they are deliberately not loosened to `.*`.
_WORDNUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def grab(pattern, name, group=None, flags=0):
    m = re.search(pattern, src, flags)
    if not m:
        return None
    if group is not None:
        return _WORDNUM.get(m.group(group), m.group(group))
    return tuple(_WORDNUM.get(g, g) for g in m.groups())


E["inline"] = {
    # abstract / results
    "abstract_divergence": grab(r"select different best measured placements in (\d+) of (\d+) workload groups, including (\d+) involving generative", None),
    "abstract_joint_top1": grab(r"Top-1 accuracy at ([\d.]+)\\% on the GPU and ([\d.]+)\\% on the NPU", None),
    "direct_rho_top1": grab(r"group-mean Spearman \$([\d.]+)/([\d.]+)\$ and Top-1 \$([\d.]+)/([\d.]+)\$", None),
    "selected_ratio_sec4": grab(r"at least \$(\d+)\\%\$ of the best measured score\..*?"
                                r"mean ratios are \$([\d.]+)\\%\$ on the GPU and \$([\d.]+)\\%\$ on the NPU",
                                None, flags=re.S),
    "divergence_main": grab(r"differs between the GPU and NPU in (\d+) of (\d+) workload groups, "
                            r"including (\d+) of (\d+) generative groups and (\w+) of (\d+) vision-only", None),
    "divergence_beta05": grab(r"\\beta\{=\}0\.5\$, the number of disagreements increases to (\d+) of (\d+)", None),
    "divergence_exhaustive": grab(r"Among the (\d+) exhaustively measured groups, (\d+) of \d+ generative groups differ", None),
    "divergence_exhaustive_split": grab(r"groups, (\d+) of (\d+) generative groups differ, while no divergence "
                                       r"occurs in the (\d+) vision-only", None),
    # NOTE: captured but not consumed by compare.py (was already unused before this
    # realignment). The counts are now spelled out in the body, so they are normalized
    # to digits by grab().
    "sweep_claim": grab(r"differs from the baseline in at most (\w+) workload sets on the GPU "
                        r"but in as many as (\w+) on the NPU", None),
    "zero_shot_npu": grab(r"reduces Spearman correlation from \$([\d.]+)\$ to \$([\d.]+)\$ and Top-1 accuracy from \$([\d.]+)\$ to \$([\d.]+)\$", None),
    "zero_shot_gpu": grab(r"Spearman correlation decreases from \$([\d.]+)\$ to \$([\d.]+)\$, while Top-1 accuracy remains \$([\d.]+)\$", None),
    "joint_reach": grab(r"reaching Spearman \$([\d.]+)/([\d.]+)\$ and Top-1 \$([\d.]+)/([\d.]+)\$", None),
    "greedy_main": grab(r"Top-1 accuracies of \$([\d.]+)\$ on the GPU and \$([\d.]+)\$ on the NPU when evaluated", None),
    "greedy_bounds_caption": grab(r"45-group lower bounds are ([\d.]+) \(GPU\) and ([\d.]+) \(NPU\)", None),
    "decomp_pooled": grab(r"pooled instance-level Spearman correlations of \$([\d.]+)/([\d.]+)\$", None),
    "decomp_top1": grab(r"placement-level Top-1 accuracies of \$([\d.]+)/([\d.]+)\$", None),
    # Moved: the Sec.4 sentence that printed these was dropped in the Sec.3 rewrite.
    # The values still appear in the App.E table, so the check follows them there
    # rather than being deleted.
    "decomp_recon": grab(r"Measured-attainment.*?\$[\d.]+ / ([\d.]+) / [\d.]+\$\s*&\s*"
                         r"\$[\d.]+ / ([\d.]+) / [\d.]+\$", None, flags=re.S),
    # appendix A -- VLM pinning rationale (2026-08-17 신설). 정수 앵커(appA_exhaustive
    # 등)와 별개로 백분율을 따로 잡고, 재현 측에서는 같은 카운트에서 유도해 대조한다.
    "appA_vlm_cost": grab(r"CPU time to first token is \$([\d.]+)\$\\,s against \$([\d.]+)\$"
                          r"\\,s on the GPU and \$([\d.]+)\$\\,s on the NPU, so a single "
                          r"(\d+)-token request takes about \$(\d+)\$\\,s on the CPU and "
                          r"roughly (\w+) complete within a (\d+)\\,s window", None),
    "appA_rates": grab(r"corresponding rates are \$([\d.]+)\\%\$ against \$([\d.]+)\\%\$ "
                       r"over all (\d+) groups at \$\\beta\{=\}1\.0\$, and \$([\d.]+)\\%\$ "
                       r"against \$([\d.]+)\\%\$ at \$\\beta\{=\}0\.5\$", None),
    "appA_exhaustive": grab(r"the GPU and NPU disagree on the best measured placement in (\d+) groups: (\d+) of (\d+) generative groups and none of the (\d+) vision-only", None),
    "appA_beta05_exhaustive": grab(r"At \$\\beta\{=\}0\.5\$, (\d+) of the (\d+) exhaustive groups diverge", None),
    "appA_counts": grab(r"yields (\d+) placement configurations\..*?produces (\d+) measurement "
                        r"windows and (\d+) workload groups", None, flags=re.S),
    # appendix B (y3)
    "appB_masking": grab(r"trained only on the (\d+) windows per platform[^.]*\. "
                         r"The remaining (\d+) windows consist of (\d+) vision-only windows,"
                         r".*?and (\d+) windows in which the LLM", None, flags=re.S),
    "appB_window_totals": grab(r"recorded window totals range from \$([\d.]+)\$ to \$([\d.]+)\$ tok/s", None),
    "appB_extrap": grab(r"range from \$([\d.]+)\$ to \$([\d.]+)\$ on the GPU and from \$([\d.]+)\$ to \$([\d.]+)\$ on the NPU", None),
    "appB_restored": grab(r"remain at most \$([\d.]+)\$ on the GPU and \$([\d.]+)\$ on the NPU", None),
    "appB_sensitivity": grab(r"across the (\w+) predictor--platform combinations.*?"
                             r"preserves the same (\d+) workload groups", None, flags=re.S),
    "appB_max_spearman_change": grab(r"largest absolute change in group-mean Spearman "
                                     r"correlation is \$([\d.]+)\$", None),
    # appendix D
    "appD_vlm_capacity": grab(r"VLM consumes ([\d.]+)--([\d.]+) of the estimated capacity, compared with ([\d.]+)--([\d.]+) on the GPU", None),
    "appD_cum_exceeds": grab(r"cumulative estimated load exceeds 1\.0 in (\d+) of the (\d+) NPU workload groups", None),
    # appendix E
    "appE_top5": grab(r"Top-5 accuracy[^$]*is \$([\d.]+)\$ on both platforms", None),
    # appendix H -- "What differs." 문단 (2026-08-17 신설)
    "appH_what_differs": grab(r"In all (\d+) disagreeing groups the two platforms place the "
                              r"same generative instances.*?In (\w+) of them the NPU offloads "
                              r"(\w+) to the CPU while the GPU keeps every instance on the "
                              r"accelerator", None, flags=re.S),
    # appendix H
    "appH_margins": grab(r"smallest of the (\d+) margins is \$([\d.]+)\\times10\^\{-3\}\$", None),
    "appH_median_margin": grab(r"the median is \$([\d.]+)\$\.", None),
    # (n_cells<0.01, n_margins, any: div/total, both: div/total, 상한 밴드)
    "appH_band": grab(r"Only (\w+) of the (\d+) margins fall below \$0\.01\$.*?"
                      r"divergence count would be (\d+) of (\d+)\..*?"
                      r"leaves the count unchanged at (\d+) of (\d+), and that remains "
                      r"true for bands up to \$([\d.]+)\$", None, flags=re.S),
    "appH_ratio_direct": grab(r"averages \$([\d.]+)\$/\$([\d.]+)\$ \(GPU/NPU\) for direct regression, with a worst case of \$([\d.]+)\$/\$([\d.]+)\$", None),
    "appH_ratio_decomp": grab(r"decomposed baseline averages \$([\d.]+)\$/\$([\d.]+)\$ but with "
                              r"more low-ratio cases, with a worst case of \$([\d.]+)\$/\$([\d.]+)\$ "
                              r"and (\w+) GPU group and (\w+) NPU groups falling below", None),
    "appH_ratio_greedy": grab(r"Greedy averages \$([\d.]+)\$/\$([\d.]+)\$ over measured outputs under speedup ordering, and \$([\d.]+)\$/\$([\d.]+)\$ when unmeasured", None),
    # appendix F
    "appF_grid": grab(r"four candidate pairs: \$\((\d), ([\d.]+)\)\$, \$\((\d), ([\d.]+)\)\$, "
                      r"\$\((\d), ([\d.]+)\)\$, and \$\((\d), ([\d.]+)\)\$", None),
    # subsample 과 column-sample 이 별도 값으로 분리되어 캡처가 5개 -> 6개가 되었다
    "appF_fixed": grab(r"fixed to the (squared-error) objective, (\d+) boosting rounds, "
                       r"a subsampling rate of \$([\d.]+)\$, a column-sampling rate of "
                       r"\$([\d.]+)\$, a minimum child weight of \$(\d)\$, and seed~\$(\d+)\$",
                       None),
    "appF_features": grab(r"yields (\d+) features, and the instance count \$N\$ completes "
                          r"the (\d+)-dimensional", None),
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

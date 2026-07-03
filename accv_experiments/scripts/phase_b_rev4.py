"""Phase B (rev 4) — autonomous re-aggregation for B1 + B1.5 + B2.

No new measurement. Reads existing P1 sweep CSVs and the YOLOv11s anchors
(step_a, gen_decomp, cstar, step_g2) and produces:

  results/L_onset.csv          per-detector × size group: lowest ladder level
                                where |L(s,C)| ≥ 0.005 sAP (single-stream)
  results/L_per_level.csv      per-detector × size group × ladder level: L value
  results/L_multistream.csv    per-detector × size group at N=4 Composition A bg L1_light:
                                concurrency-induced L = N=1 GPU L1 sAP − N=4 GPU-resident sAP
  results/gen_gain_v3.csv      gen_gain with explicit mean-gain column

  paper/tables/gen_decomp.tex   regenerated. Staleness block becomes L(s, L2_lm)
                                if any detector shows meaningful L at L2_lm;
                                otherwise stays at L(s, L1_light) with a footnote.
                                (Decision recorded in results/B1_decision.json.)
  paper/tables/gen_gain.tex     regenerated with 3 columns: worst gain (bold) |
                                mean gain | worst/mean ratio.

Reads:
  results/p1_baseline_<det>.csv         (24 logs × {GPU,NPU} × L0)
  results/p1_ladder_<det>.csv           (24 logs × GPU × {L1_light,L1_heavy,L2_lm,L3_vlm})
  results/p1_multistream_<det>.csv      (Composition A N=4, N=8)
  results/step_a_baseline.csv           (v11s 24-log GPU/NPU L0 global8)
  results/gen_decomp.csv                (v11s 24-log GPU L1_light single)
  results/cstar.csv                     (v11s 24-log GPU L1_heavy/L2_lm/L3_vlm)
  results/step_g2_single_mode.csv       (v11s multistream N=4/5/6/8 bg L1)
  results/step_e_size_classification.csv (sid -> size_label)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _step_d_common import load_val

RES = SCRIPT_DIR.parent / "results"
PAPER_TAB = SCRIPT_DIR.parent.parent / "paper" / "tables"
PAPER_TAB.mkdir(parents=True, exist_ok=True)

# ---- detector registry ----
DETECTORS = [
    {"name": "yolov8n",  "params_M": 3.2,  "display": "YOLOv8n"},
    {"name": "yolov8s",  "params_M": 11.2, "display": "YOLOv8s"},
    {"name": "yolov8m",  "params_M": 25.9, "display": "YOLOv8m"},
    {"name": "yolov8l",  "params_M": 43.7, "display": "YOLOv8l"},
    {"name": "yolov8x",  "params_M": 68.2, "display": "YOLOv8x"},
    {"name": "yolo11s",  "params_M": 9.4,  "display": "YOLOv11s (marker)"},
]
LADDER = ["L0", "L1_light", "L1_heavy", "L2_lm", "L3_vlm"]
SIZE_GROUPS = {
    "small-rich":   [2, 12, 13, 15, 16, 19, 22, 23],
    "medium-mixed": [0, 1, 6, 8, 10, 11, 17, 20],
    "large-rich":   [3, 4, 5, 7, 9, 14, 18, 21],
}
COMP_A_N4 = [2, 22, 3, 21]


# ============================ helpers ============================

def _norm_sap_cols(df):
    """Map sap_small/medium/large -> sap_s/m/l in-place."""
    rename = {}
    for src, dst in [("sap_small", "sap_s"), ("sap_medium", "sap_m"),
                      ("sap_large", "sap_l")]:
        if src in df.columns and dst not in df.columns:
            rename[src] = dst
    if rename:
        df = df.rename(columns=rename)
    return df


def load_ladder_sap_per_sid(det):
    """Return dict bg_level -> {sid: {'sap_5095', 'sap_s', 'sap_m', 'sap_l'}}.
    For v11s, stitches step_a (L0), gen_decomp (L1_light), cstar (heavier).
    For v8 family, stitches p1_baseline (L0) + p1_ladder."""
    out = {}
    if det == "yolo11s":
        sa = pd.read_csv(RES / "step_a_baseline.csv")
        val = load_val()
        log_to_sid = {n: i for i, n in enumerate(val["sequences"])}
        sa["sid"] = sa["log_id"].map(log_to_sid)
        sa = _norm_sap_cols(sa)
        sa_gpu = sa[sa.device == "GPU"]
        out["L0"] = {int(r["sid"]): {"sap_5095": float(r["sap_5095"]),
                                      "sap_s": float(r["sap_s"]),
                                      "sap_m": float(r["sap_m"]),
                                      "sap_l": float(r["sap_l"])}
                      for _, r in sa_gpu.iterrows()}
        gd = pd.read_csv(RES / "gen_decomp.csv")
        gd_l1 = gd[(gd.device == "GPU") & (gd.bg_level == "L1_light")]
        out["L1_light"] = {int(r["sid"]): {"sap_5095": float(r["sap_5095"]),
                                            "sap_s": float(r["sap_s"]),
                                            "sap_m": float(r["sap_m"]),
                                            "sap_l": float(r["sap_l"])}
                            for _, r in gd_l1.iterrows()}
        cs = pd.read_csv(RES / "cstar.csv")
        for bg in ["L1_heavy", "L2_lm", "L3_vlm"]:
            sub = cs[cs.bg_level == bg]
            out[bg] = {int(r["sid"]): {"sap_5095": float(r["sap_5095"]),
                                        "sap_s": float(r["sap_s"]),
                                        "sap_m": float(r["sap_m"]),
                                        "sap_l": float(r["sap_l"])}
                        for _, r in sub.iterrows()}
        return out

    # v8 family
    base = pd.read_csv(RES / f"p1_baseline_{det}.csv")
    ldr = pd.read_csv(RES / f"p1_ladder_{det}.csv")
    base_gpu = base[(base.device == "GPU") & (base.bg_level == "L0")]
    out["L0"] = {int(r["sid"]): {"sap_5095": float(r["sap_5095"]),
                                  "sap_s": float(r["sap_s"]),
                                  "sap_m": float(r["sap_m"]),
                                  "sap_l": float(r["sap_l"])}
                  for _, r in base_gpu.iterrows()}
    for bg in ["L1_light", "L1_heavy", "L2_lm", "L3_vlm"]:
        sub = ldr[ldr.bg_level == bg]
        out[bg] = {int(r["sid"]): {"sap_5095": float(r["sap_5095"]),
                                    "sap_s": float(r["sap_s"]),
                                    "sap_m": float(r["sap_m"]),
                                    "sap_l": float(r["sap_l"])}
                    for _, r in sub.iterrows()}
    return out


# ============================ B1: L per level ============================

def compute_L_per_level():
    rows = []
    for det in DETECTORS:
        name = det["name"]
        try:
            data = load_ladder_sap_per_sid(name)
        except Exception as e:
            print(f"  {name}: cannot load ladder: {e}")
            continue
        if "L0" not in data:
            print(f"  {name}: no L0 baseline available")
            continue
        L0 = data["L0"]
        for grp, sids in SIZE_GROUPS.items():
            for bg in LADDER[1:]:  # L0 baseline used as reference
                if bg not in data:
                    continue
                cur = data[bg]
                ls = []; lm = []; ll = []
                for sid in sids:
                    if sid not in L0 or sid not in cur:
                        continue
                    ls.append(L0[sid]["sap_s"] - cur[sid]["sap_s"])
                    lm.append(L0[sid]["sap_m"] - cur[sid]["sap_m"])
                    ll.append(L0[sid]["sap_l"] - cur[sid]["sap_l"])
                rows.append({
                    "detector": name, "params_M": det["params_M"],
                    "group": grp, "bg_level": bg,
                    "L_small": float(np.mean(ls)) if ls else None,
                    "L_medium": float(np.mean(lm)) if lm else None,
                    "L_large": float(np.mean(ll)) if ll else None,
                })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "L_per_level.csv", index=False)
    print(f"saved {RES / 'L_per_level.csv'} ({len(df)} rows)")
    return df


def compute_L_onset(df_L, threshold=0.005):
    """For each (detector, group, size), find the lowest ladder level where
    |L| ≥ threshold. None if never."""
    rows = []
    for (det, grp), sub in df_L.groupby(["detector", "group"]):
        sub = sub.set_index("bg_level")
        for size_col in ["L_small", "L_medium", "L_large"]:
            onset = None
            value_at_onset = None
            for bg in LADDER[1:]:
                if bg not in sub.index:
                    continue
                v = sub.loc[bg, size_col]
                if v is None or np.isnan(v):
                    continue
                if abs(v) >= threshold:
                    onset = bg
                    value_at_onset = v
                    break
            rows.append({
                "detector": det,
                "group": grp,
                "size": size_col.replace("L_", ""),
                "onset_bg": onset,
                "L_at_onset": value_at_onset,
                "threshold": threshold,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "L_onset.csv", index=False)
    print(f"saved {RES / 'L_onset.csv'} ({len(df)} rows)")
    return df


def decide_gen_decomp_L_slice(df_L, df_onset):
    """Decide which ladder level to feature in gen_decomp's L block.
    Rule: prefer L2_lm if at least one (detector, size, group) has |L|≥0.005;
    otherwise fall back to L_max."""
    sub = df_L[df_L.bg_level == "L2_lm"].copy()
    n_meaningful = 0
    for _, r in sub.iterrows():
        for c in ["L_small", "L_medium", "L_large"]:
            if r[c] is not None and not np.isnan(r[c]) and abs(r[c]) >= 0.005:
                n_meaningful += 1
    decision = "L2_lm" if n_meaningful >= 5 else "L_max"
    info = {"decision": decision, "L2_lm_meaningful_cells": n_meaningful}
    with open(RES / "B1_decision.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"B1 decision: featured slice = {decision} (L2_lm meaningful cells = {n_meaningful})")
    return decision


def L_for_table(df_L, slice_kind):
    """Compute per-detector mean L across groups for the chosen slice."""
    if slice_kind == "L2_lm":
        sub = df_L[df_L.bg_level == "L2_lm"]
        per_det = {}
        for det, group in sub.groupby("detector"):
            per_det[det] = {
                "L_small":  float(group["L_small"].mean(skipna=True)),
                "L_medium": float(group["L_medium"].mean(skipna=True)),
                "L_large":  float(group["L_large"].mean(skipna=True)),
                "slice_label": "L2_lm",
            }
        return per_det
    # L_max per detector per size: pick the ladder level with the largest |L|
    per_det = {}
    for det, group in df_L.groupby("detector"):
        row = {"slice_label": "max-over-ladder"}
        for sz in ["L_small", "L_medium", "L_large"]:
            best_v = 0.0
            best_bg = None
            for _, r in group.iterrows():
                v = r[sz]
                if v is None or np.isnan(v): continue
                if abs(v) > abs(best_v):
                    best_v = v; best_bg = r["bg_level"]
            row[sz] = best_v
            row[sz + "_bg"] = best_bg
        per_det[det] = row
    return per_det


# ============================ B1.5: multi-stream L ============================

def compute_multistream_L():
    """Concurrency-induced L estimate:
       For each detector at N=4 Composition A bg L1_light, SizeBlindRev placement
       (large→NPU, small→GPU): the GPU-resident streams are the small-rich ones.
       Compare per-size sAP of GPU-resident streams vs N=1 GPU L1_light sAP for
       the same sids. The drop = concurrency-induced L_g for small-rich.

       Similarly, SizeAware places large-rich on GPU → concurrency L_g for large-rich.

       Output: per-detector, per-group, L_multistream.
    """
    rows = []
    # 1. Per-detector N=1 GPU L1_light sAP per sid (already in ladder data)
    for det in DETECTORS:
        name = det["name"]
        try:
            data = load_ladder_sap_per_sid(name)
        except Exception:
            continue
        gpu_l1_per_sid = data.get("L1_light", {})

        # 2. N=4 placements
        if name == "yolo11s":
            g2 = pd.read_csv(RES / "step_g2_single_mode.csv")
            sub = g2[(g2.n_streams == 4) & (g2.bg_level == "L1") &
                     (g2.placement_spec.isin(['["NPU", "NPU", "GPU", "GPU"]',
                                              '["GPU", "GPU", "NPU", "NPU"]']))]
            ms_rows = sub.to_dict("records")
        else:
            fp = RES / f"p1_multistream_{name}.csv"
            if not fp.exists(): continue
            df = pd.read_csv(fp)
            ms_rows = df[(df.n_streams == 4) &
                          (df.placement_name.isin(["SizeAware", "SizeBlindRev"]))].to_dict("records")

        # 3. For each multi-stream row, extract GPU-resident per-stream sAP
        for r in ms_rows:
            placement = json.loads(r["placement_spec"]) if isinstance(r["placement_spec"], str) else r["placement_spec"]
            pname = r.get("placement_name", "")
            # Build per-stream view
            for i in range(4):
                dev = r.get(f"s{i}_dev", placement[i])
                if dev != "GPU":
                    continue
                sid = int(r[f"s{i}_sid"])
                # Determine group of this sid
                grp = None
                for g, gsids in SIZE_GROUPS.items():
                    if sid in gsids:
                        grp = g
                        break
                if grp is None:
                    continue
                # Reference: N=1 GPU L1_light sAP for same sid
                ref = gpu_l1_per_sid.get(sid)
                if ref is None:
                    continue
                for size in ["s", "m", "l"]:
                    n1_v = ref[f"sap_{size}"]
                    n4_v = float(r[f"s{i}_sap_{size}"]) if f"s{i}_sap_{size}" in r and pd.notna(r[f"s{i}_sap_{size}"]) else None
                    if n4_v is None: continue
                    L_multi = n1_v - n4_v
                    rows.append({
                        "detector": name, "params_M": det["params_M"],
                        "placement": pname, "stream_idx": i,
                        "sid": sid, "group": grp, "size": size,
                        "N1_GPU_L1_sAP": n1_v,
                        "N4_GPU_sAP": n4_v,
                        "L_multistream": L_multi,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "L_multistream_raw.csv", index=False)
    # Aggregate to per-(detector, group, size) at SizeBlindRev (small-rich on GPU)
    # and SizeAware (large-rich on GPU); average L
    if not len(df):
        print("  no multistream L rows extracted")
        return df
    agg = df.groupby(["detector", "params_M", "placement", "group", "size"])["L_multistream"].mean().reset_index()
    agg.to_csv(RES / "L_multistream.csv", index=False)
    print(f"saved {RES / 'L_multistream_raw.csv'} ({len(df)} rows)")
    print(f"saved {RES / 'L_multistream.csv'} ({len(agg)} rows)")
    return agg


# ============================ B2: gen_gain restructure ============================

def write_gen_gain_v3():
    df = pd.read_csv(RES / "gen_gain_v2.csv")
    out_rows = []
    for det in DETECTORS:
        sub = df[df.detector == det["name"]]
        if not len(sub):
            out_rows.append({"detector": det["display"], "family":
                              ("YOLOv11" if det["name"] == "yolo11s" else "YOLOv8"),
                              "inverts": "TBD", "worst_gain": None,
                              "mean_gain": None, "ratio": None})
            continue
        r = sub.iloc[0]
        out_rows.append({
            "detector": det["display"],
            "family": ("YOLOv11" if det["name"] == "yolo11s" else "YOLOv8"),
            "inverts": "yes" if r["inverts"] else "no",
            "worst_gain": float(r["worst_gain"]),
            "mean_gain": float(r["mean_gain"]),
            "ratio": float(r["worst_mean_ratio"]),
        })
    summary = pd.DataFrame(out_rows)
    summary.to_csv(RES / "gen_gain_v3.csv", index=False)
    # Write LaTeX
    tex_path = PAPER_TAB / "gen_gain.tex"
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by accv_experiments/scripts/phase_b_rev4.py (B2).\n")
        f.write("% Worst-stream gain SizeAware - SizeBlindRev at N=4 bg L1_light,\n")
        f.write("% Composition A. Worst gain is bold (operative claim); mean gain shown\n")
        f.write("% separately so reader sees the noisy ratio denominator directly.\n")
        f.write("\\begin{tabular}{ll|cccc}\n\\toprule\n")
        f.write("Detector & family & inverts? & worst gain & mean gain & worst/mean \\\\\n")
        f.write("\\midrule\n")
        for r in out_rows:
            wg = r["worst_gain"]; mg = r["mean_gain"]; ratio = r["ratio"]
            if wg is None:
                f.write(f"{r['detector']} & {r['family']} & TBD & TBD & TBD & TBD \\\\\n")
                continue
            wg_s = f"\\textbf{{${wg:+.3f}$}}"
            mg_s = f"${mg:+.3f}$"
            if ratio is None or ratio != ratio:
                ratio_s = "---"
            elif ratio == float("inf") or ratio == float("-inf"):
                ratio_s = "$\\infty$"
            else:
                ratio_s = f"${ratio:+.1f}\\times$"
            f.write(f"{r['detector']} & {r['family']} & {r['inverts']} & {wg_s} & {mg_s} & {ratio_s} \\\\\n")
        f.write("\\midrule\n")
        f.write("RT-DETR / PicoDet & DETR/anchor-free & \\multicolumn{4}{c}{\\emph{future work (no INT8 NPU export available)}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {tex_path}")
    return summary


# ============================ B1 table writer ============================

def write_gen_decomp_v3(L_per_det, slice_label, df_L):
    """Q stays the same (single-stream L0 sap gap). L block uses the chosen slice."""
    qd = pd.read_csv(RES / "gen_decomp_v2.csv")
    qd = qd.set_index("detector")
    tex_path = PAPER_TAB / "gen_decomp.tex"
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by accv_experiments/scripts/phase_b_rev4.py (B1).\n")
        f.write(f"% L block uses single-stream {slice_label} (averaged over the three size groups).\n")
        f.write("% Q(s) is unchanged from gen_decomp_v2 (single-stream L0 sap gap GPU - NPU).\n")
        f.write("\\begin{tabular}{ll|ccc|ccc}\n\\toprule\n")
        slice_hdr = "L_2" if slice_label == "L2_lm" else "C^{\\max}"
        f.write("& & \\multicolumn{3}{c|}{$Q(s)$ (quantization)} & "
                "\\multicolumn{3}{c}{$L(s,\\text{" + slice_label.replace('_','\\_') + "})$ (staleness)} \\\\\n")
        f.write("Detector & params & small & medium & large & small & medium & large \\\\\n")
        f.write("\\midrule\n")
        for det in DETECTORS:
            name = det["name"]
            disp = det["display"]; pm = det["params_M"]
            if name not in qd.index:
                f.write(f"{disp} & {pm}\\,M & TBD & TBD & TBD & TBD & TBD & TBD \\\\\n")
                continue
            qrow = qd.loc[name]
            Qs = float(qrow["Q_small"]); Qm = float(qrow["Q_medium"]); Ql = float(qrow["Q_large"])
            L = L_per_det.get(name)
            if L is None:
                Ls_s = Lm_s = Ll_s = "TBD"
            else:
                Ls_s = f"${L['L_small']:+.3f}$"
                Lm_s = f"${L['L_medium']:+.3f}$"
                Ll_s = f"${L['L_large']:+.3f}$"
            f.write(f"{disp} & {pm}\\,M & ${Qs:+.3f}$ & ${Qm:+.3f}$ & ${Ql:+.3f}$ & "
                    f"{Ls_s} & {Lm_s} & {Ll_s} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {tex_path}")


# ============================ inline main_vision.tex update ============================

def update_main_vision_tex(L_per_det, slice_label, gain_rows):
    """Replace the inline tab:gen-decomp body (TBD rows) and tab:gen-gain body."""
    fp = Path(__file__).resolve().parent.parent.parent / "paper" / "main_vision.tex"
    src = fp.read_text()
    # Find tab:gen-decomp body. We replace the YOLOv8n..YOLOv8x rows plus the
    # YOLOv11s row inside the tabular.
    # Look for the line "Detector & params & small & medium & large & small & medium & large \\\\"
    # then \midrule, then the 5 v8 rows, then \midrule, then v11s row.
    # We'll splice based on text anchors.
    import re

    # B1 — Replace tab:gen-decomp block (from "\midrule" after header to "\bottomrule")
    # First find the unique header anchor for gen-decomp.
    header = "Detector & params & small & medium & large & small & medium & large \\\\"
    pos = src.find(header)
    if pos < 0:
        print("[update_main_vision_tex] gen-decomp header not found")
    else:
        mid = src.find("\\midrule", pos) + len("\\midrule")
        bottom = src.find("\\bottomrule", mid)
        # Build new body
        body = ["\n"]
        qd = pd.read_csv(RES / "gen_decomp_v2.csv").set_index("detector")
        for i, det in enumerate(DETECTORS):
            if det["name"] == "yolo11s":
                body.append("\\midrule\n")
            qrow = qd.loc[det["name"]]
            Qs = float(qrow["Q_small"]); Qm = float(qrow["Q_medium"]); Ql = float(qrow["Q_large"])
            L = L_per_det.get(det["name"])
            if L is None:
                Ls_s = Lm_s = Ll_s = "TBD"
            else:
                Ls_s = f"${L['L_small']:+.3f}$"
                Lm_s = f"${L['L_medium']:+.3f}$"
                Ll_s = f"${L['L_large']:+.3f}$"
            body.append(f"{det['display'].replace(' (marker)','')} & {det['params_M']}\\,M & "
                        f"${Qs:+.3f}$ & ${Qm:+.3f}$ & ${Ql:+.3f}$ & {Ls_s} & {Lm_s} & {Ll_s} \\\\\n")
        new_body = "".join(body)
        src = src[:mid] + new_body + src[bottom:]
        # Also patch the column header line right BEFORE the body header so
        # it shows the chosen slice (L_2 or C^max).
        # The line above the header looks like:
        # & & \multicolumn{3}{c|}{$Q(s)$ ...} & \multicolumn{3}{c}{$L(s,\text{L1})$ (staleness)} \\
        old_lhdr = "\\multicolumn{3}{c}{$L(s,\\text{L1})$ (staleness)}"
        if old_lhdr in src:
            new_lhdr = "\\multicolumn{3}{c}{$L(s,\\text{" + slice_label.replace("_", "\\_") + "})$ (staleness)}"
            src = src.replace(old_lhdr, new_lhdr, 1)

    # B2 — Replace tab:gen-gain body. Header line:
    # Detector & family & inverts? & worst gain & worst/mean ratio \\\\
    old_header = "Detector & family & inverts? & worst gain & worst/mean ratio \\\\"
    new_header = "Detector & family & inverts? & worst gain & mean gain & worst/mean \\\\"
    if old_header in src:
        src = src.replace(old_header, new_header, 1)
    if new_header in src:
        # Adjust tabular spec from {ll|ccc} to {ll|cccc}
        # only at the first occurrence after the gen-gain label.
        gg_label_pos = src.find("\\label{tab:gen-gain}")
        # Find first {ll|ccc} after the label
        ll3_pos = src.find("{ll|ccc}", gg_label_pos)
        if ll3_pos >= 0:
            src = src[:ll3_pos] + "{ll|cccc}" + src[ll3_pos + len("{ll|ccc}"):]

        # Replace body rows
        # Anchor: after the new_header, find next \midrule, then body up to \midrule
        h_pos = src.find(new_header)
        mid1 = src.find("\\midrule", h_pos) + len("\\midrule")
        mid2 = src.find("\\midrule", mid1)
        # build body
        body = ["\n"]
        for r in gain_rows:
            wg = r["worst_gain"]; mg = r["mean_gain"]; ratio = r["ratio"]
            inv = r["inverts"]
            disp = r["detector"].replace(" (marker)", "")
            fam = r["family"]
            if wg is None:
                body.append(f"{disp} & {fam} & TBD & TBD & TBD & TBD \\\\\n")
                continue
            wg_s = f"\\textbf{{${wg:+.3f}$}}"
            mg_s = f"${mg:+.3f}$"
            if ratio is None or ratio != ratio:
                ratio_s = "---"
            elif ratio == float("inf") or ratio == float("-inf"):
                ratio_s = "$\\infty$"
            else:
                ratio_s = f"${ratio:+.1f}\\times$"
            body.append(f"{disp} & {fam} & {inv} & {wg_s} & {mg_s} & {ratio_s} \\\\\n")
        src = src[:mid1] + "".join(body) + src[mid2:]
        # Update the \multicolumn count in the future-work row from {3}{c} to {4}{c}
        src = src.replace("\\multicolumn{3}{c}{\\emph{future work (no INT8 NPU export available)}}",
                          "\\multicolumn{4}{c}{\\emph{future work (no INT8 NPU export available)}}", 1)

    fp.write_text(src)
    print(f"updated {fp}")


# ============================ main ============================

def main():
    print("\n=== B1: L per ladder level ===")
    df_L = compute_L_per_level()
    print("\n=== B1: L onset detection ===")
    df_onset = compute_L_onset(df_L)
    print("\nOnset summary:")
    for (det, grp), sub in df_onset.groupby(["detector", "group"]):
        for _, r in sub.iterrows():
            print(f"  {det:<10s} {grp:<14s} {r['size']:<6s} -> {r['onset_bg'] or 'never'}", end="")
            if r['L_at_onset'] is not None:
                print(f"  (|L|={abs(r['L_at_onset']):.4f})")
            else:
                print("")

    print("\n=== B1: gen_decomp slice decision ===")
    slice_label = decide_gen_decomp_L_slice(df_L, df_onset)
    L_per_det = L_for_table(df_L, slice_label)
    print(f"\nL table (slice={slice_label}):")
    for det, row in L_per_det.items():
        print(f"  {det:<10s} L_small={row['L_small']:+.3f}  L_medium={row['L_medium']:+.3f}  L_large={row['L_large']:+.3f}")

    print("\n=== B1.5: multi-stream L ===")
    df_ms = compute_multistream_L()
    if len(df_ms):
        print("\nMulti-stream L summary (per detector, group at N=4 L1_light):")
        for (det, grp, pl), sub in df_ms.groupby(["detector", "group", "placement"]):
            for _, r in sub.iterrows():
                print(f"  {det:<10s} {grp:<14s} {pl:<14s} size={r['size']:<2s}  L_multi={r['L_multistream']:+.4f}")

    print("\n=== B2: gen_gain restructure ===")
    gain_summary = write_gen_gain_v3()
    gain_rows = gain_summary.to_dict("records")

    print("\n=== Write tables and update main_vision.tex ===")
    write_gen_decomp_v3(L_per_det, slice_label, df_L)
    update_main_vision_tex(L_per_det, slice_label, gain_rows)


if __name__ == "__main__":
    main()

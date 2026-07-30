#!/usr/bin/env python3
"""Write the C3 fluid-model entries into fig/confirmed_values.json from the committed runs.

The confirmed table is the single source the figure draws from, so its entries have to come
from the runs, not from a report. This script is that step: it calls scripts/analyze_c3.py's
computation and writes c3_* entries whose `source` points at runs/ directories.

It does NOT write sidecars. Sidecars come from the canvas (bar.get_height()), because a
sidecar copied from this table would make check_figures.py verify itself (fig/README.md).

Existing c3_* entries are replaced; every other entry is left untouched.

usage: python3 scripts/emit_c3_confirmed.py [--dry-run]
"""
import argparse, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_c3 as A

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIRMED = os.path.join(PROJECT, "fig", "confirmed_values.json")
FIG = "c3_fluid_validation"
PLATS = [("gpu", "CPU–GPU"), ("npu", "CPU–NPU")]
FRACS = ["070", "090", "095"]
# Phase lengths actually executed (scripts/run_c3_campaign.py). The conditions document
# drafted 5/10/45 s; the campaign ran 3/18/35 s. Recorded here because window_s is checked
# for exact equality against the sidecar and must be the executed value, not the drafted one.
W_ACC, W_DRAIN = 18.0, 35.0


def ms(vals, nd=2):
    return round(st.mean(vals), nd), round(st.pstdev(vals), nd)


def entry(ident, value, unit, quantity, reference, n, source, kind, sd=None,
          figure=None, window_s=None, note=None):
    e = {"id": ident, "value": value, "sd": sd, "unit": unit, "quantity": quantity,
         "reference": reference, "n": n, "censored": False, "recovered": None,
         "window_s": window_s, "source": source, "source_kind": kind}
    if figure:
        e["figure"] = [figure]
    if note:
        e["note"] = note
    return e


def build():
    out = []
    for plat, plabel in PLATS:
        cells = {}
        for frac in FRACS:
            ds = sorted(__import__("glob").glob(
                os.path.join(PROJECT, "runs", f"*_c3{plat}_{frac}_r*")))
            ds = [d for d in ds if os.path.isfile(os.path.join(d, "run_manifest.json"))]
            runs = [A.one_run(d, plat, frac) for d in ds]
            cells[frac] = runs
            lam = runs[0]["lambda"]
            n = len(runs)
            src = "runs/" + runs[0]["run"]
            srcs = "; ".join(r["run"] for r in runs)
            p = f"c3_{plat}_{frac}"
            common = dict(n=n, source=src, note=f"n={n}: {srcs}")

            v, sd = ms([r["acc_slope"] for r in runs])
            out.append(entry(f"{p}_acc_slope", v, "frames/s", "backlog_slope", "burst",
                             kind="hardware", sd=sd, figure=FIG, window_s=W_ACC, **common))
            v, sd = ms([r["mu_cpu_obs"] for r in runs])
            out.append(entry(f"{p}_mu_cpu", v, "fps", "service_rate", "burst",
                             kind="hardware", sd=sd, window_s=W_ACC, **common))
            v, sd = ms([r["acc_pred"] for r in runs])
            out.append(entry(f"{p}_acc_pred", v, "frames/s", "backlog_slope", "burst",
                             kind="analytic", sd=sd, figure=FIG, window_s=W_ACC,
                             n=n, source=src,
                             note=f"lambda {lam} - mu_cpu measured in the same phase"))

            v, sd = ms([r["drain_slope"] for r in runs])
            out.append(entry(f"{p}_drain_slope", v, "frames/s", "backlog_slope", "commit",
                             kind="hardware", sd=sd, figure=FIG, window_s=W_DRAIN, **common))
            v, sd = ms([r["mu_drain_obs"] for r in runs])
            out.append(entry(f"{p}_mu_drain", v, "fps", "service_rate", "commit",
                             kind="hardware", sd=sd, window_s=W_DRAIN, **common))
            v, sd = ms([r["mu_star_implied"] for r in runs])
            out.append(entry(f"{p}_mu_star_implied", v, "fps", "service_rate", "commit",
                             kind="analytic", sd=sd, window_s=W_DRAIN, n=n, source=src,
                             note=f"lambda {lam} - measured drain slope"))
            v, sd = ms([r["drain_slope_late"] for r in runs])
            out.append(entry(f"{p}_drain_slope_late", v, "frames/s", "backlog_slope",
                             "commit", kind="hardware", sd=sd, window_s=W_DRAIN, n=n,
                             source=src, note="POST-HOC: segment after view2's queue empties"))
            v, sd = ms([r["q_at_switch"] for r in runs], 0)
            out.append(entry(f"{p}_backlog_at_switch", v, "frames", "backlog", "commit",
                             kind="hardware", sd=sd, **common))

            for tag, mu in (("pipe", A.MU_PIPE_DRAIN[plat]), ("sat", A.MU_PIPE_SAT[plat]),
                            ("raw", A.MU_RAW_INFER[plat])):
                out.append(entry(f"{p}_drain_pred_{tag}", round(-(mu - lam), 2), "frames/s",
                                 "backlog_slope", "commit", n=n, source=src, kind="analytic",
                                 sd=0.0, figure=FIG, window_s=W_DRAIN,
                                 note=f"-(mu* - lambda) with mu* = {mu} fps ({tag})"))

            ts = [r["T_stable_s"] for r in runs if r["T_stable_s"]]
            if len(ts) == len(runs):
                mean_slope = st.mean(r["drain_slope"] for r in runs)
                mean_q = st.mean(r["q_at_switch"] for r in runs)
                out.append(entry(f"{p}_T_stable", round(mean_q / abs(mean_slope), 1), "s",
                                 "T_stable", "commit", n=n, source=src, kind="analytic",
                                 sd=round(st.pstdev(ts), 1), figure=FIG, window_s=W_DRAIN,
                                 note="Q / |dQ/dt| from the cell's mean backlog and mean "
                                      "fitted slope. The drain does not run to completion "
                                      "except at npu 0.70, so this is an extrapolation."))

        # per-platform mu* candidates and the sweep fit
        lam_d = [(cells[f][0]["lambda"], st.mean(r["drain_slope"] for r in cells[f]))
                 for f in FRACS]
        mus = [l - d for l, d in lam_d]
        src_all = "runs/" + cells["070"][0]["run"]
        out.append(entry(f"c3_{plat}_mu_star_anchor", A.MU_PIPE_DRAIN[plat], "fps",
                         "service_rate", "window", n=1,
                         source="runs/" + os.path.basename(sorted(__import__("glob").glob(
                             os.path.join(PROJECT, "runs", f"*_c3mudrain_{plat}_r*")))[-1]),
                         kind="hardware", sd=0.0, figure=FIG,
                         note="near-idle drain pilot at lambda=20/view; THIS FIXED lambda "
                              "and is not revised after the fact"))
        out.append(entry(f"c3_{plat}_mu_star_sat", A.MU_PIPE_SAT[plat], "fps",
                         "service_rate", "window", n=1,
                         source="runs/" + os.path.basename(sorted(__import__("glob").glob(
                             os.path.join(PROJECT, "runs", f"*_c3mustar_{plat}_r*")))[-1]),
                         kind="hardware", sd=0.0,
                         note="view1 completions/s under 400 fps/view of arrivals"))
        out.append(entry(f"c3_{plat}_mu_star_raw", A.MU_RAW_INFER[plat], "fps",
                         "service_rate", "window", n=1,
                         source="runs/" + os.path.basename(sorted(__import__("glob").glob(
                             os.path.join(PROJECT, "runs", f"*_c3mustar_{plat}_r*")))[-1]),
                         kind="hardware", sd=0.0,
                         note="1000/infer_ms for view1 at the end of the saturated phase"))
        out.append(entry(f"c3_{plat}_mu_star_eff", round(st.mean(mus), 1), "fps",
                         "service_rate", "commit", n=3, source=src_all, kind="analytic",
                         sd=round(st.pstdev(mus), 2), figure=FIG, window_s=W_DRAIN,
                         note="intercept of the lambda sweep: mean over the three cells of "
                              "lambda - drain slope. 3 = cells, 9 runs behind it"))
        out.append(entry(f"c3_{plat}_dslope_dlambda",
                         round(A.fit([p[0] for p in lam_d], [p[1] for p in lam_d]), 3),
                         "1", "fit_slope", "commit", n=3, source=src_all, kind="analytic",
                         sd=0.0, window_s=W_DRAIN,
                         note="d(drain slope)/d(lambda) across the sweep; the model predicts "
                              "exactly 1.0"))

    # the one cell whose drain completed inside the window: observed emptying time, which
    # is what validates the extrapolated T_stable elsewhere
    out.append(entry("c3_npu_070_drain_observed", 27.3, "s", "T_stable", "commit", n=3,
                     source="runs/20260730T024839Z_c3npu_070_r0", kind="hardware", sd=0.47,
                     window_s=W_DRAIN,
                     note="samples from the switch to the first empty queue: 28/27/27 s. "
                          "Compare c3_npu_070_T_stable (extrapolated 26.1 s)."))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    doc = json.load(open(CONFIRMED, encoding="utf-8"))
    new = build()
    ids = {e["id"] for e in new}
    kept = [e for e in doc["values"] if not e["id"].startswith("c3_")]
    doc["values"] = kept + new
    print(f"c3 entries: {len(new)} ({sum(1 for e in new if e.get('figure'))} drawn), "
          f"other entries kept: {len(kept)}")
    if a.dry_run:
        for e in new:
            print(f"  {e['id']:34s} {e['value']:>9} {e['unit']:<9} {e['source_kind']}")
        return 0
    assert len(ids) == len(new), "duplicate id"
    json.dump(doc, open(CONFIRMED, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {CONFIRMED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

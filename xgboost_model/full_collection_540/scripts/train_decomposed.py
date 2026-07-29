"""P6: Kim-style decomposed modeling — per-view degradation prediction + composition.

View-level target (VISION views only; per-view tokens were not preserved, so y3
cannot be decomposed):
    r = min(1.0, throughput_fps / min(infps, capacity_fps))
with the number of clipped views recorded. Features are the view's own 12 plan
features (the suite's _view_features) plus co-runner summaries — all plan-level,
never window-level measurements (target leakage).

CV protocol matches P1 exactly: the view inherits its window's (models,
rate_factor) group and P1's group->fold assignment (read from
groupkfold_{platform}_metrics.json), the same 4-combo grid, group-aware inner CV.

Composition per window (group stats are the group's MEASURED stats, matching the
ranking protocol's rank-preserving within-group normalization):
    y1_decomp = sum(r_pred * min(infps, cap)) over vision views / Fmax_meas(group)
    y2_decomp = clip01(mean(1 - r_pred)) min-max'd with the group's measured
                clipped-y2 lo/hi  — an APPROXIMATE reconstruction of window y2
    y3        = P1's set-level OOF y3 prediction, shared verbatim
    S_decomp  = y1_decomp - 0.3*y2_decomp + 1.0*y3 (generative sets)
A measured-ratio composition (same formulas with r_meas) isolates the
composition-approximation error from the model error.

Outputs: analysis/decomposed_results.md, decomposed_metrics.json,
decomposed_view_oof.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as ac
from xgboost_model.deploy_selector_xgb_suite import (
    _device_static, _is_vision, _norm_exec, _view_features, load_static_profiles)

VIEW_FEATURES = [
    "view.infps", "view.exec_cpu", "view.exec_gpu", "view.exec_npu",
    "view.is_vision", "view.is_llm", "view.static_infer_sel",
    "view.static_load_sel", "view.static_tokens_sel", "view.capacity_fps",
    "view.load_factor", "x.infps__static_infer_sel",
    "co.same_dev_load_sum", "co.same_dev_count",
    "co.other_dev_load_sum", "co.other_dev_count",
    "co.set_has_gen", "co.gen_on_accel",
]


def build_view_dataset(platform, S):
    """One row per (window, view). Vision rows get a target; gen rows ratio only."""
    imap = ac.infps_map_for(platform)
    rows = []
    for w in ac.load_windows(platform):
        combo = str(w["combination"])
        vmap = imap.get(combo, {})
        views = []
        for v in w["models"].values():
            m, dev = v["model"], _norm_exec(v["execution"])
            infps = float(vmap.get((m, dev), 0.0))
            s_infer, _, _ = _device_static(m, dev, S)
            cap = 1000.0 / s_infer if np.isfinite(s_infer) and s_infer > 0 else np.nan
            eff = min(infps, cap) if np.isfinite(cap) else infps
            thr = float(v.get("throughput_fps", 0.0) or 0.0)
            views.append({"model": m, "dev": dev, "infps": infps, "cap": cap,
                          "eff": eff, "thr": thr,
                          "lf": infps / cap if np.isfinite(cap) and cap > 0 else 0.0,
                          "is_vision": _is_vision(m)})
        for i, me in enumerate(views):
            feat = _view_features(me["model"], me["dev"], me["infps"], S)
            same = [o for j, o in enumerate(views) if j != i and o["dev"] == me["dev"]]
            other = [o for j, o in enumerate(views) if j != i and o["dev"] != me["dev"]]
            gen = [o for o in views if not o["is_vision"]]
            feat.update({
                "co.same_dev_load_sum": sum(o["lf"] for o in same),
                "co.same_dev_count": float(len(same)),
                "co.other_dev_load_sum": sum(o["lf"] for o in other),
                "co.other_dev_count": float(len(other)),
                "co.set_has_gen": 1.0 if gen else 0.0,
                "co.gen_on_accel": 1.0 if any(o["dev"] in ("gpu", "npu") for o in gen)
                                   else 0.0,
            })
            ratio_raw = me["thr"] / me["eff"] if me["eff"] > 1e-9 else np.nan
            rows.append({**{k: feat.get(k, 0.0) for k in VIEW_FEATURES},
                         "combination": combo, "model": me["model"], "dev": me["dev"],
                         "is_vision": me["is_vision"], "eff": me["eff"],
                         "ratio_raw": ratio_raw,
                         "target": min(1.0, ratio_raw) if np.isfinite(ratio_raw)
                                   else np.nan})
    return pd.DataFrame(rows)


def main():
    S = load_static_profiles(ac.STATIC_JSON)
    md_rows, res_all = [], {}
    for platform in ac.PLATFORMS:
        # ---- window-level context (groups, folds from P1, measured stats) ----
        _, Yraw, M = ac.load_dataset(platform)
        Yn = ac.normalize_targets(Yraw, M)
        p1 = json.loads((ac.ANALYSIS / f"groupkfold_{platform}_metrics.json").read_text())
        g2f = p1["group_to_fold"]
        w = pd.DataFrame({
            "combination": M["combination"].astype(str), "models": M["models"],
            "rate_factor": M["rate_factor"], "has_gen": M["has_gen"],
            "group": [str((m, r)) for m, r in zip(M["models"], M["rate_factor"])],
            "y1_raw": Yraw["y1_total_throughput_fps"],
            "y2_raw": np.clip(Yraw["y2_deadline_miss_rate"], 0, 1),
            "y1_meas": Yn["y1_total_throughput_fps"],
            "y2_meas": Yn["y2_deadline_miss_rate"],
            "y3_meas": Yn["y3_total_tokens_per_s"],
        })
        w["fold"] = w["group"].map(g2f)
        assert w["fold"].notna().all(), "P1 fold mapping must cover every group"
        p1_oof = pd.read_csv(ac.ANALYSIS / f"groupkfold_{platform}_oof.csv", comment="#")
        w = w.merge(p1_oof[["combination", "y3_pred"]], on="combination", how="left")
        gstat = w.groupby("group").agg(fmax=("y1_raw", "max"),
                                       y2_lo=("y2_raw", "min"), y2_hi=("y2_raw", "max"))

        # ---- view dataset ----
        V = build_view_dataset(platform, S)
        V = V.merge(w[["combination", "group", "fold"]], on="combination", how="left")
        vis = V["is_vision"].values.astype(bool)
        n_clip = int((V.loc[vis, "ratio_raw"] > 1.0).sum())
        Xv = V.loc[vis, VIEW_FEATURES].values
        yv = V.loc[vis, "target"].values.astype(float)
        grp = V.loc[vis, "group"].reset_index(drop=True)
        fold = V.loc[vis, "fold"].values.astype(int)

        pred = np.full(len(yv), np.nan)
        chosen = {}
        for f in range(ac.N_FOLDS):
            tr, te = fold != f, fold == f
            params, inner_mae = ac.cv_select_params(
                Xv[tr], yv[tr], grp[tr].reset_index(drop=True),
                feature_names=VIEW_FEATURES)
            bst = ac.train_booster(Xv[tr], yv[tr], params, feature_names=VIEW_FEATURES)
            pred[te] = ac.predict_booster(bst, Xv[te], feature_names=VIEW_FEATURES)
            chosen[f"fold{f}"] = {"params": {k: params[k] for k in ("max_depth", "eta")},
                                  "inner_mae": inner_mae}
        V.loc[vis, "pred"] = pred

        view_stats = {
            "n_vision_views": int(vis.sum()), "n_gen_views": int((~vis).sum()),
            "n_clipped_targets": n_clip,
            "oof_spearman_pooled": ac.spearman(yv, pred),
            "oof_mae": float(np.mean(np.abs(yv - pred))),
            "hyperparams": chosen,
        }

        # ---- composition (predicted and measured-ratio variants) ----
        agg = {}
        for name, col in (("pred", "pred"), ("meas", "target")):
            g = V[vis].groupby("combination")
            y1_sum = g.apply(lambda d: float((d[col] * d["eff"]).sum()),
                             include_groups=False)
            miss = g.apply(lambda d: float(np.clip((1.0 - d[col]).mean(), 0.0, 1.0)),
                           include_groups=False)
            agg[name] = pd.DataFrame({f"y1_sum_{name}": y1_sum,
                                      f"miss_{name}": miss})
        comp = w.set_index("combination").join(agg["pred"]).join(agg["meas"]).reset_index()
        comp = comp.join(gstat, on="group")
        span = (comp["y2_hi"] - comp["y2_lo"]).values
        for name in ("pred", "meas"):
            comp[f"y1_decomp_{name}"] = comp[f"y1_sum_{name}"] / comp["fmax"]
            comp[f"y2_decomp_{name}"] = np.where(
                span > 1e-9, (comp[f"miss_{name}"] - comp["y2_lo"]) / np.where(
                    span > 1e-9, span, 1.0), 0.0)

        # y2 reconstruction error: measured-ratio composition vs actual y2 (raw scale)
        recon = {
            "y2_raw_mae": float(np.mean(np.abs(comp["miss_meas"] - comp["y2_raw"]))),
            "y2_raw_spearman": ac.spearman(comp["miss_meas"], comp["y2_raw"]),
            "y2_group_spearman_mean": float(np.nanmean(list(ac.per_group_spearman(
                comp.assign(a=comp["miss_meas"], b=comp["y2_raw"]),
                "group", "a", "b").values()))),
            "y1_raw_relerr_mean": float(np.mean(
                np.abs(comp["y1_sum_meas"] - comp["y1_raw"])
                / comp["y1_raw"].clip(lower=1e-9))),
        }

        # ---- score-level ranking ----
        comp["s_meas"] = ac.scores_from(comp["y1_meas"], comp["y2_meas"],
                                        comp["y3_meas"], comp["has_gen"])
        comp["s_pred"] = ac.scores_from(comp["y1_decomp_pred"], comp["y2_decomp_pred"],
                                        comp["y3_pred"], comp["has_gen"])
        ranking = ac.ranking_metrics(comp)

        res_all[platform] = {"view_level": view_stats, "y2_reconstruction": recon,
                             "ranking": ranking}
        with open(ac.ANALYSIS / "decomposed_view_oof.csv",
                  "w" if platform == ac.PLATFORMS[0] else "a") as fh:
            if platform == ac.PLATFORMS[0]:
                fh.write(ac.prov_csv_comment(experiment="P6 view-level oof"))
                fh.write("platform,combination,model,dev,eff_fps,ratio_raw,target,pred\n")
            for _, r in V[vis].iterrows():
                fh.write(f"{platform},{r['combination']},{r['model']},{r['dev']},"
                         f"{r['eff']:.3f},{r['ratio_raw']:.4f},{r['target']:.4f},"
                         f"{r['pred']:.4f}\n")
        print(platform, "view rho", round(view_stats["oof_spearman_pooled"], 3),
              "mae", round(view_stats["oof_mae"], 4), "clip", n_clip,
              "| ranking", {k: round(v, 3) for k, v in ranking.items()})

    # ---- outputs ----
    out = dict(res_all)
    out["_provenance"] = ac.provenance(
        experiment="P6 decomposed modeling",
        p1_reference="analysis/groupkfold_* (folds, y3_pred)",
        note="vision views only; per-view tokens not preserved -> y3 not decomposable")
    (ac.ANALYSIS / "decomposed_metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    p1m = {pl: json.loads((ac.ANALYSIS / f"groupkfold_{pl}_metrics.json").read_text())
           for pl in ac.PLATFORMS}
    greedy = pd.read_csv(ac.ANALYSIS / "greedy_results.csv", comment="#")
    md = [ac.prov_md_header(experiment="P6 decomposed modeling")]
    md.append("# P6 — 분해 모델링 (뷰 단위 저하율 예측 → 합성)\n")
    md.append("target r = min(1, throughput/min(infps, capacity)). vision 뷰만 학습 "
              "(뷰별 tokens 미보존 → y3 분해 불가, P1의 세트 수준 ŷ3 공유). "
              "fold·그룹은 P1과 동일.\n")
    md.append("## 뷰 수준 OOF\n")
    md.append("| 플랫폼 | vision 뷰 | clip된 target | OOF Spearman | OOF MAE |")
    md.append("|---|---|---|---|---|")
    for pl in ac.PLATFORMS:
        v = res_all[pl]["view_level"]
        md.append(f"| {pl} | {v['n_vision_views']} | {v['n_clipped_targets']} "
                  f"| {v['oof_spearman_pooled']:+.3f} | {v['oof_mae']:.4f} |")
    md.append("\n## y2 재구성 오차 (실측 저하율 합성 vs 실측 y2 — 모델 오차와 분리된 "
              "합성 근사 자체의 오차)\n")
    md.append("| 플랫폼 | raw MAE | pooled ρ | 그룹 ρ 평균 | (참고) y1 합성 상대오차 |")
    md.append("|---|---|---|---|---|")
    for pl in ac.PLATFORMS:
        r = res_all[pl]["y2_reconstruction"]
        md.append(f"| {pl} | {r['y2_raw_mae']:.4f} | {r['y2_raw_spearman']:+.3f} "
                  f"| {r['y2_group_spearman_mean']:+.3f} "
                  f"| {r['y1_raw_relerr_mean']:.4f} |")
    md.append("\n(ŷ2_decomp = 뷰별 (1−r)의 비가중 평균은 창 수준 y2의 **근사**다: 실제 "
              "y2는 요청 수 가중이 다르고 생성 뷰도 포함한다. 생성 뷰는 합성에서 제외.)\n")
    md.append("## 점수 수준 3열 비교 (45그룹, S = ŷ1 − 0.3·ŷ2 + 1.0·ŷ3)\n")
    md.append("| 플랫폼 | 방법 | Top-1 | Top-5 | 그룹 Spearman |")
    md.append("|---|---|---|---|---|")
    for pl in ac.PLATFORMS:
        r1 = p1m[pl]["ranking"]
        md.append(f"| {pl} | P1 직접 회귀 | {r1['top1']:.3f} | {r1['top5']:.3f} "
                  f"| {r1['group_spearman_mean']:+.3f} |")
        r6 = res_all[pl]["ranking"]
        md.append(f"| {pl} | P6 분해+합성 | {r6['top1']:.3f} | {r6['top5']:.3f} "
                  f"| {r6['group_spearman_mean']:+.3f} |")
        for var in ("speedup", "load_factor"):
            g = greedy[(greedy["platform"] == pl) & (greedy["variant"] == var)]
            coll = g[g["collected"] == 1]
            unc = int((g["collected"] == 0).sum())
            md.append(f"| {pl} | P5 greedy({var}) | {coll['top1'].mean():.3f}"
                      f"(수집 {len(coll)}/45, 미수집 {unc}) | — | — |")
    md.append("\n(P5는 배치 1개만 내므로 Top-5·Spearman 미정의. 세부: "
              "`decomposed_metrics.json`, `decomposed_view_oof.csv`, "
              "`greedy_results.csv`.)")
    (ac.ANALYSIS / "decomposed_results.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()

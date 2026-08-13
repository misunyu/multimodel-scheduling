"""P4: unified predictor — one model trained on both platforms' 1080 rows.

Same GroupKFold protocol as P1 (3-fold, seed=42, group-aware inner CV over the
same 4-combo grid, y3 masked per combination), with the CV group extended to
(platform, models, rate_factor) — 90 groups. Features stay the shipped 37: the
views.*.view.exec_gpu / exec_npu one-hots are what lets a single model separate
the platforms. Targets are normalized within (models, workload, rate) per
platform, exactly as P1, so scores remain within-group comparable.

Per-platform OOF metrics of the unified model are reported side by side with
P1's specialized predictors. Judgement criteria are stated but NOT applied —
that call belongs to the paper.

Outputs: analysis/unified_metrics.json, analysis/unified_oof.csv,
analysis/unified_vs_specialized.md. (No boosters are exported: nothing
downstream consumes a unified booster.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as ac


def main():
    Xs, Ys, Ms = [], [], []
    for pl in ac.PLATFORMS:
        X, Yraw, M = ac.load_dataset(pl)
        Yn = ac.normalize_targets(Yraw, M)
        Xs.append(X); Ys.append(Yn); Ms.append(M)
    X = pd.concat(Xs, ignore_index=True)
    Yn = pd.concat(Ys, ignore_index=True)
    M = pd.concat(Ms, ignore_index=True)
    assert len(X) == 1080

    groups = ac.group_keys(M, with_platform=True)
    fold = ac.assign_group_folds(groups).values
    Xv = X.values
    y3_valid = M["y3_valid"].values.astype(bool)

    oof = pd.DataFrame({
        "platform": M["platform"], "combination": M["combination"],
        "models": M["models"], "rate_factor": M["rate_factor"],
        "y3_valid": y3_valid, "has_gen": M["has_gen"], "fold": fold,
        "y1_meas": Yn["y1_total_throughput_fps"],
        "y2_meas": Yn["y2_deadline_miss_rate"],
        "y3_meas": Yn["y3_total_tokens_per_s"],
    })
    chosen = {}
    for tag, col in ac.TARGETS:
        yv = Yn[col].values.astype(float)
        valid = y3_valid if tag == "y3" else np.ones(len(yv), bool)
        pred = np.full(len(yv), np.nan)
        for f in range(ac.N_FOLDS):
            tr = (fold != f) & valid
            te = fold == f
            params, inner_mae = ac.cv_select_params(
                Xv[tr], yv[tr], groups[tr].reset_index(drop=True))
            bst = ac.train_booster(Xv[tr], yv[tr], params)
            pred[te] = ac.predict_booster(bst, Xv[te])
            chosen[f"{tag}/fold{f}"] = {
                "params": {k: params[k] for k in ("max_depth", "eta")},
                "inner_mae": inner_mae}
        oof[f"{tag}_pred"] = pred

    oof["group"] = [str((m, r)) for m, r in zip(M["models"], M["rate_factor"])]
    oof["s_meas"] = ac.scores_from(oof["y1_meas"], oof["y2_meas"], oof["y3_meas"],
                                   oof["has_gen"])
    oof["s_pred"] = ac.scores_from(oof["y1_pred"], oof["y2_pred"], oof["y3_pred"],
                                   oof["has_gen"])

    res = {"n_rows": 1080, "n_groups": int(groups.nunique()),
           "fold_row_counts": np.bincount(fold, minlength=ac.N_FOLDS).tolist(),
           "hyperparams": chosen, "per_platform": {}}
    p1 = {pl: json.loads((ac.ANALYSIS / f"groupkfold_{pl}_metrics.json").read_text())
          for pl in ac.PLATFORMS}
    for pl in ac.PLATFORMS:
        sub = oof[oof["platform"] == pl]
        entry = {"targets": {}}
        for tag, _ in ac.TARGETS:
            v = sub["y3_valid"] if tag == "y3" else pd.Series(True, index=sub.index)
            s = sub[v]
            gsp = [x for x in ac.per_group_spearman(s, "group", f"{tag}_meas",
                                                    f"{tag}_pred").values()
                   if np.isfinite(x)]
            entry["targets"][tag] = {
                "pooled_spearman": ac.spearman(s[f"{tag}_meas"], s[f"{tag}_pred"]),
                "group_spearman_mean": float(np.mean(gsp)) if gsp else None,
                "mae": float(np.mean(np.abs(s[f"{tag}_meas"] - s[f"{tag}_pred"])))}
        entry["ranking"] = ac.ranking_metrics(sub)
        res["per_platform"][pl] = entry
    res["_provenance"] = ac.provenance(experiment="P4 unified predictor",
                                       p1_reference="analysis/groupkfold_*_metrics.json")
    (ac.ANALYSIS / "unified_metrics.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    with open(ac.ANALYSIS / "unified_oof.csv", "w") as fh:
        fh.write(ac.prov_csv_comment(experiment="P4 unified oof"))
        oof.drop(columns=["s_meas", "s_pred"]).to_csv(fh, index=False)

    # ---- comparison md ----
    md = [ac.prov_md_header(experiment="P4 unified vs specialized")]
    md.append("# P4 — 통합 예측기(1080행) vs 플랫폼별 예측기(P1)\n")
    md.append("동일 프로토콜(GroupKFold 3-fold seed=42, 그룹 키에 플랫폼 포함 → 90그룹). "
              "feature 37차원 그대로 — exec one-hot이 플랫폼을 구분.\n")
    md.append("| 플랫폼 | 예측기 | 점수 그룹-ρ | Top-1 | Top-5 | y1 ρ | y2 ρ | y3 ρ |")
    md.append("|---|---|---|---|---|---|---|---|")
    for pl in ac.PLATFORMS:
        for name, src in (("플랫폼별(P1)", p1[pl]),
                          ("통합(P4)", None)):
            if src is not None:
                r, t = src["ranking"], src["targets"]
                md.append(f"| {pl} | {name} | {r['group_spearman_mean']:+.3f} "
                          f"| {r['top1']:.3f} | {r['top5']:.3f} "
                          f"| {t['y1']['oof_spearman_pooled']:+.3f} "
                          f"| {t['y2']['oof_spearman_pooled']:+.3f} "
                          f"| {t['y3']['oof_spearman_pooled']:+.3f} |")
            else:
                e = res["per_platform"][pl]
                r, t = e["ranking"], e["targets"]
                md.append(f"| {pl} | {name} | {r['group_spearman_mean']:+.3f} "
                          f"| {r['top1']:.3f} | {r['top5']:.3f} "
                          f"| {t['y1']['pooled_spearman']:+.3f} "
                          f"| {t['y2']['pooled_spearman']:+.3f} "
                          f"| {t['y3']['pooled_spearman']:+.3f} |")
    md.append("\n## 세트별 점수 Spearman (통합 − 플랫폼별, 그룹 평균)\n")
    md.append("| 플랫폼 | 개선 그룹 | 동일(±0.02) | 악화 그룹 |")
    md.append("|---|---|---|---|")
    detail = {}
    for pl in ac.PLATFORMS:
        sub = oof[oof["platform"] == pl]
        uni = {}
        for k, g in sub.groupby("group", sort=False):
            uni[k] = ac.spearman(
                ac.scores_from(g["y1_meas"], g["y2_meas"], g["y3_meas"], g["has_gen"]),
                ac.scores_from(g["y1_pred"], g["y2_pred"], g["y3_pred"], g["has_gen"]))
        oof_p1 = pd.read_csv(ac.ANALYSIS / f"groupkfold_{pl}_oof.csv", comment="#")
        oof_p1["group"] = [str((m, r)) for m, r in
                           zip(oof_p1["models"], oof_p1["rate_factor"])]
        spec = {}
        for k, g in oof_p1.groupby("group", sort=False):
            spec[k] = ac.spearman(
                ac.scores_from(g["y1_meas"], g["y2_meas"], g["y3_meas"], g["has_gen"]),
                ac.scores_from(g["y1_pred"], g["y2_pred"], g["y3_pred"], g["has_gen"]))
        diffs = {k: uni[k] - spec[k] for k in uni if np.isfinite(uni[k]) and
                 np.isfinite(spec.get(k, np.nan))}
        up = sum(1 for d in diffs.values() if d > 0.02)
        dn = sum(1 for d in diffs.values() if d < -0.02)
        md.append(f"| {pl} | {up} | {len(diffs) - up - dn} | {dn} |")
        detail[pl] = diffs
    md.append("\n## 판정 기준 (판정은 논문 쪽에서)\n")
    md.append("- 대등: 플랫폼별 대비 점수 그룹-ρ 차이 ≤ 0.01 그리고 Top-1 차이 ≤ 1그룹(0.022)")
    md.append("- 열세: 위 기준 초과 하락. 세트별 세부는 `unified_metrics.json`·"
              "`unified_oof.csv` 참조.")
    (ac.ANALYSIS / "unified_vs_specialized.md").write_text("\n".join(md) + "\n")

    for pl in ac.PLATFORMS:
        e = res["per_platform"][pl]
        print(pl, "unified grp-rho", round(e["ranking"]["group_spearman_mean"], 3),
              "top1", round(e["ranking"]["top1"], 3),
              "| P1", round(p1[pl]["ranking"]["group_spearman_mean"], 3),
              round(p1[pl]["ranking"]["top1"], 3))


if __name__ == "__main__":
    main()

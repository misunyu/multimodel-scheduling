"""P1: GroupKFold retraining with the fold leakage removed.

Fixes two defects of the shipped OOF protocol:
  1. folds were assigned per-row at random (RandomState(42).randint), splitting the
     rows of one (models, rate) group across folds;
  2. target normalization saw the whole dataset before the split.
Here folds are assigned per (models, rate_factor) GROUP (3-fold, seed=42), so a
group's 12 placements are ranked by a model that never saw ANY row of that group.
Normalization statistics are strictly within-group (see analysis_common.normalize_
targets docstring): with group-atomic folds no statistic crosses a fold boundary,
and within a held-out group the transform is monotone, hence rank-preserving —
not leakage for a ranking evaluation.

Outputs (analysis/): groupkfold_{platform}_metrics.json, groupkfold_{platform}_oof.csv,
groupkfold_ranking.md. Boosters (artifacts/): gkf_cpu_{platform}_{y1,y2,y3}.json
(full-data, for P3/P4/P6 reuse) + per-fold gkf_cpu_{platform}_{tag}_fold{k}.json
(needed by P3's cross-platform transfer) + features/coverage sidecars.
Never overwrites the shipped deploy_* artifacts.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as ac
from xgboost_model.deploy_selector_xgb_suite import write_coverage


def run_platform(platform: str, with_infps: bool = True, params_by_fold=None):
    """Full GroupKFold OOF for one platform. Returns (result dict, oof DataFrame).

    params_by_fold: optional {(tag, fold): params} to reuse (for the zero-infps
    variant, so the variant differs from the primary run only in features).
    """
    X, Yraw, M = ac.load_dataset(platform, with_infps=with_infps)
    Yn = ac.normalize_targets(Yraw, M)
    groups = ac.group_keys(M)
    fold = ac.assign_group_folds(groups).values
    Xv = X.values
    y3_valid = M["y3_valid"].values.astype(bool)

    oof = pd.DataFrame({
        "combination": M["combination"], "models": M["models"],
        "workload": M["workload"], "rate_factor": M["rate_factor"],
        "y3_valid": y3_valid, "has_gen": M["has_gen"], "fold": fold,
        "y1_meas": Yn["y1_total_throughput_fps"],
        "y2_meas": Yn["y2_deadline_miss_rate"],
        "y3_meas": Yn["y3_total_tokens_per_s"],
    })
    chosen = {}          # (tag, fold or "final") -> {"params":.., "inner_mae":..}
    for tag, col in ac.TARGETS:
        yv = Yn[col].values.astype(float)
        valid = y3_valid if tag == "y3" else np.ones(len(yv), bool)
        pred = np.full(len(yv), np.nan)
        for f in range(ac.N_FOLDS):
            tr = (fold != f) & valid
            te = fold == f
            if params_by_fold is not None:
                params = params_by_fold[(tag, f)]["params"]
                inner_mae = None
            else:
                params, inner_mae = ac.cv_select_params(
                    Xv[tr], yv[tr], groups[tr].reset_index(drop=True))
            bst = ac.train_booster(Xv[tr], yv[tr], params)
            # y3 boosters are trained on valid rows only but PREDICT every held-out
            # row: the production predictor scores all 12 placements of a generative
            # set, including LLM-on-CPU ones whose measured y3 is masked noise.
            pred[te] = ac.predict_booster(bst, Xv[te])
            chosen[(tag, f)] = {"params": {k: params[k] for k in ("max_depth", "eta")},
                                "inner_mae": inner_mae}
            if with_infps and params_by_fold is None:
                bst.save_model(str(ac.ARTIFACTS / f"gkf_cpu_{platform}_{tag}_fold{f}.json"))
        oof[f"{tag}_pred"] = pred

        if with_infps and params_by_fold is None:
            params, inner_mae = ac.cv_select_params(
                Xv[valid], yv[valid], groups[valid].reset_index(drop=True))
            bst = ac.train_booster(Xv[valid], yv[valid], params)
            bst.save_model(str(ac.ARTIFACTS / f"gkf_cpu_{platform}_{tag}.json"))
            chosen[(tag, "final")] = {"params": {k: params[k] for k in ("max_depth", "eta")},
                                      "inner_mae": inner_mae}

    # ---- metrics ----
    oof["group"] = groups.astype(str).values
    res = {"platform": platform, "with_infps": with_infps, "targets": {}}
    for tag, col in ac.TARGETS:
        valid = y3_valid if tag == "y3" else np.ones(len(oof), bool)
        m, p = oof[f"{tag}_meas"][valid], oof[f"{tag}_pred"][valid]
        gsp = ac.per_group_spearman(oof[valid], "group", f"{tag}_meas", f"{tag}_pred")
        finite = [v for v in gsp.values() if np.isfinite(v)]
        res["targets"][tag] = {
            "oof_spearman_pooled": ac.spearman(m, p),
            "oof_mae": float(np.mean(np.abs(m - p))),
            "group_spearman_mean": float(np.mean(finite)) if finite else None,
            "group_spearman": gsp,
            "n_rows_evaluated": int(valid.sum()),
            "hyperparams": {str(k[1]): v for k, v in chosen.items() if k[0] == tag},
        }

    # score-level ranking
    oof["s_meas"] = ac.scores_from(oof["y1_meas"], oof["y2_meas"], oof["y3_meas"],
                                   oof["has_gen"])
    oof["s_pred"] = ac.scores_from(oof["y1_pred"], oof["y2_pred"], oof["y3_pred"],
                                   oof["has_gen"])
    res["ranking"] = ac.ranking_metrics(oof)
    res["fold_row_counts"] = np.bincount(fold, minlength=ac.N_FOLDS).tolist()
    res["group_to_fold"] = {str(g): int(f) for g, f in
                            sorted(set(zip(groups.values, fold)), key=str)}
    return res, oof, chosen


def main():
    ac.ANALYSIS.mkdir(exist_ok=True)
    ranking_rows = []
    for platform in ac.PLATFORMS:
        print(f"=== {platform}: primary (real infps) ===")
        res, oof, chosen = run_platform(platform, with_infps=True)
        print(f"=== {platform}: variant (infps=0, legacy features) ===")
        res0, oof0, _ = run_platform(platform, with_infps=False,
                                     params_by_fold=chosen)
        res["variant_infps_zero"] = {
            "targets": {t: {"oof_spearman_pooled": res0["targets"][t]["oof_spearman_pooled"],
                            "group_spearman_mean": res0["targets"][t]["group_spearman_mean"],
                            "oof_mae": res0["targets"][t]["oof_mae"]}
                        for t in res0["targets"]},
            "ranking": res0["ranking"],
        }
        res["_provenance"] = ac.provenance(
            experiment="P1 groupkfold", platform=platform,
            boosters=f"artifacts/gkf_cpu_{platform}_*.json")
        out = ac.ANALYSIS / f"groupkfold_{platform}_metrics.json"
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        csv_out = ac.ANALYSIS / f"groupkfold_{platform}_oof.csv"
        with open(csv_out, "w") as fh:
            fh.write(ac.prov_csv_comment(experiment="P1 groupkfold oof",
                                         platform=platform))
            oof.drop(columns=["s_meas", "s_pred"]).to_csv(fh, index=False)
        (ac.ARTIFACTS / f"gkf_cpu_{platform}_features.json").write_text(
            json.dumps(ac.FEATURE_ORDER))
        _, _, Mfull = ac.load_dataset(platform)
        write_coverage(Mfull, ac.ARTIFACTS / f"gkf_cpu_{platform}")
        ranking_rows.append((platform, res))
        for tag in ("y1", "y2", "y3"):
            t = res["targets"][tag]
            print(f"  {tag}: pooled ρ={t['oof_spearman_pooled']:.3f} "
                  f"group-ρ={t['group_spearman_mean']:.3f} MAE={t['oof_mae']:.4f}")
        print(f"  ranking: {res['ranking']}")

    # ---- groupkfold_ranking.md ----
    md = [ac.prov_md_header(experiment="P1 groupkfold ranking",
                            boosters="artifacts/gkf_cpu_{gpu,npu}_*.json")]
    md.append("# P1 — GroupKFold OOF 결과 (fold 누수 제거)\n")
    md.append("fold: `(models, rate_factor)` 그룹 단위 3-fold (seed=42), "
              "정규화 통계는 그룹 내부로 한정.\n")
    md.append("## 점수 수준 (S = ŷ1 − 0.3·ŷ2 + 1.0·ŷ3[생성 세트만], 45그룹)\n")
    md.append("| 플랫폼 | Top-1 | Top-5 | 그룹 Spearman 평균 |")
    md.append("|---|---|---|---|")
    for platform, res in ranking_rows:
        r = res["ranking"]
        md.append(f"| CPU-{platform.upper()} | {r['top1']:.3f} | {r['top5']:.3f} "
                  f"| {r['group_spearman_mean']:+.3f} |")
    md.append("\n## target별 OOF (pooled / 그룹평균 Spearman, MAE)\n")
    md.append("| 플랫폼 | target | pooled ρ | 그룹 ρ 평균 | MAE | 평가 행수 |")
    md.append("|---|---|---|---|---|---|")
    for platform, res in ranking_rows:
        for tag in ("y1", "y2", "y3"):
            t = res["targets"][tag]
            md.append(f"| {platform} | {tag} | {t['oof_spearman_pooled']:+.3f} "
                      f"| {t['group_spearman_mean']:+.3f} | {t['oof_mae']:.4f} "
                      f"| {t['n_rows_evaluated']} |")
    md.append("\n## 참고: infps=0 변형 (기존 아티팩트의 feature 결함 재현)\n")
    md.append("기존 학습 경로는 창에 schedule 힌트가 없어 view.infps 계열 feature가 "
              "전부 0으로 들어갔다. 동일 fold·동일 하이퍼파라미터로 feature만 바꾼 비교:\n")
    md.append("| 플랫폼 | 변형 | Top-1 | Top-5 | 그룹 Spearman | y1 ρ | y2 ρ | y3 ρ (pooled) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for platform, res in ranking_rows:
        r = res["ranking"]
        t = res["targets"]
        md.append(f"| {platform} | 실제 infps | {r['top1']:.3f} | {r['top5']:.3f} "
                  f"| {r['group_spearman_mean']:+.3f} "
                  f"| {t['y1']['oof_spearman_pooled']:+.3f} "
                  f"| {t['y2']['oof_spearman_pooled']:+.3f} "
                  f"| {t['y3']['oof_spearman_pooled']:+.3f} |")
        v = res["variant_infps_zero"]
        md.append(f"| {platform} | infps=0 | {v['ranking']['top1']:.3f} "
                  f"| {v['ranking']['top5']:.3f} "
                  f"| {v['ranking']['group_spearman_mean']:+.3f} "
                  f"| {v['targets']['y1']['oof_spearman_pooled']:+.3f} "
                  f"| {v['targets']['y2']['oof_spearman_pooled']:+.3f} "
                  f"| {v['targets']['y3']['oof_spearman_pooled']:+.3f} |")
    md.append("\n채택 하이퍼파라미터·fold 구성은 `groupkfold_{platform}_metrics.json` 참조.")
    (ac.ANALYSIS / "groupkfold_ranking.md").write_text("\n".join(md) + "\n")
    print("wrote groupkfold_ranking.md")


if __name__ == "__main__":
    main()

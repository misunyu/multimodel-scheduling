"""A (#7): joint 예측기를 paired fold 로 재학습·재평가.

현행 joint(P4)는 그룹 키에 플랫폼을 포함해 90그룹으로 나눈다. 그 결과 같은
(set, rate)의 GPU 행과 NPU 행이 서로 다른 fold 에 떨어질 수 있고, 그러면 한 플랫폼의
동일 워크로드가 학습에 들어간 채로 다른 플랫폼의 같은 워크로드를 평가하게 된다.

여기서는 그룹 키에서 플랫폼을 빼서 (models, rate_factor) = 45 paired 그룹으로 나눈다.
한 (set, rate)의 GPU·NPU 행 전부가 항상 같은 fold 에 들어간다.

프로토콜은 그 외 전부 P4와 동일: 3-fold, seed=42, 그룹 인지 내부 CV, 동일 4조합 그리드,
y3 는 y3_valid 행으로만 학습하되 held-out 전 행을 예측, feature 37개 그대로.

기존 아티팩트/analysis 는 건드리지 않는다. 출력은 이 폴더에만.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "xgboost_model" / "full_collection_540" / "scripts"))

import analysis_common as ac  # noqa: E402


def build():
    Xs, Ys, Ms = [], [], []
    for pl in ac.PLATFORMS:
        X, Yraw, M = ac.load_dataset(pl)
        Yn = ac.normalize_targets(Yraw, M)
        Xs.append(X); Ys.append(Yn); Ms.append(M)
    X = pd.concat(Xs, ignore_index=True)
    Yn = pd.concat(Ys, ignore_index=True)
    M = pd.concat(Ms, ignore_index=True)
    assert len(X) == 1080, len(X)
    return X, Yn, M


def run(X, Yn, M, groups, tag):
    """P4 와 동일한 루프. groups 만 바꿔 넣는다."""
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
    for t, col in ac.TARGETS:
        yv = Yn[col].values.astype(float)
        valid = y3_valid if t == "y3" else np.ones(len(yv), bool)
        pred = np.full(len(yv), np.nan)
        for f in range(ac.N_FOLDS):
            tr = (fold != f) & valid
            te = fold == f
            params, inner_mae = ac.cv_select_params(
                Xv[tr], yv[tr], groups[tr].reset_index(drop=True))
            bst = ac.train_booster(Xv[tr], yv[tr], params)
            pred[te] = ac.predict_booster(bst, Xv[te])
            chosen[f"{t}/fold{f}"] = {
                "params": {k: params[k] for k in ("max_depth", "eta")},
                "inner_mae": inner_mae}
            print(f"    [{tag}] {t} fold{f}: {params['max_depth']}/{params['eta']} "
                  f"inner_mae={inner_mae:.4f}", flush=True)
        oof[f"{t}_pred"] = pred

    # 평가 그룹은 두 분할 모두 (models, rate) — 플랫폼별로 나눠서 본다 (P4와 동일)
    oof["group"] = [str((m, r)) for m, r in zip(M["models"], M["rate_factor"])]
    res = {"split": tag, "n_rows": int(len(X)),
           "n_cv_groups": int(groups.nunique()),
           "fold_row_counts": np.bincount(fold, minlength=ac.N_FOLDS).tolist(),
           "hyperparams": chosen, "per_platform": {}}
    for pl in ac.PLATFORMS:
        sub = oof[oof["platform"] == pl]
        entry = {"targets": {}}
        for t, _ in ac.TARGETS:
            v = sub["y3_valid"] if t == "y3" else pd.Series(True, index=sub.index)
            s = sub[v]
            gsp = [x for x in ac.per_group_spearman(
                s, "group", f"{t}_meas", f"{t}_pred").values() if np.isfinite(x)]
            entry["targets"][t] = {
                "pooled_spearman": ac.spearman(s[f"{t}_meas"], s[f"{t}_pred"]),
                "group_spearman_mean": float(np.mean(gsp)) if gsp else None,
                "mae": float(np.mean(np.abs(s[f"{t}_meas"] - s[f"{t}_pred"])))}
        sub = sub.copy()
        sub["s_meas"] = ac.scores_from(sub["y1_meas"], sub["y2_meas"], sub["y3_meas"],
                                       sub["has_gen"])
        sub["s_pred"] = ac.scores_from(sub["y1_pred"], sub["y2_pred"], sub["y3_pred"],
                                       sub["has_gen"])
        entry["ranking"] = ac.ranking_metrics(sub)
        # 부트스트랩·재현용 그룹별 결과
        per_group = []
        for g, gg in sub.groupby("group", sort=False):
            meas, pr = gg["s_meas"].values, gg["s_pred"].values
            best = meas.max()
            oracle = set(np.where(np.abs(meas - best) < 1e-9)[0])
            order = np.argsort(-pr, kind="stable")
            per_group.append({"group": g, "top1": int(order[0] in oracle),
                              "spearman": ac.spearman(meas, pr)})
        entry["per_group"] = per_group
        res["per_platform"][pl] = entry
    return res, oof, fold


def main():
    X, Yn, M = build()

    g_old = ac.group_keys(M, with_platform=True)     # 90 groups (현행)
    g_new = ac.group_keys(M)                         # 45 paired groups (신규)
    f_old = ac.assign_group_folds(g_old)
    f_new = ac.assign_group_folds(g_new)

    # ---- pair straddle: 구 분할에서 같은 (set,rate) 의 GPU/NPU 가 다른 fold 인 쌍 ----
    tmp = pd.DataFrame({"platform": M["platform"], "key": g_new.astype(str),
                        "fold_old": f_old.values, "fold_new": f_new.values})
    straddle, pair_rows = [], []
    for k, g in tmp.groupby("key", sort=False):
        folds_by_pl = {pl: sorted(set(g.loc[g["platform"] == pl, "fold_old"]))
                       for pl in ac.PLATFORMS}
        same = folds_by_pl["gpu"] == folds_by_pl["npu"]
        pair_rows.append({"group": k, "gpu_fold_old": folds_by_pl["gpu"],
                          "npu_fold_old": folds_by_pl["npu"],
                          "fold_new": sorted(set(g["fold_new"]))[0],
                          "straddles_old": not same})
        if not same:
            straddle.append(k)
    # 신규 분할에서는 정의상 0 이어야 한다
    new_straddle = [r["group"] for r in pair_rows
                    if len({f for f in [r["fold_new"]]}) != 1]
    assert not new_straddle

    print(f"pair straddle (old 90-group split): {len(straddle)}/45", flush=True)

    print("=== old split (90 groups, platform in key) ===", flush=True)
    res_old, oof_old, _ = run(X, Yn, M, g_old, "old90")
    print("=== new split (45 paired groups) ===", flush=True)
    res_new, oof_new, _ = run(X, Yn, M, g_new, "new45")

    stored = json.loads((ac.ANALYSIS / "unified_metrics.json").read_text())
    reproduces = {
        pl: (round(res_old["per_platform"][pl]["ranking"]["top1"], 6)
             == round(stored["per_platform"][pl]["ranking"]["top1"], 6)
             and round(res_old["per_platform"][pl]["ranking"]["group_spearman_mean"], 6)
             == round(stored["per_platform"][pl]["ranking"]["group_spearman_mean"], 6))
        for pl in ac.PLATFORMS}

    p1 = {pl: json.loads((ac.ANALYSIS / f"groupkfold_{pl}_metrics.json").read_text())
          for pl in ac.PLATFORMS}

    out = {
        "pair_straddle": {"n": len(straddle), "of": 45, "groups": sorted(straddle),
                          "detail": pair_rows},
        "old90": res_old, "new45": res_new,
        "stored_unified_metrics_reproduced": reproduces,
        "platform_specific_reference": {
            pl: {"top1": p1[pl]["ranking"]["top1"],
                 "group_spearman_mean": p1[pl]["ranking"]["group_spearman_mean"]}
            for pl in ac.PLATFORMS},
        "delta_new_minus_old": {
            pl: {"top1": (res_new["per_platform"][pl]["ranking"]["top1"]
                          - res_old["per_platform"][pl]["ranking"]["top1"]),
                 "group_spearman_mean": (
                     res_new["per_platform"][pl]["ranking"]["group_spearman_mean"]
                     - res_old["per_platform"][pl]["ranking"]["group_spearman_mean"])}
            for pl in ac.PLATFORMS},
        "_provenance": ac.provenance(
            experiment="A(#7) joint paired-fold retraining",
            note="no artifacts written outside this run directory"),
    }
    (HERE / "paired_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for name, o in (("old90", oof_old), ("new45", oof_new)):
        o.to_csv(HERE / f"joint_oof_{name}.csv", index=False)

    print("\n| 플랫폼 | 분할 | 그룹-ρ | Top-1 |")
    for pl in ac.PLATFORMS:
        for tag, r in (("old90", res_old), ("new45", res_new)):
            k = r["per_platform"][pl]["ranking"]
            print(f"| {pl} | {tag} | {k['group_spearman_mean']:+.4f} | {k['top1']:.4f} |")
    print("stored reproduced:", reproduces)


if __name__ == "__main__":
    main()

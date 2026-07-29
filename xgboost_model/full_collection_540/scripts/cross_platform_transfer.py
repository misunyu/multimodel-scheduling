"""P3: cross-platform transfer of the P1 GroupKFold boosters.

Four train->eval cells: (GPU->GPU), (GPU->NPU), (NPU->GPU), (NPU->NPU).
Diagonal cells reuse P1's OOF predictions verbatim (no retraining — the numbers
must match groupkfold_*_metrics.json). Off-diagonal cells apply the TRAIN
platform's three per-fold boosters to all 540 rows of the EVAL platform and use
the mean of the three fold predictions (their spread is recorded).

Eval-side normalization uses the eval platform's own group stats (ranking
evaluation; rank-preserving within a group).

Feature caveat, reported not corrected: an accelerator one-hot column that was
constantly 0 in training (exec_npu for a GPU-trained booster) becomes the live
column on the other platform. Two variants are evaluated:
  - asis: eval features unchanged (the naive transfer);
  - swap: views.{sum,mean,max}.view.exec_gpu <-> exec_npu swapped, so the
    trained accelerator column carries the eval platform's accelerator flag.
The static_*_sel / capacity / load_factor features always come from the eval
platform's device statics — that difference IS the transfer gap being measured.

Outputs: analysis/cross_platform.md, analysis/cross_platform_metrics.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as ac

EXEC_COLS = [(f"views.{agg}.view.exec_gpu", f"views.{agg}.view.exec_npu")
             for agg in ("sum", "mean", "max")]


def load_oof(platform: str) -> pd.DataFrame:
    df = pd.read_csv(ac.ANALYSIS / f"groupkfold_{platform}_oof.csv", comment="#")
    df["group"] = list(zip(df["models"], df["rate_factor"]))
    return df


def eval_frame(df: pd.DataFrame, pred_cols=("y1_pred", "y2_pred", "y3_pred")) -> dict:
    """Per-target Spearman + score-level ranking on a frame with meas/pred cols."""
    out = {"targets": {}}
    for tag, pcol in zip(("y1", "y2", "y3"), pred_cols):
        valid = df["y3_valid"].values.astype(bool) if tag == "y3" else np.ones(len(df), bool)
        sub = df[valid]
        gsp = [v for v in ac.per_group_spearman(sub, "group", f"{tag}_meas", pcol).values()
               if np.isfinite(v)]
        out["targets"][tag] = {
            "pooled_spearman": ac.spearman(sub[f"{tag}_meas"], sub[pcol]),
            "group_spearman_mean": float(np.mean(gsp)) if gsp else None,
            "mae": float(np.mean(np.abs(sub[f"{tag}_meas"] - sub[pcol]))),
        }
    s_meas = ac.scores_from(df["y1_meas"], df["y2_meas"], df["y3_meas"], df["has_gen"])
    s_pred = ac.scores_from(df[pred_cols[0]], df[pred_cols[1]], df[pred_cols[2]],
                            df["has_gen"])
    tmp = pd.DataFrame({"group": df["group"], "s_meas": s_meas, "s_pred": s_pred})
    out["ranking"] = ac.ranking_metrics(tmp)
    return out


def transfer_predict(train_pl: str, eval_X: pd.DataFrame) -> tuple:
    """Mean and spread of the 3 fold-booster predictions per target."""
    preds, spreads = {}, {}
    for tag, _ in ac.TARGETS:
        ps = []
        for f in range(ac.N_FOLDS):
            bst = ac.load_booster(ac.ARTIFACTS / f"gkf_cpu_{train_pl}_{tag}_fold{f}.json")
            ps.append(ac.predict_booster(bst, eval_X.values))
        ps = np.vstack(ps)
        preds[tag] = ps.mean(axis=0)
        spreads[tag] = float(ps.std(axis=0).mean())
    return preds, spreads


def main():
    data = {}
    for pl in ac.PLATFORMS:
        X, Yraw, M = ac.load_dataset(pl)
        Yn = ac.normalize_targets(Yraw, M)
        base = pd.DataFrame({
            "group": list(zip(M["models"], M["rate_factor"])),
            "y3_valid": M["y3_valid"].astype(bool), "has_gen": M["has_gen"],
            "y1_meas": Yn["y1_total_throughput_fps"],
            "y2_meas": Yn["y2_deadline_miss_rate"],
            "y3_meas": Yn["y3_total_tokens_per_s"],
        })
        data[pl] = (X, base)

    results = {}
    for train_pl in ac.PLATFORMS:
        for eval_pl in ac.PLATFORMS:
            cell = f"{train_pl}->{eval_pl}"
            if train_pl == eval_pl:
                res = eval_frame(load_oof(eval_pl))
                res["source"] = "P1 OOF (verbatim)"
                results[(cell, "asis")] = res
                results[(cell, "swap")] = res       # diagonal has no swap question
                continue
            X, base = data[eval_pl]
            for variant in ("asis", "swap"):
                Xv = X.copy()
                if variant == "swap":
                    for a, b in EXEC_COLS:
                        Xv[a], Xv[b] = X[b].values.copy(), X[a].values.copy()
                preds, spreads = transfer_predict(train_pl, Xv)
                df = base.copy()
                for tag in ("y1", "y2", "y3"):
                    df[f"{tag}_pred"] = preds[tag]
                res = eval_frame(df)
                res["fold_pred_spread_mean"] = spreads
                res["source"] = f"gkf_cpu_{train_pl}_*_fold{{0,1,2}} mean"
                results[(cell, variant)] = res

    # ---- outputs ----
    out_json = {f"{cell}|{var}": res for (cell, var), res in results.items()}
    out_json["_provenance"] = ac.provenance(
        experiment="P3 cross-platform transfer",
        boosters="artifacts/gkf_cpu_{gpu,npu}_{y1,y2,y3}_fold{0,1,2}.json (P1)")
    (ac.ANALYSIS / "cross_platform_metrics.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False))

    md = [ac.prov_md_header(experiment="P3 cross-platform transfer",
                            boosters="artifacts/gkf_* fold boosters (P1)")]
    md.append("# P3 — 교차 플랫폼 전이 (P1 fold 부스터, 재학습 없음)\n")
    md.append("대각선 = P1 OOF 그대로. 비대각선 = 학습 플랫폼 fold 부스터 3개 예측의 평균. "
              "평가측 정규화는 평가 플랫폼 그룹 통계(순위 보존).\n")
    md.append("가속기 one-hot 주의: 전이 시 평가 플랫폼의 가속기 열은 학습에서 상수 0이던 "
              "열이다. `asis`는 무보정, `swap`은 exec_gpu↔exec_npu 스왑 — 두 변형의 차이 "
              "자체가 결과다. static/capacity feature는 항상 평가 플랫폼 값이다.\n")
    for variant in ("asis", "swap"):
        md.append(f"## 변형: {variant}\n")
        md.append("| 학습→평가 | y1 ρ | y2 ρ | y3 ρ | 점수 그룹-ρ | Top-1 | Top-5 |")
        md.append("|---|---|---|---|---|---|---|")
        for train_pl in ac.PLATFORMS:
            for eval_pl in ac.PLATFORMS:
                res = results[(f"{train_pl}->{eval_pl}", variant)]
                t, r = res["targets"], res["ranking"]
                md.append(f"| {train_pl}→{eval_pl} | {t['y1']['pooled_spearman']:+.3f} "
                          f"| {t['y2']['pooled_spearman']:+.3f} "
                          f"| {t['y3']['pooled_spearman']:+.3f} "
                          f"| {r['group_spearman_mean']:+.3f} "
                          f"| {r['top1']:.3f} | {r['top5']:.3f} |")
        md.append("")
    diag = np.mean([results[(f"{p}->{p}", "asis")]["ranking"]["group_spearman_mean"]
                    for p in ac.PLATFORMS])
    for variant in ("asis", "swap"):
        off = np.mean([results[(f"{a}->{b}", variant)]["ranking"]["group_spearman_mean"]
                       for a in ac.PLATFORMS for b in ac.PLATFORMS if a != b])
        md.append(f"- 점수 그룹-ρ 평균: 대각선 {diag:+.3f} → 비대각선({variant}) {off:+.3f} "
                  f"(하락 {diag - off:.3f})")
    sp = {v: results[(f"gpu->npu", v)]["fold_pred_spread_mean"] for v in ("asis", "swap")}
    md.append(f"- fold 예측 표준편차(행 평균, gpu→npu 예): asis {sp['asis']} / swap {sp['swap']}")
    md.append("\n기계가독 전체 수치: `cross_platform_metrics.json`")
    (ac.ANALYSIS / "cross_platform.md").write_text("\n".join(md) + "\n")

    for (cell, var), res in results.items():
        if var == "asis" or cell.split("->")[0] != cell.split("->")[1]:
            print(cell, var, "grp-rho", round(res["ranking"]["group_spearman_mean"], 3),
                  "top1", round(res["ranking"]["top1"], 3))


if __name__ == "__main__":
    main()

"""F (#20): oracle ratio / regret 지표 — direct · greedy · decomposed.

각 방법이 그룹에서 고른 배치의 **측정** 점수를 그 그룹의 최고 측정 점수로 나눈 값
(oracle ratio)과 1 - 그 값(regret)을 45그룹에서 계산한다. Top-1/Top-5 의 보조 지표로
"틀렸을 때 얼마나 틀렸나"를 보여주는 표를 만든다.

점수 기저는 파이프라인과 동일한 그룹 정규화 (alpha=0.3, beta=1.0, analysis_common).
greedy 는 측정 산출이 있는 그룹만 값이 정의되므로 D 의 하한(미측정=ratio 0)을 병기한다.

기존 OOF 예측·측정값만 사용. 재학습 없음.
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

AN = ac.ANALYSIS
TIE = 1e-9


def pick_stats(df, group_col, meas_col, pred_col):
    """그룹별 (선택 배치의 측정 점수 / 그룹 최고 측정 점수) 및 Top-1."""
    out = []
    for g, gg in df.groupby(group_col, sort=False):
        meas = gg[meas_col].values.astype(float)
        pred = gg[pred_col].values.astype(float)
        best = float(meas.max())
        i = int(np.argsort(-pred, kind="stable")[0])
        oracle = set(np.where(np.abs(meas - best) < TIE)[0])
        out.append({"group": str(g), "picked_meas": float(meas[i]),
                    "oracle_meas": best,
                    "ratio": float(meas[i] / best) if abs(best) > 1e-12 else np.nan,
                    "top1": int(i in oracle)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ direct (P1)
direct = {}
for pl in ac.PLATFORMS:
    o = pd.read_csv(AN / f"groupkfold_{pl}_oof.csv", comment="#")
    o["s_meas"] = ac.scores_from(o["y1_meas"], o["y2_meas"], o["y3_meas"], o["has_gen"])
    o["s_pred"] = ac.scores_from(o["y1_pred"], o["y2_pred"], o["y3_pred"], o["has_gen"])
    o["group"] = [str((m, r)) for m, r in zip(o["models"], o["rate_factor"])]
    direct[pl] = pick_stats(o, "group", "s_meas", "s_pred")

# -------------------------------------------------------------- decomposed (P6)
# train_decomposed.py:150-186 의 합성을 저장된 뷰 OOF 로 재현한다 (재학습 없음).
vo = pd.read_csv(AN / "decomposed_view_oof.csv", comment="#")
decomp = {}
for pl in ac.PLATFORMS:
    _, Yraw, M = ac.load_dataset(pl)
    Yn = ac.normalize_targets(Yraw, M)
    w = pd.DataFrame({
        "combination": M["combination"].astype(str), "has_gen": M["has_gen"],
        "group": [str((m, r)) for m, r in zip(M["models"], M["rate_factor"])],
        "y1_raw": Yraw["y1_total_throughput_fps"],
        "y2_raw": np.clip(Yraw["y2_deadline_miss_rate"], 0, 1),
        "y1_meas": Yn["y1_total_throughput_fps"],
        "y2_meas": Yn["y2_deadline_miss_rate"],
        "y3_meas": Yn["y3_total_tokens_per_s"]})
    p1 = pd.read_csv(AN / f"groupkfold_{pl}_oof.csv", comment="#")
    p1["combination"] = p1["combination"].astype(str)
    w = w.merge(p1[["combination", "y3_pred"]], on="combination", how="left")
    gstat = w.groupby("group").agg(fmax=("y1_raw", "max"), y2_lo=("y2_raw", "min"),
                                   y2_hi=("y2_raw", "max"))
    V = vo[vo["platform"] == pl].copy()
    V["combination"] = V["combination"].astype(str)
    agg = V.groupby("combination").apply(
        lambda d: pd.Series({"y1_sum": float((d["pred"] * d["eff_fps"]).sum()),
                             "miss": float(np.clip((1.0 - d["pred"]).mean(), 0.0, 1.0))}),
        include_groups=False)
    comp = w.set_index("combination").join(agg).reset_index().join(gstat, on="group")
    span = (comp["y2_hi"] - comp["y2_lo"]).values
    comp["y1_dec"] = comp["y1_sum"] / comp["fmax"]
    comp["y2_dec"] = np.where(span > 1e-9,
                              (comp["miss"] - comp["y2_lo"]) / np.where(span > 1e-9, span, 1.0),
                              0.0)
    comp["s_meas"] = ac.scores_from(comp["y1_meas"], comp["y2_meas"], comp["y3_meas"],
                                    comp["has_gen"])
    comp["s_pred"] = ac.scores_from(comp["y1_dec"], comp["y2_dec"], comp["y3_pred"],
                                    comp["has_gen"])
    decomp[pl] = pick_stats(comp, "group", "s_meas", "s_pred")

# ------------------------------------------------------------------ greedy (P5)
gr = pd.read_csv(AN / "greedy_results.csv", comment="#")
greedy = {}
for (pl, var), g in gr.groupby(["platform", "variant"]):
    c = g[g["collected"] == 1]
    greedy[f"{pl}/{var}"] = {
        "n_measured": int(len(c)), "n_groups": int(len(g)),
        "ratios_measured": c["ratio_norm"].astype(float).tolist(),
        "top1_hits": int(c["top1"].sum()),
    }


def summ(vals, nd=4):
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": round(float(a.mean()), nd),
            "min": round(float(a.min()), nd),
            "p10": round(float(np.percentile(a, 10)), nd),
            "median": round(float(np.median(a)), nd),
            "p90": round(float(np.percentile(a, 90)), nd),
            "max": round(float(a.max()), nd),
            "n_exact": int(np.sum(np.abs(a - 1.0) < TIE)),
            "n_below_0.9": int(np.sum(a < 0.9)),
            "n_below_0.5": int(np.sum(a < 0.5))}


table = []
for name, per_pl in (("direct (P1)", direct), ("decomposed (P6)", decomp)):
    for pl in ac.PLATFORMS:
        df = per_pl[pl]
        r = df["ratio"].values
        table.append({"method": name, "platform": pl, "denominator": "45 groups",
                      "top1": round(float(df["top1"].mean()), 4),
                      "oracle_ratio": summ(r),
                      "regret": summ(1.0 - r)})
for key, gd in sorted(greedy.items()):
    pl, var = key.split("/")
    rm = np.asarray(gd["ratios_measured"], float)
    # 하한: 미측정 그룹을 ratio 0 (= regret 1) 로 채운다
    lb = np.concatenate([rm, np.zeros(gd["n_groups"] - gd["n_measured"])])
    table.append({"method": f"greedy ({var})", "platform": pl,
                  "denominator": f"{gd['n_measured']} measured groups",
                  "top1": round(gd["top1_hits"] / gd["n_measured"], 4),
                  "oracle_ratio": summ(rm), "regret": summ(1.0 - rm)})
    table.append({"method": f"greedy ({var})", "platform": pl,
                  "denominator": "45 groups (unmeasured = ratio 0)",
                  "top1": round(gd["top1_hits"] / gd["n_groups"], 4),
                  "oracle_ratio": summ(lb), "regret": summ(1.0 - lb)})

out = {
    "definition": {
        "oracle_ratio": "그룹에서 그 방법이 고른 배치의 **측정** 점수 / 그룹 최고 측정 점수",
        "regret": "1 - oracle_ratio",
        "score_basis": "그룹 정규화 (analysis_common.scores_from, alpha=0.3, beta=1.0)",
        "note": "정규화 점수는 음수가 될 수 있어 ratio 가 0 미만이 될 수 있다 "
                "(regret > 1). 하한 행의 0 채움은 '완전 실패'의 관례적 표기다.",
        "sources": {"direct": "analysis/groupkfold_{pl}_oof.csv",
                    "decomposed": "analysis/decomposed_view_oof.csv 재합성 "
                                  "(train_decomposed.py:150-186 동일 식) + P1 y3_pred",
                    "greedy": "analysis/greedy_results.csv (ratio_norm)"},
    },
    "table": table,
}
(HERE / "regret_facts.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

for name, per_pl in (("direct", direct), ("decomposed", decomp)):
    for pl in ac.PLATFORMS:
        per_pl[pl].to_csv(HERE / f"per_group_{name}_{pl}.csv", index=False)

md = ["| 방법 | 플랫폼 | 분모 | Top-1 | Oracle ratio 평균 | 중앙값 | p10 | 최소 | "
      "regret 평균 | ratio<0.9 |", "|" + "---|" * 10]
for t in table:
    o, g = t["oracle_ratio"], t["regret"]
    md.append(f"| {t['method']} | {t['platform'].upper()} | {t['denominator']} "
              f"| {t['top1']:.3f} | **{o['mean']:.3f}** | {o['median']:.3f} "
              f"| {o['p10']:.3f} | {o['min']:.3f} | {g['mean']:.3f} | {o['n_below_0.9']} |")
(HERE / "regret_table.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

# 검산: direct 의 Top-1 이 저장된 P1 값과 같은가 / decomposed 가 P6 값과 같은가
p1m = {pl: json.loads((AN / f"groupkfold_{pl}_metrics.json").read_text())
       for pl in ac.PLATFORMS}
p6m = json.loads((AN / "decomposed_metrics.json").read_text())
checks = {
    **{f"direct_top1_matches_P1_{pl}":
       bool(abs(direct[pl]["top1"].mean() - p1m[pl]["ranking"]["top1"]) < 1e-9)
       for pl in ac.PLATFORMS},
    **{f"decomposed_top1_matches_P6_{pl}":
       bool(abs(decomp[pl]["top1"].mean() - p6m[pl]["ranking"]["top1"]) < 1e-9)
       for pl in ac.PLATFORMS},
    **{f"n_groups_45_{k}_{pl}": bool(len(v[pl]) == 45)
       for k, v in (("direct", direct), ("decomposed", decomp)) for pl in ac.PLATFORMS},
}
out["checks"] = checks
(HERE / "regret_facts.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print()
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
if not all(checks.values()):
    sys.exit(1)

#!/usr/bin/env python3
"""Compute agreement statistics between prediction_dir and runtime_dir.

Metrics:
- Exact match rate of best schedule (predicted vs measured-oracle).
- Top-5 score-group inclusion rate (grouped by equal scores in measured results).
- Score-gap summaries for predicted choice vs measured oracle.

Usage:
  python compute_prediction_vs_runtime_stats.py --pred xgboost_model/performance_results/prediction_test --runtime xgboost_model/performance_results/runtime_test
"""
import argparse, json, re, math, os
import pandas as pd
import numpy as np

KEY_RE = re.compile(r"model_schedules_(.*)_test\.json$")

def get_any(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

def load_dir(dir_path):
    rows=[]
    for name in os.listdir(dir_path):
        if not name.lower().endswith(".json") or "__macosx" in name.lower():
            continue
        m=KEY_RE.search(name)
        if not m:
            continue
        key=m.group(1)
        full_path = os.path.join(dir_path, name)
        with open(full_path, 'r') as f:
            data=json.load(f)
            rows.append({
                "key": key,
                "file": name,
                "schedule_file": get_any(data, ["schedule_file","schedule file","schedule"]),
                "best_deployment": get_any(data, ["best_deployment","best deployment","best"]),
                "best_score": float(get_any(data, ["best_score","score"])) if get_any(data, ["best_score","score"]) is not None else np.nan,
                "data": data.get("data", [])
            })
    return pd.DataFrame(rows)

def oracle_from_data(data_list):
    if not data_list:
        return None, np.nan
    best=max(data_list, key=lambda x: float(x.get("score", -1e9)))
    return best.get("combination"), float(best.get("score"))

def runtime_score_for_combo(runtime_list, combo):
    for x in runtime_list:
        if x.get("combination")==combo:
            return float(x.get("score"))
    return np.nan

def group_rank(runtime_list, score, round_decimals=6):
    scores = [round(float(x.get("score")), round_decimals) for x in runtime_list]
    uniq = sorted(set(scores), reverse=True)
    s = round(score, round_decimals) if not math.isnan(score) else None
    if s is None or s not in uniq:
        return np.nan, len(uniq)
    return uniq.index(s)+1, len(uniq)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred", default="xgboost_model/performance_results/prediction_test", help="prediction directory (XGBoost-based)")
    ap.add_argument("--runtime", default="xgboost_model/performance_results/runtime_test", help="runtime directory (measurement-based)")
    ap.add_argument("--round_decimals", type=int, default=6, help="rounding for score-group ties")
    args=ap.parse_args()

    pred_df=load_dir(args.pred).rename(columns={"best_deployment":"pred_deployment","best_score":"pred_best_score"})
    run_df=load_dir(args.runtime)
    run_df[["oracle_deployment","oracle_score"]] = run_df["data"].apply(lambda lst: pd.Series(oracle_from_data(lst)))

    merged = pd.merge(
        pred_df[["key","schedule_file","pred_deployment","pred_best_score"]],
        run_df[["key","oracle_deployment","oracle_score","data"]].rename(columns={"data":"runtime_data"}),
        on="key", how="inner"
    )

    merged["pred_runtime_score"] = merged.apply(lambda r: runtime_score_for_combo(r["runtime_data"], r["pred_deployment"]), axis=1)
    merged["score_gap_to_oracle"] = merged["oracle_score"] - merged["pred_runtime_score"]
    ranks = merged.apply(lambda r: group_rank(r["runtime_data"], r["pred_runtime_score"], round_decimals=args.round_decimals), axis=1)
    merged["pred_group_rank"] = [x[0] for x in ranks]
    merged["num_groups"] = [x[1] for x in ranks]

    exact_match = (merged["pred_deployment"]==merged["oracle_deployment"]).mean()
    top5_group = (merged["pred_group_rank"]<=5).mean()

    merged_valid = merged.dropna(subset=["pred_runtime_score","oracle_score"])
    mean_gap = merged_valid["score_gap_to_oracle"].mean()
    median_gap = merged_valid["score_gap_to_oracle"].median()
    p90_gap = merged_valid["score_gap_to_oracle"].quantile(0.9)
    within_0_02 = (merged_valid["score_gap_to_oracle"]<=0.02).mean()
    within_1pct = (merged_valid["score_gap_to_oracle"] <= 0.01*merged_valid["oracle_score"]).mean()
    within_5pct = (merged_valid["score_gap_to_oracle"] <= 0.05*merged_valid["oracle_score"]).mean()

    summary = pd.DataFrame({
        "Metric":[
            "Exact best-schedule match rate",
            "Top-5 score-group inclusion rate (ties grouped)",
            "Mean score gap to oracle (oracle - predicted-choice score)",
            "Median score gap to oracle",
            "90th-percentile score gap to oracle",
            "Share with gap ≤ 0.02 (absolute)",
            "Share with gap ≤ 1% of oracle score",
            "Share with gap ≤ 5% of oracle score",
            "Num compared scenarios (intersection of directories)"
        ],
        "Value":[
            f"{exact_match*100:.1f}%",
            f"{top5_group*100:.1f}%",
            f"{mean_gap:.4f}",
            f"{median_gap:.4f}",
            f"{p90_gap:.4f}",
            f"{within_0_02*100:.1f}%",
            f"{within_1pct*100:.1f}%",
            f"{within_5pct*100:.1f}%",
            str(len(merged_valid))
        ]
    })

    print(summary.to_string(index=False))
    out_csv="prediction_vs_runtime_detailed.csv"
    merged.drop(columns=["runtime_data"]).to_csv(out_csv, index=False)
    print(f"\nSaved per-scenario details to: {out_csv}")

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import zipfile, json, os, re

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

pred_dir = os.path.join(project_root, "xgboost_model/performance_results/prediction_test")
run_dir = os.path.join(project_root, "xgboost_model/performance_results/runtime_test")


def compute_best_score(obj, best):
    for row in obj.get("data", []):
        if row.get("combination")==best:
            return float(row.get("score"))
    return None

def extract(dir_path, kind):
    rows=[]
    for filename in os.listdir(dir_path):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'r') as f:
            obj=json.load(f)
            best=obj.get("best deployment") or obj.get("best_schedule_name")
            score=obj.get("score") or compute_best_score(obj,best)
            mm=re.search(r"model_schedules_(.+?)_test\.json$", filename)
            scenario=mm.group(1) if mm else os.path.splitext(filename)[0]
            rows.append((scenario,kind,best,float(score)))
    return pd.DataFrame(rows,columns=["scenario","kind","schedule","score"])

df=pd.concat([extract(run_dir,"Runtime"), extract(pred_dir,"Predicted")])
global_max_score = df["score"].max()

def save_plot(df_sub, output_name, title):
    scenarios = sorted(df_sub["scenario"].unique(), key=lambda s: (s.count("_"), s))
    if not scenarios:
        print(f"No scenarios for {output_name}, skipping.")
        return
    xpos = {s: i for i, s in enumerate(scenarios)}

    plt.figure(figsize=(max(8, len(scenarios) * 0.8), 5))

    # jittered scatter
    for kind, color, dx in [("Runtime", "blue", -0.1), ("Predicted", "red", 0.1)]:
        sub = df_sub[df_sub.kind == kind]
        xs = [xpos[s] + dx for s in sub.scenario if s in xpos]
        plt.scatter(xs, sub.score, color=color, label=kind, s=60)

    plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha="right")
    plt.ylabel("Best Score")
    plt.title(title)
    plt.legend()
    plt.ylim(0, global_max_score + 0.2)
    plt.tight_layout()

    plt.savefig(output_name)
    print(f"Saved plot to {output_name}")

# split scenarios: single model (no '_') vs multi model (has '_')
df_single = df[~df['scenario'].str.contains('_')]
df_multi = df[df['scenario'].str.contains('_')]

save_plot(df_single, "best_schedule_pointplot_single.pdf", "Best Schedule Scores (Single Model)")
save_plot(df_multi, "best_schedule_pointplot_multi.pdf", "Best Schedule Scores (Multi Model)")


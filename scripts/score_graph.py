import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import zipfile, json, os, re

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

pred_dir = os.path.join(project_root, "xgboost_model/test/performance_results/prediction")
run_dir = os.path.join(project_root, "xgboost_model/test/performance_results/runtime")


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
            
            # Extract scenario from "schedule file" field if present, otherwise from filename
            schedule_file = obj.get("schedule file") or obj.get("schedule_name")
            if schedule_file:
                scenario = os.path.splitext(schedule_file)[0]
            else:
                scenario = os.path.splitext(filename)[0]
            
            # Standardize scenario name: remove known prefixes and suffixes
            scenario = re.sub(r"^(predict|recompute)_performance_\d{8}_\d{6}_", "", scenario)
            scenario = re.sub(r"^model_schedules_", "", scenario)
            scenario = re.sub(r"_test$", "", scenario)
            
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
    plt.legend(loc='lower right')
    plt.ylim(0, global_max_score + 0.2)
    plt.subplots_adjust(bottom=0.3)  # plt.tight_layout() 대신 사용하거나 추가
    plt.savefig(output_name)
    print(f"Saved plot to {output_name}")

# split scenarios: single model (no '_') vs multi model (has '_')
df_single = df[~df['scenario'].str.contains('_')]
df_multi = df[df['scenario'].str.contains('_')]

save_plot(df_single, "best_schedule_pointplot_single.pdf", "Best Schedule Scores (Single Model)")
save_plot(df_multi, "best_schedule_pointplot_multi.pdf", "Best Schedule Scores (Multi Model)")

# Save to Excel for ChatGPT with descriptions
excel_path = "best_schedule_summary.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Single Model Sheet
    df_single_pivot = df_single.pivot(index='scenario', columns='kind', values=['schedule', 'score'])
    # Flatten multi-index columns for easier reading
    df_single_pivot.columns = [f"{col[1]} {col[0]}" for col in df_single_pivot.columns]
    df_single_pivot = df_single_pivot.reset_index()
    
    # Description for Single Model
    desc_single = pd.DataFrame([
        ["Description:", "This table shows the best deployment schedule and its corresponding score for scenarios involving a single model."],
        ["Graph Reference:", "Refer to 'best_schedule_pointplot_single.pdf' for a visual representation."],
        ["Note:", "Runtime represents the actual measured score, while Predicted represents the score estimated by the XGBoost model."],
        []
    ])
    desc_single.to_excel(writer, sheet_name='Single Model', index=False, header=False)
    df_single_pivot.to_excel(writer, sheet_name='Single Model', startrow=len(desc_single), index=False)

    # Multi Model Sheet
    df_multi_pivot = df_multi.pivot(index='scenario', columns='kind', values=['schedule', 'score'])
    df_multi_pivot.columns = [f"{col[1]} {col[0]}" for col in df_multi_pivot.columns]
    df_multi_pivot = df_multi_pivot.reset_index()

    # Description for Multi Model
    desc_multi = pd.DataFrame([
        ["Description:", "This table shows the best deployment schedule and its corresponding score for scenarios involving multiple models (multi-model scheduling)."],
        ["Graph Reference:", "Refer to 'best_schedule_pointplot_multi.pdf' for a visual representation."],
        ["Note:", "Scenario names with '_' indicate combinations of multiple models."],
        []
    ])
    desc_multi.to_excel(writer, sheet_name='Multi Model', index=False, header=False)
    df_multi_pivot.to_excel(writer, sheet_name='Multi Model', startrow=len(desc_multi), index=False)

print(f"Saved Excel summary to {excel_path}")


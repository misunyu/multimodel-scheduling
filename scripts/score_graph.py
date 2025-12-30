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

# sort scenarios
scenarios=sorted(df["scenario"].unique(), key=lambda s:(s.count("_"),s))
xpos={s:i for i,s in enumerate(scenarios)}

plt.figure(figsize=(14,5))

# jittered scatter
for kind,color,dx in [("Runtime","blue",-0.1),("Predicted","red",0.1)]:
    sub=df[df.kind==kind]
    xs=[xpos[s]+dx for s in sub.scenario]
    plt.scatter(xs, sub.score, color=color, label=kind, s=60)

plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha="right")
plt.ylabel("Best Score")
plt.title("Best Schedule Scores per Scenario (Point Plot)")
plt.legend()
plt.ylim(bottom=0)
plt.tight_layout()

out="best_schedule_pointplot.pdf"
plt.savefig(out)
print(f"Saved plot to {out}")

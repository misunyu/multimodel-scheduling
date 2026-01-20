import pandas as pd
from pathlib import Path

def load_results(csv_path):
    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found.")
        return None
    # Skip the first row which contains 'Average'
    df = pd.read_csv(csv_path, skiprows=1)
    return df[['schedule_file', 'best_combination']]

def compare():
    base_file = "best_deployment_results.csv"
    compare_files = {
        "XGBoost": "xgb_best_results.csv",
        "Random Search": "random_search_best_results.csv",
        "Latency-based": "latency_based_best_results.csv"
    }

    base_df = load_results(base_file)
    if base_df is None:
        return

    base_df = base_df.rename(columns={'best_combination': 'ground_truth'})

    for name, file_path in compare_files.items():
        comp_df = load_results(file_path)
        if comp_df is None:
            continue
        
        # Merge on schedule_file
        merged = pd.merge(base_df, comp_df, on='schedule_file', how='inner')
        
        if merged.empty:
            print(f"No matching schedules found between {base_file} and {file_path}")
            continue

        # Count matches
        matches = (merged['ground_truth'] == merged['best_combination']).sum()
        total = len(merged)
        ratio = (matches / total) * 100 if total > 0 else 0
        
        print(f"[{name}] Match Rate: {ratio:.2f}% ({matches}/{total})")

if __name__ == "__main__":
    compare()

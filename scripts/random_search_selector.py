import json
import csv
import yaml
import random
from pathlib import Path
import pandas as pd
import numpy as np

def get_performance_index(results_dir):
    perf_index = {} # schedule_file_name -> { combination_name -> performance_data }
    results_path = Path(results_dir)
    
    for p_file in results_path.glob("*.json"):
        try:
            with open(p_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            sched_file = content.get("schedule file") or content.get("schedule_file")
            if not sched_file:
                continue
                
            # 스케줄 파일 이름만 키로 사용 (예: model_schedules_g.yaml)
            sched_name = Path(sched_file).name
            
            if sched_name not in perf_index:
                perf_index[sched_name] = {}
                
            data_list = content.get("data", [])
            for item in data_list:
                comb_name = item.get("combination")
                if comb_name:
                    perf_index[sched_name][comb_name] = item
                    
        except Exception as e:
            print(f"Error indexing {p_file}: {e}")
            
    return perf_index

def random_search_best_combination(sched_name, perf_index, num_samples=10):
    # 해당 스케줄에 대해 측정된 모든 combination 목록 가져오기
    pure_sched_name = Path(sched_name).name
    sched_perf = perf_index.get(pure_sched_name, {})
    
    if not sched_perf:
        return None
    
    available_combs = list(sched_perf.keys())
    
    # 가능한 배치 방법 중에서 무작위로 10가지(또는 그 이하) 선택
    sampled_combs = random.sample(available_combs, min(len(available_combs), num_samples))
    
    best_comb_name = None
    max_score = -1.0
    
    for comb_name in sampled_combs:
        perf_item = sched_perf[comb_name]
        score = perf_item.get("score", 0.0)
        if score > max_score:
            max_score = score
            best_comb_name = comb_name
            
    return best_comb_name

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Random Search Selector for best combinations.")
    parser.add_argument("-k", type=int, default=5, help="Number of samples for random search (default: 5)")
    parser.add_argument("--test_csv", default="xgboost_model/dataset/gpu/test_schedules_x3.csv", help="Test schedules CSV")
    args = parser.parse_args()
    
    num_samples = args.k
    test_schedules_csv = args.test_csv
    output_csv = "random_search_best_results.csv"
    results_recompute_dir = "results_recompute"
    
    print(f"Random Search with k={num_samples}")
    print("Indexing performance data from results_recompute...")
    perf_index = get_performance_index(results_recompute_dir)
    print(f"Indexed performance data for {len(perf_index)} schedules.")
    
    print("Loading test schedules...")
    schedules_df = pd.read_csv(test_schedules_csv)
    
    results = []
    
    for _, row in schedules_df.iterrows():
        sched_name = row['schedule_name']
        print(f"Processing schedule: {sched_name}")
        
        best_comb_name = random_search_best_combination(sched_name, perf_index, num_samples=num_samples)
        
        if not best_comb_name:
            print(f"  [LOG] No performance data found for {sched_name}")
            results.append({
                'schedule_file': sched_name,
                'best_combination': '-',
                'normalized_throughput': '-',
                'drop_rate': '-',
                'score': '-',
                'models_count': 0
            })
            continue
            
        # 성능 데이터 추출
        pure_sched_name = Path(sched_name).name
        perf_item = perf_index[pure_sched_name][best_comb_name]
        derived = perf_item.get('derived', {})
        models_count = len(perf_item.get('models', {}))
        
        results.append({
            'schedule_file': sched_name,
            'best_combination': best_comb_name,
            'normalized_throughput': derived.get('throughput_norm'),
            'drop_rate': derived.get('drop_rate_norm'),
            'score': perf_item.get('score'),
            'models_count': models_count
        })
        
    if results:
        output_df = pd.DataFrame(results)
        
        # 평균값 계산 (숫자 데이터만)
        numeric_throughput = pd.to_numeric(output_df['normalized_throughput'], errors='coerce')
        numeric_drop_rate = pd.to_numeric(output_df['drop_rate'], errors='coerce')
        numeric_score = pd.to_numeric(output_df['score'], errors='coerce')
        
        avg_throughput = round(numeric_throughput.mean(), 2) if not numeric_throughput.isna().all() else 0.0
        avg_drop_rate = round(numeric_drop_rate.mean(), 2) if not numeric_drop_rate.isna().all() else 0.0
        avg_score = round(numeric_score.mean(), 2) if not numeric_score.isna().all() else 0.0
        
        # 모델 개수별 누적 평균값 계산 (>= 3, 4, 5, 6, 7, 8)
        avg_rows = []
        for n in range(8, 2, -1):
            mask = output_df['models_count'] >= n
            subset = output_df[mask]
            if not subset.empty:
                avg_t = round(pd.to_numeric(subset['normalized_throughput'], errors='coerce').mean(), 2)
                avg_d = round(pd.to_numeric(subset['drop_rate'], errors='coerce').mean(), 2)
                avg_s = round(pd.to_numeric(subset['score'], errors='coerce').mean(), 2)
                avg_rows.append([f'Average (>= {n} models)', '', avg_t, avg_d, avg_s])
            else:
                avg_rows.append([f'Average (>= {n} models)', '', 'nan', 'nan', 'nan'])

        output_path = Path("experimental_results") / output_csv
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV 저장
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for row in avg_rows:
                writer.writerow(row)
            writer.writerow(['Average', '', avg_throughput, avg_drop_rate, avg_score])
            writer.writerow(['schedule_file', 'best_combination', 'normalized_throughput', 'drop_rate', 'score'])
            
            for _, row in output_df.iterrows():
                writer.writerow([
                    row['schedule_file'],
                    row['best_combination'],
                    row['normalized_throughput'],
                    row['drop_rate'],
                    row['score']
                ])
                
        print(f"Saved {len(results)} results to {output_csv}")
        print(f"Average Throughput: {avg_throughput}, Drop Rate: {avg_drop_rate}, Score: {avg_score}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()

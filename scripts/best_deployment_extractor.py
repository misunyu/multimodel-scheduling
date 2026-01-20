import json
import csv
from pathlib import Path
import pandas as pd

def get_performance_index(results_dir):
    perf_index = {} # schedule_file_name -> { 'best_deployment': name, 'data': { combination_name -> performance_data } }
    results_path = Path(results_dir)
    
    for p_file in results_path.glob("*.json"):
        try:
            with open(p_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            sched_file = content.get("schedule file") or content.get("schedule_file")
            if not sched_file:
                continue
                
            sched_name = Path(sched_file).name
            best_dep = content.get("best deployment")
            
            if sched_name not in perf_index:
                perf_index[sched_name] = {'best_deployment': best_dep, 'data': {}}
            else:
                # 같은 스케줄 파일에 대해 여러 결과 파일이 있을 경우, best deployment가 있는 것을 우선하거나 업데이트
                if best_dep:
                    perf_index[sched_name]['best_deployment'] = best_dep
                
            data_list = content.get("data", [])
            for item in data_list:
                comb_name = item.get("combination")
                if comb_name:
                    perf_index[sched_name]['data'][comb_name] = item
                    
        except Exception as e:
            print(f"Error indexing {p_file}: {e}")
            
    return perf_index

def main():
    results_recompute_dir = "results_recompute"
    test_schedules_csv = "xgboost_model/dataset/gpu/test_schedules_x3.csv"
    output_csv = "best_deployment_results.csv"
    
    print("Indexing performance data from results_recompute...")
    perf_index = get_performance_index(results_recompute_dir)
    print(f"Indexed performance data for {len(perf_index)} schedules.")
    
    print("Loading test schedules...")
    try:
        schedules_df = pd.read_csv(test_schedules_csv)
    except Exception as e:
        print(f"Error loading {test_schedules_csv}: {e}")
        return

    results = []
    
    for _, row in schedules_df.iterrows():
        sched_name = row['schedule_name']
        pure_sched_name = Path(sched_name).name
        
        print(f"Processing schedule: {pure_sched_name}")
        
        sched_info = perf_index.get(pure_sched_name)
        if not sched_info:
            print(f"  [LOG] No data found for schedule {pure_sched_name}")
            results.append({
                'schedule_file': sched_name,
                'best_combination': '-',
                'normalized_throughput': '-',
                'drop_rate': '-',
                'score': '-',
                'models_count': 0
            })
            continue
            
        best_comb_name = sched_info.get('best_deployment')
        if not best_comb_name:
            print(f"  [LOG] No 'best deployment' specified for {pure_sched_name}")
            results.append({
                'schedule_file': sched_name,
                'best_combination': '-',
                'normalized_throughput': '-',
                'drop_rate': '-',
                'score': '-',
                'models_count': 0
            })
            continue
            
        perf_item = sched_info['data'].get(best_comb_name)
        if not perf_item:
            print(f"  [LOG] Best combination '{best_comb_name}' not found in data for {pure_sched_name}")
            results.append({
                'schedule_file': sched_name,
                'best_combination': best_comb_name,
                'normalized_throughput': '-',
                'drop_rate': '-',
                'score': '-',
                'models_count': 0
            })
            continue
            
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
        
        # 평균값 계산
        numeric_throughput = pd.to_numeric(output_df['normalized_throughput'], errors='coerce')
        numeric_drop_rate = pd.to_numeric(output_df['drop_rate'], errors='coerce')
        numeric_score = pd.to_numeric(output_df['score'], errors='coerce')
        
        avg_throughput = round(numeric_throughput.mean(), 2)
        avg_drop_rate = round(numeric_drop_rate.mean(), 2)
        avg_score = round(numeric_score.mean(), 2)
        
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

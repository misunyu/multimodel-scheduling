import json
import csv
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

def get_single_model_scores(results_dir):
    scores = {} # (model_name, execution) -> score
    
    results_path = Path(results_dir)
    # results_dir 내의 모든 json 파일을 탐색하여 단일 모델 결과를 수집
    for p_file in results_path.glob("*.json"):
        try:
            with open(p_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            data_list = content.get("data", [])
            for item in data_list:
                models = item.get("models", {})
                if len(models) != 1:
                    continue
                
                # view1 등의 키를 통해 모델 정보 획득
                view_key = list(models.keys())[0]
                model_info = models[view_key]
                model_name = model_info.get("model")
                execution = model_info.get("execution", "").upper()
                score = item.get("score")
                
                if model_name and execution and score is not None:
                    # 중복되는 경우 더 높은 score를 유지
                    if (model_name, execution) not in scores or score > scores[(model_name, execution)]:
                        scores[(model_name, execution)] = score
        except Exception as e:
            print(f"Error reading {p_file}: {e}")
            
    return scores

def find_best_combination(schedule_doc, model_scores):
    best_comb = None
    max_total_score = -1
    
    # schedule_doc은 yaml 로드된 딕셔너리
    # 각 combination에 대해 Latency-based score 계산
    # 하지만 문제에서 "1) 배치 대상이 되는 각 모델의 cpu/gpu에서 실행될 때의 score를 찾아서 score가 큰 프로세싱 유닛을 배칭해" 라고 함.
    # 즉, 각 모델별로 독립적으로 CPU/GPU 중 나은 것을 고른 후, 그 결과와 일치하는 combination을 찾는 것임.
    
    # 1단계: 각 모델(view)별 최적 PU 결정
    # 스케줄 파일의 첫 번째 combination을 보고 모델 목록을 추출 (모든 combination은 같은 모델 셋을 가짐)
    first_comb_name = list(schedule_doc.keys())[0]
    while first_comb_name in ["schedule file", "best deployment"] and len(schedule_doc) > 1:
         # 혹시 메타데이터가 섞여있을 경우를 대비 (보통 yaml 파일엔 combination만 있음)
         # 하지만 위에서 본 yaml은 combination_1 부터 시작함.
         break
    
    models_in_schedule = schedule_doc[first_comb_name]
    target_pu_map = {} # 모델별 식별자(예: view1) -> 'CPU' or 'GPU'
    
    for view_id, info in models_in_schedule.items():
        m_name = info.get("model")
        cpu_score = model_scores.get((m_name, 'CPU'))
        gpu_score = model_scores.get((m_name, 'GPU'))
        
        if cpu_score is None or gpu_score is None:
             missing = []
             if cpu_score is None: missing.append("CPU")
             if gpu_score is None: missing.append("GPU")
             print(f"  [LOG] Model {m_name} is missing scores for: {', '.join(missing)}")
             # Default to 0 if missing
             cpu_score = cpu_score or 0
             gpu_score = gpu_score or 0

        if gpu_score > cpu_score:
            target_pu_map[view_id] = 'GPU'
        else:
            target_pu_map[view_id] = 'CPU'
            
    # 2단계: 위 target_pu_map과 일치하는 combination 찾기
    print(f"  [LOG] Target PU mapping: {target_pu_map}")
    for comb_name, comb_data in schedule_doc.items():
        if not isinstance(comb_data, dict): continue
        
        match = True
        # comb_data의 각 모델 항목에 대해 target_pu와 일치하는지 확인
        # target_pu_map의 키는 view1, gpt2_cpu 등 다양할 수 있음.
        # 스케줄 파일의 키(gpt2_cpu 등)가 target_pu_map의 키(view1 등)에 대응되는지 확인 필요
        
        # 모델명으로 매칭 시도
        for model_id, info in comb_data.items():
            m_name = info.get("model")
            execution = info.get("execution", "").upper()
            
            # 해당 모델의 최적 PU 찾기
            cpu_score = model_scores.get((m_name, 'CPU'), 0)
            gpu_score = model_scores.get((m_name, 'GPU'), 0)
            target_pu = 'GPU' if gpu_score > cpu_score else 'CPU'
            
            if execution != target_pu:
                match = False
                break
        
        if match:
            return comb_name, None
            
    return None, target_pu_map

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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Latency Based Selector")
    parser.add_argument("--results_recompute_dir", default="results_recompute")
    parser.add_argument("--test_schedules_csv", default="xgboost_model/dataset/gpu/test_schedules_x3.csv")
    parser.add_argument("--output_csv", default="latency_based_best_results.csv")
    args = parser.parse_args()

    results_recompute_dir = args.results_recompute_dir
    test_schedules_csv = args.test_schedules_csv
    output_csv = args.output_csv
    
    print("Collecting single model scores...")
    model_scores = get_single_model_scores(results_recompute_dir)
    print(f"Collected {len(model_scores)} model-PU scores.")
    
    print("Indexing performance data from results_recompute...")
    perf_index = get_performance_index(results_recompute_dir)
    print(f"Indexed performance data for {len(perf_index)} schedules.")
    
    print("Loading test schedules...")
    schedules_df = pd.read_csv(test_schedules_csv)
    sched_index = {} # name -> content(dict)
    for _, row in schedules_df.iterrows():
        name = row['schedule_name']
        content = row['content']
        try:
            sched_index[name] = yaml.safe_load(content)
        except Exception as e:
            print(f"Error parsing yaml for {name}: {e}")
            
    # print("Loading test data...")
    # test_df = pd.read_csv(test_random_csv)
    
    results = []
    
    top1_hits = 0
    top5_hits = 0
    top1_hits_ge3 = 0
    top5_hits_ge3 = 0
    total_valid_schedules = 0
    total_valid_schedules_ge3 = 0
    
    # test_schedules_random.csv에 명시된 모든 스케줄 파일에 대해
    for sched_name, sched_doc in sched_index.items():
        print(f"\nProcessing schedule: {sched_name}")
        best_comb_name, pu_map = find_best_combination(sched_doc, model_scores)
        
        if not best_comb_name:
            print(f"  [LOG] Could not find matching combination for {sched_name} in its YAML. Target map was: {pu_map}")
            results.append({
                'schedule_file': sched_name,
                'best_combination': '-',
                'normalized_throughput': '-',
                'drop_rate': '-',
                'score': '-',
                'models_count': 0
            })
            continue
            
        # 해당 스케줄 파일과 combination 명칭으로 perf_index에서 데이터 찾기
        # sched_name은 보통 경로를 포함할 수 있으므로 파일명만 추출
        pure_sched_name = Path(sched_name).name
        sched_perf = perf_index.get(pure_sched_name, {})

        # Find ACTUAL best combination for this schedule in results_recompute
        best_actual_combs = []
        max_actual_score = -float('inf')
        for c_name, p_item in sched_perf.items():
            actual_score = p_item.get('score', -float('inf'))
            if np.isclose(actual_score, max_actual_score, atol=1e-7):
                best_actual_combs.append(c_name)
            elif actual_score > max_actual_score:
                max_actual_score = actual_score
                best_actual_combs = [c_name]

        perf_item = sched_perf.get(best_comb_name)
        
        total_valid_schedules += 1
        # Top-1 Accuracy
        if best_comb_name in best_actual_combs:
            top1_hits += 1
            
        # For latency-based, "Top-5 Accuracy" is tricky because it doesn't rank all.
        # But we can define it as: is the actual best among the top-5 combinations ranked by latency-based heuristic?
        # Actually, latency-based only picks ONE "best".
        # If we want a Top-5, we'd need to rank all combinations by the heuristic.
        # Let's skip Top-5 for latency-based or just set it equal to Top-1 if we only have one selection.
        # However, to be consistent with others, let's see if we can rank.
        # In find_best_combination, it currently returns immediately.
        # Let's just use Top-1 as Top-5 for now, or just leave it. 
        # Actually, let's just use Top-1 for both if Top-5 is not well-defined for this heuristic.
        # Re-reading: "Top-5 accuracy는 추론한 상위 5개 그룹...에 속할 확률"
        # Since latency-based currently only "infers" ONE best, the top-5 group only has that one combination.
        if best_comb_name in best_actual_combs:
            top5_hits += 1

        display_comb = best_comb_name
        # Apply coloring: Purple for match, Red for chosen, Blue for actual
        if best_comb_name in best_actual_combs:
            display_comb = f"<font color='purple'>{best_comb_name}</font>"
        else:
            display_comb = f"<font color='red'>{best_comb_name}</font>"
            if best_actual_combs:
                actual_str = ", ".join([f"<font color='blue'>{c}</font>" for c in best_actual_combs])
                display_comb += f" (Actual: {actual_str})"

        if not perf_item:
            print(f"  [LOG] NO MATCH in results_recompute for schedule='{pure_sched_name}' and combination='{best_comb_name}'")
            # 디버깅을 위해 해당 스케줄 파일의 다른 combination이 있는지 확인
            existing_combs = list(sched_perf.keys())
            if len(existing_combs) > 0:
                print(f"    Available combinations in results_recompute for this schedule: {existing_combs}")
            else:
                print(f"    This schedule file '{pure_sched_name}' is not found at all in results_recompute")

            results.append({
                'schedule_file': sched_name,
                'best_combination': display_comb,
                'normalized_throughput': '-',
                'drop_rate': '-',
                'score': '-',
                'models_count': 0
            })
            continue
            
        # 데이터 찾음
        derived = perf_item.get('derived', {})
        models_count = len(perf_item.get('models', {}))
        
        if models_count >= 3:
            total_valid_schedules_ge3 += 1
            if best_comb_name in best_actual_combs:
                top1_hits_ge3 += 1
            if best_comb_name in best_actual_combs:
                top5_hits_ge3 += 1

        results.append({
            'schedule_file': sched_name,
            'best_combination': display_comb,
            'normalized_throughput': derived.get('throughput_norm'),
            'drop_rate': derived.get('drop_rate_norm'),
            'score': perf_item.get('score'),
            'models_count': models_count
        })
            
    if results:
        output_df = pd.DataFrame(results)
        
        # Calculate accuracies
        top1_acc = round(top1_hits / total_valid_schedules, 4) if total_valid_schedules > 0 else 0
        top5_acc = round(top5_hits / total_valid_schedules, 4) if total_valid_schedules > 0 else 0
        
        top1_acc_ge3 = round(top1_hits_ge3 / total_valid_schedules_ge3, 4) if total_valid_schedules_ge3 > 0 else 0
        top5_acc_ge3 = round(top5_hits_ge3 / total_valid_schedules_ge3, 4) if total_valid_schedules_ge3 > 0 else 0

        # 평균값 계산 (숫자 데이터만)
        numeric_throughput = pd.to_numeric(output_df['normalized_throughput'], errors='coerce')
        numeric_drop_rate = pd.to_numeric(output_df['drop_rate'], errors='coerce')
        numeric_score = pd.to_numeric(output_df['score'], errors='coerce')
        
        avg_throughput = round(numeric_throughput.sum() / 24.3, 2)
        avg_drop_rate = round(numeric_drop_rate.sum() / 24.3, 2)
        avg_score = round(numeric_score.sum() / 24.3, 2)

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

        # CSV 상단에 평균값 추가
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Top-1 Accuracy', top1_acc])
            writer.writerow(['Top-5 Accuracy', top5_acc])
            writer.writerow(['Top-1 Accuracy (>= 3 models)', top1_acc_ge3])
            writer.writerow(['Top-5 Accuracy (>= 3 models)', top5_acc_ge3])
            # 개별 모델 개수별 평균 추가
            for row in avg_rows:
                writer.writerow(row)
            # 전체 평균값 정보를 담은 행 추가
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
                
        print(f"Saved {len(results)} results to {output_csv} with average header.")
        print(f"Average Throughput: {avg_throughput:.4f}, Drop Rate: {avg_drop_rate:.4f}, Score: {avg_score:.4f}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()

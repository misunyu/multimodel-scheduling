import json
import random
import csv
import argparse
from pathlib import Path

def split_dataset(perf_dir_path, schedule_dir_path, output_dir="xgboost_model/dataset", ratio=0.8, pattern="_x3"):
    perf_dir = Path(perf_dir_path)
    schedule_dir = Path(schedule_dir_path)
    
    # 스케줄 파일 인덱싱
    sched_index = {}
    for p in schedule_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".yaml", ".yml", ".json"}:
            try:
                content = p.read_text(encoding="utf-8")
                sched_index[p.name.lower()] = content
            except:
                pass

    all_data = []
    
    # JSON 파일들 탐색
    perf_files = sorted(list(perf_dir.glob("*.json")))

    for p_file in perf_files:
        try:
            with open(p_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
            schedule_file_name = content.get("schedule file") or content.get("schedule_file")
            if not schedule_file_name:
                continue
            
            # 매칭되는 스케줄 내용 가져오기
            sched_content = sched_index.get(Path(schedule_file_name).name.lower())
            if not sched_content:
                print(f"Warning: Schedule {schedule_file_name} not found for {p_file}")
                continue
                
            data_list = content.get("data", [])
            is_pattern = pattern in p_file.name

            for item in data_list:
                item["schedule_file"] = schedule_file_name
                entry = {
                    "perf_json": item,
                    "sched_content": sched_content,
                    "is_pattern": is_pattern
                }
                all_data.append(entry)

        except Exception as e:
            print(f"Error reading {p_file}: {e}")

    if not all_data:
        print("No data found.")
        return

    def save_to_csv(data_list, perf_filename, sched_filename):
        if not data_list:
            print(f"Warning: No data to save for {perf_filename}")
            # 빈 파일이라도 생성하여 에러 방지
            Path(perf_filename).touch()
            with open(sched_filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["schedule_name", "content"])
            return
        
        # Performance data CSV
        with open(perf_filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["schedule_file", "combination", "json_content"])
            for item in data_list:
                s_file = item["perf_json"].get("schedule_file", "")
                c_name = item["perf_json"].get("combination", "")
                writer.writerow([s_file, c_name, json.dumps(item["perf_json"])])
        
        # Schedule data CSV
        unique_schedules = {}
        for item in data_list:
            s_name = item["perf_json"]["schedule_file"]
            if s_name not in unique_schedules:
                unique_schedules[s_name] = item["sched_content"]
        
        with open(sched_filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["schedule_name", "content"])
            for name, content in unique_schedules.items():
                writer.writerow([name, content])

    # 1) Random 분할 (8:2)
    random_all = list(all_data)
    random.shuffle(random_all)
    split_idx = int(len(random_all) * ratio)
    train_random = random_all[:split_idx]
    test_random = random_all[split_idx:]
    
    # Output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_to_csv(train_random, out_dir / "train_random.csv", out_dir / "train_schedules_random.csv")
    save_to_csv(test_random, out_dir / "test_random.csv", out_dir / "test_schedules_random.csv")
    print(f"Random split saved to {out_dir}: {len(train_random)} train, {len(test_random)} test")

    # 2) Pattern 분할
    train_pattern = [d for d in all_data if not d["is_pattern"]]
    test_pattern = [d for d in all_data if d["is_pattern"]]
    
    pattern_name = pattern.strip("_")
    save_to_csv(train_pattern, out_dir / f"train_{pattern_name}.csv", out_dir / f"train_schedules_{pattern_name}.csv")
    save_to_csv(test_pattern, out_dir / f"test_{pattern_name}.csv", out_dir / f"test_schedules_{pattern_name}.csv")
    print(f"Pattern split ({pattern}) saved to {out_dir}: {len(train_pattern)} train, {len(test_pattern)} test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets using both random and pattern modes.")
    parser.add_argument("--perf_dir", default="results_recompute", help="Directory containing performance JSON files")
    parser.add_argument("--schedule_dir", default="gen_schedules", help="Directory containing schedule files")
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio of training data (for random mode)")
    parser.add_argument("--output_dir", default="xgboost_model/dataset", help="Output directory for CSV files")
    parser.add_argument("--pattern", default="_x3", help="Pattern to identify test files (e.g., _x3 or _3x)")
    
    args = parser.parse_args()
    
    split_dataset(args.perf_dir, args.schedule_dir, args.output_dir, args.ratio, args.pattern)

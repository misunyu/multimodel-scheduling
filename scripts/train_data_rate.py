import json
import glob
import os

# performance 디렉토리의 모든 JSON 파일에서 데이터 개수 카운트
perf_files = glob.glob("performance/*.json")
total_items = 0

for file_path in sorted(perf_files):
    with open(file_path, 'r') as f:
        data = json.load(f)
        items = len(data.get('data', []))
        total_items += items
        print(f"{os.path.basename(file_path)}: {items} items")

print(f"\n전체 performance 파일의 총 데이터 개수: {total_items}")

# merged_train.json의 데이터 개수
with open("xgboost_model/train_data/merged_train.json", 'r') as f:
    merged_data = json.load(f)
    merged_items = len(merged_data.get('data', []))

print(f"merged_train.json의 데이터 개수: {merged_items}")

# 비율 계산
percentage = (merged_items / total_items * 100) if total_items > 0 else 0
print(f"\n포함 비율: {percentage:.2f}%")
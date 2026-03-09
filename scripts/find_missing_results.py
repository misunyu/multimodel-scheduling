import os
import re

def main():
    gen_schedules_dir = 'gen_schedules'
    results_dirs = ['results_recompute', 'results_recompute_additional']

    if not os.path.exists(gen_schedules_dir):
        print(f"Error: {gen_schedules_dir} directory not found.")
        return
    
    for d in results_dirs:
        if not os.path.exists(d):
            print(f"Warning: {d} directory not found.")

    # 1. gen_schedules 폴더의 파일명 (확장자 제외)
    gen_files = set()
    for f in os.listdir(gen_schedules_dir):
        if f.endswith('.yaml'):
            gen_files.add(f[:-5]) # .yaml 제거

    # 2. results_recompute 폴더의 파일명에서 스케줄 파일명 추출
    # 예: recompute_performance_20260307_045901_model_schedules_vgg19.json -> model_schedules_vgg19
    recomputed_files = set()
    # 패턴: recompute_performance_YYYYMMDD_HHMMSS_
    pattern = re.compile(r'^recompute_performance_\d{8}_\d{6}_(.+)\.json$')
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            for f in os.listdir(results_dir):
                match = pattern.match(f)
                if match:
                    schedule_name = match.group(1)
                    recomputed_files.add(schedule_name)

    # 3. gen_schedules에는 있지만 results_recompute에는 없는 파일 찾기
    missing_files = sorted(list(gen_files - recomputed_files))

    print(f"Total files in gen_schedules: {len(gen_files)}")
    print(f"Total recomputed files found: {len(recomputed_files)}")
    print(f"Number of missing files: {len(missing_files)}")
    
    if missing_files:
        print("\nMissing schedule files:")
        for f in missing_files:
            print(f"- {f}.yaml")
    else:
        print(f"\nAll files are present in {', '.join(results_dirs)}.")

if __name__ == "__main__":
    main()

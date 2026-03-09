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
    # 이미 _x2, _x3, _x4가 붙은 파일들이 있으므로 베이스 이름을 추출합니다.
    all_gen_files = sorted([f[:-5] for f in os.listdir(gen_schedules_dir) if f.endswith('.yaml')])
    base_gen_files = set()
    suffix_pattern = re.compile(r'(.+)_x[234]$')
    for f in all_gen_files:
        match = suffix_pattern.match(f)
        if match:
            base_gen_files.add(match.group(1))
        else:
            base_gen_files.add(f)
    
    base_gen_files = sorted(list(base_gen_files))

    # 2. results 폴더들에서 이미 처리된 파일명 추출
    recomputed_files = set()
    pattern = re.compile(r'^recompute_performance_\d{8}_\d{6}_(.+)\.json$')
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            for f in os.listdir(results_dir):
                match = pattern.match(f)
                if match:
                    recomputed_files.add(match.group(1))

    # 3. 필요한 접미사들
    # 사용자 명시: _x2, _x3, _x4, x1-5, x2-5, x3-5
    # 실제 파일명 관찰 결과: _x2, _x3, _x4, _x1-5, _x2-5, _x3-5
    required_suffixes = ['_x2', '_x3', '_x4', '_x1-5', '_x2-5', '_x3-5']
    
    missing_variants = []
    for base_name in base_gen_files:
        for suffix in required_suffixes:
            target_name = f"{base_name}{suffix}"
            if target_name not in recomputed_files:
                missing_variants.append(f"{target_name}.json")

    print(f"Total base schedules (deduplicated): {len(base_gen_files)}")
    print(f"Total recomputed files found: {len(recomputed_files)}")
    print(f"Number of missing performance variants: {len(missing_variants)}")
    
    if missing_variants:
        print("\nMissing performance result files (some examples):")
        for f in missing_variants[:20]: # 상위 20개만 출력
            print(f"- {f}")
        if len(missing_variants) > 20:
            print(f"... and {len(missing_variants) - 20} more.")
    else:
        print(f"\nAll required performance variants for each base schedule are present.")

if __name__ == "__main__":
    main()

import os
import glob
import argparse
from typing import Dict, List, Tuple, Optional

# Constants
PREFIX = "performance_"
MODEL_MARKER = "_model_"
EXTENSION = ".json"
SUFFIXES = ["_x2", "_x3", "_x4", "_x1-5", "_x2-5", "_x3-5"]
BASE_SUFFIX = "base"

def parse_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses a filename to extract the group name and suffix.
    
    Expected format: performance_<timestamp>_model_<group><suffix>.json
    
    Returns:
        (group, suffix) if valid, otherwise (None, None)
    """
    if not (filename.startswith(PREFIX) and filename.endswith(EXTENSION)):
        return None, None
    
    # Remove extension
    core_name = filename[:-len(EXTENSION)]
    
    # Find the end of timestamp/prefix part
    # performance_<timestamp>_model_ is the prefix
    marker_idx = core_name.find(MODEL_MARKER)
    if marker_idx == -1:
        return None, None
    
    # Everything including model_ is <group><suffix>
    # The requirement is that the group name should include 'model_'
    group_with_suffix = core_name[marker_idx + 1:] # +1 to skip the underscore before 'model'
    
    if not group_with_suffix:
        return None, None
    
    # Check for known suffixes at the end
    found_suffix = BASE_SUFFIX
    group_name = group_with_suffix
    
    for s in SUFFIXES:
        if group_with_suffix.endswith(s):
            found_suffix = s
            group_name = group_with_suffix[:-len(s)]
            break
            
    return group_name, found_suffix

def collect_groups(results_dir: str) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Collects files from the directory and groups them.
    
    Returns:
        A dictionary mapping group_name -> {suffix: filename}
        and a list of warnings for invalid filenames.
    """
    pattern = os.path.join(results_dir, "*.json")
    files = glob.glob(pattern)
    
    groups = {}
    warnings = []
    
    for file_path in files:
        filename = os.path.basename(file_path)
        group_name, suffix = parse_filename(filename)
        
        if group_name is None:
            warnings.append(f"Invalid filename format: {filename}")
            continue
            
        if group_name not in groups:
            groups[group_name] = {}
            
        groups[group_name][suffix] = filename
        
    return groups, warnings

def check_missing_sets(group_data: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Checks which suffixes are missing for a group.
    
    Returns:
        (existing_suffixes, missing_suffixes)
    """
    all_required = [BASE_SUFFIX] + SUFFIXES
    existing = []
    missing = []
    
    for s in all_required:
        if s in group_data:
            existing.append(s)
        else:
            missing.append(s)
            
    return existing, missing

def parse_schedule_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses a schedule filename to extract the group name and suffix.
    
    Expected format: <group><suffix>.yaml
    
    Returns:
        (group, suffix) if valid, otherwise (None, None)
    """
    if not filename.endswith(".yaml"):
        return None, None
        
    core_name = filename[:-len(".yaml")]
    
    found_suffix = BASE_SUFFIX
    group_name = core_name
    
    for s in SUFFIXES:
        if core_name.endswith(s):
            found_suffix = s
            group_name = core_name[:-len(s)]
            break
            
    return group_name, found_suffix

def collect_schedule_groups(dirs: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Collects group names from schedule directories.
    """
    groups = {}
    warnings = []
    
    for d in dirs:
        if not os.path.isdir(d):
            continue
        
        pattern = os.path.join(d, "*.yaml")
        files = glob.glob(pattern)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            group_name, suffix = parse_schedule_filename(filename)
            
            if group_name is None:
                warnings.append(f"Invalid schedule filename format: {filename} in {d}")
                continue
                
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(filename)
            
    return groups, warnings

def print_results(groups: Dict[str, Dict[str, str]], warnings: List[str], schedule_groups: Dict[str, List[str]] = None):
    """
    Prints incomplete groups and summary.
    """
    if warnings:
        print("--- Warnings ---")
        for w in warnings:
            print(f"WARNING: {w}")
        print()

    incomplete_count = 0
    
    # Sort group names for consistent output
    sorted_group_names = sorted(groups.keys())
    
    print("--- Incomplete Groups (Missing Suffixes) ---")
    for group_name in sorted_group_names:
        existing, missing = check_missing_sets(groups[group_name])
        
        if missing:
            incomplete_count += 1
            print(f"Group: {group_name}")
            print(f"  Existing (OK): {', '.join(existing)}")
            print(f"  MISSING (!!!): {', '.join(missing)}")
            print(f"  Files:")
            for s in existing:
                print(f"    - {groups[group_name][s]}")
            print("-" * 40)
            
    if incomplete_count == 0:
        print("No incomplete groups found based on suffixes.")
    
    print(f"\nTotal incomplete groups (missing suffixes): {incomplete_count}")

    if schedule_groups:
        print("\n" + "="*40)
        print("--- Groups with No Results ---")
        unmatched_count = 0
        for group_name in sorted(schedule_groups.keys()):
            if group_name not in groups:
                unmatched_count += 1
                print(f"Group: {group_name}")
                print(f"  Schedule files: {', '.join(schedule_groups[group_name])}")
        
        if unmatched_count == 0:
            print("All schedule groups have at least one result file.")
        
        print(f"\nTotal groups with no results: {unmatched_count}")

def main():
    parser = argparse.ArgumentParser(description="Check for incomplete result sets in the results directory.")
    parser.add_argument("--results_dir", default="results", help="Directory containing JSON results (default: results)")
    parser.add_argument("--schedule_dirs", nargs="+", default=["gen_schedules", "gen_schedules_additional"], 
                        help="Directories containing schedule YAML files (default: gen_schedules gen_schedules_additional)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Error: Directory '{args.results_dir}' does not exist.")
        return

    groups, warnings = collect_groups(args.results_dir)
    
    schedule_groups, schedule_warnings = collect_schedule_groups(args.schedule_dirs)
    warnings.extend(schedule_warnings)
    
    print_results(groups, warnings, schedule_groups)

if __name__ == "__main__":
    main()

# Example usage:
# python scripts/find_incomplete_result_groups.py --results_dir results
# python scripts/find_incomplete_result_groups.py (uses default 'results' directory)

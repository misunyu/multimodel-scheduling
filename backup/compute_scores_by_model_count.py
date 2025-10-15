import os
import json
import argparse
from typing import Dict, List, Tuple

try:
    import yaml
except Exception:
    yaml = None


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError('PyYAML is required to parse YAML files in tests/. Please install pyyaml.')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_yaml_for_perf(perf: dict, yaml_dirs: List[str]) -> str:
    sched_file = perf.get('schedule file') or perf.get('schedule_file')
    if not sched_file:
        return ''
    for d in yaml_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn == sched_file:
                    return os.path.join(root, fn)
    # fallback: return first yaml in dirs
    for d in yaml_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith('.yaml') or fn.lower().endswith('.yml'):
                    return os.path.join(root, fn)
    return ''


def count_models_in_combination(schedule: dict, combination: str) -> Tuple[int, int]:
    """
    Returns (model_count, total_infps) for the given combination name using the YAML schedule.
    If the combination doesn't exist or cannot be parsed, returns (0, 0).
    """
    try:
        combo_cfg = schedule.get(combination)
        if not isinstance(combo_cfg, dict):
            return 0, 0
        model_count = 0
        total_infps = 0
        for _, entry in combo_cfg.items():
            if not isinstance(entry, dict):
                continue
            if 'model' in entry:
                model_count += 1
            infps = entry.get('infps')
            try:
                if infps is not None:
                    total_infps += int(infps)
            except Exception:
                pass
        return model_count, total_infps
    except Exception:
        return 0, 0


def process_performance_files(perf_dir: str, yaml_dirs: List[str]) -> Dict[int, List[Tuple[Tuple[int, str], str]]]:
    """
    Scans performance JSONs and groups best deployments by model count.
    Returns a dict: {model_count: [((total_infps, perf_basename), output_line_str), ...]}
    """
    groups: Dict[int, List[Tuple[Tuple[int, str], str]]] = {}
    for root, _, files in os.walk(perf_dir):
        for fn in files:
            if not fn.lower().endswith('.json'):
                continue
            path = os.path.join(root, fn)
            try:
                perf = load_json(path)
            except Exception:
                continue
            best_combo = perf.get('best deployment') or perf.get('best_deployment')
            data = perf.get('data', [])
            if not best_combo or not data:
                continue
            # find record for best combination to get score
            score = None
            for rec in data:
                if rec.get('combination') == best_combo:
                    score = rec.get('score')
                    break
            if score is None:
                # fallback: max score
                try:
                    best_rec = max(data, key=lambda r: r.get('score', float('-inf')))
                    score = best_rec.get('score')
                    best_combo = best_rec.get('combination')
                except Exception:
                    continue
            # find yaml and compute model_count and total_infps
            yaml_path = find_yaml_for_perf(perf, yaml_dirs)
            if not yaml_path:
                continue
            try:
                schedule = load_yaml(yaml_path)
            except Exception:
                continue
            mcount, total_infps = count_models_in_combination(schedule, best_combo)
            if mcount == 0:
                continue
            # Construct output line: "(index) score". Index will be filled later when sorting.
            perf_base = os.path.basename(path)
            line = f"score={score:.4f} | file={perf_base} | combo={best_combo} | total_infps={total_infps}"
            groups.setdefault(mcount, []).append(((total_infps, perf_base), line))
    # sort within each group by total_infps ascending, then by filename for stability
    for mcount, items in groups.items():
        items.sort(key=lambda t: (t[0][0], t[0][1]))
    return groups


def write_outputs(groups: Dict[int, List[Tuple[Tuple[int, str], str]]], output_prefix: str) -> None:
    os.makedirs('../results', exist_ok=True)
    master_lines: List[str] = []
    for mcount, items in sorted(groups.items()):
        out_path = os.path.join('../results', f"scores_best_deployments_{mcount}models.txt")
        idx = 1
        lines: List[str] = []
        for _, line in items:
            formatted = f"({idx}) {line}"
            lines.append(formatted)
            master_lines.append(f"[{mcount} models] {formatted}")
            idx += 1
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"Saved: {out_path} ({len(lines)} lines)")
    # Also write a combined file
    combined_path = os.path.join('../results', f"{output_prefix}_all.txt")
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(master_lines) + '\n')
    print(f"Saved combined: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute scores grouped by model count from performance JSONs and tests YAMLs.')
    parser.add_argument('--perf_dir', default='performance', help='Directory containing performance JSON files')
    parser.add_argument('--yaml_dirs', nargs='*', default=['tests'], help='Directory(ies) to search for schedule YAML files')
    parser.add_argument('--output_prefix', default='scores_output', help='Prefix for combined output filename')
    args = parser.parse_args()

    groups = process_performance_files(args.perf_dir, args.yaml_dirs)
    if not groups:
        print('No valid data found.')
        return
    write_outputs(groups, args.output_prefix)


if __name__ == '__main__':
    main()

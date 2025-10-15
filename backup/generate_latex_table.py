import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

try:
    import yaml
except Exception:
    yaml = None


MODEL_SHORT = {
    'resnet50_small': 'R50-s',
    'resnet50': 'R50',
    'yolov3_small': 'Y3-s',
    'yolov3': 'Y3',
    'yolov3_big': 'Y3-b',
}


def short_model_name(name: str) -> str:
    if not name:
        return ''
    n = name.strip()
    if n in MODEL_SHORT:
        return MODEL_SHORT[n]
    # heuristic: suffix -s or -b if present
    base = n
    suffix = ''
    if n.endswith('_small'):
        base = n.replace('_small', '')
        suffix = '-s'
    elif n.endswith('_big'):
        base = n.replace('_big', '')
        suffix = '-b'
    base_short = MODEL_SHORT.get(base, base)
    return f"{base_short}{suffix}" if suffix else base_short


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Please install pyyaml to parse YAML files.")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_best_record(perf: dict) -> Tuple[str, dict]:
    # Accept both dict and list formats
    if isinstance(perf, list):
        # treat as list of records already
        data = [r for r in perf if isinstance(r, dict)]
        best_combo = None
    else:
        best_combo = perf.get('best deployment') or perf.get('best_deployment')
        data = perf.get('data', [])
    if not data:
        raise ValueError('No data entries in performance JSON')
    if best_combo:
        for rec in data:
            if rec.get('combination') == best_combo:
                return best_combo, rec
    # fallback: pick highest score
    best_rec = max(data, key=lambda r: r.get('score', float('-inf')))
    return best_rec.get('combination', ''), best_rec


def device_key(name: str) -> str:
    n = (name or '').lower()
    if n in ('cpu',):
        return 'cpu'
    if n in ('npu0', 'npu_0', 'npu-0', 'npu 0', '0'):
        return 'npu0'
    if n in ('npu1', 'npu_1', 'npu-1', 'npu 1', '1'):
        return 'npu1'
    return n


def collect_deployment_from_yaml(schedule: dict, combination: str) -> Tuple[Dict[str, List[str]], List[int]]:
    combo_cfg = schedule.get(combination)
    if not isinstance(combo_cfg, dict):
        return {}, []
    per_device: Dict[str, List[str]] = {'cpu': [], 'npu0': [], 'npu1': []}
    # Capture per-model infps keyed by canonical model name
    infps_by_model: Dict[str, int] = {}
    for _, entry in combo_cfg.items():
        if not isinstance(entry, dict):
            continue
        model_name = entry.get('model')
        exec_dev = device_key(entry.get('execution', ''))
        if 'infps' in entry and model_name:
            try:
                infps_by_model[str(model_name)] = int(entry['infps'])
            except Exception:
                pass
        if exec_dev in per_device:
            per_device[exec_dev].append(short_model_name(model_name))
    # Build ordered list according to required priority
    priority = ['resnet50_small', 'yolov3_small', 'yolov3_big', 'resnet50_big']
    ordered = [infps_by_model[m] for m in priority if m in infps_by_model]
    return per_device, ordered


def collect_deployment_from_json(rec: dict) -> Dict[str, List[str]]:
    per_device: Dict[str, List[str]] = {'cpu': [], 'npu0': [], 'npu1': []}
    models = rec.get('../models', {})
    for _, m in models.items():
        model_name = m.get('model')
        exec_dev = device_key(m.get('execution', ''))
        if exec_dev in per_device:
            per_device[exec_dev].append(short_model_name(model_name))
    return per_device


def format_device_models(names: List[str]) -> str:
    if not names:
        return ''
    return ', '.join(names)


def build_input_freq(window_sec: float, infps_list: List[int]) -> str:
    # Construct a tuple-like string ONLY from per-model input FPS values.
    # The previous implementation incorrectly prefixed window_sec (e.g., 10).
    # We now exclude window_sec and show exactly one value per model.
    parts: List[str] = []
    if infps_list:
        # keep original order; often corresponds to view1/view2 ordering in YAML
        parts.extend([str(int(i)) for i in infps_list if i is not None])
    return f"({','.join(parts)})"


def generate_table(perf_json_path: str, schedule_yaml_path: str) -> str:
    perf = load_json(perf_json_path)
    schedule = load_yaml(schedule_yaml_path)

    best_combo, rec = find_best_record(perf)
    window_sec = rec.get('window_sec')
    fps = rec.get('total', {}).get('total_throughput_fps')
    drops = rec.get('derived', {}).get('drop_count')
    score = rec.get('score')

    per_device_yaml, infps_list = collect_deployment_from_yaml(schedule, best_combo)
    # if YAML missing, fall back to JSON-derived deployment
    per_device = per_device_yaml or collect_deployment_from_json(rec)

    input_freq = build_input_freq(window_sec, infps_list)

    cpu_str = format_device_models(per_device.get('cpu', []))
    npu0_str = format_device_models(per_device.get('npu0', []))
    npu1_str = format_device_models(per_device.get('npu1', []))

    # Prepare LaTeX table
    header = (
        "\\begin{tabular}{ccccccc}\n"
        "\\hline \n"
        "\\multirow{2}{*}{\\textbf{Input Freq.}} & \\multirow{2}{*}{\\textbf{FPS}} & \\multirow{2}{*}{\\textbf{\\# Drops}} & \\multirow{2}{*}{\\textbf{Score}} & \\multicolumn{3}{c}{\\textbf{Deployment}}\\tabularnewline\n"
        "\\cline{5-7}\n"
        " &  &  &  & \\textbf{CPU} & \\textbf{NPU0} & \\textbf{NPU1}\\tabularnewline\n"
        "\\hline \n"
    )

    row = (
        f"\\textbf{{{input_freq}}} & {fps:.2f} & {int(drops) if drops is not None else ''} & {score:.2f} & {cpu_str} & {npu0_str} & {npu1_str}\\tabularnewline\n"
    )

    footer = "\\hline \n\\end{tabular}\n"

    return header + row + footer


def _render_header() -> str:
    return (
        "\\begin{tabular}{ccccccc}\n"
        "\\hline \n"
        "\\multirow{2}{*}{\\textbf{Input Freq.}} & \\multirow{2}{*}{\\textbf{FPS}} & \\multirow{2}{*}{\\textbf{\\# Drops}} & \\multirow{2}{*}{\\textbf{Score}} & \\multicolumn{3}{c}{\\textbf{Deployment}}\\tabularnewline\n"
        "\\cline{5-7}\n"
        " &  &  &  & \\textbf{CPU} & \\textbf{NPU0} & \\textbf{NPU1}\\tabularnewline\n"
        "\\hline \n"
    )


def _render_footer() -> str:
    return "\\hline \n\\end{tabular}\n"


def _build_row_from_perf(perf: dict, schedule: Optional[dict]) -> Tuple[str, int]:
    best_combo, rec = find_best_record(perf)
    window_sec = rec.get('window_sec')
    fps = rec.get('total', {}).get('total_throughput_fps')
    drops = rec.get('derived', {}).get('drop_count')
    score = rec.get('score')

    per_device_yaml, infps_list = (collect_deployment_from_yaml(schedule, best_combo) if schedule else ({}, []))
    per_device = per_device_yaml or collect_deployment_from_json(rec)

    input_freq = build_input_freq(window_sec, infps_list)

    cpu_str = format_device_models(per_device.get('cpu', []))
    npu0_str = format_device_models(per_device.get('npu0', []))
    npu1_str = format_device_models(per_device.get('npu1', []))

    row = (
        f"\\textbf{{{input_freq}}} & {fps:.2f} & {int(drops) if drops is not None else ''} & {score:.2f} & {cpu_str} & {npu0_str} & {npu1_str}\\tabularnewline\n"
    )
    # model count across all devices
    model_count = sum(len(per_device.get(k, [])) for k in ('cpu', 'npu0', 'npu1'))
    return row, model_count


def _find_yaml_for_perf(perf: dict, yaml_search_dirs: List[str]) -> Optional[str]:
    # Some performance JSONs might be a list of records instead of a dict wrapper
    # Normalize by extracting the first element if a list is provided
    if isinstance(perf, list):
        # try to find a dict in the list that has schedule info
        candidate = None
        for item in perf:
            if isinstance(item, dict) and (('schedule file' in item) or ('schedule_file' in item)):
                candidate = item
                break
        if candidate is None:
            # fallback to first dict-like item
            for item in perf:
                if isinstance(item, dict):
                    candidate = item
                    break
        perf = candidate or {}

    schedule_name = None
    if isinstance(perf, dict):
        schedule_name = perf.get('schedule file') or perf.get('schedule_file')
        # Some formats nest under a top-level 'meta' or similar
        if not schedule_name and isinstance(perf.get('meta'), dict):
            meta = perf['meta']
            schedule_name = meta.get('schedule file') or meta.get('schedule_file')
    if not schedule_name:
        return None
    # If it's a path already, try it directly relative to repo root
    if os.path.isabs(schedule_name) and os.path.exists(schedule_name):
        return schedule_name
    # Try basename match under provided dirs
    base = os.path.basename(schedule_name)
    for root_dir in yaml_search_dirs:
        for root, _, files in os.walk(root_dir):
            for fn in files:
                if fn == base and fn.lower().endswith(('.yaml', '.yml')):
                    return os.path.join(root, fn)
    return None


def generate_grouped_tables(perf_paths: List[str], yaml_search_dirs: List[str]) -> Dict[int, str]:
    # We will keep both the row text and a numeric key for sorting by Input Freq.
    grouped_rows: Dict[int, List[Tuple[Tuple[int, ...], str]]] = {}
    for p in perf_paths:
        try:
            perf = load_json(p)
        except Exception:
            continue
        yaml_path = _find_yaml_for_perf(perf, yaml_search_dirs)
        schedule = None
        if yaml_path and os.path.exists(yaml_path):
            try:
                schedule = load_yaml(yaml_path)
            except Exception:
                schedule = None
        try:
            row, mcount = _build_row_from_perf(perf, schedule)
        except Exception:
            # If perf is a list of records, try wrapping into expected dict structure
            try:
                if isinstance(perf, list):
                    perf_wrapped = {'data': perf}
                    row, mcount = _build_row_from_perf(perf_wrapped, schedule)
                else:
                    continue
            except Exception:
                continue
        # Extract numeric tuple from the rendered row's Input Freq. part
        key_tuple: Tuple[int, ...] = ()
        try:
            # row starts with \textbf{(a,b,...)} & ...
            start = row.find('{(')
            end = row.find(')}', start)
            if start != -1 and end != -1:
                inside = row[start+2:end]
                nums = [int(x) for x in inside.split(',') if x.strip()]
                key_tuple = tuple(nums)
        except Exception:
            key_tuple = ()
        grouped_rows.setdefault(mcount, []).append((key_tuple, row))

    # Render tables per model count in ascending order of Input Freq.
    tables: Dict[int, str] = {}
    for mcount, items in grouped_rows.items():
        items.sort(key=lambda t: t[0])
        header = _render_header()
        body = ''.join([r for _, r in items])
        footer = _render_footer()
        tables[mcount] = header + body + footer
    return tables


def _discover_perf_jsons(json_dirs: List[str]) -> List[str]:
    perf_paths: List[str] = []
    for d in json_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith('.json'):
                    perf_paths.append(os.path.join(root, fn))
    return perf_paths


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX table(s) for best deployments.')
    sub = parser.add_subparsers(dest='mode')

    # Single-file mode (backward compatible)
    p_single = sub.add_parser('single', help='Generate a table for a single perf JSON and schedule YAML')
    p_single.add_argument('--perf_json', required=True, help='Path to performance JSON file')
    p_single.add_argument('--schedule_yaml', required=True, help='Path to schedule YAML file')
    p_single.add_argument('--output', '-o', default='', help='Output .tex file path (optional). If omitted, saves next to perf_json with _table.tex')

    # Batch mode
    p_batch = sub.add_parser('batch', help='Scan directories and group best deployments by model count into separate tables')
    p_batch.add_argument('--json_dirs', nargs='*', default=['./performance'], help='Directories to scan for performance JSON files')
    p_batch.add_argument('--yaml_dirs', nargs='*', default=['./tests'], help='Directories to search for schedule YAML files')
    p_batch.add_argument('--dirs', nargs='*', default=[], help='[Deprecated] Directories to scan for both JSON and YAML; if provided, used for both')
    p_batch.add_argument('--output_prefix', default='aggregated_best_deployments', help='Prefix for output .tex files')

    args = parser.parse_args()

    if args.mode == 'single' or (args.mode is None and hasattr(args, 'perf_json')):
        # Support calling without subcommand for backward compatibility
        perf_json = getattr(args, 'perf_json', None)
        schedule_yaml = getattr(args, 'schedule_yaml', None)
        output = getattr(args, 'output', '')
        if not perf_json or not schedule_yaml:
            parser.error('single mode requires --perf_json and --schedule_yaml')
        table_tex = generate_table(perf_json, schedule_yaml)
        print(table_tex)
        out_path = output or (os.path.splitext(perf_json)[0] + '_best_deployment_table.tex')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(table_tex)
        print(f"Saved LaTeX table to: {out_path}")
        return

    if args.mode == 'batch':
        # Determine directories: prefer explicit --json_dirs/--yaml_dirs; if --dirs is provided, use it for both
        json_dirs = args.json_dirs if getattr(args, 'json_dirs', None) else []
        yaml_dirs = args.yaml_dirs if getattr(args, 'yaml_dirs', None) else []
        if getattr(args, 'dirs', None):
            if not json_dirs:
                json_dirs = args.dirs
            if not yaml_dirs:
                yaml_dirs = args.dirs
        if not json_dirs:
            json_dirs = ['./performance']
        if not yaml_dirs:
            yaml_dirs = ['./tests']
        perf_paths = _discover_perf_jsons(json_dirs)
        tables = generate_grouped_tables(perf_paths, yaml_dirs)
        if not tables:
            print('No tables generated (no valid performance JSONs found).')
            return
        for mcount, tex in sorted(tables.items()):
            out_path = f"{args.output_prefix}_{mcount}models.tex"
            print(tex)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(tex)
            print(f"Saved LaTeX table (model_count={mcount}) to: {out_path}")
        return

    # If no mode provided, show help
    parser.print_help()


if __name__ == '__main__':
    main()

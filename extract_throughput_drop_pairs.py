#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys


def parse_args():
    p = argparse.ArgumentParser(description=(
        "Extract (total_throughput_fps, drop_count) pairs from a performance JSON and "
        "print them as 'xx.xx yyy \\' lines; also writes them to file(s).\n"
        "If combinations contain different numbers of models, results are grouped and "
        "written per model-count to <input_basename>_<N>models_throughput_drop.txt."
    ))
    p.add_argument("input", help="Path to performance JSON file (e.g., results/performance_YYYYMMDD_HHMMSS.json)")
    p.add_argument("-o", "--output", help="Output text file path. If all combinations have the same model count, this single file is used; consider including the model count in your filename. Otherwise ignored.")
    p.add_argument("--output-dir", help="Directory to place output files (defaults to input file's directory)")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 1

    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: failed to parse JSON: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
        print("Error: JSON format unexpected: missing top-level 'data' list", file=sys.stderr)
        return 1

    # Group lines by number of models in the combination
    grouped = {}  # model_count -> list[str]
    for item in data["data"]:
        try:
            total_thr = float(item["total"]["total_throughput_fps"])
            drop_cnt = int(item["derived"]["drop_count"])
            models_field = item.get("models", {})
            model_count = len(models_field) if isinstance(models_field, dict) else None
        except (KeyError, TypeError, ValueError):
            # Skip malformed entries
            continue
        if not model_count:
            # If model count is unavailable, skip entry to avoid mis-grouping
            continue
        line = f"{total_thr:.2f} {drop_cnt} \\\\"  # produces: 24.89 177 \
        grouped.setdefault(model_count, []).append(line)

    # Print to stdout in deterministic order by model_count
    for mc in sorted(grouped):
        for line in grouped[mc]:
            print(line)

    # Determine output directory
    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent

    # Decide output files
    if len(grouped) == 0:
        print("Warning: no valid entries found to write.", file=sys.stderr)
    elif len(grouped) == 1:
        # Single group: honor --output if provided, else use <basename>_<N>models_throughput_drop.txt
        (mc, lines) = next(iter(grouped.items()))
        if args.output:
            # Ensure model count appears in filename even when a custom output path is provided
            custom = Path(args.output)
            if custom.is_dir():
                out_path = custom / f"{in_path.stem}_{mc}models_throughput_drop.txt"
            else:
                # If user passed a file path, append model-count suffix before extension
                stem = custom.stem
                suffix = ''.join(custom.suffixes)
                out_path = custom.with_name(f"{stem}_{mc}models{suffix if suffix else ''}")
        else:
            out_path = out_dir / f"{in_path.stem}_{mc}models_throughput_drop.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        # Multiple groups: ignore --output (to avoid ambiguity), write one file per model count
        for mc, lines in grouped.items():
            out_path = out_dir / f"{in_path.stem}_{mc}models_throughput_drop.txt"
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

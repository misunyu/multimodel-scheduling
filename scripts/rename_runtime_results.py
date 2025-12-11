#!/usr/bin/env python3
"""
Rename runtime_test performance result files to include the schedule filename.

Existing runtime_test files look like:
  xgboost_model/performance_results/runtime_test/performance_YYYYMMDD_HHMMSS.json

This script renames them to include the schedule yaml stem found at the top of the file:
  xgboost_model/performance_results/runtime_test/performance_YYYYMMDD_HHMMSS_<schedule_yaml_stem>.json

Example:
  { "schedule file": "model_schedules_m_r_r_y.yaml", ... }
  → performance_20251211_152025_model_schedules_m_r_r_y.json

Usage:
  python scripts/rename_runtime_results.py \
    --dir xgboost_model/performance_results/runtime_test \
    --dry-run

Options:
  --dir     Directory to scan. Defaults to xgboost_model/performance_results/runtime_test
  --dry-run Show what would be renamed without actually renaming files
  --yes     Proceed without confirmation prompt

Notes:
  - Files already containing a schedule name will be skipped.
  - If the target filename exists, a numeric suffix _1, _2, ... is appended to avoid overwrite.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RUNTIME_DIR_DEFAULT = Path("xgboost_model/performance_results/runtime_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", dest="dir", type=Path, default=RUNTIME_DIR_DEFAULT,
                   help=f"Directory with runtime_test results (default: {RUNTIME_DIR_DEFAULT})")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="Only print planned renames")
    p.add_argument("--yes", dest="assume_yes", action="store_true", help="Do not prompt for confirmation")
    return p.parse_args()


FNAME_RE = re.compile(r"^performance_(\d{8}_\d{6})(?:_.+)?\.json$")


def extract_timestamp(name: str) -> str | None:
    m = FNAME_RE.match(name)
    return m.group(1) if m else None


def safe_with_suffix(dst: Path) -> Path:
    if not dst.exists():
        return dst
    base = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        cand = parent / f"{base}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def needs_rename(path: Path) -> bool:
    # Skip non-json and non-matching names
    if path.suffix != ".json":
        return False
    if not FNAME_RE.match(path.name):
        return False
    # If it already contains a schedule stem, we assume it's already renamed (has an extra _ after timestamp)
    # e.g., performance_YYYYMMDD_HHMMSS_model_schedules_...json
    parts = path.stem.split("_")
    return len(parts) == 3  # ['performance', 'YYYYMMDD', 'HHMMSS'] → actually stem is 'performance_YYYYMMDD_HHMMSS'


def get_schedule_stem(json_path: Path) -> str | None:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    # Try a few key variants
    for key in ("schedule file", "schedule_file", "schedule", "schedule yaml"):
        if key in data and isinstance(data[key], str):
            stem = Path(data[key]).stem
            return stem
    return None


def main() -> int:
    args = parse_args()
    root: Path = args.dir
    if not root.exists() or not root.is_dir():
        print(f"Directory does not exist: {root}")
        return 2

    candidates = [p for p in sorted(root.iterdir()) if needs_rename(p)]
    if not candidates:
        print("No runtime result files that need renaming were found.")
        return 0

    planned: list[tuple[Path, Path]] = []
    skipped_missing_schedule: list[Path] = []

    for src in candidates:
        ts = extract_timestamp(src.name)
        if not ts:
            continue
        schedule_stem = get_schedule_stem(src)
        if not schedule_stem:
            skipped_missing_schedule.append(src)
            continue
        dst = src.with_name(f"performance_{ts}_{schedule_stem}.json")
        dst = safe_with_suffix(dst)
        if dst == src:
            continue
        planned.append((src, dst))

    if not planned and not skipped_missing_schedule:
        print("Nothing to do. All files look already renamed.")
        return 0

    print("Planned renames:")
    for src, dst in planned:
        print(f"  {src.name} -> {dst.name}")
    if skipped_missing_schedule:
        print("\nSkipped (could not read schedule file from JSON):")
        for p in skipped_missing_schedule:
            print(f"  {p.name}")

    if args.dry_run:
        print("\nDry run: no files were renamed.")
        return 0

    if not args.assume_yes:
        reply = input("\nProceed with these renames? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted by user.")
            return 1

    for src, dst in planned:
        src.rename(dst)
    print(f"Done. Renamed {len(planned)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

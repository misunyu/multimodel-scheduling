#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
지정한 폴더(기본: 프로젝트의 results 폴더) 안의 모든 .json 파일을 확인하여,
각 파일의 JSON 내용 중 "schedule file" 값(예: "exp_2_model_schedules_1.yaml")의
베이스 이름(확장자 제거 및 경로 제거, 예: "exp_2_model_schedules_1")이
현재 파일 이름(확장자 제외)에 포함되어 있지 않으면 확장자 앞에 붙여서 파일명을 변경한다.

동작:
- 변경 시: "변경: <원본파일명> -> <수정된파일명>" 출력
- 미변경 시: "미변경: <파일명>" 출력
- 예외/오류 시: "오류: <파일명> - <사유>" 출력

사용법:
- 기본(프로젝트 results 폴더 기준):
    python fix_results_json_filenames.py
- 임의의 폴더를 지정:
    python fix_results_json_filenames.py /full/path/to/dir
    python fix_results_json_filenames.py ../some/relative/dir

참고:
- 입력 디렉토리는 절대/상대 경로, ~ (홈) 표기가 모두 가능합니다.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

DEFAULT_RESULTS_DIR = (Path(__file__).parent / "../results").resolve()


def ensure_schedule_in_filename(json_path: Path) -> None:
    """Check a single JSON file and rename it if needed.

    Rules:
    - Read JSON and get value at key "schedule file".
    - Extract schedule base name (basename without extension).
    - If base name not contained in current filename stem, rename to
      f"{stem}_{base}{suffix}".
    - Additionally, when renaming performance JSON files, append the execution
      window seconds ("window_sec") before the extension in the form
      "_w실행시간" (e.g., "_w30"). If the value is a whole number, omit
      the decimal part; otherwise keep it (e.g., "_w30.5").
    - If target path exists already, append numeric suffix "_1", "_2", ...
      to avoid overwrite.
    """
    try:
        if not json_path.is_file():
            print(f"오류: {json_path.name} - 파일이 아닙니다")
            return

        # Load JSON content
        with json_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"오류: {json_path.name} - JSON 파싱 실패: {e}")
                return

        if not isinstance(data, dict):
            print(f"오류: {json_path.name} - 최상위가 객체(JSON object)가 아닙니다")
            return

        if "schedule file" not in data:
            print(f"오류: {json_path.name} - 'schedule file' 키가 없습니다")
            return

        schedule_file_value = data["schedule file"]
        if not isinstance(schedule_file_value, str) or not schedule_file_value.strip():
            print(f"오류: {json_path.name} - 'schedule file' 값이 비어있거나 문자열이 아닙니다")
            return

        # Derive base name (without dir and extension)
        schedule_base = Path(schedule_file_value).name  # remove directories
        schedule_base = os.path.splitext(schedule_base)[0]  # remove extension

        # Try to extract window_sec (execution time window in seconds)
        def extract_window_sec_anywhere(obj) -> float | None:
            """Recursively search for a numeric 'window_sec' in dicts/lists."""
            try:
                if isinstance(obj, dict):
                    # direct key
                    if isinstance(obj.get("window_sec"), (int, float)):
                        return float(obj["window_sec"])
                    # common nesting
                    derived = obj.get("derived")
                    if isinstance(derived, dict) and isinstance(derived.get("window_sec"), (int, float)):
                        return float(derived["window_sec"])
                    # search all values
                    for v in obj.values():
                        val = extract_window_sec_anywhere(v)
                        if val is not None:
                            return val
                elif isinstance(obj, list):
                    for v in obj:
                        val = extract_window_sec_anywhere(v)
                        if val is not None:
                            return val
            except Exception:
                pass
            return None

        window_val = extract_window_sec_anywhere(data)

        def format_window(v: float | None) -> str | None:
            if v is None:
                return None
            # Normalize formatting: no trailing .0 for integer values
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return str(v)

        window_str = format_window(window_val)

        stem = json_path.stem
        suffix = json_path.suffix  # typically .json

        # Already contains schedule_base?
        if schedule_base in stem:
            # We only append window info when we are performing a rename due to
            # missing schedule base, to keep current behavior minimal.
            print(f"미변경: {json_path.name}")
            return

        # Build new name and avoid collisions
        new_stem = f"{stem}_{schedule_base}"

        # Append window seconds if available and not already present in the new stem
        if window_str and f"w{window_str}" not in new_stem:
            new_stem = f"{new_stem}_w{window_str}"

        target = json_path.with_name(new_stem + suffix)
        counter = 1
        while target.exists():
            target = json_path.with_name(f"{new_stem}_{counter}{suffix}")
            counter += 1

        # Rename
        json_path.rename(target)
        print(f"변경: {json_path.name} -> {target.name}")

    except Exception as e:
        print(f"오류: {json_path.name} - {e}")


def process_dir(dir_path: Path) -> None:
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"오류: 디렉토리를 찾을 수 없습니다: {dir_path}")
        return

    json_files = sorted(p for p in dir_path.iterdir() if p.suffix.lower() == ".json")
    if not json_files:
        print(f"정보: 지정한 디렉토리에 .json 파일이 없습니다: {dir_path}")
        return

    for json_path in json_files:
        ensure_schedule_in_filename(json_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="지정한 디렉토리 안의 .json 파일명을 'schedule file' 기반으로 정리합니다.")
    parser.add_argument(
        "dir",
        nargs="?",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"처리할 디렉토리 경로 (기본: {DEFAULT_RESULTS_DIR})",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Normalize the directory path: support ~ and relative paths
    dir_arg = Path(str(args.dir)).expanduser()
    if not dir_arg.is_absolute():
        dir_arg = (Path.cwd() / dir_arg).resolve()

    process_dir(dir_arg)


if __name__ == "__main__":
    main()

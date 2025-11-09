#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
results 폴더 안의 모든 .json 파일을 확인하여,
각 파일의 JSON 내용 중 "schedule file" 값(예: "exp_2_model_schedules_1.yaml")의
베이스 이름(확장자 제거 및 경로 제거, 예: "exp_2_model_schedules_1")이
현재 파일 이름(확장자 제외)에 포함되어 있지 않으면 확장자 앞에 붙여서 파일명을 변경한다.

동작:
- 변경 시: "변경: <원본파일명> -> <수정된파일명>" 출력
- 미변경 시: "미변경: <파일명>" 출력
- 예외/오류 시: "오류: <파일명> - <사유>" 출력

사용법:
- 프로젝트 루트에서 실행: python fix_results_json_filenames.py
"""

from __future__ import annotations
import json
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "../results"


def ensure_schedule_in_filename(json_path: Path) -> None:
    """Check a single JSON file and rename it if needed.

    Rules:
    - Read JSON and get value at key "schedule file".
    - Extract schedule base name (basename without extension).
    - If base name not contained in current filename stem, rename to
      f"{stem}_{base}{suffix}".
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

        stem = json_path.stem
        suffix = json_path.suffix  # typically .json

        # Already contains schedule_base?
        if schedule_base in stem:
            print(f"미변경: {json_path.name}")
            return

        # Build new name and avoid collisions
        new_stem = f"{stem}_{schedule_base}"
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


def main() -> None:
    if not RESULTS_DIR.exists() or not RESULTS_DIR.is_dir():
        print(f"오류: results 디렉토리를 찾을 수 없습니다: {RESULTS_DIR}")
        return

    json_files = sorted(p for p in RESULTS_DIR.iterdir() if p.suffix.lower() == ".json")
    if not json_files:
        print("정보: results 디렉토리에 .json 파일이 없습니다")
        return

    for json_path in json_files:
        ensure_schedule_in_filename(json_path)


if __name__ == "__main__":
    main()

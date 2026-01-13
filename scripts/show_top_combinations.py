#!/usr/bin/env python3
"""
성능 JSON 파일을 읽어 score 상위 N개 combination을 출력하는 유틸 스크립트.

사용 예시:
  python scripts/show_top_combinations.py \
    /home/msyu/PycharmProjects/multimodel-scheduling/xgboost_model/test/performance_results/prediction/predict_performance_20251215_150050_model_schedules_g_r_r.json

옵션:
  --top N  출력할 상위 개수 (기본값: 5)
"""

import argparse
import json
import sys
from typing import Any, Dict, List


def pick_score(item: Dict[str, Any]) -> float:
    """항목에서 점수를 추출한다. score가 없으면 total.total_throughput_fps를 사용."""
    score = item.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    total = item.get("total", {})
    return float(total.get("total_throughput_fps", float("nan")))


def get_float(d: Dict[str, Any], *keys: str) -> float:
    """중첩 dict에서 숫자 값을 안전하게 꺼낸다. 없거나 숫자가 아니면 NaN 반환."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return float("nan")
        cur = cur.get(k)
    return float(cur) if isinstance(cur, (int, float)) else float("nan")


def fmt(value: float) -> str:
    """값이 유효한 숫자면 소수점 둘째 자리까지, 아니면 '-'로 포맷."""
    try:
        if value != value:  # NaN 체크
            return "-"
        return f"{value:.2f}"
    except Exception:
        return "-"


def load_data(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}", file=sys.stderr)
        sys.exit(1)

    data = obj.get("data")
    if not isinstance(data, list) or len(data) == 0:
        print("유효한 'data' 항목이 없습니다.", file=sys.stderr)
        sys.exit(1)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="성능 JSON 파일에서 score 상위 N개 combination을 출력"
    )
    parser.add_argument("path", help="성능 JSON 파일 경로")
    parser.add_argument(
        "--top", type=int, default=5, help="출력할 상위 개수 (기본값: 5)"
    )
    args = parser.parse_args()

    data = load_data(args.path)

    # 정렬: score 내림차순, tie-breaker로 combination 이름 오름차순
    def sort_key(item: Dict[str, Any]):
        name = str(item.get("combination", ""))
        score = pick_score(item)
        return (-score, name)

    sorted_items = sorted(data, key=sort_key)
    top_n = sorted_items[: args.top]

    if len(top_n) == 0:
        print("표시할 항목이 없습니다.")
        return

    print("Top combinations by score:")
    for idx, item in enumerate(top_n, start=1):
        name = item.get("combination", "<unknown>")
        score = pick_score(item)
        total_tput = get_float(item, "total", "total_throughput_fps")
        avg_tput = get_float(item, "total", "avg_throughput_fps")
        drop_rate = get_float(item, "derived", "drop_rate_fps")

        # 공백 구분, 값이 없으면 '-' 표시, 소수점 둘째 자리 고정
        print(
            f"{idx}. {name} "
            f"Score: {fmt(score)} "
            f"Total: {fmt(total_tput)} "
            f"Avg: {fmt(avg_tput)} "
            f"Drop: {fmt(drop_rate)}"
        )


if __name__ == "__main__":
    main()

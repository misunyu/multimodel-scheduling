#!/usr/bin/env python3
"""
Prepare ONLY train dataset (a sampled subset) for XGBoost from ./performance JSON files.

This script:
- Loads all JSON files under ./performance (or given --perf_dir)
- Concatenates their "data" entries
- Randomly samples a subset (by --fraction and/or --max_items)
- Saves the result to:
    - xgboost_model/train_data/merged_train.json  (by default)
  Optionally splits into multiple chunks with --chunk size.

Usage examples:
  # Take 30% of all items (deterministic with seed) and save to train_data
  python -m xgboost_model.prepare_train_from_perf \
      --perf_dir ./performance \
      --out_train_dir xgboost_model/train_data \
      --fraction 0.3 \
      --seed 42

  # Cap to maximum 500 items regardless of fraction
  python -m xgboost_model.prepare_train_from_perf --max_items 500

After generation, you can train with:
  python xgboost_model/deploy_selector_xgb_suite.py train \
      --perf_glob "xgboost_model/train_data/*.json" \
      --out_dir xgboost_model/artifacts
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any


essential_fields = [
    "timestamp",
    "window_sec",
    "combination",
    "models",
    "total",
    "devices",
    "derived",
    "score",
]


def load_all_items(perf_glob: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    files = sorted(glob.glob(perf_glob))
    for fp in files:
        try:
            with open(fp, "r") as f:
                doc = json.load(f)
            data = doc.get("data", [])
            if not isinstance(data, list):
                continue
            items.extend(data)
        except Exception as e:
            print(f"WARN: failed to read {fp}: {e}")
    return items


def filter_minimal_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only essential fields widely used by training utilities."""
    return {k: item[k] for k in essential_fields if k in item}


def main():
    ap = argparse.ArgumentParser(description="Prepare a sampled train-only dataset from performance JSONs")
    ap.add_argument("--perf_dir", default="./performance", help="Directory containing performance JSON files")
    ap.add_argument("--out_train_dir", default="xgboost_model/train_data", help="Output directory for train JSONs")
    ap.add_argument("--fraction", type=float, default=0.3, help="Fraction of total items to sample (0~1]. If <=0, use max_items only.")
    ap.add_argument("--max_items", type=int, default=0, help="Upper bound on number of sampled items (0: no cap)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--keep_full", action="store_true", help="Keep full item fields instead of minimal subset")
    ap.add_argument("--chunk", type=int, default=0, help="If >0, split output into multiple chunk files with given chunk size")
    args = ap.parse_args()

    perf_glob = os.path.join(args.perf_dir, "*.json")
    all_items = load_all_items(perf_glob)

    if not all_items:
        raise SystemExit(f"No items found under {perf_glob}")

    # Optionally reduce fields to minimize size and ensure compatibility
    if not args.keep_full:
        all_items = [filter_minimal_fields(it) for it in all_items]

    # Deterministic sampling
    rnd = random.Random(args.seed)
    n_total = len(all_items)

    if args.fraction is not None and args.fraction > 0:
        n_sample = max(1, int(round(n_total * min(args.fraction, 1.0))))
    else:
        n_sample = n_total

    if args.max_items and args.max_items > 0:
        n_sample = min(n_sample, args.max_items)

    indices = list(range(n_total))
    rnd.shuffle(indices)
    pick_idx = sorted(indices[:n_sample])
    train_items = [all_items[i] for i in pick_idx]

    out_train = Path(args.out_train_dir)
    out_train.mkdir(parents=True, exist_ok=True)

    def write_json(path: Path, data_items: List[Dict[str, Any]]):
        obj = {
            "best deployment": None,
            "schedule file": None,
            "data": data_items,
        }
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    if args.chunk and args.chunk > 0:
        # Split into multiple files of given chunk size
        total = len(train_items)
        idx = 0
        chunk_id = 1
        while idx < total:
            part = train_items[idx: idx + args.chunk]
            out_path = out_train / f"merged_train_part{chunk_id:03d}.json"
            write_json(out_path, part)
            idx += args.chunk
            chunk_id += 1
    else:
        write_json(out_train / "merged_train.json", train_items)

    print(json.dumps({
        "n_total": n_total,
        "n_train_sampled": len(train_items),
        "fraction": args.fraction,
        "max_items": args.max_items,
        "train_dir": str(out_train.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

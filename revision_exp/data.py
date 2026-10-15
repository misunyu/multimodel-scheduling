"""Data loading for the CPU-GPU retraining pipeline.

Reads the existing dataset CSVs unchanged. Features come from the existing builder
(xgboost_model/deploy_selector_xgb_suite.py: featurize_window), which is imported, not copied.
"""
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "revision_exp"
OUT = EXP / "out"
DS = REPO / "xgboost_model" / "dataset"
ALPHA = 0.3
MASTER_SEED = 42

SETS = {
    "train": ("gpu/train_x3.csv", "gpu/train_schedules_x3.csv", ["results_recompute"]),
    "x3": ("gpu/test_x3.csv", "gpu/test_schedules_x3.csv", ["results_recompute"]),
    "x1-5": ("gpu_additional/test_x1-5.csv", "gpu_additional/test_schedules_x1-5.csv", ["results_recompute_additional"]),
    "x2-5": ("gpu_additional/test_x2-5.csv", "gpu_additional/test_schedules_x2-5.csv", ["results_recompute_additional"]),
    "x3-5": ("gpu_additional/test_x3-5.csv", "gpu_additional/test_schedules_x3-5.csv", ["results_recompute_additional"]),
}

# columns that must never appear among the features
FORBIDDEN_FEATURE_TOKENS = ("throughput", "drop", "score", "norm", "inference_count", "latency", "avg_", "wait", "_fps",
                            "actual", "pred", "oracle", "regret", "top1", "top5")
ALLOWED_FEATURE_RE = re.compile(
    r"^(pairhash_\d+_on_(cpu|gpu|npu0|npu1)"
    r"|views\.count\.views"
    r"|views\.(sum|mean|max)\.(model_hash_\d+(_on_(cpu|gpu|npu0|npu1))?|view\.exec_(cpu|gpu|npu0|npu1)|view\.infps(_on_(cpu|gpu|npu0|npu1))?))$")

_SUITE = None
MANIFEST = {}  # path -> (bytes, sha256), filled by every reader in this package


def suite():
    """Import the existing module once (never copied). Bytecode writing is disabled."""
    global _SUITE
    if _SUITE is None:
        os.environ.setdefault("MPLBACKEND", "Agg")
        p = str(REPO / "xgboost_model")
        if p not in sys.path:
            sys.path.insert(0, p)
        import deploy_selector_xgb_suite as s  # noqa: E402
        _SUITE = s
        note_input(REPO / "xgboost_model" / "deploy_selector_xgb_suite.py")
    return _SUITE


def note_input(path):
    path = Path(path)
    key = str(path.relative_to(REPO)) if path.is_absolute() else str(path)
    if key not in MANIFEST:
        h = hashlib.sha256()
        with open(REPO / key, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        MANIFEST[key] = (os.path.getsize(REPO / key), h.hexdigest())
    return REPO / key


def write_manifest():
    rows = [{"path": k, "bytes": v[0], "sha256": v[1]} for k, v in sorted(MANIFEST.items())]
    pd.DataFrame(rows).to_csv(OUT / "input_manifest.csv", index=False)
    return len(rows)


def comb_num(name):
    m = re.search(r"(\d+)$", str(name))
    if not m:
        raise ValueError(f"combination id without a number: {name!r}")
    return int(m.group(1))


def family_key(schedule_file):
    """Schedule file name without the rate suffix (_x2, _x3, _x1-5, ...). 1x files have no suffix."""
    base = os.path.basename(schedule_file)
    return re.sub(r"(_x\d(-\d)?)?\.yaml$", "", base)


def rate_key(schedule_file):
    m = re.search(r"_x(\d(?:-\d)?)\.yaml$", os.path.basename(schedule_file))
    return ("x" + m.group(1)) if m else "x1"


def _source_index(dirs):
    """(schedule_file, combination, timestamp) -> result JSON it came from (for the dedup audit trail)."""
    idx = {}
    for d in dirs:
        for p in sorted(glob.glob(str(REPO / d / "*.json"))):
            doc = json.load(open(p))
            rel = str(Path(p).relative_to(REPO))
            for e in doc.get("data", []):
                idx[(doc.get("schedule file"), e.get("combination"), e.get("timestamp"))] = rel
    return idx


def load_set(name, dedup=True):
    """Return (df, dropped). One row per measured candidate; df['_x'] holds the feature dict from featurize_window.

    Duplicated (group, combination) pairs from re-run files: the row with the later run timestamp is kept.
    If a pair cannot be ordered by timestamp the function raises instead of choosing.
    """
    s = suite()
    perf_rel, sched_rel, src_dirs = SETS[name]
    perf_csv = note_input(DS / perf_rel)
    sched_csv = note_input(DS / sched_rel)
    sched_index = s._index_schedules_from_csv(sched_csv)
    raw = pd.read_csv(perf_csv)
    rows = []
    for i, r in raw.iterrows():
        w = json.loads(r["json_content"])
        sf = w.get("schedule_file") or w.get("schedule file")
        comb = w.get("combination")
        ts = w.get("timestamp")
        s_doc = s._find_schedule(sched_index, sf)
        if s_doc is None:
            raise RuntimeError(f"[{name}] schedule content missing for {sf}")
        infps_map = s._build_infps_lookup(s_doc, comb)
        x, (y1, y2), _meta = s.featurize_window(w, infps_map)
        if not (np.isfinite(y1) and np.isfinite(y2)):
            raise RuntimeError(f"[{name}] non-finite label at csv row {i}")
        devs = [str(m.get("execution", "")).upper() for m in w["models"].values()]
        rows.append({
            "row_uid": f"{sf}|{comb}|{ts}", "set": name, "csv_row": i,
            "schedule_file": sf, "group": sf, "group_id_legacy": s.get_group_id(w),
            "family": family_key(sf), "rate": rate_key(sf),
            "combination": comb, "comb_num": comb_num(comb), "timestamp": ts,
            "N": len(w.get("models", {})),
            "placement": ";".join(f"{m.get('model')}@{str(m.get('execution')).upper()}" for m in w["models"].values()),
            "n_gpu": sum(d == "GPU" for d in devs), "n_cpu": sum(d == "CPU" for d in devs),
            "all_gpu": all(d == "GPU" for d in devs), "all_cpu": all(d == "CPU" for d in devs),
            "T": float(y1), "D": float(y2), "_x": x,
        })
    df = pd.DataFrame(rows)
    assert df["row_uid"].is_unique, f"[{name}] row_uid not unique"
    # group key must partition the rows exactly as get_group_id does
    a = df.groupby("group")["group_id_legacy"].nunique()
    b = df.groupby("group_id_legacy")["group"].nunique()
    assert (a == 1).all() and (b == 1).all(), f"[{name}] schedule-file key differs from get_group_id partition"
    assert df.groupby("group")["N"].nunique().eq(1).all()

    dropped = pd.DataFrame()
    dup = df.duplicated(["group", "combination"], keep=False)
    df["n_runs_for_candidate"] = df.groupby(["group", "combination"])["row_uid"].transform("size")
    if dup.any():
        src = _source_index(src_dirs)
        recs = []
        keep_mask = np.ones(len(df), dtype=bool)
        for (g, c), sub in df[dup].groupby(["group", "combination"]):
            ts = pd.to_datetime(sub["timestamp"], errors="coerce")
            if ts.isna().any() or ts.nunique() != len(sub):
                raise RuntimeError(f"[{name}] cannot order duplicate runs by timestamp for ({g}, {c}); stopping as instructed")
            keep_idx = ts.idxmax()
            for idx in sub.index:
                if idx != keep_idx:
                    keep_mask[df.index.get_loc(idx)] = False
                    recs.append({
                        "set": name, "group": g, "combination": c,
                        "kept_row_uid": df.at[keep_idx, "row_uid"], "kept_timestamp": df.at[keep_idx, "timestamp"],
                        "kept_source_file": src.get((g, c, df.at[keep_idx, "timestamp"]), "NOT FOUND"),
                        "kept_T": df.at[keep_idx, "T"], "kept_D": df.at[keep_idx, "D"],
                        "dropped_row_uid": df.at[idx, "row_uid"], "dropped_timestamp": df.at[idx, "timestamp"],
                        "dropped_source_file": src.get((g, c, df.at[idx, "timestamp"]), "NOT FOUND"),
                        "dropped_T": df.at[idx, "T"], "dropped_D": df.at[idx, "D"],
                        "reason": "later run timestamp kept",
                    })
        dropped = pd.DataFrame(recs)
        # which run each row belongs to (so that groups mixing two normalisations can be reported)
        df["source_file"] = [src.get((g, c, t), "NOT FOUND") for g, c, t in zip(df["group"], df["combination"], df["timestamp"])]
        if dedup:
            df = df[keep_mask].reset_index(drop=True)
    if "source_file" not in df.columns:
        df["source_file"] = ""
    df = df.sort_values(["group", "comb_num", "timestamp"], kind="stable").reset_index(drop=True)
    return df, dropped


def build_features(df, columns=None):
    """Feature matrix aligned to df (index = row_uid). Test sets are re-indexed to the training columns with 0 fill,
    exactly as the existing predict path does."""
    X = pd.DataFrame(list(df["_x"])).fillna(0.0)
    X.index = df["row_uid"].values
    if columns is None:
        columns = sorted(X.columns)  # same ordering rule as train_double()
    X = X.reindex(columns=columns, fill_value=0.0)
    assert len(X) == len(df) and (X.index == df["row_uid"].values).all(), "row alignment broken"
    bad = [c for c in X.columns if not ALLOWED_FEATURE_RE.match(c) or any(t in c.lower() for t in FORBIDDEN_FEATURE_TOKENS)]
    assert not bad, f"leakage / unexpected feature columns: {bad[:5]}"
    assert np.isfinite(X.values).all()
    return X


def check_against_existing_builder(name, df_nodedup):
    """The row-wise call above must reproduce build_dataset_from_csv() exactly (same rows, same values)."""
    s = suite()
    perf_rel, sched_rel, _ = SETS[name]
    X0, Y0, M0 = s.build_dataset_from_csv(DS / perf_rel, None, DS / sched_rel)
    d = df_nodedup.sort_values("csv_row")
    X1 = pd.DataFrame(list(d["_x"])).fillna(0.0)
    cols = sorted(set(X0.columns) | set(X1.columns))
    same_x = np.array_equal(X0.reindex(columns=cols, fill_value=0.0).values, X1.reindex(columns=cols, fill_value=0.0).values)
    same_y = np.allclose(Y0["y1"].values, d["T"].values) and np.allclose(Y0["y2"].values, d["D"].values)
    same_g = list(M0["group_id"]) == list(d["group_id_legacy"])
    return {"set": name, "rows_existing_builder": len(X0), "rows_this_loader": len(X1), "features_equal": bool(same_x),
            "labels_equal": bool(same_y), "group_ids_equal": bool(same_g)}


def summarize(name, df_nodedup, df, dropped):
    g = df.groupby("group")
    ge3 = df[df["N"] >= 3]
    return {
        "set": name, "rows_csv": len(df_nodedup), "rows_after_dedup": len(df), "rows_dropped": len(dropped),
        "groups": g.ngroups, "families": df["family"].nunique(), "rates": " ".join(sorted(df["rate"].unique())),
        "groups_N>=3": ge3["group"].nunique(), "rows_N>=3": len(ge3),
        "groups_le5_candidates_N>=3": int((ge3.groupby("group").size() <= 5).sum()),
        "groups_le5_candidates_all": int((g.size() <= 5).sum()),
        "min_candidates_N>=3": int(ge3.groupby("group").size().min()) if len(ge3) else 0,
        "groups_without_all_gpu_N>=3": int((~ge3.groupby("group")["all_gpu"].any()).sum()) if len(ge3) else 0,
        "groups_mixing_runs_after_dedup": int((df[df["source_file"] != ""].groupby("group")["source_file"].nunique() > 1).sum()),
    }

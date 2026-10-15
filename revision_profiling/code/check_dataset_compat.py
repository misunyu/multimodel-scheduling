"""Read a revision_profiling dataset folder with the UNMODIFIED revision_exp/data.py loader and existing feature builder.

  python3 -B code/check_dataset_compat.py --data dataset_pilot_check --report dataset_pilot_check/compat_check.json
  python3 -B code/check_dataset_compat.py --data dataset --report dataset/compat_check.json --final

Nothing is trained or evaluated here: parsing, feature building and consistency checks only.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(REPO / "revision_exp"))
import data as D  # noqa: E402  (revision_exp/data.py, read-only)

LABEL_KEYS = ("completed", "dropped", "generated", "q_i", "latency_p50_ms", "throughput_norm", "drop_rate_norm", "drop_agg_norm", "drop_max",
              "drop_rate_fps", "throughput_fps", "F_max", "R_max")


def point_loader_at(data_dir):
    D.DS = Path(data_dir)
    D.SETS = {"train": ("train_x3.csv", "train_schedules_x3.csv", []), "x3": ("test_x3.csv", "test_schedules_x3.csv", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--final", action="store_true", help="also require the 1,2,4x / 3x split and complete workloads")
    a = ap.parse_args()
    data_dir = ROOT / a.data
    point_loader_at(data_dir)
    s = D.suite()
    rep, checks = {"data": a.data, "sets": {}}, {}
    sets = {}
    for name in ("train", "x3"):
        raw = pd.read_csv(data_dir / D.SETS[name][0])
        assert list(raw.columns) == ["schedule_file", "combination", "json_content"], list(raw.columns)
        df, dropped = D.load_set(name, dedup=True)          # raises on missing schedule / non-finite label / key mismatch
        lab = pd.read_csv(data_dir / f"labels_{'train' if name == 'train' else 'test'}.csv")
        m = df.merge(lab, on=["schedule_file", "combination"], how="left", validate="one_to_one", suffixes=("", "_lab"))
        sets[name] = df
        # infps the builder sees must equal the scheduled rate of every instance (otherwise the rate features are silently 0)
        sched_index = s._index_schedules_from_csv(data_dir / D.SETS[name][1])
        inst = pd.read_csv(data_dir / "labels_instances_all.csv")
        lam = {(r.family + "_" + r.rate + ".yaml", r.combination, r.model, str(r.device).upper()): r.lambda_i for r in inst.itertuples()}
        n_inf, bad_inf = 0, 0
        for r in df.itertuples():
            mp = s._build_infps_lookup(s._find_schedule(sched_index, r.schedule_file), r.combination)
            for (model, dev), v in mp.items():
                n_inf += 1
                bad_inf += not np.isclose(v, lam[(r.schedule_file, r.combination, model, dev)])
            bad_inf += len(mp) != r.N
        # placement string of the loader against the G/C mask (instance order = sorted model names, bit i = instance i)
        bad_place = 0
        for r in m.itertuples():
            devs = [x.split("@")[1] for x in r.placement.split(";")]
            names = [x.split("@")[0] for x in r.placement.split(";")]
            mask = "".join("G" if d == "GPU" else "C" for d in devs)
            num = sum((1 << i) for i, d in enumerate(devs) if d == "GPU")
            bad_place += (mask != r.placement_lab) or (num != r.comb_num) or (names != sorted(names)) or ("+".join(names) != r.family)
        info = {"csv_rows": len(raw), "rows_loaded": len(df), "rows_dropped_as_duplicates": len(dropped), "groups": int(df["group"].nunique()),
                "families": int(df["family"].nunique()), "rates": sorted(df["rate"].unique()),
                "dtypes": {c: str(raw[c].dtype) for c in raw.columns},
                "T_equals_throughput_norm": bool(np.allclose(m["T"], m["throughput_norm"])),
                "D_equals_D_macro": bool(np.allclose(m["D"], m["D_macro_mean"])),
                "max_abs_D_minus_D_macro": float(np.abs(m["D"] - m["D_macro_mean"]).max()),
                "rows_where_D_macro_differs_from_D_agg": int((~np.isclose(m["D_macro_mean"], m["drop_agg_norm"])).sum()),
                "derived_has_drop_agg_norm_and_drop_max": bool(all(("drop_agg_norm" in json.loads(j)["derived"]) and ("drop_max" in json.loads(j)["derived"])
                                                                   for j in raw["json_content"])),
                "infps_values_checked": n_inf, "infps_mismatches": int(bad_inf), "placement_or_combination_mismatches": int(bad_place),
                "group_key_1to1_with_get_group_id": True,   # asserted inside load_set
                "group_equals_family_rate": bool((df["group"] == df["family"] + "_" + df["rate"] + ".yaml").all()),
                "N>=3_groups": int(df[df["N"] >= 3]["group"].nunique()),
                "candidates_per_group_equal_2^N": bool((df.groupby("group").size() == 2 ** df.groupby("group")["N"].first()).all())}
        rep["sets"][name] = info
    X = D.build_features(sets["train"])                     # asserts: allowed names only, no forbidden token, finite
    cols = list(X.columns)
    Xt = D.build_features(sets["x3"], cols)
    old = (REPO / "revision_exp" / "out" / "feature_columns.txt").read_text().split()
    test_only = sorted(set(pd.DataFrame(list(sets["x3"]["_x"])).columns) - set(cols))
    leak = [c for c in cols if any(k.lower() in c.lower() for k in LABEL_KEYS)]
    rep["features"] = {"new_columns": len(cols), "old_columns": len(old), "common": len(set(cols) & set(old)),
                       "only_in_old": sorted(set(old) - set(cols)), "only_in_new": sorted(set(cols) - set(old)),
                       "same_order_on_common": [c for c in cols if c in set(old)] == [c for c in old if c in set(cols)],
                       "test_only_feature_columns(dropped by reindex)": test_only,
                       "label_or_counter_names_in_features": leak, "all_zero_columns_train": [c for c in cols if not X[c].any()],
                       "rate_feature_nonzero": bool(X["views.sum.view.infps"].gt(0).all()),
                       "train_shape": list(X.shape), "test_shape": list(Xt.shape)}
    (ROOT / a.data / "feature_columns.txt").write_text("\n".join(cols) + "\n")
    tr, te = sets["train"], sets["x3"]
    rep["split"] = {"train_rates": sorted(tr["rate"].unique()), "test_rates": sorted(te["rate"].unique()),
                    "row_uid_disjoint": bool(set(tr["row_uid"]).isdisjoint(te["row_uid"])),
                    "groups_disjoint": bool(set(tr["group"]).isdisjoint(te["group"])),
                    "families_train": int(tr["family"].nunique()), "families_test": int(te["family"].nunique()),
                    "families_in_both": int(len(set(tr["family"]) & set(te["family"]))),
                    "duplicate_rows_train": int(tr.duplicated(["group", "combination"]).sum()), "duplicate_rows_test": int(te.duplicated(["group", "combination"]).sum()),
                    "workloads_train": int(tr["group"].nunique()), "workloads_test": int(te["group"].nunique()),
                    "workloads_test_N>=3": int(te[te["N"] >= 3]["group"].nunique()), "candidates_train": len(tr), "candidates_test": len(te)}
    ok = all(v["T_equals_throughput_norm"] and v["D_equals_D_macro"] and v["derived_has_drop_agg_norm_and_drop_max"] and v["infps_mismatches"] == 0
             and v["placement_or_combination_mismatches"] == 0 and v["group_equals_family_rate"] and v["rows_dropped_as_duplicates"] == 0
             for v in rep["sets"].values())
    ok = ok and not leak and rep["features"]["rate_feature_nonzero"] and rep["split"]["row_uid_disjoint"] and rep["split"]["groups_disjoint"]
    if a.final:
        ok = ok and rep["split"]["train_rates"] == ["x1", "x2", "x4"] and rep["split"]["test_rates"] == ["x3"] \
            and all(v["candidates_per_group_equal_2^N"] for v in rep["sets"].values())
    rep["pass"] = bool(ok)
    Path(ROOT / a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

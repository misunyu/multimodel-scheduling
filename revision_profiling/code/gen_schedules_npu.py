"""Schedules for the CPU-NPU board (Phase N1 sections 0, 2-3).

Candidate set: the 11 families and their combinations are read (read-only) from gen_schedules_antara/model_schedules_<f>.yaml
(base files; the _2x/_3x/_4x files are checked to hold the same placements). Each instance keeps its name and its variant
(resnet50_big, yolov3_small, ...); only the rate is replaced.
Independent check: all feasible placements are enumerated again (CPU: any number; npu0/npu1: at most one instance each)
and must equal the YAML set exactly, per family and in total (93).

Rate rule (placement and variant independent): lambda_i = beta * Cmax[model(i)] * k, model(i) in {resnet50, yolov3},
Cmax[m] = max over (device, variant) of the isolated closed-loop throughput (isolated/offline_profile.json).
Asserted per workload (family, k): every combination has the same lambda vector.

Registry key per instance: antara_<model>_cpu on the CPU (common ONNX), antara_<model>_<variant>_npu on npu0/npu1.

  python3 -B code/gen_schedules_npu.py --check                                  # set equality only, no output
  python3 -B code/gen_schedules_npu.py --family rbybys --beta 0.1 --mult 1 --out x.yaml [--combinations c1,c5]
  python3 -B code/gen_schedules_npu.py --single resnet50:big:npu0 --rate 2 --out x.yaml
  python3 -B code/gen_schedules_npu.py --closed resnet50:big:npu0 --out x.yaml   # isolated profile schedule
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import re
from pathlib import Path

import yaml

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parent.parent
SRC = REPO / "gen_schedules_antara"
PROFILE = REPO / "revision_profiling_npu" / "isolated" / "offline_profile.json"
NPUS = ("npu0", "npu1")
FAMILIES = ("rb", "rs", "yb", "ys", "rbrs", "rsyb", "ybys", "rbrsyb", "rbybys", "rsybys", "rbrsybys")


def split_name(inst_model):
    m = re.fullmatch(r"(resnet50|yolov3)_(big|small)", inst_model)
    assert m, inst_model
    return m.group(1), m.group(2)


def registry_key(model, variant, device):
    return f"antara_{model}_cpu" if device == "cpu" else f"antara_{model}_{variant}_npu"


def read_family(fam, suffix=""):
    """{combination: {instance name: device}} from the original YAML."""
    y = yaml.safe_load((SRC / f"model_schedules_{fam}{suffix}.yaml").read_text())
    out = {}
    for comb, insts in y.items():
        d = {}
        for _key, v in insts.items():
            split_name(v["model"])
            assert v["model"] not in d, (fam, comb, v["model"])
            d[v["model"]] = v["execution"]
        out[comb] = d
    return out, y


def enumerate_feasible(names):
    res = []

    def rec(i, cur, free):
        if i == len(names):
            res.append(dict(cur))
            return
        cur[names[i]] = "cpu"
        rec(i + 1, cur, free)
        for n in sorted(free):
            cur[names[i]] = n
            rec(i + 1, cur, free - {n})
        del cur[names[i]]

    rec(0, {}, set(NPUS))
    return res


def check_all():
    total, rows = 0, []
    for fam in FAMILIES:
        base, _ = read_family(fam)
        names = sorted(next(iter(base.values())))
        assert all(sorted(v) == names for v in base.values()), fam
        got = sorted(tuple(sorted(v.items())) for v in base.values())
        want = sorted(tuple(sorted(v.items())) for v in enumerate_feasible(names))
        assert got == want, f"{fam}: YAML placements != feasible set ({len(got)} vs {len(want)})"
        assert len(set(got)) == len(got), f"{fam}: duplicate placements"
        for m in ("_2x", "_3x", "_4x"):
            other, _ = read_family(fam, m)
            assert other == base, f"{fam}{m}: placements differ from the base file"
        rows.append((fam, len(names), len(got)))
        total += len(got)
    assert total == 93, total
    return rows, total


def cmax():
    prof = json.loads(PROFILE.read_text())
    return {m: prof["cmax"][m]["throughput_fps"] for m in ("resnet50", "yolov3")}


def build_family(fam, beta, mult, combos=None, cm=None):
    cm = cm or cmax()
    base, raw = read_family(fam)
    out = {}
    for comb, devs in base.items():
        if combos and comb not in combos:
            continue
        legacy = {v["model"]: v.get("infps") for v in raw[comb].values()}
        inst = {}
        for name in sorted(devs):
            m, v = split_name(name)
            dev = devs[name]
            rate = beta * cm[m] * mult
            inst[name] = {"model": registry_key(m, v, dev), "logical_model": m, "variant": v, "execution": dev,
                          "base_rate": cm[m], "beta": beta, "multiplier": mult, "rate": rate, "infps": rate,
                          "legacy_infps_x1": legacy[name]}
        out[comb] = {"placement": ",".join(f"{n}:{inst[n]['execution']}" for n in sorted(inst)), "instances": inst}
    # same lambda vector for every candidate of the workload
    vecs = {tuple((n, c["instances"][n]["rate"]) for n in sorted(c["instances"])) for c in out.values()}
    assert len(vecs) == 1, f"lambda vector differs across candidates of {fam} x{mult}: {vecs}"
    return {"meta": {"family": fam, "models": sorted(next(iter(base.values()))), "n_instances": len(next(iter(base.values()))),
                     "beta": beta, "multiplier": mult,
                     "rate_rule": "rate = beta * Cmax[model] * multiplier; Cmax = max isolated throughput over (device, variant); "
                                  "placement and variant independent",
                     "rate_source": "revision_profiling_npu/isolated/offline_profile.json",
                     "candidate_source": f"gen_schedules_antara/model_schedules_{fam}.yaml",
                     "candidates": len(base), "family_index": FAMILIES.index(fam)},
            "combinations": out}


def parse_mvd(s):
    m, v, d = s.split(":")
    return m, v, d


def single(spec, rate, closed=False):
    m, v, d = parse_mvd(spec)
    name = f"{m}_{v}"
    inst = {name: {"model": registry_key(m, v, d), "logical_model": m, "variant": v, "execution": d,
                   "base_rate": None, "beta": None, "multiplier": None, "rate": None if closed else rate,
                   "infps": None if closed else rate}}
    return {"meta": {"family": name, "models": [name], "n_instances": 1, "beta": None, "multiplier": None,
                     "rate_rule": "closed loop" if closed else f"fixed {rate} req/s (smoke)"},
            "combinations": {"combination_000": {"placement": f"{name}:{d}", "instances": inst}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--family")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--mult", type=float)
    ap.add_argument("--combinations")
    ap.add_argument("--single")
    ap.add_argument("--closed")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows, total = check_all()
    if a.check:
        for r in rows:
            print(*r)
        print("total", total, "OK")
        return
    if a.single:
        y = single(a.single, a.rate)
    elif a.closed:
        y = single(a.closed, None, closed=True)
    else:
        y = build_family(a.family, a.beta, a.mult, a.combinations.split(",") if a.combinations else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(y, sort_keys=False))


if __name__ == "__main__":
    main()

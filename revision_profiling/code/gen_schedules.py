"""Schedule generation (spec v2.2 section 2.7, D3): r_m = beta * GPU isolated throughput_m * multiplier, independent of placement.

Families are read from the existing gen_schedules/*.yaml (read-only): only the model list of each family is taken.
Combination id = bit mask over the sorted instance names (bit i = 1 -> instance i on GPU), all 2^N by default.

  python3 -B code/gen_schedules.py --family g_m_r_t --beta 0.5 --mult 2 --out schedules/x.yaml [--masks all_gpu,all_cpu,half]
  python3 -B code/gen_schedules.py --single --beta 0.5 --mult 1 --out-dir pilot/smoke/schedules
  python3 -B code/gen_schedules.py --list-families
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import re
from pathlib import Path

import yaml

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
REPO = ROOT.parent
PROFILE = ROOT / "offline_profile" / "offline_profile.json"


def existing_families():
    """family name -> sorted model list, from the base (1x) schedule files of the original campaign."""
    fam = {}
    for p in sorted((REPO / "gen_schedules").glob("model_schedules_*.yaml")):
        stem = p.stem[len("model_schedules_"):]
        if re.search(r"_x\d$", stem):
            continue
        y = yaml.safe_load(p.read_text())
        first = y[sorted(y)[0]]
        models = sorted(v["model"] for v in first.values())
        assert len(models) == len(set(models)), (stem, models)
        fam[stem] = models
    return fam


EXCLUDED_MODELS = ("tiny-llama-chat-onnx",)   # Phase 1b user decision


def reconstructed_families(exclude=EXCLUDED_MODELS):
    """Spec v2.2 section 3.4: drop excluded models; N = 0 -> removed; equal to an already kept family -> removed.
    Returns {key: (models, original family name)} with key = "+".join(sorted models)."""
    fam = existing_families()
    kept = {}
    for f, models in sorted(fam.items()):
        if not set(models) & set(exclude):
            kept[tuple(models)] = f
    for f, models in sorted(fam.items(), key=lambda kv: (len(kv[1]), kv[0])):
        red = tuple(m for m in models if m not in exclude)
        if set(models) & set(exclude) and red and red not in kept:
            kept[red] = f
    return {"+".join(k): (list(k), v) for k, v in sorted(kept.items(), key=lambda kv: (len(kv[0]), kv[0]))}


def gpu_throughput():
    prof = json.loads(PROFILE.read_text())
    return {m: v["gpu"]["throughput_fps"] for m, v in prof["models"].items() if v.get("gpu")}


def build(family, models, beta, mult, masks=None, profile=None):
    profile = profile or gpu_throughput()
    names = sorted(models)
    n = len(names)
    if masks is None:
        masks = list(range(2 ** n))
    combos = {}
    for mask in masks:
        inst = {}
        for i, m in enumerate(names):
            dev = "gpu" if (mask >> i) & 1 else "cpu"
            base = profile[m]
            rate = beta * base * mult
            inst[m] = {"model": m, "execution": dev, "base_rate": base, "beta": beta, "multiplier": mult,
                       "rate": rate, "infps": rate}
        combos[f"combination_{mask:03d}"] = {"placement": "".join("G" if (mask >> i) & 1 else "C" for i in range(n)), "instances": inst}
    return {"meta": {"family": family, "models": names, "n_instances": n, "beta": beta, "multiplier": mult,
                     "rate_rule": "rate = beta * gpu_isolated_throughput * multiplier (placement independent)",
                     "rate_source": "revision_profiling/offline_profile/offline_profile.json"},
            "combinations": combos}


def named_masks(n, spec):
    out = []
    for s in spec.split(","):
        if s == "all_gpu":
            out.append(2 ** n - 1)
        elif s == "all_cpu":
            out.append(0)
        elif s.startswith("rand"):   # randK: K masks without replacement from the mixed placements 1 .. 2^n-2, SeedSequence(42)
            import numpy as np
            rng = np.random.default_rng(np.random.SeedSequence(42))
            out += sorted(int(x) for x in rng.choice(np.arange(1, 2 ** n - 1), size=int(s[4:]), replace=False))
        elif s == "half":       # first half of the sorted instance names on GPU, rest on CPU
            out.append(2 ** (n // 2) - 1)
        else:
            out.append(int(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--models")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--mult", type=float, default=1)
    ap.add_argument("--masks")
    ap.add_argument("--out")
    ap.add_argument("--out-dir")
    ap.add_argument("--list-families", action="store_true")
    a = ap.parse_args()
    fam = {k: v[0] for k, v in reconstructed_families().items()}
    if a.list_families:
        for k, v in fam.items():
            print(len(v), k)
        return
    if a.single:
        prof = gpu_throughput()
        models = a.models.split(",") if a.models else sorted(prof)
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        for m in models:
            y = build(m, [m], a.beta, a.mult, named_masks(1, a.masks) if a.masks else None, profile=prof)
            (Path(a.out_dir) / f"single_{m}_b{a.beta}_x{a.mult:g}.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
        return
    models = fam[a.family]
    masks = named_masks(len(models), a.masks) if a.masks else None
    y = build(a.family, models, a.beta, a.mult, masks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(y, sort_keys=False))


if __name__ == "__main__":
    main()

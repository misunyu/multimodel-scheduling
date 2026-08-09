"""C (#10): 세트 단위 cluster bootstrap 으로 핵심 지표의 95% CI.

리샘플링 단위는 **워킹셋 15개**다. 한 세트는 rate 3수준 = 3그룹을 통째로 데리고
들어간다(그룹은 세트 안에서 독립이 아니므로 그룹 단위 리샘플링은 CI 를 과소추정한다).
B=10,000, seed=42. 백분위수 방식 95% CI.

대상 지표
  - direct(P1) 의 그룹평균 Spearman · Top-1 (플랫폼별)
  - divergence 11/45 의 비율
  - direct - greedy Top-1 차이 (paired: 같은 세트 리샘플)
  - direct - decomposed Top-1 차이 (paired)

기존 OOF 예측·측정값만 사용. 재학습 없음.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "xgboost_model" / "full_collection_540" / "scripts"))

import analysis_common as ac  # noqa: E402

AN = ac.ANALYSIS
B = 10000
SEED = 42
TIE = 1e-9

SETS = yaml.safe_load((ROOT / "working_sets.yaml").read_text())["sets"]
SID_OF = {",".join(sorted(v)): k for k, v in SETS.items()}
SET_IDS = sorted(SETS)

# ------------------------------------------------------- 그룹 단위 지표 테이블
recs = defaultdict(dict)   # (sid, rate) -> {metric: value}


def gkey(models_str, rate):
    return (SID_OF[",".join(sorted(models_str.split(",")))], float(rate))


# direct (P1): 그룹별 Top-1 과 Spearman
for pl in ac.PLATFORMS:
    o = pd.read_csv(AN / f"groupkfold_{pl}_oof.csv", comment="#")
    o["s_meas"] = ac.scores_from(o["y1_meas"], o["y2_meas"], o["y3_meas"], o["has_gen"])
    o["s_pred"] = ac.scores_from(o["y1_pred"], o["y2_pred"], o["y3_pred"], o["has_gen"])
    for (models, rate), g in o.groupby(["models", "rate_factor"], sort=False):
        k = gkey(models, rate)
        meas, pred = g["s_meas"].values, g["s_pred"].values
        best = meas.max()
        oracle = set(np.where(np.abs(meas - best) < TIE)[0])
        recs[k][f"direct_top1_{pl}"] = float(np.argsort(-pred, kind="stable")[0] in oracle)
        recs[k][f"direct_rho_{pl}"] = ac.spearman(meas, pred)

# decomposed (P6): F 의 per-group CSV 재사용
for pl in ac.PLATFORMS:
    df = pd.read_csv(HERE.parent / "F_regret" / f"per_group_decomposed_{pl}.csv")
    for _, r in df.iterrows():
        models, rate = eval(r["group"])
        recs[gkey(models, rate)][f"decomp_top1_{pl}"] = float(r["top1"])

# greedy (P5)
gr = pd.read_csv(AN / "greedy_results.csv", comment="#")
for _, r in gr.iterrows():
    k = (r["set"], float(r["rate"]))
    v = r["variant"]
    # 하한 관례: 미측정 산출은 실패(0)
    recs[k][f"greedy_{v}_top1_lb_{r['platform']}"] = float(r["top1"]) if r["collected"] == 1 else 0.0
    recs[k][f"greedy_{v}_collected_{r['platform']}"] = float(r["collected"])

# divergence (선언 기저, beta=1.0) — 감사 모듈 재사용
spec = importlib.util.spec_from_file_location(
    "basis_audit", ROOT / "runs/20260730_190154_score_basis_audit/build_basis_audit.py")
ba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ba)
rows_pl = {pl: ba.normalize(ba.load_rows(pl)) for pl in ("gpu", "npu")}
ties = {pl: ba.tie_sets(rows_pl[pl], 0.3, 1.0, "normalized") for pl in rows_pl}
for k in ties["gpu"]:
    recs[(k[0], float(k[1]))]["divergent"] = float(not (ties["gpu"][k] & ties["npu"][k]))

T = pd.DataFrame([{"set": k[0], "rate": k[1], **v} for k, v in recs.items()])
assert len(T) == 45, len(T)
T = T.sort_values(["set", "rate"]).reset_index(drop=True)
T.to_csv(HERE / "per_group_metrics.csv", index=False)

groups_by_set = {s: T.index[T["set"] == s].to_numpy() for s in SET_IDS}
assert all(len(v) == 3 for v in groups_by_set.values())

# ------------------------------------------------------------------ bootstrap
METRICS = {}
for pl in ac.PLATFORMS:
    METRICS[f"direct_group_mean_spearman_{pl}"] = lambda t, pl=pl: np.nanmean(
        t[f"direct_rho_{pl}"].values)
    METRICS[f"direct_top1_{pl}"] = lambda t, pl=pl: t[f"direct_top1_{pl}"].mean()
    METRICS[f"decomposed_top1_{pl}"] = lambda t, pl=pl: t[f"decomp_top1_{pl}"].mean()
    for v in ("speedup", "load_factor"):
        METRICS[f"greedy_{v}_top1_lb_{pl}"] = (
            lambda t, pl=pl, v=v: t[f"greedy_{v}_top1_lb_{pl}"].mean())
        METRICS[f"diff_direct_minus_greedy_{v}_{pl}"] = (
            lambda t, pl=pl, v=v: t[f"direct_top1_{pl}"].mean()
            - t[f"greedy_{v}_top1_lb_{pl}"].mean())
    METRICS[f"diff_direct_minus_decomposed_{pl}"] = (
        lambda t, pl=pl: t[f"direct_top1_{pl}"].mean() - t[f"decomp_top1_{pl}"].mean())
METRICS["divergence_rate"] = lambda t: t["divergent"].mean()

point = {k: float(f(T)) for k, f in METRICS.items()}

rng = np.random.RandomState(SEED)
draws = {k: np.empty(B) for k in METRICS}
for b in range(B):
    picks = rng.randint(0, len(SET_IDS), len(SET_IDS))
    idx = np.concatenate([groups_by_set[SET_IDS[i]] for i in picks])
    t = T.iloc[idx]
    for k, f in METRICS.items():
        draws[k][b] = f(t)

res = {}
for k, d in draws.items():
    d = d[np.isfinite(d)]
    res[k] = {
        "point": round(point[k], 4),
        "ci95": [round(float(np.percentile(d, 2.5)), 4),
                 round(float(np.percentile(d, 97.5)), 4)],
        "boot_mean": round(float(d.mean()), 4),
        "boot_sd": round(float(d.std(ddof=1)), 4),
        "n_draws": int(d.size),
    }
    if k.startswith("diff_"):
        res[k]["p_diff_le_0"] = round(float((d <= 0).mean()), 4)
        res[k]["ci_excludes_zero"] = bool(res[k]["ci95"][0] > 0 or res[k]["ci95"][1] < 0)

out = {
    "design": {
        "resampling_unit": "워킹셋 15개 (한 세트 = rate 3수준 = 3그룹을 통째로)",
        "B": B, "seed": SEED, "ci": "percentile 95%",
        "paired": "direct - greedy / direct - decomposed 는 같은 리샘플에서 계산 (paired)",
        "greedy_convention": "미측정 산출을 실패(0)로 둔 45그룹 하한 (항목 D)",
        "divergence_basis": "declared_normalized, alpha=0.3, beta=1.0 (감사 모듈 import)",
        "sources": ["analysis/groupkfold_{pl}_oof.csv", "analysis/greedy_results.csv",
                    "F_regret/per_group_decomposed_{pl}.csv",
                    "runs/20260730_190154_score_basis_audit/build_basis_audit.py"],
    },
    "results": res,
}
(HERE / "bootstrap_facts.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

md = ["| 지표 | 점추정 | 95% CI | bootstrap SD |", "|---|---|---|---|"]
for k in sorted(res):
    r = res[k]
    extra = ""
    if "ci_excludes_zero" in r:
        extra = f"  (CI가 0 제외: {'예' if r['ci_excludes_zero'] else '아니오'})"
    md.append(f"| `{k}` | {r['point']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
              f"{extra} | {r['boot_sd']:.3f} |")
(HERE / "bootstrap_table.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

# 검산: 점추정이 저장된 값과 일치하는가
p1m = {pl: json.loads((AN / f"groupkfold_{pl}_metrics.json").read_text())
       for pl in ac.PLATFORMS}
checks = {}
for pl in ac.PLATFORMS:
    checks[f"point_direct_top1_{pl}"] = bool(
        abs(point[f"direct_top1_{pl}"] - p1m[pl]["ranking"]["top1"]) < 1e-9)
    checks[f"point_direct_rho_{pl}"] = bool(
        abs(point[f"direct_group_mean_spearman_{pl}"]
            - p1m[pl]["ranking"]["group_spearman_mean"]) < 1e-6)
checks["point_divergence_11_of_45"] = bool(
    abs(point["divergence_rate"] - 11 / 45) < 1e-9)
out["checks"] = checks
(HERE / "bootstrap_facts.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print()
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
if not all(checks.values()):
    sys.exit(1)

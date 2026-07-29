"""P2: platform comparison rerun — measured-score argmax per (set, rate) group.

Replaces compare_platforms.py's dead /tmp data path with the in-repo backup and
runs the comparison at BOTH beta=1.0 (the codebase-wide default policy) and
beta=0.5 (the value the original run used, kept for continuity with the reported
30/45 disagreement figure). Scores are computed on RAW measured window totals,
exactly as the original did — this is a data-level comparison, no predictor.

Outputs: analysis/platform_divergence.csv (one row per group x beta) and
analysis/platform_divergence.md (summary + beta-sensitive groups).
"""
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as ac
from xgboost_model.deploy_selector_xgb_suite import _norm_exec, score_combo

SETS = yaml.safe_load((ac.ROOT / "working_sets.yaml").read_text())["sets"]
SID_OF = {",".join(sorted(v)): k for k, v in SETS.items()}
BETAS = (1.0, 0.5)
ORDER = ["S1", "S2", "S3", "S4", "base1", "base2", "base3", "base4", "base5",
         "S5", "S6", "S7", "S9", "S8", "S10"]


def load_rows(platform):
    rows = []
    for w in ac.load_windows(platform):
        models = sorted(v["model"] for v in w["models"].values())
        sid = SID_OF.get(",".join(models))
        accel = tuple(sorted(v["model"] for v in w["models"].values()
                             if _norm_exec(v["execution"]) in ("gpu", "npu")))
        has_gen = any(m in ac.GEN_MODELS for m in models)
        t = w["total"]
        row = {"sid": sid, "rate": w["rate_factor"], "combo": w["combination"],
               "accel": accel}
        for beta in BETAS:
            row[f"score_b{beta}"] = score_combo(
                t["total_throughput_fps"], t["deadline_miss_rate"],
                t.get("total_tokens_per_s", 0.0) if has_gen else 0.0,
                has_gen, ac.ALPHA, beta)
        rows.append(row)
    return rows


def best_by_group(rows, beta):
    g = defaultdict(list)
    for r in rows:
        g[(r["sid"], r["rate"])].append(r)
    return {k: max(rs, key=lambda r: r[f"score_b{beta}"]) for k, rs in g.items()}


def main():
    gpu, npu = load_rows("gpu"), load_rows("npu")
    group_keys = []
    for sid in ORDER:
        for rate in sorted({r["rate"] for r in gpu if r["sid"] == sid}):
            group_keys.append((sid, rate))

    csv_lines = ["beta,set,rate,gpu_best_accel,npu_best_accel,same,"
                 "gpu_best_score,npu_best_score,gpu_best_combo,npu_best_combo"]
    summary = {}
    diff_groups = {}
    for beta in BETAS:
        bg, bn = best_by_group(gpu, beta), best_by_group(npu, beta)
        n_same, diffs = 0, []
        for k in group_keys:
            rg, rn = bg.get(k), bn.get(k)
            if rg is None or rn is None:
                continue
            same = set(rg["accel"]) == set(rn["accel"])
            n_same += same
            if not same:
                diffs.append(k)
            csv_lines.append(
                f"{beta},{k[0]},{k[1]},\"{','.join(rg['accel'])}\","
                f"\"{','.join(rn['accel'])}\",{int(same)},"
                f"{rg[f'score_b{beta}']:.3f},{rn[f'score_b{beta}']:.3f},"
                f"{rg['combo']},{rn['combo']}")
        summary[beta] = (n_same, len(group_keys) - n_same)
        diff_groups[beta] = set(diffs)

    out_csv = ac.ANALYSIS / "platform_divergence.csv"
    out_csv.write_text(ac.prov_csv_comment(
        experiment="P2 platform divergence", betas=list(BETAS),
        score_basis="raw measured window totals (as compare_platforms.py)")
        + "\n".join(csv_lines) + "\n")

    flips = sorted(diff_groups[1.0] ^ diff_groups[0.5], key=str)
    md = [ac.prov_md_header(experiment="P2 platform divergence",
                            betas=list(BETAS),
                            score_basis="raw measured window totals")]
    md.append("# P2 — 플랫폼별 최적 배치 불일치 (측정 점수 argmax, 45그룹)\n")
    md.append("점수: S = y1 − 0.3·y2 + β·y3(생성 세트만), **원시 측정 총계** 기준 "
              "(원본 compare_platforms.py와 동일 — 예측기 미사용).\n")
    md.append("| β | 동일 배치 | 다른 배치 | 불일치율 |")
    md.append("|---|---|---|---|")
    for beta in BETAS:
        same, diff = summary[beta]
        md.append(f"| {beta} | {same} | {diff} | {diff/(same+diff):.1%} |")
    md.append("")
    md.append(f"β에 민감한 그룹 (한쪽 β에서만 불일치): {len(flips)}개")
    for k in flips:
        w10 = "불일치" if k in diff_groups[1.0] else "일치"
        w05 = "불일치" if k in diff_groups[0.5] else "일치"
        md.append(f"- {k[0]} @rate{k[1]}: β=1.0 {w10} / β=0.5 {w05}")
    md.append("\n그룹별 전체 내역은 `platform_divergence.csv` 참조.")
    (ac.ANALYSIS / "platform_divergence.md").write_text("\n".join(md) + "\n")
    for beta in BETAS:
        same, diff = summary[beta]
        print(f"beta={beta}: same={same} diff={diff}")
    print(f"beta-sensitive groups: {len(flips)}")


if __name__ == "__main__":
    main()

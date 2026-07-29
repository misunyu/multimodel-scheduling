"""P5: capacity-based greedy placement baseline (no ML, no hardware).

For each (set, rate) group the greedy assigns models to the accelerator in a
sorted pass, admitting a model while cumulative accelerator load_factor stays
<= 1.0 (load_factor = infps / capacity_fps, capacity_fps = 1000 / static accel
infer ms). Two sort variants: (a) speedup = cpu_infer/accel_infer descending,
(b) accelerator load_factor descending. qwen2_vl cannot run on CPU
(DEVICE_CONSTRAINTS) so it is force-placed on the accelerator even when over
capacity, and its load still counts toward the cumulative total.

The produced placement is then looked up among the group's COLLECTED combos: if
present, its measured score (beta=1.0) is compared to the group oracle (measured
argmax); if absent it is counted as UNCOLLECTED — never mapped to a near miss.

Outputs: analysis/greedy_results.csv, analysis/greedy_summary.md.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as ac
from xgboost_model.deploy_selector_xgb_suite import _device_static, load_static_profiles

SETS = yaml.safe_load((ac.ROOT / "working_sets.yaml").read_text())["sets"]
SID_OF = {",".join(sorted(v)): k for k, v in SETS.items()}
VARIANTS = ("speedup", "load_factor")
FORCED_ACCEL = {"qwen2_vl"}          # CPU-infeasible (model_registry.DEVICE_CONSTRAINTS)


def greedy_place(models, infps_of, S, accel, variant):
    """Return (accel_set, per-model log rows) for one group and sort variant."""
    rows = []
    for m in models:
        cpu_infer, _, _ = _device_static(m, "cpu", S)
        acc_infer, _, _ = _device_static(m, accel, S)
        cap = 1000.0 / acc_infer if acc_infer and np.isfinite(acc_infer) and acc_infer > 0 else np.nan
        lf = infps_of[m] / cap if np.isfinite(cap) and cap > 0 else np.inf
        speedup = (cpu_infer / acc_infer) if (np.isfinite(cpu_infer) and np.isfinite(acc_infer)
                                              and acc_infer > 0) else np.inf
        rows.append({"model": m, "cpu_infer": cpu_infer, "accel_infer": acc_infer,
                     "capacity_fps": cap, "load_factor": lf, "speedup": speedup})
    key = "speedup" if variant == "speedup" else "load_factor"
    order = sorted(rows, key=lambda r: (-r[key], r["model"]))
    cum, placed = 0.0, []
    for r in order:
        if r["model"] in FORCED_ACCEL:
            placed.append(r["model"]); cum += r["load_factor"]
            r["decision"] = "accel (forced)"
        elif cum + r["load_factor"] <= 1.0:
            placed.append(r["model"]); cum += r["load_factor"]
            r["decision"] = "accel"
        else:
            r["decision"] = "cpu"
    return tuple(sorted(placed)), order, cum


def main():
    S = load_static_profiles(ac.STATIC_JSON)
    csv_lines = ["platform,set,rate,variant,greedy_accel,collected,matched_combo,"
                 "top1,s_norm,oracle_norm,ratio_norm,s_raw,oracle_raw,ratio_raw,cum_load"]
    md_caps = []            # generative-model capacity note rows
    summaries = {}
    for platform in ac.PLATFORMS:
        _, Yraw, M = ac.load_dataset(platform)
        Yn = ac.normalize_targets(Yraw, M)
        s_norm = ac.scores_from(Yn["y1_total_throughput_fps"], Yn["y2_deadline_miss_rate"],
                                Yn["y3_total_tokens_per_s"], M["has_gen"])
        s_raw = ac.scores_from(Yraw["y1_total_throughput_fps"],
                               np.clip(Yraw["y2_deadline_miss_rate"], 0, 1),
                               Yraw["y3_total_tokens_per_s"], M["has_gen"])
        M = M.assign(s_norm=s_norm, s_raw=s_raw)
        imap = ac.infps_map_for(platform)

        for (models_str, rate), g in M.groupby([M["models"], M["rate_factor"]], sort=False):
            sid = SID_OF.get(models_str, models_str)
            models = models_str.split(",")
            # per-model infps for this group (device-independent; take any combo's map)
            infps_of = {}
            for m in models:
                vals = {v for (mm, dev), v in imap[g["combination"].iloc[0]].items() if mm == m}
                infps_of[m] = max(vals) if vals else 0.0
            for variant in VARIANTS:
                accel_set, order, cum = greedy_place(models, infps_of, S, platform, variant)
                for r in order:
                    if r["model"] in ac.GEN_MODELS and platform == "gpu" and variant == "speedup":
                        md_caps.append((r["model"], platform, r["accel_infer"],
                                        r["capacity_fps"], infps_of[r["model"]],
                                        r["load_factor"]))
                match = g[g["accel_set"] == accel_set]
                oracle_n, oracle_r = g["s_norm"].max(), g["s_raw"].max()
                if len(match):
                    row = match.iloc[0]
                    top1 = int(abs(row["s_norm"] - oracle_n) < 1e-9)
                    csv_lines.append(
                        f"{platform},{sid},{rate},{variant},\"{','.join(accel_set)}\",1,"
                        f"{row['combination']},{top1},{row['s_norm']:.4f},{oracle_n:.4f},"
                        f"{row['s_norm']/oracle_n if oracle_n > 1e-9 else np.nan:.4f},"
                        f"{row['s_raw']:.3f},{oracle_r:.3f},"
                        f"{row['s_raw']/oracle_r if abs(oracle_r) > 1e-9 else np.nan:.4f},"
                        f"{cum:.3f}")
                    st = summaries.setdefault((platform, variant),
                                              {"top1": [], "rn": [], "rr": [], "unc": []})
                    st["top1"].append(top1)
                    st["rn"].append(row["s_norm"] / oracle_n if oracle_n > 1e-9 else np.nan)
                    st["rr"].append(row["s_raw"] / oracle_r if abs(oracle_r) > 1e-9 else np.nan)
                else:
                    csv_lines.append(
                        f"{platform},{sid},{rate},{variant},\"{','.join(accel_set)}\",0,"
                        f",,,,,,,,{cum:.3f}")
                    summaries.setdefault((platform, variant),
                                         {"top1": [], "rn": [], "rr": [], "unc": []})[
                        "unc"].append(f"{sid}@{rate}")

    (ac.ANALYSIS / "greedy_results.csv").write_text(
        ac.prov_csv_comment(experiment="P5 capacity greedy",
                            static_profile=str(ac.STATIC_JSON)) +
        "\n".join(csv_lines) + "\n")

    md = [ac.prov_md_header(experiment="P5 capacity greedy",
                            static_profile=str(ac.STATIC_JSON))]
    md.append("# P5 — Capacity greedy baseline (45그룹 × 2변형 × 2플랫폼)\n")
    md.append("배정: 정렬 순서대로 누적 load_factor ≤ 1.0이면 가속기, 아니면 CPU. "
              "qwen2_vl은 CPU 불가라 용량 초과여도 가속기 강제(누적에 반영).\n")
    md.append("| 플랫폼 | 변형 | Top-1 일치율(수집분) | 평균 oracle비(정규화) | "
              "평균 oracle비(원시) | 미수집 그룹 |")
    md.append("|---|---|---|---|---|---|")
    for (platform, variant), st in sorted(summaries.items()):
        n_c = len(st["top1"])
        md.append(f"| {platform} | {variant} | "
                  f"{np.mean(st['top1']):.3f} ({int(sum(st['top1']))}/{n_c}) | "
                  f"{np.nanmean(st['rn']):.3f} | {np.nanmean(st['rr']):.3f} | "
                  f"{len(st['unc'])}/45 |")
    md.append("\n## 미수집 그룹 목록 (greedy 산출 배치가 수집 조합에 없음)\n")
    for (platform, variant), st in sorted(summaries.items()):
        md.append(f"- {platform}/{variant}: "
                  + (", ".join(st["unc"]) if st["unc"] else "(없음)"))
    md.append("\n## 생성 모델의 capacity_fps에 쓰인 값 (판단은 논문 쪽)\n")
    md.append("capacity_fps = 1000/static_infer_sel. 생성 모델의 static_infer_sel은 "
              "prefill 지연(ms)이라 '초당 처리 가능 요청 수'로서의 의미가 제한적이다:\n")
    md.append("| 모델 | 플랫폼 | accel_infer(ms) | capacity_fps | infps | load_factor |")
    md.append("|---|---|---|---|---|---|")
    seen = set()
    for m, pl, inf, cap, ifp, lf in md_caps:
        k = (m, pl, round(ifp, 3))
        if k in seen:
            continue
        seen.add(k)
        md.append(f"| {m} | {pl} | {inf:.1f} | {cap:.2f} | {ifp:.3f} | {lf:.4f} |")
    md.append("\n(NPU에서는 llama1b 81.2ms→12.3fps, qwen2_vl 613.8ms→1.63fps로 동일 "
              "정의를 적용. 전체 내역은 `greedy_results.csv`.)")
    (ac.ANALYSIS / "greedy_summary.md").write_text("\n".join(md) + "\n")

    for (platform, variant), st in sorted(summaries.items()):
        print(f"{platform}/{variant}: top1={np.mean(st['top1']):.3f} "
              f"ratio_norm={np.nanmean(st['rn']):.3f} uncollected={len(st['unc'])}")


if __name__ == "__main__":
    main()

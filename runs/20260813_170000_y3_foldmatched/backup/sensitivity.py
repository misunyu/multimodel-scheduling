"""작업 18-r2: CPU-only 생성 후보의 y3 처리 민감도 — 재채점만.

대상: 생성 세트인데 가속기 위에 생성 인스턴스가 없는 창.
      S3(llama1b,resnet50,yolo11s)에서 llama1b 가 CPU 인 4배치 × 3 rate = 플랫폼당 12창.

A. Prediction-side  — 측정 라벨은 현행(y3=0) 고정
     A1 현행 외삽 yhat3   vs   A2 해당 후보 yhat3 = 0
B. Label-side       — 예측은 현행(A1) 고정
     B1 현행 y3=0        vs   B2 기록된 창 총계를 r3 로 복원 후 그룹 내 재정규화

재학습·재수집 없음. 저장된 OOF CSV·측정 창·(zero-shot 은) 저장된 fold 부스터의
추론만 사용한다. 코드·아티팩트·원고 미수정.
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
TIE = 1e-9
ALPHA, BETA = ac.ALPHA, ac.BETA
PAIRED = ROOT / "runs/20260809_140605_checklist_r2/A_joint_paired/joint_oof_new45.csv"
SETS = yaml.safe_load((ROOT / "working_sets.yaml").read_text())["sets"]
SID_OF = {",".join(sorted(v)): k for k, v in SETS.items()}
S3_KEY = "llama1b,resnet50,yolo11s"

EXEC_COLS = [(f"views.{a}.view.exec_gpu", f"views.{a}.view.exec_npu")
             for a in ("sum", "mean", "max")]


# ------------------------------------------------------------------ 공통 지표
def group_metrics(df, meas="s_meas", pred="s_pred"):
    """그룹별 Spearman / Top-1 / selected-score ratio. 기존 규칙(tie 1e-9) 유지."""
    rows = []
    for g, gg in df.groupby("group", sort=False):
        m = gg[meas].values.astype(float)
        p = gg[pred].values.astype(float)
        best = float(m.max())
        oracle = set(np.where(np.abs(m - best) < TIE)[0])
        i = int(np.argsort(-p, kind="stable")[0])
        rows.append({"group": str(g),
                     "spearman": ac.spearman(m, p),
                     "top1": float(i in oracle),
                     "ratio": float(m[i] / best) if abs(best) > 1e-12 else np.nan,
                     "picked_meas": float(m[i]), "oracle_meas": best})
    return pd.DataFrame(rows)


def summarize(gm):
    rho = gm["spearman"].values
    return {"n_groups": int(len(gm)),
            "mean_spearman": float(np.nanmean(rho)),
            "top1": float(gm["top1"].mean()),
            "mean_ratio": float(np.nanmean(gm["ratio"]))}


def d4(x):
    return float(f"{x:.4f}")


# ------------------------------------------------- 데이터: 측정 라벨 두 변형
raw, meta, feats = {}, {}, {}
target_rows = {}      # platform -> boolean mask of the 12 CPU-only-generative rows
r3_totals = {}
for pl in ac.PLATFORMS:
    X, Yraw, M = ac.load_dataset(pl)
    feats[pl] = X
    raw[pl] = Yraw.copy()
    meta[pl] = M
    # 창 총계 r3 (실측) — featurize 는 뷰별 합을 쓰므로 원본 창에서 다시 읽는다
    wins = ac.load_windows(pl)
    tot = {str(w["combination"]): float(w["total"].get("total_tokens_per_s", 0.0))
           for w in wins}
    r3_totals[pl] = M["combination"].astype(str).map(tot).values
    mask = (M["has_gen"].values.astype(bool)) & (~M["y3_valid"].values.astype(bool))
    target_rows[pl] = mask

meas = {}   # (platform, variant) -> DataFrame(group, s_meas, ...) aligned to M order
for pl in ac.PLATFORMS:
    M = meta[pl]
    base = pd.DataFrame({
        "combination": M["combination"].astype(str),
        "models": M["models"], "rate_factor": M["rate_factor"],
        "has_gen": M["has_gen"], "y3_valid": M["y3_valid"].astype(bool),
        "group": [str((m, r)) for m, r in zip(M["models"], M["rate_factor"])],
        "set": [SID_OF[",".join(sorted(s.split(",")))] for s in M["models"]],
        "is_target": target_rows[pl]})
    for variant in ("B1", "B2"):
        Yr = raw[pl].copy()
        if variant == "B2":
            Yr["y3_total_tokens_per_s"] = np.where(
                target_rows[pl], r3_totals[pl], Yr["y3_total_tokens_per_s"].values)
        Yn = ac.normalize_targets(Yr, M)
        d = base.copy()
        d["y1_meas"] = Yn["y1_total_throughput_fps"].values
        d["y2_meas"] = Yn["y2_deadline_miss_rate"].values
        d["y3_meas"] = Yn["y3_total_tokens_per_s"].values
        d["r3_raw"] = Yr["y3_total_tokens_per_s"].values
        d["s_meas"] = ac.scores_from(d["y1_meas"], d["y2_meas"], d["y3_meas"], d["has_gen"])
        meas[(pl, variant)] = d

# 그룹 max(r3) 가 실제로 바뀌는가 (B2 의 격리성)
gmax = {}
for pl in ac.PLATFORMS:
    for v in ("B1", "B2"):
        d = meas[(pl, v)]
        gmax[(pl, v)] = d.groupby("group")["r3_raw"].max().to_dict()
gmax_changed = {pl: sorted(g for g in gmax[(pl, "B1")]
                           if abs(gmax[(pl, "B1")][g] - gmax[(pl, "B2")][g]) > 1e-9)
                for pl in ac.PLATFORMS}
# B2 가 y3_meas 를 바꾸는 행이 전부 S3 인가
b2_changed_rows = {}
for pl in ac.PLATFORMS:
    a, b = meas[(pl, "B1")], meas[(pl, "B2")]
    ch = a.index[np.abs(a["y3_meas"].values - b["y3_meas"].values) > 1e-12]
    b2_changed_rows[pl] = {"n": int(len(ch)), "sets": sorted(set(a.loc[ch, "set"])),
                           "groups": sorted(set(a.loc[ch, "group"]))}

# ------------------------------------------------------------- 예측 4종
preds = {}   # (method, platform) -> DataFrame(combination, y1_pred,y2_pred,y3_pred)

for pl in ac.PLATFORMS:
    o = pd.read_csv(AN / f"groupkfold_{pl}_oof.csv", comment="#")
    o["combination"] = o["combination"].astype(str)
    preds[("platform-specific", pl)] = o[["combination", "y1_pred", "y2_pred", "y3_pred"]]

j = pd.read_csv(PAIRED)
j["combination"] = j["combination"].astype(str)
for pl in ac.PLATFORMS:
    preds[("joint", pl)] = j[j["platform"] == pl][
        ["combination", "y1_pred", "y2_pred", "y3_pred"]].reset_index(drop=True)

# zero-shot: 저장된 fold 부스터 3개의 예측 평균 (cross_platform_transfer.py:64-75, asis)
other = {"gpu": "npu", "npu": "gpu"}
for pl in ac.PLATFORMS:
    Xv = feats[pl]
    out = {"combination": meta[pl]["combination"].astype(str).values}
    for tag, _ in ac.TARGETS:
        ps = []
        for f in range(ac.N_FOLDS):
            bst = ac.load_booster(ac.ARTIFACTS / f"gkf_cpu_{other[pl]}_{tag}_fold{f}.json")
            ps.append(ac.predict_booster(bst, Xv.values))
        out[f"{tag}_pred"] = np.vstack(ps).mean(axis=0)
    preds[("zero-shot", pl)] = pd.DataFrame(out)

# decomposed: 뷰 OOF 재합성 + P1 yhat3 공유 (train_decomposed.py:114-115, :150-186)
vo = pd.read_csv(AN / "decomposed_view_oof.csv", comment="#")
for pl in ac.PLATFORMS:
    M = meta[pl]
    w = pd.DataFrame({
        "combination": M["combination"].astype(str),
        "group": [str((m, r)) for m, r in zip(M["models"], M["rate_factor"])],
        "y1_raw": raw[pl]["y1_total_throughput_fps"],
        "y2_raw": np.clip(raw[pl]["y2_deadline_miss_rate"], 0, 1)})
    p1 = preds[("platform-specific", pl)]
    w = w.merge(p1[["combination", "y3_pred"]], on="combination", how="left")
    gs = w.groupby("group").agg(fmax=("y1_raw", "max"), lo=("y2_raw", "min"),
                                hi=("y2_raw", "max"))
    V = vo[vo["platform"] == pl].copy()
    V["combination"] = V["combination"].astype(str)
    agg = V.groupby("combination").apply(
        lambda d: pd.Series({"y1_sum": float((d["pred"] * d["eff_fps"]).sum()),
                             "miss": float(np.clip((1.0 - d["pred"]).mean(), 0, 1))}),
        include_groups=False)
    c = w.set_index("combination").join(agg).reset_index().join(gs, on="group")
    span = (c["hi"] - c["lo"]).values
    c["y1_pred"] = c["y1_sum"] / c["fmax"]
    c["y2_pred"] = np.where(span > 1e-9,
                            (c["miss"] - c["lo"]) / np.where(span > 1e-9, span, 1.0), 0.0)
    preds[("decomposed", pl)] = c[["combination", "y1_pred", "y2_pred", "y3_pred"]]

METHODS = ["platform-specific", "zero-shot", "joint", "decomposed"]

# decomposed 가 P1 yhat3 를 그대로 공유하는지 검산
share_ok = {}
for pl in ac.PLATFORMS:
    a = preds[("decomposed", pl)].set_index("combination")["y3_pred"]
    b = preds[("platform-specific", pl)].set_index("combination")["y3_pred"]
    share_ok[pl] = bool(np.allclose(a.loc[b.index].values, b.values, atol=0, rtol=0,
                                    equal_nan=True))


def scored(method, pl, meas_variant, mask_y3):
    """예측 + 측정 라벨을 붙여 s_pred/s_meas 를 만든 프레임."""
    d = meas[(pl, meas_variant)].copy()
    p = preds[(method, pl)].set_index("combination")
    d = d.join(p, on="combination")
    y3p = d["y3_pred"].values.astype(float).copy()
    if mask_y3:
        y3p[d["is_target"].values] = 0.0
    d["y3_pred_used"] = y3p
    d["s_pred"] = ac.scores_from(d["y1_pred"], d["y2_pred"], y3p, d["has_gen"])
    return d


# ------------------------------------------------------------------ A 파트
extrap = {}
for pl in ac.PLATFORMS:
    for m in METHODS:
        d = scored(m, pl, "B1", False)
        v = d.loc[d["is_target"], "y3_pred"].values
        extrap[f"{m}/{pl}"] = {"min": d4(v.min()), "max": d4(v.max()),
                               "mean": d4(v.mean()), "n": int(len(v))}

A = {}
for m in METHODS:
    for pl in ac.PLATFORMS:
        row = {}
        gms = {}
        for tag, mask in (("A1", False), ("A2", True)):
            d = scored(m, pl, "B1", mask)
            gm = group_metrics(d)
            gms[tag] = gm
            s3 = gm[gm["group"].str.contains(S3_KEY)]
            row[tag] = {
                "all45": {k: d4(v) if isinstance(v, float) else v
                          for k, v in summarize(gm).items()},
                "S3": {"n_groups": int(len(s3)),
                       "mean_spearman": d4(float(np.nanmean(s3["spearman"]))),
                       "top1": d4(float(s3["top1"].mean())),
                       "mean_ratio": d4(float(np.nanmean(s3["ratio"]))),
                       "per_group_spearman": [d4(x) for x in s3["spearman"]],
                       "per_group_top1": [int(x) for x in s3["top1"]],
                       "per_group_ratio": [d4(x) for x in s3["ratio"]]},
            }
        a1, a2 = gms["A1"], gms["A2"]
        merged = a1.merge(a2, on="group", suffixes=("_A1", "_A2"))
        row["delta_A2_minus_A1"] = {
            "all45_mean_spearman": d4(row["A2"]["all45"]["mean_spearman"]
                                      - row["A1"]["all45"]["mean_spearman"]),
            "all45_top1": d4(row["A2"]["all45"]["top1"] - row["A1"]["all45"]["top1"]),
            "all45_mean_ratio": d4(row["A2"]["all45"]["mean_ratio"]
                                   - row["A1"]["all45"]["mean_ratio"]),
            "S3_mean_spearman": d4(row["A2"]["S3"]["mean_spearman"]
                                   - row["A1"]["S3"]["mean_spearman"]),
            "S3_top1": d4(row["A2"]["S3"]["top1"] - row["A1"]["S3"]["top1"]),
            "S3_mean_ratio": d4(row["A2"]["S3"]["mean_ratio"]
                                - row["A1"]["S3"]["mean_ratio"]),
            "n_groups_with_any_change": int(
                (np.abs(merged["spearman_A1"].fillna(0) - merged["spearman_A2"].fillna(0)) > 1e-12).sum()
                | 0),
            "n_groups_top1_changed": int((merged["top1_A1"] != merged["top1_A2"]).sum()),
        }
        A[f"{m}/{pl}"] = row

# ------------------------------------------------------------------ B 파트
Bres = {}
for m in METHODS:
    for pl in ac.PLATFORMS:
        row, gms = {}, {}
        for tag in ("B1", "B2"):
            d = scored(m, pl, tag, False)
            gm = group_metrics(d)
            gms[tag] = gm
            s3 = gm[gm["group"].str.contains(S3_KEY)]
            row[tag] = {
                "all45": {k: d4(v) if isinstance(v, float) else v
                          for k, v in summarize(gm).items()},
                "S3": {"mean_spearman": d4(float(np.nanmean(s3["spearman"]))),
                       "top1": d4(float(s3["top1"].mean())),
                       "mean_ratio": d4(float(np.nanmean(s3["ratio"]))),
                       "per_group_spearman": [d4(x) for x in s3["spearman"]],
                       "per_group_top1": [int(x) for x in s3["top1"]],
                       "per_group_ratio": [d4(x) for x in s3["ratio"]]},
            }
        merged = gms["B1"].merge(gms["B2"], on="group", suffixes=("_B1", "_B2"))
        row["delta_B2_minus_B1"] = {
            "all45_mean_spearman": d4(row["B2"]["all45"]["mean_spearman"]
                                      - row["B1"]["all45"]["mean_spearman"]),
            "all45_top1": d4(row["B2"]["all45"]["top1"] - row["B1"]["all45"]["top1"]),
            "all45_mean_ratio": d4(row["B2"]["all45"]["mean_ratio"]
                                   - row["B1"]["all45"]["mean_ratio"]),
            "S3_mean_spearman": d4(row["B2"]["S3"]["mean_spearman"]
                                   - row["B1"]["S3"]["mean_spearman"]),
            "S3_top1": d4(row["B2"]["S3"]["top1"] - row["B1"]["S3"]["top1"]),
            "S3_mean_ratio": d4(row["B2"]["S3"]["mean_ratio"]
                                - row["B1"]["S3"]["mean_ratio"]),
            "n_groups_top1_changed": int((merged["top1_B1"] != merged["top1_B2"]).sum()),
            "groups_top1_changed": merged.loc[
                merged["top1_B1"] != merged["top1_B2"], "group"].tolist(),
        }
        Bres[f"{m}/{pl}"] = row

# S3 의 best-measured placement 가 B1 -> B2 로 바뀌는가
best_meas = {}
for pl in ac.PLATFORMS:
    for tag in ("B1", "B2"):
        d = meas[(pl, tag)]
        s3 = d[d["set"] == "S3"]
        bb = {}
        for g, gg in s3.groupby("group"):
            best = gg["s_meas"].max()
            bb[g] = sorted(gg.loc[np.abs(gg["s_meas"] - best) < TIE, "combination"])
        best_meas[(pl, tag)] = bb
best_changed = {pl: sorted(g for g in best_meas[(pl, "B1")]
                           if best_meas[(pl, "B1")][g] != best_meas[(pl, "B2")][g])
                for pl in ac.PLATFORMS}

# ------------------------------------------------------- greedy (측정 라벨 의존)
gr = pd.read_csv(AN / "greedy_results.csv", comment="#")
greedy_B = {}
for pl in ac.PLATFORMS:
    accel_of = {}
    for tag in ("B1", "B2"):
        d = meas[(pl, tag)].copy()
        d["accel_set"] = meta[pl]["accel_set"].values
        accel_of[tag] = d
    for var in ("speedup", "load_factor"):
        g = gr[(gr["platform"] == pl) & (gr["variant"] == var)]
        out = {}
        for tag in ("B1", "B2"):
            d = accel_of[tag]
            hits, ratios = 0, []
            n_meas = 0
            for _, r in g.iterrows():
                if r["collected"] != 1:
                    continue
                n_meas += 1
                gg = d[d["combination"] == str(r["matched_combo"])]
                grp = gg["group"].iloc[0]
                sub = d[d["group"] == grp]
                best = sub["s_meas"].max()
                sm = float(gg["s_meas"].iloc[0])
                hits += int(abs(sm - best) < TIE)
                ratios.append(sm / best if abs(best) > 1e-12 else np.nan)
            out[tag] = {"n_measured": n_meas, "top1": d4(hits / n_meas),
                        "mean_oracle_ratio": d4(float(np.nanmean(ratios))),
                        "lower_bound_top1": d4(hits / 45)}
        out["delta_B2_minus_B1"] = {
            k: d4(out["B2"][k] - out["B1"][k])
            for k in ("top1", "mean_oracle_ratio", "lower_bound_top1")}
        greedy_B[f"{pl}/{var}"] = out

# ------------------------------------------------------------- divergence
spec = importlib.util.spec_from_file_location(
    "basis_audit", ROOT / "runs/20260730_190154_score_basis_audit/build_basis_audit.py")
ba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ba)


def divergence_with(r3_key):
    """r3_key: 'total'(감사 기본, 창 총계) 또는 'viewsum'(파이프라인 관례)."""
    rows = {}
    for pl in ("gpu", "npu"):
        rs = ba.load_rows(pl)
        if r3_key == "viewsum":
            vs = dict(zip(meas[(pl, "B1")]["combination"],
                          meas[(pl, "B1")]["r3_raw"]))
            for r in rs:
                r["r3"] = float(vs[r["combo"]])
        rows[pl] = ba.normalize(rs)
    res, tg, tn = ba.divergence(rows, ALPHA, BETA, "normalized")
    return res


div_total = divergence_with("total")
div_viewsum = divergence_with("viewsum")

# ------------------------------------------------------------------ 검산
checks = {
    "target_rows_are_12_per_platform": all(int(target_rows[pl].sum()) == 12
                                           for pl in ac.PLATFORMS),
    "target_rows_all_in_S3": all(
        set(meas[(pl, "B1")].loc[target_rows[pl], "set"]) == {"S3"}
        for pl in ac.PLATFORMS),
    "B1_y3_meas_zero_on_targets": all(
        float(np.abs(meas[(pl, "B1")].loc[target_rows[pl], "y3_meas"]).max()) == 0.0
        for pl in ac.PLATFORMS),
    "B2_changes_only_S3_rows": all(b2_changed_rows[pl]["sets"] in ([], ["S3"])
                                   for pl in ac.PLATFORMS),
    "B2_changes_no_group_max_r3": all(not gmax_changed[pl] for pl in ac.PLATFORMS),
    "decomposed_shares_P1_y3_pred": all(share_ok.values()),
    "A_leaves_measured_scores_untouched": True,   # 구성상 s_meas 는 B1 고정
    "divergence_invariant_under_A": True,          # 측정 전용 지표
    "divergence_total_is_11_of_45": div_total["n_disagree"] == 11,
}
# A 가 divergence 를 못 바꾼다는 것을 구성이 아니라 값으로도 확인
checks["A_measured_labels_identical_across_A1_A2"] = True
for m in METHODS:
    for pl in ac.PLATFORMS:
        a = scored(m, pl, "B1", False)["s_meas"].values
        b = scored(m, pl, "B1", True)["s_meas"].values
        if not np.array_equal(a, b):
            checks["A_measured_labels_identical_across_A1_A2"] = False

facts = {
    "scope": {
        "target_windows": "생성 세트인데 가속기 위 생성 인스턴스가 없는 창 = S3 의 "
                          "llama1b-on-CPU 4배치 × 3 rate = 플랫폼당 12창",
        "no_retraining": "저장된 OOF CSV·측정 창·저장된 fold 부스터 추론만 사용",
        "alpha": ALPHA, "beta": BETA, "tie_eps": TIE,
        "joint_source": str(PAIRED.relative_to(ROOT)) + " (paired-fold, 작업 14-A)",
        "zero_shot_source": "artifacts/gkf_cpu_{train}_{tag}_fold{0,1,2}.json 평균 (asis)",
    },
    "extrapolated_yhat3_on_targets": extrap,
    "restored_r3_on_targets": {
        pl: {"min": d4(float(meas[(pl, "B2")].loc[target_rows[pl], "r3_raw"].min())),
             "max": d4(float(meas[(pl, "B2")].loc[target_rows[pl], "r3_raw"].max())),
             "mean": d4(float(meas[(pl, "B2")].loc[target_rows[pl], "r3_raw"].mean())),
             "n_zero": int((meas[(pl, "B2")].loc[target_rows[pl], "r3_raw"] == 0).sum()),
             "normalized_y3_after_restore": {
                 "min": d4(float(meas[(pl, "B2")].loc[target_rows[pl], "y3_meas"].min())),
                 "max": d4(float(meas[(pl, "B2")].loc[target_rows[pl], "y3_meas"].max()))}}
        for pl in ac.PLATFORMS},
    "A_prediction_side": A,
    "B_label_side": Bres,
    "B_greedy": greedy_B,
    "B_S3_best_measured_placement_changed": best_changed,
    "B2_changed_rows": b2_changed_rows,
    "B2_group_max_r3_changed": gmax_changed,
    "divergence": {
        "window_total_convention_published": {
            "n_disagree": div_total["n_disagree"], "n_groups": div_total["n_groups"],
            "gen": div_total["gen"], "vision": div_total["vision"],
            "groups": div_total["disagree_groups"]},
        "viewsum_convention": {
            "n_disagree": div_viewsum["n_disagree"], "n_groups": div_viewsum["n_groups"],
            "gen": div_viewsum["gen"], "vision": div_viewsum["vision"],
            "groups": div_viewsum["disagree_groups"]},
        "identical": sorted(div_total["disagree_groups"]) ==
                     sorted(div_viewsum["disagree_groups"]),
    },
    "checks": checks,
}
(HERE / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False))

# ------------------------------------------------------------------ 콘솔
print("== 외삽 yhat3 (대상 12창) ==")
for k, v in extrap.items():
    print(f"  {k:26s} {v['min']:.4f} ~ {v['max']:.4f} (평균 {v['mean']:.4f})")
print("== 복원 r3 ==")
for pl in ac.PLATFORMS:
    r = facts["restored_r3_on_targets"][pl]
    print(f"  {pl}: r3 {r['min']:.4f}~{r['max']:.4f} (평균 {r['mean']:.4f}, 0인 창 "
          f"{r['n_zero']}), 정규화 y3 {r['normalized_y3_after_restore']['min']:.4f}"
          f"~{r['normalized_y3_after_restore']['max']:.4f}")
print("\n== A (예측측): A2 − A1 ==")
for k, v in A.items():
    d = v["delta_A2_minus_A1"]
    print(f"  {k:26s} 45그룹 ρ {d['all45_mean_spearman']:+.4f} Top-1 {d['all45_top1']:+.4f}"
          f" ratio {d['all45_mean_ratio']:+.4f} | S3 ρ {d['S3_mean_spearman']:+.4f}"
          f" Top-1 {d['S3_top1']:+.4f} (Top-1 바뀐 그룹 {d['n_groups_top1_changed']})")
print("\n== B (라벨측): B2 − B1 ==")
for k, v in Bres.items():
    d = v["delta_B2_minus_B1"]
    print(f"  {k:26s} 45그룹 ρ {d['all45_mean_spearman']:+.4f} Top-1 {d['all45_top1']:+.4f}"
          f" ratio {d['all45_mean_ratio']:+.4f} | S3 ρ {d['S3_mean_spearman']:+.4f}"
          f" Top-1 {d['S3_top1']:+.4f} 바뀐그룹 {d['groups_top1_changed']}")
print("\n== B greedy ==")
for k, v in greedy_B.items():
    d = v["delta_B2_minus_B1"]
    print(f"  {k:20s} Top-1 {d['top1']:+.4f} oracle-ratio {d['mean_oracle_ratio']:+.4f}")
print(f"\n== divergence: 창총계 {div_total['n_disagree']}/45, "
      f"뷰별합 {div_viewsum['n_disagree']}/45, 동일집합 {facts['divergence']['identical']}")
print(f"== S3 best-measured 변화: {best_changed}")
print("\n== 검산 ==")
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
if not all(checks.values()):
    sys.exit(1)

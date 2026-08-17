"""작업 19-r2 [3][4]: 논문의 그림·표·인라인 수치를 처음부터 다시 만든다.

원칙
  - 출력은 이 run 디렉터리 아래에만. artifacts/·analysis/·figures/·manuscript/ 미수정.
  - 파이프라인 코드는 수정하지 않는다. analysis_common 헬퍼와 감사 모듈을 그대로 import 해
    같은 순서·같은 seed(42)로 재실행한다.
  - 부스터 미저장 항목(joint, decomposed)은 재학습으로 재현한다.

단계
  1 P1 platform-specific (재학습, fold 부스터를 이 디렉터리에 저장)
  2 zero-shot transfer  (1 이 만든 fold 부스터 사용)
  3 joint paired-fold   (재학습)
  4 decomposed          (재학습; 1 의 fold·y3 예측 사용)
  5 greedy              (학습 없음)
  6 divergence / sweep / coverage (감사 모듈)
  7 bootstrap           (seed 42)
  8 selected-score ratio, y3 민감도 headline
  9 결정론 스팟체크
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "xgboost_model" / "full_collection_540" / "scripts"))

import analysis_common as ac  # noqa: E402
from xgboost_model.deploy_selector_xgb_suite import (  # noqa: E402
    _device_static, _is_vision, _norm_exec, _view_features, load_static_profiles)
import train_decomposed as td  # noqa: E402

TIE = 1e-9
SETS = yaml.safe_load((ROOT / "working_sets.yaml").read_text())["sets"]
SID_OF = {",".join(sorted(v)): k for k, v in SETS.items()}
R = {}          # reproduced values
T = {}          # stage timings


def stage(name):
    class _C:
        def __enter__(s):
            s.t = time.time(); print(f"[{name}] start", flush=True); return s
        def __exit__(s, *a):
            T[name] = round(time.time() - s.t, 1)
            print(f"[{name}] done {T[name]}s", flush=True)
    return _C()


def rank_of(df, meas="s_meas", pred="s_pred"):
    rows = []
    for g, gg in df.groupby("group", sort=False):
        m = gg[meas].values.astype(float); p = gg[pred].values.astype(float)
        best = float(m.max())
        oracle = set(np.where(np.abs(m - best) < TIE)[0])
        order = np.argsort(-p, kind="stable")
        rows.append({"group": str(g), "top1": float(order[0] in oracle),
                     "top5": float(bool(oracle & set(order[:5]))),
                     "spearman": ac.spearman(m, p),
                     "ratio": float(m[order[0]] / best) if abs(best) > 1e-12 else np.nan})
    d = pd.DataFrame(rows)
    return d, {"n_groups": int(len(d)), "top1": float(d["top1"].mean()),
               "top5": float(d["top5"].mean()),
               "group_spearman_mean": float(np.nanmean(d["spearman"])),
               "mean_ratio": float(np.nanmean(d["ratio"])),
               "min_ratio": float(np.nanmin(d["ratio"])),
               # App.H selected-placement paragraph counts the groups under 0.9
               "n_ratio_below_0.9": int((d["ratio"] < 0.9).sum())}


# ---------------------------------------------------------------- 데이터 로드
DATA = {}
with stage("load"):
    for pl in ac.PLATFORMS:
        X, Yraw, M = ac.load_dataset(pl)
        Yn = ac.normalize_targets(Yraw, M)
        groups = ac.group_keys(M)
        DATA[pl] = {"X": X, "Yraw": Yraw, "Yn": Yn, "M": M, "groups": groups,
                    "gstr": pd.Series([str((m, r)) for m, r in
                                       zip(M["models"], M["rate_factor"])]),
                    "set": [SID_OF[",".join(sorted(s.split(",")))] for s in M["models"]]}

# --------------------------------------------------------- 1. P1 재학습
P1 = {}
with stage("1_platform_specific_retrain"):
    for pl in ac.PLATFORMS:
        d = DATA[pl]
        fold = ac.assign_group_folds(d["groups"]).values
        Xv = d["X"].values
        y3v = d["M"]["y3_valid"].values.astype(bool)
        oof = pd.DataFrame({
            "combination": d["M"]["combination"].astype(str),
            "models": d["M"]["models"], "rate_factor": d["M"]["rate_factor"],
            "has_gen": d["M"]["has_gen"], "y3_valid": y3v, "fold": fold,
            "group": d["gstr"].values, "set": d["set"],
            "y1_meas": d["Yn"]["y1_total_throughput_fps"].values,
            "y2_meas": d["Yn"]["y2_deadline_miss_rate"].values,
            "y3_meas": d["Yn"]["y3_total_tokens_per_s"].values})
        tgt = {}
        for tag, col in ac.TARGETS:
            yv = d["Yn"][col].values.astype(float)
            valid = y3v if tag == "y3" else np.ones(len(yv), bool)
            pred = np.full(len(yv), np.nan)
            for f in range(ac.N_FOLDS):
                tr, te = (fold != f) & valid, fold == f
                params, _ = ac.cv_select_params(Xv[tr], yv[tr],
                                                d["groups"][tr].reset_index(drop=True))
                bst = ac.train_booster(Xv[tr], yv[tr], params)
                pred[te] = ac.predict_booster(bst, Xv[te])
                bst.save_model(str(OUT / f"gkf_cpu_{pl}_{tag}_fold{f}.json"))
            oof[f"{tag}_pred"] = pred
            m_, p_ = oof.loc[valid, f"{tag}_meas"], oof.loc[valid, f"{tag}_pred"]
            gsp = ac.per_group_spearman(oof[valid], "group", f"{tag}_meas", f"{tag}_pred")
            tgt[tag] = {"oof_spearman_pooled": ac.spearman(m_, p_),
                        "oof_mae": float(np.mean(np.abs(m_ - p_))),
                        "n_rows_evaluated": int(valid.sum()),
                        "group_spearman": gsp}
        oof["s_meas"] = ac.scores_from(oof["y1_meas"], oof["y2_meas"], oof["y3_meas"],
                                       oof["has_gen"])
        oof["s_pred"] = ac.scores_from(oof["y1_pred"], oof["y2_pred"], oof["y3_pred"],
                                       oof["has_gen"])
        per, agg = rank_of(oof)
        oof.to_csv(OUT / f"p1_oof_{pl}.csv", index=False)
        P1[pl] = {"oof": oof, "per_group": per, "ranking": agg, "targets": tgt,
                  "fold": fold}
        print(f"   {pl}: top1={agg['top1']:.4f} rho={agg['group_spearman_mean']:.4f}",
              flush=True)

R["direct"] = {pl: {k: P1[pl]["ranking"][k] for k in
                    ("n_groups", "top1", "top5", "group_spearman_mean",
                     "mean_ratio", "min_ratio", "n_ratio_below_0.9")}
               for pl in ac.PLATFORMS}
R["direct_targets_pooled"] = {
    pl: {t: {"pooled_spearman": P1[pl]["targets"][t]["oof_spearman_pooled"],
             "mae": P1[pl]["targets"][t]["oof_mae"],
             "n_rows": P1[pl]["targets"][t]["n_rows_evaluated"]}
         for t in ("y1", "y2", "y3")} for pl in ac.PLATFORMS}

# App.B 표: 세트별 · 타깃별 그룹 Spearman 평균 (rate 평균)
perset = {}
for pl in ac.PLATFORMS:
    o = P1[pl]["oof"]
    g2s = dict(zip(o["group"], o["set"]))
    for tag in ("y1", "y2", "y3"):
        for g, v in P1[pl]["targets"][tag]["group_spearman"].items():
            perset.setdefault(g2s[g], {}).setdefault(f"{pl}_{tag}", []).append(v)
R["per_set_group_spearman"] = {
    s: {k: (None if all(not np.isfinite(x) for x in v) else
            float(np.nanmean([x for x in v if np.isfinite(x)])))
        for k, v in d.items()} for s, d in perset.items()}

# ------------------------------------------------------ 2. zero-shot transfer
other = {"gpu": "npu", "npu": "gpu"}
with stage("2_zero_shot"):
    zs = {}
    for pl in ac.PLATFORMS:
        d = DATA[pl]
        base = P1[pl]["oof"][["group", "has_gen", "y3_valid",
                              "y1_meas", "y2_meas", "y3_meas", "s_meas"]].copy()
        # fold-matched (paper protocol since 2026-08-13): the source fold-f booster
        # predicts only the target's fold-f rows, so the predicting booster has seen
        # that workload group on neither platform.
        tfold = P1[pl]["fold"]
        for tag, _ in ac.TARGETS:
            pr = np.full(len(d["X"]), np.nan)
            for f in range(ac.N_FOLDS):
                m = tfold == f
                bst = ac.load_booster(OUT / f"gkf_cpu_{other[pl]}_{tag}_fold{f}.json")
                pr[m] = ac.predict_booster(bst, d["X"].values[m])
            assert np.isfinite(pr).all()
            base[f"{tag}_pred"] = pr
        base["s_pred"] = ac.scores_from(base["y1_pred"], base["y2_pred"],
                                        base["y3_pred"], base["has_gen"])
        per, agg = rank_of(base)
        tp = {}
        for tag in ("y1", "y2", "y3"):
            v = base["y3_valid"].values.astype(bool) if tag == "y3" else np.ones(len(base), bool)
            tp[tag] = {"pooled_spearman": ac.spearman(base.loc[v, f"{tag}_meas"],
                                                      base.loc[v, f"{tag}_pred"])}
        zs[pl] = {"ranking": agg, "targets": tp}
        print(f"   {other[pl]}->{pl}: top1={agg['top1']:.4f} "
              f"rho={agg['group_spearman_mean']:.4f}", flush=True)
R["zero_shot"] = {f"{other[pl]}->{pl}": zs[pl]["ranking"] for pl in ac.PLATFORMS}

# ------------------------------------------------------------ 3. joint paired
with stage("3_joint_paired_retrain"):
    Xj = pd.concat([DATA[pl]["X"] for pl in ac.PLATFORMS], ignore_index=True)
    Ynj = pd.concat([DATA[pl]["Yn"] for pl in ac.PLATFORMS], ignore_index=True)
    Mj = pd.concat([DATA[pl]["M"] for pl in ac.PLATFORMS], ignore_index=True)
    assert len(Xj) == 1080
    gj = ac.group_keys(Mj)                      # 플랫폼 제외 → 45 paired 그룹
    fj = ac.assign_group_folds(gj).values
    y3v = Mj["y3_valid"].values.astype(bool)
    oj = pd.DataFrame({"platform": Mj["platform"], "has_gen": Mj["has_gen"],
                       "y3_valid": y3v, "fold": fj,
                       "group": [str((m, r)) for m, r in
                                 zip(Mj["models"], Mj["rate_factor"])],
                       "y1_meas": Ynj["y1_total_throughput_fps"].values,
                       "y2_meas": Ynj["y2_deadline_miss_rate"].values,
                       "y3_meas": Ynj["y3_total_tokens_per_s"].values})
    for tag, col in ac.TARGETS:
        yv = Ynj[col].values.astype(float)
        valid = y3v if tag == "y3" else np.ones(len(yv), bool)
        pred = np.full(len(yv), np.nan)
        for f in range(ac.N_FOLDS):
            tr, te = (fj != f) & valid, fj == f
            params, _ = ac.cv_select_params(Xj.values[tr], yv[tr],
                                            gj[tr].reset_index(drop=True))
            bst = ac.train_booster(Xj.values[tr], yv[tr], params)
            pred[te] = ac.predict_booster(bst, Xj.values[te])
        oj[f"{tag}_pred"] = pred
    oj["s_meas"] = ac.scores_from(oj["y1_meas"], oj["y2_meas"], oj["y3_meas"], oj["has_gen"])
    oj["s_pred"] = ac.scores_from(oj["y1_pred"], oj["y2_pred"], oj["y3_pred"], oj["has_gen"])
    oj.to_csv(OUT / "joint_oof_paired.csv", index=False)
    R["joint"] = {}
    for pl in ac.PLATFORMS:
        _, agg = rank_of(oj[oj["platform"] == pl])
        R["joint"][pl] = agg
        print(f"   joint/{pl}: top1={agg['top1']:.4f} rho={agg['group_spearman_mean']:.4f}",
              flush=True)

# ----------------------------------------------------------- 4. decomposed
with stage("4_decomposed_retrain"):
    S = load_static_profiles(ac.STATIC_JSON)
    R["decomposed"] = {}
    R["decomposed_recon"] = {}
    for pl in ac.PLATFORMS:
        d = DATA[pl]
        w = pd.DataFrame({
            "combination": d["M"]["combination"].astype(str), "has_gen": d["M"]["has_gen"],
            "group": d["gstr"].values,
            "y1_raw": d["Yraw"]["y1_total_throughput_fps"].values,
            "y2_raw": np.clip(d["Yraw"]["y2_deadline_miss_rate"].values, 0, 1),
            "y1_meas": d["Yn"]["y1_total_throughput_fps"].values,
            "y2_meas": d["Yn"]["y2_deadline_miss_rate"].values,
            "y3_meas": d["Yn"]["y3_total_tokens_per_s"].values})
        w["fold"] = pd.Series(P1[pl]["fold"]).values          # P1 fold 상속 (재현본)
        w = w.merge(P1[pl]["oof"][["combination", "y3_pred"]], on="combination", how="left")
        gst = w.groupby("group").agg(fmax=("y1_raw", "max"), lo=("y2_raw", "min"),
                                     hi=("y2_raw", "max"))
        V = td.build_view_dataset(pl, S)
        V = V.merge(w[["combination", "group", "fold"]], on="combination", how="left")
        vis = V["is_vision"].values.astype(bool)
        Xv = V.loc[vis, td.VIEW_FEATURES].values
        yv = V.loc[vis, "target"].values.astype(float)
        grp = V.loc[vis, "group"].reset_index(drop=True)
        fold = V.loc[vis, "fold"].values.astype(int)
        pred = np.full(len(yv), np.nan)
        for f in range(ac.N_FOLDS):
            tr, te = fold != f, fold == f
            params, _ = ac.cv_select_params(Xv[tr], yv[tr], grp[tr].reset_index(drop=True),
                                            feature_names=td.VIEW_FEATURES)
            bst = ac.train_booster(Xv[tr], yv[tr], params, feature_names=td.VIEW_FEATURES)
            pred[te] = ac.predict_booster(bst, Xv[te], feature_names=td.VIEW_FEATURES)
        V.loc[vis, "pred"] = pred
        view_stats = {"n_vision_views": int(vis.sum()),
                      "n_clipped_targets": int((V.loc[vis, "ratio_raw"] > 1.0).sum()),
                      "oof_spearman_pooled": ac.spearman(yv, pred),
                      "oof_mae": float(np.mean(np.abs(yv - pred)))}
        agg = {}
        for name, col in (("pred", "pred"), ("meas", "target")):
            g = V[vis].groupby("combination")
            agg[name] = pd.DataFrame({
                f"y1sum_{name}": g.apply(lambda x: float((x[col] * x["eff"]).sum()),
                                         include_groups=False),
                f"miss_{name}": g.apply(
                    lambda x: float(np.clip((1.0 - x[col]).mean(), 0, 1)),
                    include_groups=False)})
        comp = w.set_index("combination").join(agg["pred"]).join(agg["meas"]).reset_index()
        comp = comp.join(gst, on="group")
        span = (comp["hi"] - comp["lo"]).values
        for n in ("pred", "meas"):
            comp[f"y1d_{n}"] = comp[f"y1sum_{n}"] / comp["fmax"]
            comp[f"y2d_{n}"] = np.where(span > 1e-9,
                                        (comp[f"miss_{n}"] - comp["lo"]) /
                                        np.where(span > 1e-9, span, 1.0), 0.0)
        recon = {"y2_raw_mae": float(np.mean(np.abs(comp["miss_meas"] - comp["y2_raw"]))),
                 "y2_raw_spearman": ac.spearman(comp["miss_meas"], comp["y2_raw"]),
                 "y2_group_spearman_mean": float(np.nanmean(list(
                     ac.per_group_spearman(comp.assign(a=comp["miss_meas"],
                                                       b=comp["y2_raw"]),
                                           "group", "a", "b").values())))}
        comp["s_meas"] = ac.scores_from(comp["y1_meas"], comp["y2_meas"], comp["y3_meas"],
                                        comp["has_gen"])
        comp["s_pred"] = ac.scores_from(comp["y1d_pred"], comp["y2d_pred"],
                                        comp["y3_pred"], comp["has_gen"])
        per, ragg = rank_of(comp)
        comp.to_csv(OUT / f"decomposed_comp_{pl}.csv", index=False)
        R["decomposed"][pl] = {**ragg, **view_stats}
        R["decomposed_recon"][pl] = recon
        print(f"   decomp/{pl}: top1={ragg['top1']:.4f} rho={ragg['group_spearman_mean']:.4f} "
              f"view_rho={view_stats['oof_spearman_pooled']:.4f}", flush=True)

# ---------------------------------------------------------------- 5. greedy
with stage("5_greedy"):
    FORCED = {"qwen2_vl"}
    def greedy_place(models, infps_of, S, accel, variant):
        rows = []
        for m in models:
            ci, _, _ = _device_static(m, "cpu", S)
            ai, _, _ = _device_static(m, accel, S)
            cap = 1000.0 / ai if ai and np.isfinite(ai) and ai > 0 else np.nan
            lf = infps_of[m] / cap if np.isfinite(cap) and cap > 0 else np.inf
            sp = (ci / ai) if (np.isfinite(ci) and np.isfinite(ai) and ai > 0) else np.inf
            rows.append({"model": m, "load_factor": lf, "speedup": sp})
        key = "speedup" if variant == "speedup" else "load_factor"
        order = sorted(rows, key=lambda r: (-r[key], r["model"]))
        cum, placed = 0.0, []
        for r in order:
            if r["model"] in FORCED:
                placed.append(r["model"]); cum += r["load_factor"]
            elif cum + r["load_factor"] <= 1.0:
                placed.append(r["model"]); cum += r["load_factor"]
        return tuple(sorted(placed))

    S = load_static_profiles(ac.STATIC_JSON)
    R["greedy"] = {}
    for pl in ac.PLATFORMS:
        d = DATA[pl]
        M = d["M"].assign(s_norm=ac.scores_from(
            d["Yn"]["y1_total_throughput_fps"], d["Yn"]["y2_deadline_miss_rate"],
            d["Yn"]["y3_total_tokens_per_s"], d["M"]["has_gen"]))
        imap = ac.infps_map_for(pl)
        for variant in ("speedup", "load_factor"):
            hits, ratios, unc, ncol = 0, [], 0, 0
            for (ms, rate), g in M.groupby([M["models"], M["rate_factor"]], sort=False):
                models = ms.split(",")
                infps_of = {}
                for m in models:
                    vals = {v for (mm, dev), v in imap[g["combination"].iloc[0]].items()
                            if mm == m}
                    infps_of[m] = max(vals) if vals else 0.0
                acc = greedy_place(models, infps_of, S, pl, variant)
                match = g[g["accel_set"] == acc]
                orc = g["s_norm"].max()
                if len(match):
                    ncol += 1
                    sm = float(match.iloc[0]["s_norm"])
                    hits += int(abs(sm - orc) < TIE)
                    ratios.append(sm / orc if abs(orc) > 1e-12 else np.nan)
                else:
                    unc += 1
            R["greedy"][f"{pl}/{variant}"] = {
                "n_measured": ncol, "n_unmeasured": unc, "top1_hits": hits,
                "top1_measured": hits / ncol, "top1_bound45": hits / 45,
                "oracle_ratio": float(np.nanmean(ratios))}
            print(f"   greedy {pl}/{variant}: {hits}/{ncol} unmeasured={unc}", flush=True)

# ------------------------------------------- 6. divergence / sweep / coverage
with stage("6_divergence_sweep_coverage"):
    spec = importlib.util.spec_from_file_location(
        "ba", ROOT / "runs/20260730_190154_score_basis_audit/build_basis_audit.py")
    ba = importlib.util.module_from_spec(spec); spec.loader.exec_module(ba)
    rows_pl = {pl: ba.normalize(ba.load_rows(pl)) for pl in ("gpu", "npu")}
    R["divergence"] = {}
    for beta in (1.0, 0.5):
        res, _, _ = ba.divergence(rows_pl, ba.BASE_ALPHA, beta, "normalized")
        R["divergence"][f"beta_{beta}"] = {
            "n_disagree": res["n_disagree"], "n_groups": res["n_groups"],
            "gen": res["gen"], "vision": res["vision"],
            "groups": sorted(res["disagree_groups"])}
    exh = ba.EXHAUSTIVE
    for beta in (1.0, 0.5):
        res, _, _ = ba.divergence(rows_pl, ba.BASE_ALPHA, beta, "normalized", keys=exh)
        R["divergence"][f"exhaustive_beta_{beta}"] = {
            "n_disagree": res["n_disagree"], "n_groups": res["n_groups"],
            "gen": res["gen"], "vision": res["vision"]}
    # Divergence margins and the App.H score-equivalence band. Recomputed from the
    # normalized rows, not read from B_divergence_gap -- the point of this suite is an
    # independent path. `any` reverts a group when EITHER margin is inside the band,
    # `both` when both are; App.H quotes the `any` number.
    from collections import defaultdict as _dd
    _by = {pl: _dd(list) for pl in rows_pl}
    for _pl, _rs in rows_pl.items():
        for _r in _rs:
            _by[_pl][(_r["sid"], _r["rate"])].append(_r)
    _ties = {pl: ba.tie_sets(rows_pl[pl], ba.BASE_ALPHA, 1.0, "normalized")
             for pl in rows_pl}
    _keys = sorted(_ties["gpu"])
    _dis = [k for k in _keys if not (_ties["gpu"][k] & _ties["npu"][k])]
    _margins = {}
    for k in _dis:
        m = {}
        for pl, other in (("gpu", "npu"), ("npu", "gpu")):
            sc = {r["placement"]: ba.score(r, ba.BASE_ALPHA, 1.0, "normalized")
                  for r in _by[pl][k]}
            m[pl] = max(sc.values()) - max(sc[p] for p in _ties[other][k] if p in sc)
        _margins[f"{k[0]}@{k[1]}"] = m
    # App.H "What differs." -- what actually distinguishes the two best placements in
    # each disagreeing group. GEN_MODELS are the generative ones; the claim is that they
    # sit identically in every disagreeing group and only the CPU-side vision model moves.
    _GEN = ("llama1b", "qwen2_vl")
    _same_gen, _npu_offloads_resnet = 0, 0
    _labels = {}
    for k in _dis:
        lab = {pl: sorted(ba.cpu_label(p_) for p_ in _ties[pl][k]) for pl in rows_pl}
        _labels[f"{k[0]}@{k[1]}"] = lab
        gen_side = {pl: {(m, d) for p_ in _ties[pl][k] for m, d in p_ if m in _GEN}
                    for pl in rows_pl}
        if gen_side["gpu"] == gen_side["npu"]:
            _same_gen += 1
        if lab["gpu"] == ["all-accel"] and lab["npu"] == ["cpu:resnet50"]:
            _npu_offloads_resnet += 1
    R["divergence_what_differs"] = {
        "n_groups": len(_dis),
        "n_same_generative_placement": _same_gen,
        "n_npu_offloads_resnet50_gpu_all_accel": _npu_offloads_resnet,
        "labels": _labels,
    }

    _flat = [v for m in _margins.values() for v in m.values()]
    R["divergence_margins"] = {
        "n_cells": len(_flat), "n_groups": len(_keys),
        "n_cells_below_0.01": sum(1 for g in _flat if g < 0.01),
        "min_margin": min(_flat), "median_margin": float(np.median(_flat)),
        "band_0.01": {
            "any": len(_dis) - sum(1 for m in _margins.values()
                                   if min(m.values()) < 0.01),
            "both": len(_dis) - sum(1 for m in _margins.values()
                                    if max(m.values()) < 0.01)},
    }

    sw = ba.sweep_setcounts(rows_pl, "normalized", ba.PRIMARY_ALPHAS, ba.PRIMARY_BETAS)
    R["sweep"] = {f"{a}/{b}": v for (a, b), v in sw.items()} if isinstance(sw, dict) else sw
    # App.G: "the two measured candidate sets are identical, every candidate places the
    # same instances on the CPU, and the CPU-side feature contribution is identical
    # across all 540 windows". The last clause is what carries the common-mode argument
    # against the thread-setting objection, so recompute it rather than assert it.
    # CPU-side contribution = the part of the 37 features that comes from CPU-placed
    # instances: their count, the sum of their isolated CPU latencies, and the sum of
    # their capacities (1000/latency). Those are identical by construction only if the
    # placements themselves are -- which is exactly what is being checked.
    _S_cpu = {r["model"]: r.get("cpu_infer")
              for r in json.loads(ac.STATIC_JSON.read_text())["total_data"]}

    def _cpu_side(pl):
        out = {}
        for w in ac.load_windows(pl):
            on_cpu = sorted(v["model"] for v in w["models"].values()
                            if _norm_exec(v["execution"]) == "cpu")
            lat = [_S_cpu[m] for m in on_cpu if _S_cpu.get(m)]
            out[str(w["combination"])] = {
                "models": tuple(on_cpu),
                "n": len(on_cpu),
                "sum_static_infer": round(sum(lat), 9),
                "sum_capacity": round(sum(1000.0 / v for v in lat), 9),
            }
        return out

    _cs = {pl: _cpu_side(pl) for pl in ac.PLATFORMS}
    _combos = sorted(set(_cs["gpu"]) | set(_cs["npu"]))
    _diff_place = [c for c in _combos
                   if _cs["gpu"].get(c, {}).get("models") != _cs["npu"].get(c, {}).get("models")]
    _diff_feat = [c for c in _combos
                  if tuple(_cs["gpu"].get(c, {}).get(k) for k in
                           ("n", "sum_static_infer", "sum_capacity"))
                  != tuple(_cs["npu"].get(c, {}).get(k) for k in
                           ("n", "sum_static_infer", "sum_capacity"))]
    R["cpu_side_identity"] = {
        "n_windows": {pl: len(_cs[pl]) for pl in ac.PLATFORMS},
        "n_windows_compared": len(_combos),
        "combination_keys_identical": sorted(_cs["gpu"]) == sorted(_cs["npu"]),
        "n_windows_cpu_placement_differs": len(_diff_place),
        "n_windows_cpu_feature_subtotal_differs": len(_diff_feat),
        "differing_combinations": (_diff_place + _diff_feat)[:10],
        "n_cpu_instances_total": sum(v["n"] for v in _cs["gpu"].values()),
        "n_windows_with_no_cpu_instance": sum(1 for v in _cs["gpu"].values() if v["n"] == 0),
    }

    # coverage
    cov = []
    for sid, models in SETS.items():
        free = [m for m in models if "cpu" in __import__("model_registry").allowed_devices(m)]
        n = len(models)
        meas = {pl: len({tuple(sorted((v["model"], _norm_exec(v["execution"]))
                                      for v in w["models"].values()))
                         for w in ac.load_windows(pl)
                         if SID_OF[",".join(sorted(v["model"] for v in w["models"].values()))] == sid})
                for pl in ac.PLATFORMS}
        cov.append({"set": sid, "N": n, "pow2": 2 ** n, "feasible": 2 ** len(free),
                    "measured_gpu": meas["gpu"], "measured_npu": meas["npu"]})
    R["coverage"] = cov

# --------------------------------------------------------------- 7. bootstrap
with stage("7_bootstrap"):
    T_ = []
    for pl in ac.PLATFORMS:
        pg = P1[pl]["per_group"].set_index("group")
        dg = pd.read_csv(OUT / f"decomposed_comp_{pl}.csv")
        dper, _ = rank_of(dg)
        dper = dper.set_index("group")
        for g in pg.index:
            T_.append({"platform": pl, "group": g,
                       "set": P1[pl]["oof"].set_index("group").loc[g, "set"]
                              if not isinstance(P1[pl]["oof"].set_index("group").loc[g, "set"], pd.Series)
                              else P1[pl]["oof"].set_index("group").loc[g, "set"].iloc[0],
                       "direct_top1": pg.loc[g, "top1"], "direct_rho": pg.loc[g, "spearman"],
                       "decomp_top1": dper.loc[g, "top1"]})
    B = pd.DataFrame(T_)
    # greedy per-group (하한 관례)
    gr = pd.read_csv(ac.ANALYSIS / "greedy_results.csv", comment="#")
    for _, r in gr.iterrows():
        gkey = None
        sel = B[(B["platform"] == r["platform"]) & (B["set"] == r["set"])]
        for g in sel["group"].unique():
            if str(float(r["rate"])) in g:
                gkey = g
        if gkey is None:
            continue
        B.loc[(B["platform"] == r["platform"]) & (B["group"] == gkey),
              f"greedy_{r['variant']}_top1"] = float(r["top1"]) if r["collected"] == 1 else 0.0
    SET_IDS = sorted(SETS)
    idx_by_set = {s: {pl: B.index[(B["set"] == s) & (B["platform"] == pl)].to_numpy()
                      for pl in ac.PLATFORMS} for s in SET_IDS}
    rng = np.random.RandomState(42)
    NB = 10000
    keys = ["direct_rho", "direct_top1", "decomp_top1",
            "greedy_speedup_top1", "greedy_load_factor_top1"]
    draws = {(k, pl): np.empty(NB) for k in keys for pl in ac.PLATFORMS}
    dd = {(a, pl): np.empty(NB) for a in ("d_greedy_speedup", "d_greedy_load_factor",
                                          "d_decomp") for pl in ac.PLATFORMS}
    for b in range(NB):
        picks = rng.randint(0, len(SET_IDS), len(SET_IDS))
        for pl in ac.PLATFORMS:
            idx = np.concatenate([idx_by_set[SET_IDS[i]][pl] for i in picks])
            sub = B.loc[idx]
            for k in keys:
                draws[(k, pl)][b] = np.nanmean(sub[k].values)
            dd[("d_greedy_speedup", pl)][b] = (sub["direct_top1"].mean()
                                               - sub["greedy_speedup_top1"].mean())
            dd[("d_greedy_load_factor", pl)][b] = (sub["direct_top1"].mean()
                                                   - sub["greedy_load_factor_top1"].mean())
            dd[("d_decomp", pl)][b] = sub["direct_top1"].mean() - sub["decomp_top1"].mean()
    def ci(a):
        a = a[np.isfinite(a)]
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    R["bootstrap"] = {f"{k}/{pl}": {"point": float(np.nanmean(B[B['platform'] == pl][k])),
                                    "ci95": ci(draws[(k, pl)])}
                      for k in keys for pl in ac.PLATFORMS}
    R["bootstrap"].update({f"{a}/{pl}": {"ci95": ci(dd[(a, pl)])}
                           for a in ("d_greedy_speedup", "d_greedy_load_factor", "d_decomp")
                           for pl in ac.PLATFORMS})
    # divergence fraction bootstrap
    dg1 = set(R["divergence"]["beta_1.0"]["groups"])
    div_flag = {s: sum(1 for g in dg1 if g.startswith(s + "@")) for s in SET_IDS}
    n_per_set = {s: 3 for s in SET_IDS}
    dv = np.empty(NB)
    rng2 = np.random.RandomState(42)
    for b in range(NB):
        picks = rng2.randint(0, len(SET_IDS), len(SET_IDS))
        num = sum(div_flag[SET_IDS[i]] for i in picks)
        den = sum(n_per_set[SET_IDS[i]] for i in picks)
        dv[b] = num / den
    R["bootstrap"]["divergence_fraction"] = {"point": 11 / 45, "ci95": ci(dv)}

# ------------------------------------------------ 8. y3 민감도 headline
with stage("8_y3_sensitivity"):
    tgt_rows = {pl: (DATA[pl]["M"]["has_gen"].values.astype(bool)
                     & ~DATA[pl]["M"]["y3_valid"].values.astype(bool))
                for pl in ac.PLATFORMS}
    ext = {}
    for pl in ac.PLATFORMS:
        v = P1[pl]["oof"].loc[tgt_rows[pl], "y3_pred"].values
        ext[pl] = [float(v.min()), float(v.max())]
    wins = {pl: {str(w["combination"]): float(w["total"].get("total_tokens_per_s", 0.0))
                 for w in ac.load_windows(pl)} for pl in ac.PLATFORMS}
    restored = {}
    for pl in ac.PLATFORMS:
        d = DATA[pl]
        Yr = d["Yraw"].copy()
        tot = d["M"]["combination"].astype(str).map(wins[pl]).values
        Yr["y3_total_tokens_per_s"] = np.where(tgt_rows[pl], tot,
                                               Yr["y3_total_tokens_per_s"].values)
        Yn2 = ac.normalize_targets(Yr, d["M"])
        restored[pl] = {"r3_max": float(tot[tgt_rows[pl]].max()),
                        "y3_norm_max": float(
                            Yn2["y3_total_tokens_per_s"].values[tgt_rows[pl]].max())}
    # App.A now prints the VLM's isolated CPU cost as the reason for pinning it to the
    # accelerator. Recompute those numbers from the static profile so they are checked
    # like every other printed value rather than being asserted in prose only.
    _S = json.loads(ac.STATIC_JSON.read_text())
    _q = {r["model"]: r for r in _S["total_data"]}["qwen2_vl"]
    _VLM_NEW_TOKENS = 32          # model_processors.run_vlm_process default
    _req_s = _q["cpu_prefill_ms"] / 1000.0 + _VLM_NEW_TOKENS / _q["cpu_tokens_per_s"]
    R["vlm_cpu_cost"] = {
        "prefill_s": {d: _q[f"{d}_prefill_ms"] / 1000.0 for d in ("cpu", "gpu", "npu")},
        "cpu_tokens_per_s": _q["cpu_tokens_per_s"],
        "new_tokens": _VLM_NEW_TOKENS,
        "cpu_request_s": _req_s,
        "completions_per_180s": 180.0 / _req_s,
    }

    # Sec. 3 now claims that restoring the recorded window totals "changes neither
    # Top-1 outcomes nor divergence labels". App.B makes the same claim. Recompute both
    # halves here from the restored labels so the two statements are checked against one
    # reproduced source rather than against each other.
    _rest = {}
    for pl in ac.PLATFORMS:
        d = DATA[pl]
        Yr = d["Yraw"].copy()
        tot = d["M"]["combination"].astype(str).map(wins[pl]).values
        Yr["y3_total_tokens_per_s"] = np.where(tgt_rows[pl], tot,
                                               Yr["y3_total_tokens_per_s"].values)
        Yn2 = ac.normalize_targets(Yr, d["M"])
        has_gen = d["M"]["has_gen"]
        f = pd.DataFrame({"group": d["gstr"].values})
        f["s_meas_pub"] = ac.scores_from(d["Yn"]["y1_total_throughput_fps"],
                                         d["Yn"]["y2_deadline_miss_rate"],
                                         d["Yn"]["y3_total_tokens_per_s"], has_gen)
        f["s_meas_res"] = ac.scores_from(Yn2["y1_total_throughput_fps"],
                                         Yn2["y2_deadline_miss_rate"],
                                         Yn2["y3_total_tokens_per_s"], has_gen)
        o = P1[pl]["oof"]
        f["s_pred"] = ac.scores_from(o["y1_pred"], o["y2_pred"], o["y3_pred"], has_gen)
        _, pub = rank_of(f.rename(columns={"s_meas_pub": "s_meas"}))
        _, res = rank_of(f.rename(columns={"s_meas_res": "s_meas"}))
        # tie-aware best set per group under each labelling
        best = {}
        for tag, col in (("pub", "s_meas_pub"), ("res", "s_meas_res")):
            best[tag] = {}
            for g, gg in f.groupby("group", sort=False):
                m = gg[col].values.astype(float)
                best[tag][str(g)] = frozenset(np.flatnonzero(np.abs(m - m.max()) < TIE))
        _rest[pl] = {"top1_published": pub["top1"], "top1_restored": res["top1"],
                     "top1_delta": res["top1"] - pub["top1"],
                     "n_groups_best_set_changed":
                         sum(1 for g in best["pub"] if best["pub"][g] != best["res"][g]),
                     "_best_pub": best["pub"], "_best_res": best["res"]}
    _dis_pub = sum(1 for g in _rest["gpu"]["_best_pub"]
                   if not (_rest["gpu"]["_best_pub"][g] & _rest["npu"]["_best_pub"][g]))
    _dis_res = sum(1 for g in _rest["gpu"]["_best_res"]
                   if not (_rest["gpu"]["_best_res"][g] & _rest["npu"]["_best_res"][g]))
    for pl in ac.PLATFORMS:
        _rest[pl].pop("_best_pub"); _rest[pl].pop("_best_res")
    R["y3_restored_invariance"] = {
        "per_platform": _rest,
        "n_target_windows": {pl: int(tgt_rows[pl].sum()) for pl in ac.PLATFORMS},
        "top1_changes_total": sum(1 for pl in ac.PLATFORMS
                                  if abs(_rest[pl]["top1_delta"]) > 1e-12),
        "divergence_published": _dis_pub,
        "divergence_restored": _dis_res,
        "divergence_labels_unchanged": _dis_pub == _dis_res,
    }

    R["y3_sensitivity"] = {"n_target_windows": {pl: int(tgt_rows[pl].sum())
                                                for pl in ac.PLATFORMS},
                           "extrapolated_yhat3_range": ext, "restored": restored}

# ------------------------------- 8b. 여태 대조되지 않던 인쇄값들의 재현 (지시문 #30)
# extract_expected 가 뽑기만 하고 compare 가 쓰지 않던 캡처 8건을 닫는다. 대부분은 이미
# R 에 대응값이 있었고, App.D 의 두 건과 App.B 의 0.013 만 새 계산이 필요하다.
with stage("8b_uncompared_printed_values"):
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location(
        "gb", ROOT / "xgboost_model/full_collection_540/scripts/greedy_baseline.py")
    gb = _ilu.module_from_spec(_sp); _sp.loader.exec_module(gb)
    from xgboost_model.deploy_selector_xgb_suite import load_static_profiles as _lsp
    _S = _lsp(ac.STATIC_JSON)

    # --- App.D: VLM 용량 소비 범위, cum>1.0 그룹 수, load-factor 에서 VLM 앞 admit 수
    gd = {}
    for pl in ac.PLATFORMS:
        M = DATA[pl]["M"]
        imap = ac.infps_map_for(pl)
        lf_vlm, admit_before, cum_over = [], [], {v: 0 for v in gb.VARIANTS}
        n_groups = 0
        for (models_str, rate), g in M.groupby([M["models"], M["rate_factor"]], sort=False):
            models = models_str.split(",")
            infps_of = {}
            for m in models:
                vals = {v for (mm, dev), v in imap[g["combination"].iloc[0]].items() if mm == m}
                infps_of[m] = max(vals) if vals else 0.0
            n_groups += 1
            for var in gb.VARIANTS:
                _, order, cum = gb.greedy_place(models, infps_of, _S, pl, var)
                if cum > 1.0:
                    cum_over[var] += 1
                if var == "load_factor" and "qwen2_vl" in models:
                    names = [r["model"] for r in order]
                    i = names.index("qwen2_vl")
                    lf_vlm.append(order[i]["load_factor"])
                    # 원고는 "already been admitted" 라고 쓴다 -- 스캔 순서가 아니라
                    # 실제로 가속기에 admit 된 수를 센다 (거부된 인스턴스는 제외).
                    admit_before.append(
                        sum(1 for r in order[:i] if r["decision"].startswith("accel")))
        gd[pl] = {"n_groups": n_groups,
                  "vlm_load_factor_min": min(lf_vlm), "vlm_load_factor_max": max(lf_vlm),
                  "n_groups_cum_exceeds_1": cum_over,
                  "vlm_admit_before_min": min(admit_before),
                  "vlm_admit_before_max": max(admit_before),
                  "n_groups_with_vlm": len(admit_before)}
    R["greedy_detail"] = gd

    # --- App.B: 외삽 yhat3 를 0 으로 두는 개입의 group-mean Spearman 변화 (8조합 최대)
    tgt = {pl: (DATA[pl]["M"]["has_gen"].values.astype(bool)
                & ~DATA[pl]["M"]["y3_valid"].values.astype(bool)) for pl in ac.PLATFORMS}
    jo = pd.read_csv(OUT / "joint_oof_paired.csv")
    cells = {}
    for pl in ac.PLATFORMS:
        d = DATA[pl]
        gstr, has_gen = d["gstr"].values, d["M"]["has_gen"]
        s_meas = ac.scores_from(d["Yn"]["y1_total_throughput_fps"],
                                d["Yn"]["y2_deadline_miss_rate"],
                                d["Yn"]["y3_total_tokens_per_s"], has_gen)
        cand = {}
        o = P1[pl]["oof"]
        cand["platform-specific"] = (o["y1_pred"].values, o["y2_pred"].values,
                                     o["y3_pred"].values)
        # zero-shot: 저장된 소스 부스터를 fold-matched 로 다시 태운다 (2단계와 같은 경로)
        _other = {"gpu": "npu", "npu": "gpu"}   # 상위 `other` 는 이 시점에 재사용됨
        tfold, zp = P1[pl]["fold"], {}
        for tag, _ in ac.TARGETS:
            pr = np.full(len(d["X"]), np.nan)
            for f in range(ac.N_FOLDS):
                m = tfold == f
                bst = ac.load_booster(OUT / f"gkf_cpu_{_other[pl]}_{tag}_fold{f}.json")
                pr[m] = ac.predict_booster(bst, d["X"].values[m])
            zp[tag] = pr
        cand["zero-shot"] = (zp["y1"], zp["y2"], zp["y3"])
        j = jo[jo["platform"] == pl].reset_index(drop=True)
        cand["joint"] = (j["y1_pred"].values, j["y2_pred"].values, j["y3_pred"].values)
        dc = pd.read_csv(OUT / f"decomposed_comp_{pl}.csv")
        dc = dc.set_index(dc["combination"].astype(str)).loc[
            d["M"]["combination"].astype(str)]
        cand["decomposed"] = (dc["y1d_pred"].values, dc["y2d_pred"].values,
                              dc["y3_pred"].values)
        for name, (p1_, p2_, p3_) in cand.items():
            rho = {}
            for tag, y3v in (("A1", p3_), ("A2", np.where(tgt[pl], 0.0, p3_))):
                sp = ac.scores_from(pd.Series(p1_), pd.Series(p2_), pd.Series(y3v), has_gen)
                fr = pd.DataFrame({"group": gstr, "s_meas": s_meas, "s_pred": sp})
                rho[tag] = rank_of(fr)[1]["group_spearman_mean"]
            cells[f"{name}/{pl}"] = {"A1": rho["A1"], "A2": rho["A2"],
                                     "abs_delta": abs(rho["A2"] - rho["A1"])}
    _arg = max(cells, key=lambda k: cells[k]["abs_delta"])
    R["appB_spearman_change"] = {
        "n_combinations": len(cells), "per_cell": cells,
        "max_abs_delta": cells[_arg]["abs_delta"], "attained_by": _arg,
        "rounded_3dp": round(cells[_arg]["abs_delta"], 3)}

# ------------------------------------------------------- 9. 결정론 스팟체크
with stage("9_determinism_spotcheck"):
    pl = "gpu"
    d = DATA[pl]
    fold = ac.assign_group_folds(d["groups"]).values
    yv = d["Yn"]["y1_total_throughput_fps"].values.astype(float)
    outs = []
    for _ in range(2):
        tr = fold != 0
        params, mae = ac.cv_select_params(d["X"].values[tr], yv[tr],
                                          d["groups"][tr].reset_index(drop=True))
        bst = ac.train_booster(d["X"].values[tr], yv[tr], params)
        outs.append((params["max_depth"], params["eta"], round(mae, 12),
                     float(ac.predict_booster(bst, d["X"].values[fold == 0]).sum())))
    R["determinism_spotcheck"] = {"run1": outs[0], "run2": outs[1],
                                  "identical": outs[0] == outs[1]}

R["_timings_sec"] = T
(HERE / "reproduced.json").write_text(json.dumps(R, indent=2, ensure_ascii=False,
                                                 default=str))
print("\n=== timings ===")
for k, v in T.items():
    print(f"  {k}: {v}s")
print("total", round(sum(T.values()), 1), "s")

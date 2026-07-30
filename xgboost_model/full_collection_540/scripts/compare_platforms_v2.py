"""P2: platform comparison — measured-score argmax per (set, rate) group.

Two score bases, chosen with --basis (default declared_normalized):

  declared_normalized  the manuscript's Eq. (1) per-group normalization. The score
      definitions are NOT reimplemented here: normalize()/score()/tie_sets() are
      imported from runs/20260730_190154_score_basis_audit/build_basis_audit.py, and
      the resulting disagreement set is asserted against that run's
      divergence_by_group.csv before anything is written.
  raw  the superseded v1 basis: raw measured window totals, kept so the historical
      29/45 and 30/45 figures stay reproducible.

Output paths are basis-scoped and disjoint — neither mode can touch the other's files:
  declared_normalized -> analysis/platform_divergence.{csv,md}
  raw                 -> analysis/platform_divergence_raw_v1.{csv,md}

Tie handling is the 1e-9 tie-set rule in BOTH modes (two platforms agree iff their
optimum sets intersect). The original first-max implementation is gone; on the raw
basis no group has a tie, so the historical numbers are unchanged by the switch.

--generated-at pins the provenance timestamp so a re-run can be diffed byte-for-byte
against the committed artifact.

  python compare_platforms_v2.py                      # declared basis (default)
  python compare_platforms_v2.py --basis raw
"""
import argparse
import csv as _csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as ac

BASIS_CHOICES = ("declared_normalized", "raw")
AUDIT_RUN = ac.ROOT / "runs" / "20260730_190154_score_basis_audit"
AUDIT_CSV = AUDIT_RUN / "divergence_by_group.csv"
BETAS = (1.0, 0.5)
# mode == exhaustive sets (runs/20260730_172053_coverage)
EXHAUSTIVE = ["S1", "S2", "S3", "S4", "base1", "base2", "base3", "base4", "base5", "S5"]
OUT_NAME = {"declared_normalized": "platform_divergence",
            "raw": "platform_divergence_raw_v1"}
SCORE_BASIS_LABEL = {"declared_normalized": "declared_normalized (Eq. 1)",
                     "raw": "raw measured window totals"}
# Row order of the v1 csv, kept so raw mode reproduces the preserved artifact
# byte-for-byte. The declared csv is ordered by sorted((set, rate)) instead, matching
# the audit run's divergence_by_group.csv it is checked against.
V1_SET_ORDER = ["S1", "S2", "S3", "S4", "base1", "base2", "base3", "base4", "base5",
                "S5", "S6", "S7", "S9", "S8", "S10"]


def load_audit_module():
    """Import the audit run's build_basis_audit; its definitions are the single source."""
    path = AUDIT_RUN / "build_basis_audit.py"
    spec = importlib.util.spec_from_file_location("build_basis_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audit_csv_diff_groups(basis, beta):
    out = set()
    with open(AUDIT_CSV) as fh:
        for r in _csv.DictReader(fh):
            if r["basis"] == basis and float(r["beta"]) == beta and int(r["agree"]) == 0:
                out.add(f"{r['set']}@{float(r['rate'])}")
    return out


def _rows_and_mode(bba, basis):
    if basis == "declared_normalized":
        return ({pl: bba.normalize(bba.load_rows(pl), "r1_totals")
                 for pl in ("gpu", "npu")}, "normalized")
    return {pl: bba.load_rows(pl) for pl in ("gpu", "npu")}, "raw"


def compute(bba, basis):
    """{beta: {(sid, rate): (gpu_tie_set, npu_tie_set)}} for both betas."""
    rows, mode = _rows_and_mode(bba, basis)
    out = {}
    for beta in BETAS:
        tg = bba.tie_sets(rows["gpu"], bba.ALPHA, beta, mode)
        tn = bba.tie_sets(rows["npu"], bba.ALPHA, beta, mode)
        out[beta] = {k: (tg[k], tn[k]) for k in tg}
    return out


def check_asserts(res, basis):
    """Declared mode: the disagreement set must match the audit run's csv exactly."""
    if basis != "declared_normalized":
        return {"checked": False,
                "reason": "raw basis is verified by reproducing 29/45 and 30/45, "
                          "not against the audit csv"}
    detail = {}
    for beta in BETAS:
        got = {f"{k[0]}@{k[1]}" for k, (tg, tn) in res[beta].items() if not (tg & tn)}
        want = audit_csv_diff_groups("declared_normalized", beta)
        assert got == want, (
            f"beta={beta}: declared-basis disagreement set differs from the audit csv\n"
            f"  computed-only: {sorted(got - want)}\n"
            f"  csv-only:      {sorted(want - got)}")
        detail[str(beta)] = len(got)
    return {"checked": True, "passed": True, "n_disagree_by_beta": detail,
            "source": str(AUDIT_CSV.relative_to(ac.ROOT))}


def labels(bba, ties):
    return "|".join(sorted(bba.cpu_label(p) for p in ties))


def write_csv(bba, res, basis, path):
    prov = {"score_basis": SCORE_BASIS_LABEL[basis], "alpha": bba.ALPHA,
            "betas": list(BETAS), "group": "(set, rate)",
            "tie_rule": "1e-9 tie set; intersecting tie sets = agreement",
            "source": str(AUDIT_CSV.relative_to(ac.ROOT)),
            "source_sha256": hashlib.sha256(AUDIT_CSV.read_bytes()).hexdigest()}
    prov["supersedes" if basis == "declared_normalized" else "superseded_by"] = (
        "platform_divergence_raw_v1.csv (raw measured window totals)"
        if basis == "declared_normalized"
        else "platform_divergence.csv (declared_normalized, Eq. 1)")
    cols = ["basis", "beta", "set", "rate", "has_gen", "gpu_best", "npu_best", "agree"]
    lines = ["# provenance: " + json.dumps(prov, ensure_ascii=False), ",".join(cols)]
    for beta in BETAS:
        for k in sorted(res[beta]):
            tg, tn = res[beta][k]
            lines.append(
                f"{basis},{beta},{k[0]},{k[1]},{int(bba.HAS_GEN[k[0]])},"
                f'"{labels(bba, tg)}","{labels(bba, tn)}",{int(bool(tg & tn))}')
    path.write_text("\n".join(lines) + "\n")
    return len(lines) - 2


def write_csv_v1_schema(bba, res, path, generated_at):
    """Raw mode keeps the v1 column schema so the preserved artifact stays comparable.

    The declared mode's unified schema would silently drop v1's best-score and
    best-combination columns, so raw mode reproduces the original layout instead:
    accelerator sets, the agreement flag, both best scores and both best combos.
    """
    rows, _ = _rows_and_mode(bba, "raw")
    by_group = {pl: defaultdict(list) for pl in ("gpu", "npu")}
    for pl in ("gpu", "npu"):
        for r in rows[pl]:
            by_group[pl][(r["sid"], r["rate"])].append(r)
    prov = {"generated_at": generated_at,
            "score_basis": SCORE_BASIS_LABEL["raw"], "alpha": bba.ALPHA,
            "betas": list(BETAS), "group": "(set, rate)",
            "tie_rule": "1e-9 tie set; intersecting tie sets = agreement",
            "generator": "scripts/compare_platforms_v2.py --basis raw",
            "original_generation": "2026-07-29T18:32:09+09:00, commit "
                                   "475a2089eb49a1906d663508055d3aeb5d5e2d43",
            "superseded_by": "platform_divergence.csv (declared_normalized, Eq. 1)"}
    cols = ["beta", "set", "rate", "gpu_best_accel", "npu_best_accel", "same",
            "gpu_best_score", "npu_best_score", "gpu_best_combo", "npu_best_combo"]
    lines = ["# provenance: " + json.dumps(prov, ensure_ascii=False), ",".join(cols)]
    order = [k for sid in V1_SET_ORDER
             for k in sorted((kk for kk in res[BETAS[0]] if kk[0] == sid),
                             key=lambda kk: kk[1])]
    for beta in BETAS:
        for k in order:
            tg, tn = res[beta][k]
            best = {}
            for pl, ties in (("gpu", tg), ("npu", tn)):
                # ties are empty of ambiguity on the raw basis; take the top row and,
                # if a tie ever appears, the lexicographically first placement in it
                cand = [r for r in by_group[pl][k] if r["placement"] in ties]
                r = sorted(cand, key=lambda r: r["placement"])[0]
                accel = ",".join(sorted(m for m, d in r["placement"] if d != "cpu"))
                best[pl] = (accel, bba.score(r, bba.ALPHA, beta, "raw"), r["combo"])
            lines.append(
                f'{beta},{k[0]},{k[1]},"{best["gpu"][0]}","{best["npu"][0]}",'
                f'{int(bool(tg & tn))},{best["gpu"][1]:.3f},{best["npu"][1]:.3f},'
                f'{best["gpu"][2]},{best["npu"][2]}')
    path.write_text("\n".join(lines) + "\n")
    return len(lines) - 2


def summarize(bba, res, beta, keys=None):
    items = [(k, v) for k, v in res[beta].items() if keys is None or k[0] in keys]
    dis = [k for k, (tg, tn) in items if not (tg & tn)]
    gen_n = sum(1 for k, _ in items if bba.HAS_GEN[k[0]])
    gen_d = sum(1 for k in dis if bba.HAS_GEN[k[0]])
    return {"n": len(items), "same": len(items) - len(dis), "diff": len(dis),
            "rate": len(dis) / len(items) if items else 0.0,
            "gen": (gen_d, gen_n), "vision": (len(dis) - gen_d, len(items) - gen_n),
            "groups": sorted(f"{k[0]}@{k[1]}" for k in dis)}


def by_model_count(bba, res, beta):
    acc = defaultdict(lambda: [0, 0])
    for k, (tg, tn) in res[beta].items():
        acc[len(bba.SETS[k[0]])][1] += 1
        if not (tg & tn):
            acc[len(bba.SETS[k[0]])][0] += 1
    return {n: tuple(v) for n, v in sorted(acc.items())}


def flip_groups(res):
    a = {f"{k[0]}@{k[1]}" for k, (tg, tn) in res[1.0].items() if not (tg & tn)}
    b = {f"{k[0]}@{k[1]}" for k, (tg, tn) in res[0.5].items() if not (tg & tn)}
    return sorted(a ^ b)


def own_optimum_moves(bba, basis):
    """Per platform, groups whose OWN tie set moves between beta=1.0 and beta=0.5."""
    rows, mode = _rows_and_mode(bba, basis)
    out = {}
    for pl in ("gpu", "npu"):
        t10 = bba.tie_sets(rows[pl], bba.ALPHA, 1.0, mode)
        t05 = bba.tie_sets(rows[pl], bba.ALPHA, 0.5, mode)
        out[pl] = sorted(f"{k[0]}@{k[1]}" for k in t10 if not (t10[k] & t05[k]))
    return out


def render_md(blocks):
    """Join md blocks and collapse runs of blank lines to exactly one.

    Blocks are appended with and without trailing newlines depending on the section,
    so normalize here instead of hand-tuning every append.
    """
    text = "\n".join(blocks)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.rstrip("\n") + "\n"


def prov_header(basis, generated_at, extra_lines=()):
    head = ["<!-- provenance", f"  generated_at: {generated_at}",
            f"  score_basis: {SCORE_BASIS_LABEL[basis]}"]
    if basis == "declared_normalized":
        head += [
            "  basis_definition: group = (set, rate); y1 = r1/max(r1), y3 = r3/max(r3),",
            "    y2 = (r2-min r2)/(max r2-min r2) with y2 = 0 when max == min;",
            "    S = y1 - alpha*y2 + beta*y3 (beta term only for generative sets)"]
    head += [f"  alpha: {ac.ALPHA}", f"  betas: {list(BETAS)}",
             "  tie_rule: 1e-9 tie set; intersecting tie sets count as agreement",
             "  predictor_involved: no (measured windows only)",
             "  input_data: full_collection_540/{cpu_gpu,cpu_npu}/"
             "performance_{gpu,npu}_full540.json",
             f"  generator: scripts/compare_platforms_v2.py --basis {basis}",
             "  score_definitions_from: runs/20260730_190154_score_basis_audit/"
             "build_basis_audit.py"]
    head += list(extra_lines)
    head.append("-->")
    return "\n".join(head) + "\n\n"


def md_declared(bba, res, generated_at, n_csv_rows):
    s10, s05 = summarize(bba, res, 1.0), summarize(bba, res, 0.5)
    e10 = summarize(bba, res, 1.0, keys=EXHAUSTIVE)
    e05 = summarize(bba, res, 0.5, keys=EXHAUSTIVE)
    bn = by_model_count(bba, res, 1.0)
    flips = flip_groups(res)
    moves = own_optimum_moves(bba, "declared_normalized")
    extra = [
        "  recomputation_run: runs/20260730_190154_score_basis_audit",
        "  source_rows: runs/20260730_190154_score_basis_audit/divergence_by_group.csv",
        "    (basis == declared_normalized)",
        "  supersedes: platform_divergence_raw_v1.{md,csv}",
        "  figure: analysis/figures/fig_divergence.{pdf,png}",
        "    (scripts/make_figures.py --fig f1 --basis declared_normalized)",
    ]
    bn_txt = ", ".join((f"**N={n} {d}/{t}**" if n == 4 else f"N={n} {d}/{t}")
                       for n, (d, t) in bn.items())
    L = [prov_header("declared_normalized", generated_at, extra)]
    L.append("# P2 (v2) — 플랫폼별 최적 배치 불일치, **선언 기저(식 1)** 기준\n")
    L.append("점수: `S = y1 − 0.3·y2 + β·y3` (β 항은 생성 모델 포함 세트만), "
             "**그룹 (세트, rate) 내\n정규화된 측정값** 기준. 예측기 미사용.\n")
    L.append("## v1(raw 기저)과의 차이\n")
    L.append("v1은 같은 데이터에 **원시 측정 창 총계**를 그대로 넣어 점수화했다"
             "(β=1.0에서 29/45 불일치).\n"
             "그 기저에서는 y1이 8–455 fps인데 α·y2 ≤ 0.3이므로 argmax가 사실상 "
             "`y1 + β·y3`로 결정되고,\n"
             "**GPU의 y3 최대 180.8 tok/s vs NPU 22.6 tok/s — 약 8배의 단위 스케일 격차**가 "
             "그대로 플랫폼\n간 argmax 차이로 환산되었다. 식 (1)의 그룹 정규화를 적용하면 "
             "y3가 그룹 내 [0,1]로 압축되어\n"
             f"이 격차가 사라지고, 불일치는 **29/45 → {s10['diff']}/{s10['n']}** 로 "
             "떨어진다. 즉 v1 수치의 상당 부분은 플랫폼\n"
             "간 실제 배치 선호 차이가 아니라 y3 단위 스케일의 반영이었다. "
             "기저 감사·재계산 전체 근거는\n"
             "`runs/20260730_190154_score_basis_audit/basis_audit.md` "
             "(파트 1 기저 감사표, 파트 2a/2b).\n")
    L.append("v1 산출물은 `platform_divergence_raw_v1.{md,csv}`로 보존되어 있다 "
             "(`--basis raw`로 재생성).\n")
    L.append("## 요약\n")
    L.append("| β | 동일 배치 | 다른 배치 | 불일치율 | 생성 세트 | vision-only |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| **1.0** | {s10['same']} | **{s10['diff']}** | **{s10['rate']:.1%}** | "
             f"{s10['gen'][0]} / {s10['gen'][1]} | "
             f"**{s10['vision'][0]} / {s10['vision'][1]}** |")
    L.append(f"| 0.5 | {s05['same']} | **{s05['diff']}** | {s05['rate']:.1%} | "
             f"{s05['gen'][0]} / {s05['gen'][1]} | "
             f"{s05['vision'][0]} / {s05['vision'][1]} |")
    L.append("")
    L.append(f"(v1 raw 기저 대비: β=1.0 29/45 → {s10['diff']}/{s10['n']}, "
             f"β=0.5 30/45 → {s05['diff']}/{s05['n']}.)\n")
    L.append("tie가 발생한 그룹은 양 β에서 0이므로 tie 규칙은 결과에 영향을 주지 않는다.\n")
    L.append(f"## β=1.0 불일치 {s10['diff']}그룹\n")
    L.append(", ".join(f"`{g}`" for g in s10["groups"]) + "\n")
    L.append(f"모델 수 N별 분해: {bn_txt}.\n")
    L.append("**vision-only 0/12 구조는 이 기저에서 성립하지 않는다.** `S10@2.0`이 "
             "vision-only이면서\n불일치한다 (GPU 최적 `cpu:mobilenet_v2` vs NPU 최적 "
             "`all-accel`). raw 기저에서는 α·y2가\n"
             "y1에 압도되어 vision 세트의 argmax가 양 플랫폼에서 항상 같았지만, 정규화하면 "
             "α·y2가 실제로\n작동하기 때문이다. v1의 \"생성 29/33 · vision 0/12\" 서술은 "
             f"v2에서 \"생성 {s10['gen'][0]}/{s10['gen'][1]} · "
             f"vision {s10['vision'][0]}/{s10['vision'][1]}\"로\n교체된다.\n")
    L.append("## β-민감 그룹\n")
    L.append(f"**일치/불일치 판정이 뒤집히는 그룹 {len(flips)}개** "
             "(전부 β=1.0 일치 → β=0.5 불일치):\n")
    L.append(", ".join(f"`{g}`" for g in flips) + "\n")
    L.append("참고로 **자기 optimum이 이동하는** 그룹(판정 뒤집힘과는 다른 양)은 "
             f"GPU {len(moves['gpu'])}개("
             + ", ".join(f"`{g}`" for g in moves["gpu"])
             + f"),\nNPU {len(moves['npu'])}개("
             + ", ".join(f"`{g}`" for g in moves["npu"]) + ")다.\n")
    L.append("v1(raw 기저)의 판정 뒤집힘은 `S8@1.5` 1개였다 — **v2와 목록이 전혀 "
             "겹치지 않는다.** β 민감도\n서술 역시 점수 기저에 의존한다.\n")
    L.append("## exhaustive 그룹 한정 (부록 E용)\n")
    L.append("mode == exhaustive인 10세트 30그룹"
             "(`runs/20260730_172053_coverage` 기준) 한정:\n")
    L.append("| β | raw 기저 (v1) | **선언 기저 (v2)** |")
    L.append("|---|---|---|")
    L.append(f"| 1.0 | 22/30 (73.3%) | **{e10['diff']}/{e10['n']} "
             f"({e10['rate']:.1%})** |")
    L.append(f"| 0.5 | 22/30 | **{e05['diff']}/{e05['n']} ({e05['rate']:.1%})** |")
    L.append("")
    L.append(f"선언 기저 β=1.0의 {e10['diff']}그룹: "
             + ", ".join(f"`{g}`" for g in e10["groups"])
             + f". vision-only는 {e10['vision'][0]}/{e10['vision'][1]} "
               "(raw와 동일 — `S10@2.0`은 sampled\n세트라 이 30그룹에 포함되지 않는다).\n")
    L.append("주의: raw 기저에서는 \"exhaustive 한정 불일치율이 전체보다 높다\""
             "(73.3% vs 64.4%)고 쓸 수\n"
             f"있었으나, 선언 기저에서는 {e10['rate']:.1%} vs {s10['rate']:.1%}로 "
             "차이가 거의 없다.\n")
    L.append("## 그룹별 전체 내역\n")
    L.append(f"`platform_divergence.csv` ({n_csv_rows}행 = 2 β × {s10['n']}그룹). 컬럼:\n"
             "`basis, beta, set, rate, has_gen, gpu_best, npu_best, agree`.\n"
             "`gpu_best`/`npu_best`는 tie set을 CPU 배치 모델 라벨로 표기한다\n"
             "(`all-accel` = 전 모델 가속기, `cpu:X+Y` = X와 Y만 CPU).")
    return render_md(L)


def md_raw(bba, res, generated_at, n_csv_rows):
    s10, s05 = summarize(bba, res, 1.0), summarize(bba, res, 0.5)
    flips = flip_groups(res)
    extra = [
        "  original_generation: 2026-07-29T18:32:09+09:00, commit "
        "475a2089eb49a1906d663508055d3aeb5d5e2d43",
        "  superseded_by: platform_divergence.{md,csv} (declared_normalized, Eq. 1)",
        "  status: retained for continuity with the pre-audit reports; the manuscript's",
        "    Eq. (1) basis is the declared_normalized variant",
    ]
    L = [prov_header("raw", generated_at, extra)]
    L.append("# P2 (v1, 폐기) — 플랫폼별 최적 배치 불일치, **원시 측정 총계** 기준\n")
    L.append("점수: `S = y1 − 0.3·y2 + β·y3` (생성 세트만), **원시 측정 창 총계** 기준 "
             "— 예측기 미사용.\n")
    L.append("> **이 기저는 원고 식 (1)이 선언한 기저가 아니다.** 식 (1)은 그룹 "
             "(세트, rate) 내 정규화를\n"
             "> 규정하므로, 논문 수치는 `platform_divergence.{md,csv}`(선언 기저)를 "
             "써야 한다. 이 파일은\n"
             "> 감사 이전 보고서와의 연속성 확인용으로만 유지된다. 근거: "
             "`runs/20260730_190154_score_basis_audit`.\n")
    L.append("| β | 동일 배치 | 다른 배치 | 불일치율 | 생성 세트 | vision-only |")
    L.append("|---|---|---|---|---|---|")
    for s, b in ((s10, "**1.0**"), (s05, "0.5")):
        L.append(f"| {b} | {s['same']} | **{s['diff']}** | {s['rate']:.1%} | "
                 f"{s['gen'][0]} / {s['gen'][1]} | {s['vision'][0]} / {s['vision'][1]} |")
    L.append("")
    L.append(f"β에 민감한 그룹 (한쪽 β에서만 불일치): {len(flips)}개 — "
             + ", ".join(f"`{g}`" for g in flips) + "\n")
    L.append(f"그룹별 전체 내역은 `platform_divergence_raw_v1.csv` "
             f"({n_csv_rows}행 = 2 β × {s10['n']}그룹) 참조.")
    return render_md(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=list(BASIS_CHOICES),
                    default="declared_normalized",
                    help="score basis (default: the manuscript's Eq. (1) basis)")
    ap.add_argument("--generated-at", default=None,
                    help="pin the provenance timestamp for byte-identical re-runs")
    args = ap.parse_args()
    basis = args.basis
    generated_at = args.generated_at or datetime.now().astimezone().isoformat(
        timespec="seconds")

    bba = load_audit_module()
    res = compute(bba, basis)
    checks = check_asserts(res, basis)

    stem = OUT_NAME[basis]
    # Hard guard: a mode must never be able to write the other mode's artifacts.
    forbidden = {OUT_NAME[b] for b in BASIS_CHOICES if b != basis}
    assert stem not in forbidden, stem
    csv_path, md_path = ac.ANALYSIS / f"{stem}.csv", ac.ANALYSIS / f"{stem}.md"

    n_rows = (write_csv(bba, res, basis, csv_path)
              if basis == "declared_normalized"
              else write_csv_v1_schema(bba, res, csv_path, generated_at))
    md_fn = md_declared if basis == "declared_normalized" else md_raw
    md_path.write_text(md_fn(bba, res, generated_at, n_rows))

    s10, s05 = summarize(bba, res, 1.0), summarize(bba, res, 0.5)
    print(json.dumps({
        "basis": basis, "generated_at": generated_at,
        "outputs": [str(csv_path.relative_to(ac.ROOT)),
                    str(md_path.relative_to(ac.ROOT))],
        "beta_1.0": {"diff": s10["diff"], "n": s10["n"], "gen": s10["gen"],
                     "vision": s10["vision"]},
        "beta_0.5": {"diff": s05["diff"], "n": s05["n"], "gen": s05["gen"],
                     "vision": s05["vision"]},
        "asserts": checks,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ac.ROOT,
                                     capture_output=True, text=True).stdout.strip(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

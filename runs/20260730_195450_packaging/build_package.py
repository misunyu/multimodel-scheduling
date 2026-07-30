"""MLForSys submission packaging, driven by the manuscript/ read-only mirror.

manuscript/ is never written to: this script reads and copies only.

Stages:
  0b  mirror currency gates (all must pass)
  1   dependency collection parsed OUT OF the .tex (no hardcoded file list)
  2   isolated compile (pdflatex -> bibtex -> pdflatex x2) + acceptance criteria
  3   dist/mlforsys_submission_<YYYYMMDD>.zip, only if stage 2 passes

Stages 1-3 stop at the first hard failure and record why; nothing is packaged from
an incomplete or unverified input set.

Usage: python runs/<ts>_packaging/build_package.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
OUT = Path(__file__).resolve().parent
TEX = ROOT / "manuscript" / "mlforsys_main.tex"
FIGDIR = ROOT / "xgboost_model" / "full_collection_540" / "analysis" / "figures"
DIST = ROOT / "dist"

# The v2 declared-basis figure. Any other copy is a stale pre-audit render.
FIG_DIVERGENCE_SHA256 = \
    "3c13b1dfe82814c0df6a7a4b7b715b35cd08912c316eaa215e0a575c098f1fd5"

# LaTeX packages shipped with any TeX distribution -> not local dependencies.
STOCK_PACKAGES = {
    "inputenc", "fontenc", "hyperref", "url", "booktabs", "amsmath", "amssymb",
    "amsfonts", "graphicx", "multirow", "xcolor", "enumitem", "caption", "natbib",
}
# Strings that must NOT survive into the rendered PDF (superseded raw-basis figures).
FORBIDDEN_IN_PDF = ["29 of 45", "29/45", "22/30", "7 of 15"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------- stage 0b ----------
def mirror_gates(tex_text: str) -> dict:
    gates = {
        "revision_marker_r2": "REVISION: 2026-07-30 r2" in tex_text,
        "has_11_of_45": "11 of 45" in tex_text,
        "no_29_of_45": "29 of 45" not in tex_text,
        "no_architecture_overview": "architecture_overview" not in tex_text,
    }
    return {"gates": gates, "passed": all(gates.values())}


# ---------- stage 1 ----------
def parse_dependencies(tex_text: str) -> dict:
    """Derive the dependency list from the .tex itself."""
    body = "\n".join(l for l in tex_text.splitlines() if not l.lstrip().startswith("%"))
    bibs = [f"{n.strip()}.bib"
            for m in re.findall(r"\\bibliography\{([^}]*)\}", body)
            for n in m.split(",")]
    stys = []
    for m in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", body):
        for n in (x.strip() for x in m.split(",")):
            if n and n not in STOCK_PACKAGES:
                stys.append(f"{n}.sty")
    figs = [m.strip() for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}",
                                         body)]
    return {"bib": bibs, "sty": sorted(set(stys)), "figures": figs}


def locate(name: str) -> list[Path]:
    """manuscript/ -> analysis/figures/ -> whole repo (git objects excluded)."""
    hits = []
    for cand in (TEX.parent / name, FIGDIR / name):
        if cand.is_file():
            hits.append(cand)
    if not hits:
        for p in ROOT.rglob(name):
            if ".git" in p.parts or ".venv" in p.parts:
                continue
            if p.is_file():
                hits.append(p)
    # de-dup, keep search order
    seen, ordered = set(), []
    for p in hits:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            ordered.append(p)
    return ordered


def collect(deps: dict) -> dict:
    found, missing, notes = {}, [], []
    wanted = [("bib", n) for n in deps["bib"]] + \
             [("sty", n) for n in deps["sty"]] + \
             [("figure", n) for n in deps["figures"]]
    for kind, name in wanted:
        hits = locate(name)
        if not hits:
            missing.append({"kind": kind, "name": name,
                            "searched": ["manuscript/", "analysis/figures/",
                                         "whole repo (rglob)"]})
            continue
        pick = hits[0]
        if name == "fig_divergence.pdf":
            matching = [p for p in hits if sha256(p) == FIG_DIVERGENCE_SHA256]
            if not matching:
                return {"found": found, "missing": missing,
                        "hard_failure": (
                            "fig_divergence.pdf found but NO copy matches the required "
                            f"sha256 {FIG_DIVERGENCE_SHA256} -- a stale pre-audit render "
                            "would be packaged"),
                        "candidates": [{"path": str(p.relative_to(ROOT)),
                                        "sha256": sha256(p)} for p in hits],
                        "notes": notes}
            pick = matching[0]
            if len(hits) > 1:
                notes.append(f"{name}: {len(hits)} copies found, took the sha256 match "
                             f"({pick.relative_to(ROOT)})")
        found[name] = {"kind": kind, "path": str(pick.relative_to(ROOT)),
                       "sha256": sha256(pick), "bytes": pick.stat().st_size,
                       "other_copies": [str(p.relative_to(ROOT)) for p in hits[1:]]}
    return {"found": found, "missing": missing, "hard_failure": None, "notes": notes}


# ---------- stage 2 ----------
def toolchain() -> dict:
    tools = ["pdflatex", "bibtex", "pdftotext"]
    return {t: (shutil.which(t) or None) for t in tools}


def compile_isolated(collected: dict, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEX, workdir / TEX.name)
    for name, info in collected["found"].items():
        shutil.copy2(ROOT / info["path"], workdir / name)
    stem = TEX.stem
    logs = []
    for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem],
                ["bibtex", stem],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem]):
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        logs.append({"cmd": " ".join(cmd), "returncode": r.returncode,
                     "tail": (r.stdout or "")[-2000:]})
        if r.returncode != 0 and cmd[0] == "pdflatex":
            return {"ran": True, "logs": logs, "passed": False,
                    "reason": f"{cmd[0]} exited {r.returncode}"}
    pdf = workdir / f"{stem}.pdf"
    if not pdf.is_file():
        return {"ran": True, "logs": logs, "passed": False, "reason": "no PDF produced"}
    log_text = (workdir / f"{stem}.log").read_text(errors="replace")
    full = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                          text=True).stdout
    p5 = subprocess.run(["pdftotext", "-f", "5", "-l", "5", str(pdf), "-"],
                        capture_output=True, text=True).stdout
    crit = {
        "no_errors": "! " not in log_text,
        "no_undefined_refs": "There were undefined references" not in log_text
                             and "Citation" not in log_text.replace(
                                 "Citation(s) may have changed", ""),
        "body_ends_page_4_refs_start_page_5":
            p5.strip().splitlines()[0].strip() == "References" if p5.strip() else False,
        "no_superseded_strings": {s: full.count(s) for s in FORBIDDEN_IN_PDF},
        "no_red_placeholders": "[TODO:" not in full,
    }
    crit["no_superseded_strings_ok"] = all(v == 0 for v in
                                          crit["no_superseded_strings"].values())
    passed = (crit["no_errors"] and crit["no_undefined_refs"]
              and crit["body_ends_page_4_refs_start_page_5"]
              and crit["no_superseded_strings_ok"] and crit["no_red_placeholders"])
    return {"ran": True, "logs": logs, "criteria": crit, "passed": passed,
            "pdf": str(pdf)}


def main():
    tex_text = TEX.read_text()
    report = {
        "run_id": OUT.name,
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manuscript": {"path": str(TEX.relative_to(ROOT)), "sha256": sha256(TEX),
                       "modified_by_this_run": False},
        "stage_0b_mirror": mirror_gates(tex_text),
    }
    if not report["stage_0b_mirror"]["passed"]:
        report["stopped_at"] = "0b"
        report["stop_reason"] = "mirror is stale; see failing gates"
        finish(report)
        return report

    deps = parse_dependencies(tex_text)
    report["stage_1_dependencies"] = {"parsed_from_tex": deps}
    collected = collect(deps)
    report["stage_1_dependencies"]["collected"] = {
        k: v for k, v in collected.items() if k != "found"}
    report["stage_1_dependencies"]["found"] = collected["found"]
    report["stage_1_dependencies"]["figure_count_is_2"] = len(deps["figures"]) == 2

    if collected["hard_failure"]:
        report["stopped_at"] = "1"
        report["stop_reason"] = collected["hard_failure"]
        finish(report)
        return report
    if collected["missing"]:
        report["stopped_at"] = "1"
        report["stop_reason"] = (
            "missing dependencies: "
            + ", ".join(m["name"] for m in collected["missing"])
            + ". Normal recovery: copy them into manuscript/ and re-run.")
        report["stage_2_compile"] = {"ran": False,
                                     "reason": "blocked by stage 1",
                                     "toolchain": toolchain()}
        report["stage_3_package"] = {"created": False, "reason": "blocked by stage 1"}
        finish(report)
        return report

    tools = toolchain()
    if not tools["pdflatex"] or not tools["bibtex"]:
        report["stopped_at"] = "2"
        report["stop_reason"] = ("LaTeX toolchain unavailable: "
                                 + ", ".join(t for t, v in tools.items() if not v)
                                 + " not installed")
        report["stage_2_compile"] = {"ran": False, "toolchain": tools}
        report["stage_3_package"] = {"created": False, "reason": "blocked by stage 2"}
        finish(report)
        return report

    comp = compile_isolated(collected, OUT / "build")
    comp["toolchain"] = tools
    report["stage_2_compile"] = comp
    if not comp["passed"]:
        report["stopped_at"] = "2"
        report["stop_reason"] = comp.get("reason", "acceptance criteria not met")
        report["stage_3_package"] = {"created": False, "reason": "stage 2 did not pass"}
        finish(report)
        return report

    report["stage_3_package"] = package(collected, Path(comp["pdf"]), tex_text)
    report["stopped_at"] = None
    finish(report)
    return report


def package(collected, pdf: Path, tex_text: str) -> dict:
    import zipfile
    DIST.mkdir(exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    zpath = DIST / f"mlforsys_submission_{day}.zip"
    rev = next((l.strip("% ").strip() for l in tex_text.splitlines()
                if "REVISION:" in l), "unknown")
    members = [(TEX.name, TEX)] + [(n, ROOT / i["path"])
                                   for n, i in collected["found"].items()]
    notes = ["# PACKAGING NOTES", "",
             f"- Base revision (verbatim from the .tex): `{rev}`", "",
             "## 보류 사항", "",
             "- **neurips_2026.sty 미교체 — CFP는 2026 포맷 요구. 스타일 교체 후 이 작업 "
             "재실행 필요. 현 zip은 예비본.**", "",
             "## 포함 파일", ""]
    for n, p in members + [(pdf.name, pdf)]:
        notes.append(f"- `{n}` — sha256 `{sha256(p)}` ({p.stat().st_size} B)")
    notes_path = OUT / "PACKAGING_NOTES.md"
    notes_path.write_text("\n".join(notes) + "\n")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for n, p in members:
            z.write(p, n)
        z.write(pdf, pdf.name)
        z.write(notes_path, "PACKAGING_NOTES.md")
    return {"created": True, "zip": str(zpath.relative_to(ROOT)),
            "zip_sha256": sha256(zpath),
            "members": [n for n, _ in members] + [pdf.name, "PACKAGING_NOTES.md"]}


def finish(report):
    (OUT / "run_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main()

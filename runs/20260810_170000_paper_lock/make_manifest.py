"""작업 20 run_manifest.json 생성."""
import hashlib, json, subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
F = json.loads((HERE / "lock_facts.json").read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


r = subprocess.run(["sha256sum", "-c", "paper.lock"], cwd=ROOT,
                   capture_output=True, text=True)
m = {
    "run": HERE.name,
    "task": "작업 20 — paper.lock 생성",
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
                  + " (pre-commit HEAD)",
    "scope": {
        "created": ["paper.lock", f"runs/{HERE.name}/*"],
        "modified": ["REPRODUCE.md (§5 무결성 절 — paper.lock 과 대조하도록 갱신)"],
        "note": "그 외 파일 미수정. 측정 데이터·코드·원고 불변.",
    },
    "source_manifest": "runs/20260806_103943_model_guide/run_manifest.json"
                       " (input_data_files)",
    "manuscript_revision": F["manuscript_marker"],
    "precheck": {"n_recorded": F["n_entries"], "n_recomputed": F["n_entries"],
                 "n_mismatch": len(F["mismatches"]), "n_missing": len(F["missing"]),
                 "passed": not F["mismatches"] and not F["missing"]},
    "lock": {"path": "paper.lock", "sha256": sha(ROOT / "paper.lock"),
             "n_entries": F["n_entries"], "format": "sha256sum -c 호환",
             "sort": "경로 사전순", "header": "# 주석 (생성일·근거·원고 리비전·검증 커맨드)"},
    "verification": {"command": "sha256sum -c paper.lock",
                     "exit_code": r.returncode,
                     "n_ok": sum(1 for l in r.stdout.splitlines()
                                 if l.strip().endswith(": OK")),
                     "n_entries": F["n_entries"],
                     "stderr": r.stderr.strip(),
                     "all_ok": r.returncode == 0 and not r.stderr.strip()},
    "entries": F["entries"],
    "outputs": {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.iterdir())
                if p.is_file() and p.name != "run_manifest.json"},
}
(HERE / "run_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
print("precheck:", m["precheck"])
print("verification:", {k: v for k, v in m["verification"].items() if k != "stderr"})
print("stderr:", repr(m["verification"]["stderr"]))

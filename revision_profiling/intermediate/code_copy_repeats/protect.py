"""Phase 4b: sha256 of the protected files (Phase 4 code / adapter / schedules / rep0 results / datasets / artifacts / runtime / profile).

  python3 -B revision_profiling/intermediate/code_copy_repeats/protect.py --start    -> intermediate/logs/protected_sha256_start.json
  python3 -B revision_profiling/intermediate/code_copy_repeats/protect.py --end      -> intermediate/logs/protected_sha256_end.json + comparison

git status does not see changes of untracked files, so this comparison is the criterion. The file list is fixed at --start
(files that exist then); --end hashes the same list and reports changed / missing files and new files inside protected folders.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
IROOT = HERE.parent                 # revision_profiling/intermediate
ROOT = IROOT.parent                 # revision_profiling
REPO = ROOT.parent
LOGS = IROOT / "logs"
RUNTIME_FILES = ("executor.py", "feeder.py", "counters.py", "shmqueue.py", "worker_base.py", "worker_cls.py", "worker_yolo.py", "lm_runner.py",
                 "image_processing.py", "model_registry.yaml")

# folders whose whole content is protected (no file may change, appear or disappear)
PROTECTED_DIRS = [IROOT / "code_copy", IROOT / "schedules", IROOT / "results", IROOT / "dataset", IROOT / "smoke",
                  REPO / "revision_exp" / "out_intermediate", ROOT / "dataset", ROOT / "results"]
PROTECTED_FILES = ([IROOT / f for f in ("CHANGES.md", "RUN_PLAN.md", "REPORT_INTERMEDIATE.md")]
                   + [REPO / "revision_exp_adapter" / "run_intermediate.py",
                      REPO / "revision_exp" / "out_reprofiled" / "artifacts" / "multi" / "T.json",
                      REPO / "revision_exp" / "out_reprofiled" / "artifacts" / "multi" / "D.json",
                      REPO / "revision_exp" / "out_phase3" / "artifacts" / "per_instance" / "t_ratio.json",
                      REPO / "revision_exp" / "out_phase3" / "artifacts" / "per_instance" / "q_i.json",
                      ROOT / "offline_profile" / "offline_profile.json"]
                   + [ROOT / "code" / f for f in RUNTIME_FILES] + [ROOT / "code" / "recompute_scores.py", ROOT / "code" / "run_campaign.py",
                                                                    ROOT / "code" / "gen_schedules.py"])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_list(existing_logs=None):
    files = []
    for d in PROTECTED_DIRS:
        files += [p for p in sorted(d.rglob("*")) if p.is_file()]
    files += PROTECTED_FILES
    if existing_logs is None:      # the Phase 4 log files that exist before Phase 4b
        existing_logs = [p for p in sorted(LOGS.rglob("*")) if p.is_file()]
    return files + list(existing_logs)


def snapshot(files):
    return {str(p.relative_to(REPO)): sha(p) for p in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--end", action="store_true")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    if a.start:
        out = LOGS / "protected_sha256_start.json"
        assert not out.exists(), "start snapshot already exists"
        files = current_list()
        j = {"taken_at": datetime.now().isoformat(sep=" ", timespec="seconds"), "files": len(files),
             "protected_dirs": [str(d.relative_to(REPO)) for d in PROTECTED_DIRS], "sha256": snapshot(files)}
        out.write_text(json.dumps(j, indent=1))
        print(f"start snapshot: {len(files)} files -> {out}")
    if a.end:
        st = json.loads((LOGS / "protected_sha256_start.json").read_text())
        files = [REPO / k for k in st["sha256"]]
        missing = [str(p.relative_to(REPO)) for p in files if not p.exists()]
        now = snapshot([p for p in files if p.exists()])
        changed = [k for k, v in now.items() if v != st["sha256"][k]]
        new_in_dirs = sorted(str(p.relative_to(REPO)) for d in PROTECTED_DIRS for p in d.rglob("*") if p.is_file() and str(p.relative_to(REPO)) not in st["sha256"])
        j = {"taken_at": datetime.now().isoformat(sep=" ", timespec="seconds"), "files": len(now), "sha256": now,
             "compare_to_start": {"files_in_start": len(st["sha256"]), "changed": changed, "missing": missing, "new_files_in_protected_dirs": new_in_dirs,
                                  "identical": not changed and not missing and not new_in_dirs}}
        (LOGS / "protected_sha256_end.json").write_text(json.dumps(j, indent=1))
        print(json.dumps(j["compare_to_start"], indent=1))
        return j["compare_to_start"]["identical"]


if __name__ == "__main__":
    main()

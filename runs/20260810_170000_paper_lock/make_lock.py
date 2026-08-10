"""작업 20: paper.lock 생성.

근거: runs/20260806_103943_model_guide/run_manifest.json 의 input_data_files
      (경로 → sha256) 18건. 현재 파일에서 재계산해 전건 일치할 때만 생성한다.
형식: sha256sum -c 호환 ("<sha256>  <경로>"), 경로 사전순, 헤더는 # 주석.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
SRC = ROOT / "runs/20260806_103943_model_guide/run_manifest.json"
TEX = ROOT / "manuscript/mlforsys_main.tex"
LOCK = ROOT / "paper.lock"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


rec = json.loads(SRC.read_text())["input_data_files"]
entries, mismatches, missing = [], [], []
for path, recorded in sorted(rec.items()):
    if recorded is None:
        continue
    f = ROOT / path
    if not f.exists():
        missing.append(path)
        continue
    cur = sha256(f)
    entries.append((path, cur))
    if cur != recorded:
        mismatches.append({"path": path, "recorded": recorded, "current": cur})

print(f"근거 매니페스트 기록 {len([v for v in rec.values() if v])}건 "
      f"→ 재계산 {len(entries)}건, 불일치 {len(mismatches)}, 부재 {len(missing)}")
if mismatches or missing:
    for m in mismatches:
        print(f"  MISMATCH {m['path']}\n    기록 {m['recorded']}\n    현재 {m['current']}")
    for m in missing:
        print(f"  MISSING  {m}")
    print("\n일치하지 않는 항목이 있어 paper.lock 을 생성하지 않고 중단한다.")
    sys.exit(1)

marker = TEX.read_text().splitlines()[0].lstrip("% ").strip()
lines = [
    "# paper.lock -- integrity manifest for the measurement inputs behind the paper's numbers.",
    "#",
    f"# generated      : {date.today().isoformat()}",
    f"# source         : {SRC.relative_to(ROOT)} (input_data_files)",
    f"# manuscript     : manuscript/mlforsys_main.tex -- {marker}",
    f"# entries        : {len(entries)} files, paths sorted lexicographically",
    "#",
    "# verify         : sha256sum -c paper.lock",
    "#",
    "# These are the inputs that cannot be regenerated without the measurement hardware.",
    "# Everything downstream of them is reproducible on CPU; see REPRODUCE.md.",
    "#",
]
lines += [f"{h}  {p}" for p, h in entries]
LOCK.write_text("\n".join(lines) + "\n")
print(f"\nwrote {LOCK.relative_to(ROOT)} ({len(entries)} entries)")

r = subprocess.run(["sha256sum", "-c", "paper.lock"], cwd=ROOT,
                   capture_output=True, text=True)
ok = sum(1 for l in r.stdout.splitlines() if l.strip().endswith(": OK"))
print(f"검증: sha256sum -c paper.lock → exit={r.returncode}, OK {ok}/{len(entries)}")
if r.stderr.strip():
    print("stderr:", r.stderr.strip())
(HERE / "verify_output.txt").write_text(
    f"$ sha256sum -c paper.lock\n{r.stdout}exit={r.returncode}\n"
    + (f"stderr:\n{r.stderr}" if r.stderr.strip() else ""))
json.dump({"n_entries": len(entries), "n_ok": ok, "exit": r.returncode,
           "mismatches": mismatches, "missing": missing,
           "manuscript_marker": marker,
           "entries": [{"path": p, "sha256": h} for p, h in entries]},
          open(HERE / "lock_facts.json", "w"), indent=2, ensure_ascii=False)
sys.exit(0 if (r.returncode == 0 and ok == len(entries)) else 1)

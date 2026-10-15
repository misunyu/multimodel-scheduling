"""sha256 of every registered file (revision_profiling_npu/registry/files_sha256.json) against the files on disk, plus the
host ONNX read for the registry. Writes a snapshot JSON; with --compare <earlier snapshot> reports differences.
  python3 -B code/check_registry_sha.py --out snap.json [--compare earlier.json]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    a = ap.parse_args()
    reg = json.loads((REPO / "revision_profiling_npu" / "registry" / "files_sha256.json").read_text())
    want = dict(reg["files"])
    want.update(reg.get("host_onnx_read_only", {}))
    got = {rel: sha(REPO / rel) for rel in want}
    snap = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": got,
            "match_registry": {rel: got[rel] == want[rel] for rel in want}}
    Path(a.out).write_text(json.dumps(snap, indent=1))
    bad = [r for r, ok in snap["match_registry"].items() if not ok]
    print(f"{len(got)} files, mismatch vs registry: {bad or 'none'}")
    if a.compare:
        prev = json.loads(Path(a.compare).read_text())["files"]
        diff = [r for r in got if prev.get(r) != got[r]]
        print(f"changed vs {a.compare}: {diff or 'none'}")


if __name__ == "__main__":
    main()

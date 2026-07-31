#!/usr/bin/env python3
"""Generate paper/paper.lock from the distribution copy.

The figure list is PARSED from main.tex's \\includegraphics calls, never hand-written: a
hand-maintained list drifts silently, and the whole point of the lock is that it cannot.
figures/ holds many more artefacts than the paper uses; what the paper uses is exactly what
this parse finds.

Run this only when a NEW main.tex arrives from the canonical copy outside the repo. Running
it against an edited main.tex would launder the edit into the lock, which is the one thing
the lock exists to prevent -- so --check is the default and writing requires --write.

usage:
  python3 scripts/gen_paper_lock.py            # verify the committed lock reproduces
  python3 scripts/gen_paper_lock.py --write    # re-sync after receiving a new main.tex
"""
import argparse, hashlib, json, os, re, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(PROJECT, "paper")
TEX = os.path.join(PAPER, "main.tex")
LOCK = os.path.join(PAPER, "paper.lock")
FIGDIR = os.path.join(PROJECT, "figures")

# \includegraphics[...]{name.pdf} -- the optional argument is skipped, and a line whose
# first non-space character is % is a LaTeX comment and does not count as a reference.
INC = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def referenced_figures(tex_path):
    out = []
    for line in open(tex_path, encoding="utf-8"):
        if line.lstrip().startswith("%"):
            continue
        out += INC.findall(line)
    return sorted(set(out))


def build():
    figs = referenced_figures(TEX)
    entries, missing = {}, []
    for f in figs:
        p = os.path.join(FIGDIR, f)
        if not os.path.isfile(p):
            missing.append(f)
            continue
        entries[f] = sha(p)
    return figs, entries, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    figs, entries, missing = build()
    if missing:
        print(f"[FAIL] main.tex references {len(missing)} figure(s) not in figures/: {missing}")
        return 1

    old = json.load(open(LOCK, encoding="utf-8")) if os.path.isfile(LOCK) else {}
    meta = old.get("_meta", {})
    doc = {"_meta": meta,
           "main_tex_sha256": sha(TEX),
           "main_tex_bytes": os.path.getsize(TEX),
           "figures": entries}

    if not a.write:
        same = (old.get("main_tex_sha256") == doc["main_tex_sha256"]
                and old.get("figures") == doc["figures"])
        print(f"parsed {len(figs)} figures from main.tex")
        if same:
            print("[OK] committed paper.lock reproduces from main.tex + figures/")
            return 0
        print("[FAIL] committed paper.lock does NOT reproduce:")
        if old.get("main_tex_sha256") != doc["main_tex_sha256"]:
            print(f"  main.tex {doc['main_tex_sha256'][:16]}… != lock "
                  f"{str(old.get('main_tex_sha256'))[:16]}…")
        for f in sorted(set(entries) | set(old.get("figures", {}))):
            if entries.get(f) != old.get("figures", {}).get(f):
                print(f"  {f}: figures/ {str(entries.get(f))[:16]}… != lock "
                      f"{str(old.get('figures', {}).get(f))[:16]}…")
        return 1

    json.dump(doc, open(LOCK, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {LOCK}: main.tex + {len(entries)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())

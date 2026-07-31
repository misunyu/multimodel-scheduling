#!/usr/bin/env python3
"""Propose `%!cv <id>` annotations for numerals in the paper's evaluation section.

The annotations themselves have to be applied to the canonical main.tex OUTSIDE the repo --
the distribution copy is read-only and editing it would break its hash. So this emits a
WORKLIST: line number, the line, and the confirmed-table ids whose values appear on it.

Matching is by value, not by name: for every numeral on a line, every confirmed entry whose
value equals it (within tolerance) is a candidate. That produces false positives on round
numbers -- "5 runs", "$T=3$\\,s" -- so single-digit integers and values shared by more than
a handful of entries are suppressed, and the output is meant to be curated down, not applied
wholesale. Proposing too much and cutting is safe; proposing too little is not, because a
numeral nobody proposes is one nobody notices has gone stale.

usage: python3 scripts/propose_cv_annotations.py [--section-from 399] [--section-to 827]
"""
import argparse, json, os, re, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(PROJECT, "paper", "main.tex")
CONFIRMED = os.path.join(PROJECT, "fig", "confirmed_values.json")

NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
# A numeral is only worth tracking if it could go stale. Section numbers, \ref targets and
# LaTeX lengths cannot.
SKIP_CONTEXT = re.compile(r"\\(?:ref|label|cite|includegraphics|vspace|hspace|columnwidth"
                          r"|textwidth|section|subsection)\b")


def numerals(line):
    body = re.sub(r"\\(?:ref|label|cite)\{[^}]*\}", " ", line)
    body = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", " ", body)
    body = re.sub(r"\\[a-zA-Z]+\s*(?:\[[^\]]*\])?", " ", body)   # drop control sequences
    return [float(m) for m in NUM.findall(body)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section-from", type=int, default=399)
    ap.add_argument("--section-to", type=int, default=827)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    table = json.load(open(CONFIRMED, encoding="utf-8"))["values"]
    by_value = {}
    for e in table:
        v = e.get("value")
        if v is None:
            continue
        by_value.setdefault(round(float(v), 4), []).append(e["id"])

    lines = open(TEX, encoding="utf-8").read().splitlines()
    proposals, untracked = [], []
    for i in range(a.section_from - 1, min(a.section_to, len(lines))):
        line = lines[i]
        if line.lstrip().startswith("%") or not line.strip():
            continue
        nums = numerals(line)
        if not nums:
            continue
        hits, unmatched = {}, []
        for n in nums:
            ids = by_value.get(round(n, 4), [])
            # a value carried by many entries identifies nothing; a bare single digit is
            # almost always a count, a section number or a parameter, not a measurement
            if ids and len(ids) <= 6 and not (n < 10 and n == int(n)):
                for k in ids:
                    hits.setdefault(n, []).append(k)
            else:
                unmatched.append(n)
        if hits:
            proposals.append({"line": i + 1, "text": line.strip(),
                              "matches": {str(k): v for k, v in hits.items()},
                              "unmatched_numerals": unmatched})
        elif unmatched:
            untracked.append({"line": i + 1, "n_numerals": len(unmatched),
                              "text": line.strip()[:110]})

    print(f"lines with a confirmed-value match : {len(proposals)}")
    print(f"lines with numerals but no match   : {len(untracked)}")
    print()
    for p in proposals:
        ids = sorted({i for v in p["matches"].values() for i in v})
        print(f"L{p['line']}  %!cv {','.join(ids)}")
        for val, k in sorted(p["matches"].items(), key=lambda x: float(x[0])):
            print(f"        {val:>10} -> {','.join(k)}")
    if a.json:
        json.dump({"proposals": proposals, "untracked": untracked},
                  open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

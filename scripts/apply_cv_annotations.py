#!/usr/bin/env python3
"""Apply the `%!cv` worklist to the CANONICAL main.tex, which lives outside this repo.

This is the one step of the loop the repo cannot do for itself. paper/main.tex is a
distribution copy; annotating it here would break its hash and, worse, would create a second
editable copy of the paper -- the exact failure v17 was decided to prevent.

So: run this against the canonical file, wherever it is. It refuses to touch anything inside
paper/, and it refuses to write a line whose sha256 does not match the worklist, because a
worklist built against an older revision would otherwise attach values to whatever sentence
now occupies that line number.

  python3 scripts/apply_cv_annotations.py --tex ~/paper/main.tex --dry-run
  python3 scripts/apply_cv_annotations.py --tex ~/paper/main.tex

Afterwards, bring the annotated main.tex back with a regenerated paper.lock
(`make lock` once it is in place) and run `make check`.
"""
import argparse, hashlib, json, os, sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLIST = os.path.join(PROJECT, "docs", "paper_cv_annotations.json")
DISTRIBUTION = os.path.realpath(os.path.join(PROJECT, "paper"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True, help="정본 main.tex 경로 (저장소 밖)")
    ap.add_argument("--worklist", default=WORKLIST)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    target = os.path.realpath(a.tex)
    if target.startswith(DISTRIBUTION + os.sep):
        print(f"[거부] {a.tex} 는 배포본이다. 배포본은 편집하지 않는다 — 정본에 적용하라 "
              f"(paper/README.md).")
        return 1

    doc = json.load(open(a.worklist, encoding="utf-8"))
    lines = open(target, encoding="utf-8").read().splitlines()
    applied, skipped = [], []
    for item in doc["annotations"]:
        ln = item["line"]
        if ln - 1 >= len(lines):
            skipped.append((ln, "줄 번호가 파일 밖"))
            continue
        text = lines[ln - 1]
        if "%!cv" in text:
            skipped.append((ln, "이미 주석이 있다"))
            continue
        got = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if got != item["line_sha256"]:
            skipped.append((ln, f"줄 해시 불일치 — 목록이 낡았거나 줄이 밀렸다 "
                                f"({got[:12]}… != {item['line_sha256'][:12]}…)"))
            continue
        lines[ln - 1] = text.rstrip() + "  " + item["annotation"]
        applied.append((ln, item["annotation"]))

    for ln, ann in applied:
        print(f"  L{ln}  {ann}")
    for ln, why in skipped:
        print(f"  [건너뜀] L{ln}: {why}")
    print(f"\n{len(applied)} applied, {len(skipped)} skipped")

    if skipped:
        print("건너뛴 항목이 있다. 줄이 밀렸다면 저장소에서 "
              "`python3 scripts/gen_cv_worklist.py` 를 새 main.tex 로 다시 생성해야 한다.")
    if a.dry_run:
        print("(--dry-run: 쓰지 않았다)")
        return 0 if not skipped else 1
    if applied:
        open(target, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"wrote {target}")
    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(main())

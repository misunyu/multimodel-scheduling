#!/usr/bin/env python3
"""논문 배포본을 확정표와 대조한다. `fig/check_figures.py`의 논문판이다.

`check_figures.py`가 **그림↔확정표**를 보는 자리에서, 이것은 **논문↔확정표**를 본다.
새 실험이 확정표를 갱신하면 논문의 어느 줄이 낡았는지 이 검사가 짚는다 — 지금까지 사람이
매번 다시 하던 대조다.

검사 다섯:

1. **배포본 해시** — `main.tex`와 참조된 그림이 `paper.lock`과 일치하는가.
   어긋나면 저장소에서 편집했거나 정본이 갱신된 것이며, 어느 쪽이든 실패다.
2. **수치 대조** — `%!cv <id>[,<id>...]` 주석이 달린 줄에 대해, 나열된 각 id의 확정표 값이
   그 줄에 실제로 나타나는가. 주석은 줄 **끝**에 있어야 하고, 자기가 끝맺는 **문단** 전체를
   범위로 삼는다 — LaTeX 원고는 한 문장이 여러 줄로 접히고 그때 값은 앞 줄에 있다.
3. **미추적 목록** — §IV에서 `%!cv` 없이 등장하는 수치. **경고이지 실패가 아니다** —
   검사가 무엇을 덮지 못하는지 보이게 하는 것이 목적이다.
4. **그림 참조** — `\\includegraphics` 대상이 `figures/`에 전부 있는가.
5. **삭제된 그림** — 미사용으로 표시된 그림(`confirmed_values.json` `_meta.removed_figures`)을
   아직 참조하고 있지 않은가.

### 단위 환산을 왜 검사기가 하는가

확정표는 계측 단위로 적는다(`p99`는 ms, `miss`는 비율). 논문은 읽기 좋은 단위로 적는다
(33.7 s, 63 %). 같은 값이다. 주석에 배율을 손으로 적게 하면 그 배율이 낡으므로, 검사기가
확정표 항목의 `unit`을 보고 환산한다.

### 비교 규칙 — 논문이 인쇄한 자릿수로 반올림해 정확히 일치

허용오차를 백분율로 잡으면 33.7과 33.9가 통과한다. 대신 **확정표 값을 그 줄이 인쇄한
자릿수로 반올림**해 정확히 일치를 요구한다. 33.656 → 33.7 ✓, 33.9 ✗. 실패 메시지가
"논문은 X를 적었는데 확정표는 Y로 반올림된다"가 되어 고칠 곳이 바로 보인다.

usage:
  python3 scripts/check_paper.py
  python3 scripts/check_paper.py --paper scripts/testdata/paper_fixture   # 음성 테스트용
exit 0 = 불일치 없음, exit 1 = 불일치 있음
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

INC = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
# 주석은 줄 끝에서 끝나야 한다. `$` 고정이 없으면 "%!cv 주석" 같은 산문 언급까지 주석으로 읽힌다
# (이 픽스처를 쓰다가 실제로 걸렸다).
CV = re.compile(r"%!cv\s+([A-Za-z0-9_.,\-]+)\s*$")
NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def strip_latex(line):
    """수치가 아닌 곳의 숫자를 뺀다: \\ref/\\label/\\cite 인자, 그림 파일명, 제어열."""
    s = re.sub(r"%!cv\s+[A-Za-z0-9_.,\-]+", " ", line)
    s = re.sub(r"\\(?:ref|label|cite|eqref)\{[^}]*\}", " ", s)
    s = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\s*(?:\[[^\]]*\])?", " ", s)
    return s


def decimals(text):
    return len(text.split(".")[1]) if "." in text else 0


def representations(entry):
    """확정표 항목이 논문에 인쇄될 수 있는 형태들. 단위별 환산을 여기 모은다."""
    v = float(entry["value"])
    unit = (entry.get("unit") or "").lower()
    reps = [(v, unit or "-")]
    if unit == "ms":
        reps.append((v / 1000.0, "s"))
    elif unit == "s":
        reps.append((v * 1000.0, "ms"))
    elif unit in ("ratio", "fraction"):
        reps.append((v * 100.0, "%"))
    elif unit in ("%", "percent"):
        reps.append((v / 100.0, "ratio"))
    return reps


def line_matches(entry, nums_text):
    """줄의 수치 중 하나가 이 항목과 일치하는가. (일치여부, 설명) 반환."""
    tried = []
    for value, unit in representations(entry):
        for t in nums_text:
            r = round(value, decimals(t))
            if abs(r - float(t)) < 1e-9:
                return True, None
            tried.append(f"{r:g}{'' if unit == '-' else ' ' + unit}")
    return False, sorted(set(tried), key=len)


def load_confirmed(path):
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return {e["id"]: e for e in doc["values"]}, doc.get("_meta", {})


def section_bounds(lines, name):
    start = end = None
    for i, l in enumerate(lines):
        if l.startswith("\\section{"):
            if name in l and start is None:
                start = i + 1
            elif start is not None and end is None:
                end = i
    return start, (end if end is not None else len(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="paper", help="main.tex 와 paper.lock 이 있는 디렉터리")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--confirmed", default="fig/confirmed_values.json")
    ap.add_argument("--section", default="Experimental Evaluation",
                    help="미추적 수치를 세는 절")
    ap.add_argument("--list-untracked", action="store_true")
    a = ap.parse_args()

    paper = pathlib.Path(a.paper)
    tex, lockp = paper / "main.tex", paper / "paper.lock"
    figdir = pathlib.Path(a.figdir)
    confirmed, meta = load_confirmed(a.confirmed)
    lock = json.loads(lockp.read_text(encoding="utf-8"))
    lines = tex.read_text(encoding="utf-8").splitlines()
    problems, warnings = [], []

    # ---- 1. 배포본 해시 --------------------------------------------------------------
    got = sha(tex)
    if got != lock.get("main_tex_sha256"):
        problems.append(
            f"main.tex 해시 불일치 — {got[:16]}… != lock {str(lock.get('main_tex_sha256'))[:16]}…\n"
            f"    저장소에서 편집했거나 정본이 갱신됐다. 배포본은 편집하지 않는다; 정본이 갱신됐다면\n"
            f"    새 main.tex 와 새 paper.lock 을 함께 받는다 (paper/README.md).")
    n_fig_ok = 0
    for f, want in lock.get("figures", {}).items():
        p = figdir / f
        if not p.is_file():
            problems.append(f"paper.lock 의 그림이 {figdir}/ 에 없다 — {f}")
            continue
        if sha(p) != want:
            problems.append(f"{f}: 해시 불일치 — 그림이 재생성됐는데 paper.lock 이 낡았다 "
                            f"({sha(p)[:16]}… != {want[:16]}…)")
        else:
            n_fig_ok += 1

    # ---- 2. 수치 대조 ----------------------------------------------------------------
    checked = 0
    annotated_lines = set()
    for i, line in enumerate(lines):
        m = CV.search(line)
        if not m:
            continue
        annotated_lines.add(i)
        ids = [x for x in m.group(1).split(",") if x]
        # 주석은 자기가 끝맺는 **문단**에 걸린다. main.tex 는 한 문단이 한 줄이라 차이가 없지만,
        # LaTeX 원고는 줄바꿈으로 접히는 경우가 흔하고 그때 값은 앞 줄에 있다. 뒤로 확장하는 범위는
        # 빈 줄 / 환경 경계 / 다른 %!cv 주석에서 멈춘다.
        block, j = [line], i - 1
        while j >= 0:
            prev = lines[j]
            if (not prev.strip() or prev.lstrip().startswith(("\\begin", "\\end", "\\section",
                                                              "\\subsection"))
                    or CV.search(prev)):
                break
            block.append(prev)
            j -= 1
        annotated_lines.update(range(j + 1, i + 1))
        nums_text = NUM.findall(strip_latex(" ".join(block)))
        where = f"main.tex:{i + 1}"
        if not nums_text:
            problems.append(f"{where}: %!cv 주석이 있는데 줄에 수치가 없다 — {','.join(ids)}")
            continue
        for ident in ids:
            checked += 1
            e = confirmed.get(ident)
            if e is None:
                problems.append(f"{where}: 확정표에 없는 id — {ident}")
                continue
            ok, tried = line_matches(e, nums_text)
            if not ok:
                problems.append(
                    f"{where}: {ident} = {e['value']}{' ' + e['unit'] if e.get('unit') else ''} "
                    f"가 이 줄에 없다\n"
                    f"    줄의 수치: {', '.join(nums_text)}\n"
                    f"    확정표를 줄의 자릿수로 반올림하면: {', '.join(tried)}")

    # ---- 3. 미추적 수치 --------------------------------------------------------------
    s0, s1 = section_bounds(lines, a.section)
    untracked = []
    if s0 is None:
        warnings.append(f"절을 찾지 못했다: {a.section} — 미추적 집계를 건너뛴다")
    else:
        for i in range(s0, s1):
            if i in annotated_lines or lines[i].lstrip().startswith("%"):
                continue
            nums = NUM.findall(strip_latex(lines[i]))
            if nums:
                untracked.append((i + 1, nums))

    # ---- 4. 그림 참조 ----------------------------------------------------------------
    refs = []
    for line in lines:
        if not line.lstrip().startswith("%"):
            refs += INC.findall(line)
    refs = sorted(set(refs))
    present = [f for f in refs if (figdir / f).is_file()]
    for f in refs:
        if f not in present:
            problems.append(f"main.tex 가 참조하는 그림이 {figdir}/ 에 없다 — {f}")
        if f not in lock.get("figures", {}):
            problems.append(f"main.tex 가 참조하는데 paper.lock 에 없다 — {f} "
                            f"(scripts/gen_paper_lock.py 를 다시 돌려야 한다)")

    # ---- 5. 삭제된 그림 --------------------------------------------------------------
    removed = meta.get("removed_figures", {})
    for name in removed:
        for f in refs:
            if pathlib.Path(f).stem == name:
                problems.append(f"미사용으로 표시된 그림을 아직 참조한다 — {f}: {removed[name]}")

    # ---- 출력 -----------------------------------------------------------------------
    n_lock_fig = len(lock.get("figures", {}))
    print(f"paper.lock:      {'OK' if not any('해시' in p for p in problems) else 'FAIL'} "
          f"(main.tex + {n_fig_ok}/{n_lock_fig} figures)")
    print(f"tracked values:  {checked} checked, "
          f"{sum(1 for p in problems if p.startswith('main.tex:'))} mismatch")
    print(f"untracked:       {sum(len(n) for _, n in untracked)} numerals on "
          f"{len(untracked)} lines of section '{a.section}' without %!cv")
    print(f"figures:         {len(refs)} referenced, {len(present)} present")

    if a.list_untracked and untracked:
        print("\n[untracked lines]")
        for ln, nums in untracked:
            print(f"  main.tex:{ln}  {', '.join(nums)}")

    for w in warnings:
        print(f"\n[warn] {w}")
    if problems:
        print(f"\n[FAIL] {len(problems)} problems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\n[OK] PAPER MATCHES CONFIRMED TABLE (0 mismatches).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

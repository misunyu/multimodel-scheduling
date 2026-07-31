#!/usr/bin/env python3
"""Re-verify the three structural claims v18 made about hot-swap resource risk (ISSRE R4).

v18's measurements are void -- they were taken on a phantom-contaminated workload where only
2 of 4 named views ran. Its STRUCTURAL half is not, because it came from reading the code and
the schedules rather than from the workload. But "came from reading the code" is only a
defence if the code still reads that way, and it has changed since (vocabulary validation,
alias removal). So this checks the three claims again, mechanically, instead of trusting a
report from thirteen versions ago.

  1. Every transition relocates a model ACROSS devices; no view keeps its device and swaps
     the model underneath. This is what makes device-memory duplication impossible: the old
     and new instances are never on the same device at once.
  2. The background generative path is kill-then-start -- BackgroundManager.sync stops a
     child that moved BEFORE launching its replacement, so there is no co-residence, and the
     cost is a service gap rather than a memory peak.
  3. There is still no pre-flight residual-resource check anywhere. R4's "fundamentally
     assumes sufficient residual resources" is a statement about the code, and this is the
     line that confirms or refutes it.

Any of the three flipping is more important than the numbers.

usage: python3 scripts/check_transition_structure.py
exit 0 = all three hold as v18 described, 1 = something changed
"""
import collections, glob, os, re, sys

try:
    import yaml
except ImportError:
    yaml = None

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# v18 used exactly this pattern; kept verbatim so the two rounds are comparable.
PREFLIGHT = re.compile(r"free_memory|headroom|can_fit|\bfits\b|sufficient|available_memory"
                       r"|nvmlDeviceGetMemoryInfo|mem_get_info|torch\.cuda\.mem")
# Hits that are about something else entirely. Listed with reasons rather than filtered by a
# blanket exclusion, so a real check appearing in one of these files would still show up.
BENIGN = {
    "model_registry.py": "buffer-size factor comment",
    "adaptive_deploy.py": "queue-depth comment ('everything fits')",
    "runtime/bg_entry.py": "dummy-frame comment",
    "runtime/llm_engine.py": "LLM prompt text",
    "best_deploy_finder_executor.py": "'insufficient score history' log",
    "unified_viewer.py": "'insufficient data' trigger status",
    "scripts/run_bounded_recovery_sweep.py": "'insufficient data' docstring",
    "fig/plot_overhead_v11.py": "axis margin comment",
    # v38's own apparatus names the measurement "headroom". Listed rather than skipped so the
    # hit stays visible: these two files report resource numbers, they never gate on them.
    "scripts/emit_hotswap_confirmed.py": "v38 confirmed-table emitter: names the metric",
    "scripts/emit_v39_confirmed.py": "v39 confirmed-table emitter: names the metric",
    "scripts/gen_cv_worklist.py": "annotation worklist: names the metric's id",
    "fig/check_figures.py": "quantity vocabulary entry for the v38 metric",
}


def check_cross_device():
    """Claim 1. Returns (n_files, [(file, view, from, to, model_a, model_b, device)])."""
    bad, n = [], 0
    for p in sorted(glob.glob(os.path.join(PROJECT, "schedules", "*.yaml")) +
                    glob.glob(os.path.join(PROJECT, "tests", "*.yaml"))):
        try:
            d = yaml.safe_load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        n += 1
        per_view = collections.defaultdict(list)
        for cname, slots in d.items():
            if not isinstance(slots, dict):
                continue
            for slot, cfg in slots.items():
                if not isinstance(cfg, dict):
                    continue
                per_view[cfg.get("display") or slot].append(
                    (cname, cfg.get("model_name") or cfg.get("model"), cfg.get("execution")))
        for view, seq in per_view.items():
            for i in range(1, len(seq)):
                (c0, m0, d0), (c1, m1, d1) = seq[i - 1], seq[i]
                if m0 and m1 and m0 != m1 and d0 == d1:
                    bad.append((os.path.relpath(p, PROJECT), view, c0, c1, m0, m1, d0))
    return n, bad


def check_kill_then_start():
    """Claim 2. The stop loop must appear before the start loop inside sync()."""
    p = os.path.join(PROJECT, "runtime", "background_llm.py")
    if not os.path.isfile(p):
        return None, "runtime/background_llm.py 없음"
    src = open(p, encoding="utf-8").read()
    m = re.search(r"def sync\(.*?\n(.*?)(?=\n    def )", src, re.S)
    if not m:
        return None, "sync() 를 찾지 못함"
    body = m.group(1)
    stop = body.find("_stop_one")
    start = body.find("_start_one")
    if stop < 0 or start < 0:
        return None, f"_stop_one={stop} _start_one={start} — 형태가 바뀌었다"
    return (stop < start), (f"sync(): _stop_one at {stop}, _start_one at {start} "
                            f"→ {'kill-then-start' if stop < start else 'START BEFORE STOP'}")


def check_no_preflight():
    hits = []
    for p in glob.glob(os.path.join(PROJECT, "**", "*.py"), recursive=True):
        rel = os.path.relpath(p, PROJECT)
        # backup/ holds frozen snapshots of earlier revisions; they are not executed, and
        # every hit in them is the same benign pattern as its live counterpart. Skipped by
        # path rather than by matching text, so a real check added to live code still shows.
        if rel.startswith((".venv", "backup/", "scripts/mem_sampler",
                           "scripts/analyze_hotswap", "scripts/run_hotswap_mem_campaign",
                           "scripts/check_transition_structure")):
            continue
        try:
            for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                if PREFLIGHT.search(line):
                    hits.append((rel, i, line.strip()[:100]))
        except Exception:
            continue
    real = [h for h in hits if h[0] not in BENIGN]
    return hits, real


def main():
    ok = True
    n, bad = check_cross_device()
    print(f"[1] cross-device only : {n} schedule files scanned, "
          f"{len(bad)} same-device model swaps")
    if bad:
        ok = False
        for b in bad[:10]:
            print(f"      {b}")
    else:
        print("      → 옛/새 인스턴스가 같은 디바이스를 동시 점유하는 전환이 배치 공간에 없다 (v18과 동일)")

    res, why = check_kill_then_start()
    print(f"[2] kill-then-start   : {why}")
    if res is not True:
        ok = False

    hits, real = check_no_preflight()
    print(f"[3] no pre-flight check: {len(hits)} pattern hits, {len(real)} unaccounted-for")
    for h in hits:
        tag = BENIGN.get(h[0])
        print(f"      {'(benign: ' + tag + ')' if tag else '*** UNACCOUNTED ***'} "
              f"{h[0]}:{h[1]}  {h[2]}")
    if real:
        ok = False
        print("      → 사전 자원 점검이 생겼을 수 있다. 위 항목을 확인하라.")
    else:
        print("      → 전환 전 잔여 VRAM/DRAM 을 확인하는 코드는 여전히 없다 (v18과 동일)")

    print(f"\n{'[OK] 세 구조적 결론 전부 v18 그대로 성립한다.' if ok else '[CHANGED] 위를 보라.'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

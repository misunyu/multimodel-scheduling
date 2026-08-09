"""작업 16: 540창 수집(2026-07-15~18)이 실행된 호스트의 CPU 를 증거로 확정.

읽기 전용. 저장소 파일과 시스템 로그를 읽기만 한다. 수정 없음.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent


def sh(*cmd, shell=False):
    r = subprocess.run(cmd if not shell else cmd[0], shell=shell,
                       capture_output=True, text=True)
    return r.stdout


# ---------------------------------------------------------------- 1. 현재 호스트
current = {
    "cpu_model_name": next((l.split(":", 1)[1].strip()
                            for l in Path("/proc/cpuinfo").read_text().splitlines()
                            if l.startswith("model name")), None),
    "n_logical_cpus": int(sh("nproc").strip()),
    "threads_per_core": next((l.split(":")[1].strip() for l in sh("lscpu").splitlines()
                              if l.startswith("Thread(s) per core")), None),
    "hostname": sh("hostname").strip(),
    "machine_id": Path("/etc/machine-id").read_text().strip(),
    "uname": sh("uname", "-a").strip(),
}

# ------------------------------------------- 2. 수집 시점 증거: 부트 구간 특정
boots = []
for l in sh("journalctl", "--list-boots", "--no-pager").splitlines():
    m = re.match(r"\s*(-?\d+)\s+([0-9a-f]{32})\s+(.+?)—(.+)$", l)
    if m:
        boots.append({"idx": int(m.group(1)), "boot_id": m.group(2),
                      "start": m.group(3).strip(), "end": m.group(4).strip()})

COLLECTION = ("2026-07-15", "2026-07-18")   # FULL_COLLECTION_*_REPORT.md 의 수집 기간
PROFILE_RUN = "2026-07-14"                  # sample_profiling_data.json profiling_runs[0]


def to_iso(s):
    """'Tue 2026-05-26 18:21:57 KST' -> '2026-05-26 18:21:57'"""
    m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", s)
    return m.group(1) if m else s


covering = [b for b in boots
            if to_iso(b["start"]) <= COLLECTION[0] + " 00:00:00"
            and to_iso(b["end"]) >= COLLECTION[1] + " 23:59:59"]

boot_facts = {}
if covering:
    b = covering[-1]
    k = sh("journalctl", "-b", str(b["idx"]), "-k", "--no-pager")
    def grab(pat):
        m = re.search(pat, k)
        return m.group(0).split("kernel: ", 1)[-1].strip() if m else None
    boot_facts = {
        "boot_index": b["idx"], "boot_id": b["boot_id"],
        "start": b["start"], "end": b["end"],
        "covers_collection_window": True,
        "covers_static_profiling_run": to_iso(b["start"]) <= PROFILE_RUN + " 00:00:00",
        "smpboot_cpu0": grab(r".*smpboot: CPU0:.*"),
        "cpu_count_line": grab(r".*smpboot: Total of \d+ processors activated.*"),
        "dmi": grab(r".*DMI: .*"),
        "nvidia_driver": grab(r".*NVRM: loading NVIDIA.*"),
        "npu_device": grab(r".*aries \S+: Probing a device.*"),
        "n_journal_entries_in_collection_window": int(sh(
            "bash", "-c",
            f"journalctl -b {b['idx']} --since '{COLLECTION[0]} 00:00' "
            f"--until '{COLLECTION[1]} 23:59' --no-pager 2>/dev/null | wc -l").strip()),
    }

# 로테이션된 커널 로그(수집 기간 포함)의 호스트명
kern4 = sh("bash", "-c", "zcat /var/log/kern.log.4.gz 2>/dev/null | "
                         "awk '{print $4}' | sort -u | head -5")
kern4_range = sh("bash", "-c",
                 "zcat /var/log/kern.log.4.gz 2>/dev/null | sed -n '1p;$p' | cut -c1-16")

# ------------------------------------------- 2a-f. 저장소 안의 호스트 흔적
def grep_repo(pattern, extra=()):
    return sh("bash", "-c",
              f"grep -rniE {pattern!r} --include=*.md --include=*.py --include=*.yaml "
              f"--include=*.txt --include=*.sh --include=*.json "
              f"--exclude-dir=.venv --exclude-dir=.git --exclude-dir=artifacts "
              f"--exclude-dir=runs {ROOT} 2>/dev/null | head -20")


repo = {
    "collection_artifacts_have_host_metadata": False,
    "collection_jsonl_keys": {},
    "static_profile_run_record": {},
    "cpu_model_mentions_in_docs": grep_repo(
        r"i7[- ]?13700|285K|Core Ultra|Ultra 9").strip().splitlines(),
    "i7_13700k_hits": grep_repo(r"i7[- ]?13700|13700K").strip().splitlines(),
}
for p in sorted((ROOT / "xgboost_model/full_collection_540").rglob("*.jsonl")):
    ks = set()
    for line in p.read_text().splitlines():
        ks |= set(json.loads(line).keys())
    repo["collection_jsonl_keys"][p.name] = sorted(ks)
for p in sorted((ROOT / "xgboost_model/full_collection_540").rglob("performance_*.json")):
    blob = json.loads(p.read_text())
    keys = set()
    for w in blob[:5]:
        keys |= set(w.keys())
    repo["collection_jsonl_keys"][p.name + " (window keys)"] = sorted(keys)
sp = json.loads((ROOT / "xgboost_model/performance_data/sample_profiling_data/"
                        "sample_profiling_data.json").read_text())
repo["static_profile_run_record"] = {
    k: v for k, v in sp["profiling_runs"][0].items()
    if k in ("run_id", "timestamp", "host", "python", "git_commit", "versions")}

# ------------------------------------------------------------- 4. 보조 확인
aux = {
    "gpu": sh("bash", "-c",
              "nvidia-smi --query-gpu=name,driver_version,memory.total "
              "--format=csv,noheader 2>/dev/null").strip(),
    "npu_device_node": sh("bash", "-c",
                          "ls /dev/aries* /dev/mobilint* 2>/dev/null").split(),
    "qbruntime": sh("bash", "-c",
                    "python -c 'import qbruntime as q; "
                    "print(getattr(q,\"__version__\",\"?\"))' 2>/dev/null").strip(),
    "input_video_present": (ROOT / "stockholm_1280x720.mp4").exists(),
    "input_video_mtime": sh("bash", "-c",
                            f"ls -l --time-style=long-iso {ROOT}/stockholm_1280x720.mp4 "
                            "2>/dev/null | awk '{print $6, $7}'").strip(),
}

# ------------------------------------------------------------- 3. 판정 재료
smp = (boot_facts.get("smpboot_cpu0") or "")
is_285k = "Ultra 9 285K" in smp
is_13700k = "13700K" in smp
checks = {
    "found_boot_covering_collection_window": bool(covering),
    "boot_kernel_names_the_cpu": bool(smp),
    "kernel_cpu_is_285K": is_285k,
    "kernel_cpu_is_i7_13700K": is_13700k,
    "no_reboot_inside_collection_window": bool(covering),
    "current_cpu_matches_boot_cpu": (current["cpu_model_name"] or "") in smp,
    "hostname_in_collection_period_logs_is_current": "5090pc" in kern4,
    "repo_has_zero_i7_13700k_hits": not repo["i7_13700k_hits"],
    "rtx5090_present": "RTX 5090" in aux["gpu"],
    "mla100_device_present": bool(aux["npu_device_node"]),
    "input_video_present": aux["input_video_present"],
}

verdict = ("285K 확정 (직접 증거)" if (checks["found_boot_covering_collection_window"]
                                   and is_285k and not is_13700k)
           else "판정 불가")

facts = {
    "question": "540창 수집(2026-07-15~18)이 돌아간 호스트의 CPU 는 무엇인가",
    "verdict": verdict,
    "current_host": current,
    "collection_window_boot": boot_facts,
    "rotated_kernel_log_covering_collection": {
        "file": "/var/log/kern.log.4.gz",
        "date_range_first_last_line": kern4_range.strip().splitlines(),
        "hostnames_seen": kern4.split(),
    },
    "repo_evidence": repo,
    "auxiliary_hardware": aux,
    "checks": checks,
}
(HERE / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False))

print("판정:", verdict)
print("현재:", current["cpu_model_name"], "|", current["hostname"],
      "|", current["machine_id"])
print("수집 포함 부트:", boot_facts.get("start"), "~", boot_facts.get("end"))
print("  smpboot:", boot_facts.get("smpboot_cpu0"))
print("  DMI    :", boot_facts.get("dmi"))
print("  NPU    :", boot_facts.get("npu_device"))
print("i7-13700K 흔적:", repo["i7_13700k_hits"] or "0건")
for k, v in checks.items():
    print(f"  {'OK ' if v else '·  '} {k} = {v}")

#!/bin/bash
# Runs each dynamic-load scenario (mode 3 / 0 / 1) in its OWN neubla/antara
# container, one after the other. Isolation contract:
#   * each technique's data collection happens in a fresh container
#     (docker run --rm -d); NPU drivers and Python interpreter state are
#     fully reset between runs.
#   * the container is force-killed (docker kill) before the next one
#     starts. No scenarios share a process or NPU handle.
#   * all three runs use the SAME schedule YAML
#     (tests/dynamic_load_views_schedule_npu.yaml), so input rates and
#     placements are identical across techniques. Only the executor
#     --adaptive-mode flag differs, which changes how placement
#     transitions are reacted to (not what the workload is).
#   * the plotting step runs only AFTER all three CSVs are written, in
#     a separate container, using the plot-only entrypoint of
#     scripts/dynamic_load_validation.py.
#
# The NPU driver occasionally hangs on shutdown after a completed run;
# the per-scenario monitor loop detects "CSV has stopped growing" and
# force-kills the container to unblock the pipeline.
#
# Usage:
#   ./run_dynload_scenarios.sh            # run all three modes, then plot
#   ./run_dynload_scenarios.sh plot       # replot only from existing CSVs
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
HOST_WORK=$(cd "$HERE/.." && pwd)
RESULTS_DIR="$HERE/results"
SCHEDULE=tests/dynamic_load_views_schedule_npu.yaml

PHASE_A=18
PHASE_B=14
PHASE_C=20
DURATION=$((PHASE_A + PHASE_B + PHASE_C))
# Generous cap. The scenario itself runs for DURATION seconds of data;
# we allow NPU init + teardown + margin.
CAP=$((DURATION + 180))

declare -A CSV=(
    [3]="results/dynamic_load_static.csv"
    [0]="results/dynamic_load_stop_and_restart.csv"
    [1]="results/dynamic_load_hotswap.csv"
)
declare -A LABEL=(
    [3]=Static
    [0]=Stop-and-restart
    [1]=Hot-swap
)

docker_run_detached() {
    local cmd=$1
    docker run --rm -d --privileged --shm-size=1g --net host \
        -e DISPLAY=:0 \
        -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v "$HOME/.Xauthority":"$HOME/.Xauthority" \
        -v "$HOME/.Xauthority":/root/.Xauthority \
        --env USER="$(whoami)" --env UID="$(id -u)" --env GID="$(id -g)" \
        --env NB_HOME=/workspace \
        --env RISCV_PATH=/workspace/antara-software/toolchains/riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-linux-ubuntu14 \
        --env TOOL_PATH=/workspace/antara-software/toolchains/riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-linux-ubuntu14/bin \
        --env QUICKPCIE_SDK_ROOT= \
        --env LD_LIBRARY_PATH=/workspace/msyu_workdir_origin/antara-software/antara-pcie-sdk/quickPCIelib/qpcie_api/release:/workspace/lib \
        -v "$HOST_WORK":/workspace \
        neubla/antara \
        /bin/bash -lc "$cmd"
}

run_one() {
    local mode=$1
    local csv=${CSV[$mode]}
    local label=${LABEL[$mode]}
    local host_csv="$HERE/$csv"

    echo "=========================================================="
    echo "  Scenario: $label (mode=$mode) -> $csv"
    echo "=========================================================="
    rm -f "$host_csv"

    local cmd="cd /workspace/multimodel-scheduling && export QT_QPA_PLATFORM=offscreen && python3 -u schedule_executor_main.py --schedule $SCHEDULE --duration $DURATION --adaptive-mode $mode --metrics-csv $csv --auto_start_all --combo-duration phase_a=$PHASE_A --combo-duration phase_b=$PHASE_B --combo-duration phase_c=$PHASE_C"
    local cid
    cid=$(docker_run_detached "$cmd")
    echo "  container: $cid"

    local start=$SECONDS
    local last_size=0
    local stable_ticks=0
    while docker ps -q --no-trunc | grep -q "$cid"; do
        sleep 2
        local now=$SECONDS
        local elapsed=$((now - start))
        local size=0
        [[ -f "$host_csv" ]] && size=$(stat -c%s "$host_csv")
        if (( size > 0 && size == last_size )); then
            stable_ticks=$((stable_ticks + 1))
        else
            stable_ticks=0
        fi
        last_size=$size
        # Exit conditions:
        # 1. Hard cap reached.
        # 2. CSV exists and hasn't grown for ~10s AND we are past the
        #    scenario duration window (so we know post-run hang started).
        if (( elapsed >= CAP )); then
            echo "  [cap] ${elapsed}s reached; killing container"
            docker kill "$cid" >/dev/null 2>&1
            break
        fi
        if (( size > 0 && stable_ticks >= 5 && elapsed >= DURATION + 20 )); then
            echo "  [stable] CSV idle for 10s at ${elapsed}s; killing container"
            docker kill "$cid" >/dev/null 2>&1
            break
        fi
    done
    wait 2>/dev/null || true
    if [[ -f "$host_csv" ]]; then
        local rows
        rows=$(($(wc -l < "$host_csv") - 1))
        echo "  [done] $csv rows=$rows size=$(stat -c%s "$host_csv")"
    else
        echo "  [fail] $csv was not produced"
    fi
}

plot_only() {
    docker_run_detached "cd /workspace/multimodel-scheduling && python3 -u scripts/dynamic_load_validation.py --no-run 2>&1"
}

if [[ ${1:-} == "plot" ]]; then
    # run plot step synchronously via -it-less container
    docker run --rm -i --net host \
        -v "$HOST_WORK":/workspace \
        neubla/antara \
        /bin/bash -lc "cd /workspace/multimodel-scheduling && python3 -u scripts/dynamic_load_validation.py --no-run"
    exit 0
fi

mkdir -p "$RESULTS_DIR"
run_one 3
run_one 0
run_one 1

echo
echo "=========================================================="
echo "  Plotting"
echo "=========================================================="
docker run --rm -i --net host \
    -v "$HOST_WORK":/workspace \
    neubla/antara \
    /bin/bash -lc "cd /workspace/multimodel-scheduling && python3 -u scripts/dynamic_load_validation.py --no-run"

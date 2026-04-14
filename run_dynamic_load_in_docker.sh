#!/bin/bash
# Runs the NPU dynamic-load validation inside the neubla/antara container
# without requiring an interactive TTY. Mirrors docker_run.sh but drops
# the -t flag so it works when launched non-interactively.
set -euo pipefail

HOST_WORK=/home/msyu/msyu_workdir_origin

docker run --rm -i --privileged --shm-size=1g --net host \
  -e DISPLAY=:0 \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v "$HOME/.Xauthority":"$HOME/.Xauthority" \
  -v "$HOME/.Xauthority":/root/.Xauthority \
  --env "USER=$(whoami)" --env "UID=$(id -u)" --env "GID=$(id -g)" \
  --env NB_HOME=/workspace \
  --env RISCV_PATH=/workspace/antara-software/toolchains/riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-linux-ubuntu14 \
  --env TOOL_PATH=/workspace/antara-software/toolchains/riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-linux-ubuntu14/bin \
  --env QUICKPCIE_SDK_ROOT= \
  --env LD_LIBRARY_PATH=/workspace/msyu_workdir_origin/antara-software/antara-pcie-sdk/quickPCIelib/qpcie_api/release:/workspace/lib \
  -v "$HOST_WORK":/workspace \
  neubla/antara \
  /bin/bash -lc "$*"

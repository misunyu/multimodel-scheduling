#!/bin/bash

readonly developers_gid=1006

shell=${SHELL:-/bin/bash}
uid="${UID:-"$(id -u)"}"
user="${USER:-"$(getent passwd "$uid" | cut -d: -f1)"}"
gid="${GID:-"$(id -g)"}"
os=${OS:-"$(uname -s | tr [:upper:] [:lower:])"}

main() {

  # FIXME: For backward capabilities. Remove it on next version.
  if [ "$2" == "-i" ]; then set -- "$1" "-li"; fi

#  sudo -E -u "$user" /usr/bin/env "NB_HOME=$NB_HOME" "RISCV_PATH=$RISCV_PATH" "PATH=$TOOL_PATH:$PATH" "QUICKPCIE_SDK_ROOT=" "$@"

#  sudo -E -u "$user" /usr/bin/env \
#  "NB_HOME=$NB_HOME" \
#  "RISCV_PATH=$RISCV_PATH" \
#  "PATH=$TOOL_PATH:$PATH" \
#  "QUICKPCIE_SDK_ROOT=$QUICKPCIE_SDK_ROOT" \
#  "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
#  "HOME=/home/$user" \
#  "$@"

# -------------------------------
# 사용자 런타임 디렉토리 설정
# -------------------------------
export XDG_RUNTIME_DIR="/tmp/runtime-$(whoami)"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms

# -------------------------------
# 환경 변수 보장
# -------------------------------
export NB_HOME="/workspace"
export RISCV_PATH="$NB_HOME/antara-software/toolchains/riscv64-unknown-elf-toolchain-10.2.0-2020.12.8-x86_64-linux-ubuntu14"
export TOOL_PATH="$RISCV_PATH/bin"
export QUICKPCIE_SDK_ROOT=""
export PATH="$TOOL_PATH:$PATH"
export LD_LIBRARY_PATH="/workspace/msyu_workdir_origin/antara-software/antara-pcie-sdk/quickPCIelib/qpcie_api/release:/workspace/lib:$LD_LIBRARY_PATH"
export HOME="/home/$(whoami)"

# -------------------------------
# 진입 명령 실행
# -------------------------------
echo "[entrypoint] whoami = $(whoami)"
echo "[entrypoint] XDG_RUNTIME_DIR = $XDG_RUNTIME_DIR"
exec "$@"



  res=$?
  return $res
}

cd "$(realpath "$(dirname "$0")")"

IFS=''
main $*
res=$?
IFS=$' \t\n'
exit $res

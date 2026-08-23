#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PID_FILE="${WORKSPACE_ROOT}/logs/vllm_fullft_bf16.pid"
PORT="${PORT:-8002}"
PY_BIN="${PY_BIN:-${WORKSPACE_ROOT}/.venv/bin/python}"
TERM_WAIT_ATTEMPTS="${TERM_WAIT_ATTEMPTS:-30}"
PORT_WAIT_ATTEMPTS="${PORT_WAIT_ATTEMPTS:-10}"
WAIT_SLEEP_SECONDS="${WAIT_SLEEP_SECONDS:-1}"

if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3)"
fi

port_is_open() {
  "${PY_BIN}" - "${PORT}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.25)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

PIDS=()
while IFS= read -r pid; do
  [[ -n "${pid}" ]] && PIDS+=("${pid}")
done < <("${PY_BIN}" - "${PORT}" <<'PY'
import os
import pathlib
import shlex
import subprocess
import sys

port = sys.argv[1]
listing = subprocess.run(
    ["ps", "-axo", "pid=,uid=,command="],
    check=True,
    capture_output=True,
    text=True,
).stdout
for line in listing.splitlines():
    try:
        pid_text, uid_text, command = line.strip().split(None, 2)
        if int(uid_text) != os.getuid():
            continue
        args = shlex.split(command)
    except (ValueError, IndexError):
        continue
    if "serve" not in args or not any("vllm" in pathlib.Path(arg).name.lower() for arg in args):
        continue
    has_port = any(
        arg == f"--port={port}"
        or (arg == "--port" and index + 1 < len(args) and args[index + 1] == port)
        for index, arg in enumerate(args)
    )
    if has_port:
        print(pid_text)
PY
)

if (( ${#PIDS[@]} > 0 )); then
  echo "Stopping vLLM process(es) on port ${PORT}: ${PIDS[*]}"
  kill "${PIDS[@]}"
  for _ in $(seq 1 "${TERM_WAIT_ATTEMPTS}"); do
    alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
        break
      fi
    done
    if (( alive == 0 )) && ! port_is_open; then
      break
    fi
    sleep "${WAIT_SLEEP_SECONDS}"
  done
else
  echo "No same-user vLLM process found for port ${PORT}."
fi

if (( ${#PIDS[@]} > 0 )); then
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Process ${pid} is still alive; sending SIGKILL."
      kill -9 "${pid}" || true
    fi
  done
fi

for _ in $(seq 1 "${PORT_WAIT_ATTEMPTS}"); do
  if ! port_is_open; then
    break
  fi
  sleep "${WAIT_SLEEP_SECONDS}"
done

if port_is_open; then
  echo "ERROR: port ${PORT} is still accepting connections after the vLLM stop attempt." >&2
  exit 1
fi

rm -f "${PID_FILE}"
echo "Stopped vLLM service on port ${PORT}."

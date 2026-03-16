#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PID_FILE="${WORKSPACE_ROOT}/logs/vllm_fullft_bf16.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No PID file found at ${PID_FILE}"
  exit 0
fi

PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
if [[ -z "${PID}" ]]; then
  echo "PID file is empty. Removing it."
  rm -f "${PID_FILE}"
  exit 0
fi

if kill -0 "${PID}" 2>/dev/null; then
  echo "Stopping vLLM process ${PID}..."
  kill "${PID}"
  for _ in $(seq 1 30); do
    if ! kill -0 "${PID}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${PID}" 2>/dev/null; then
    echo "Process still alive, sending SIGKILL."
    kill -9 "${PID}" || true
  fi
else
  echo "Process ${PID} is not running."
fi

rm -f "${PID_FILE}"
echo "Stopped."

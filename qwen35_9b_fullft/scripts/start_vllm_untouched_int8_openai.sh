#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

VLLM_BIN="${WORKSPACE_ROOT}/.venv/bin/vllm"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b-untouched-int8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
API_KEY="${API_KEY:-}"
READY_WAIT_SEC="${READY_WAIT_SEC:-900}"
PY_HEADERS_ROOT="${PY_HEADERS_ROOT:-${WORKSPACE_ROOT}/.local_py312dev/usr/include}"

LOG_DIR="${WORKSPACE_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/vllm_untouched_int8_${TS}.log"
PID_FILE="${LOG_DIR}/vllm_untouched_int8.pid"

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "ERROR: vLLM binary not found at ${VLLM_BIN}"
  exit 1
fi

PY_HEADERS_PYTHON312="${PY_HEADERS_ROOT}/python3.12"
PY_HEADERS_MULTIARCH="${PY_HEADERS_ROOT}/aarch64-linux-gnu"
if [[ -d "${PY_HEADERS_ROOT}" ]]; then
  export CPATH="${PY_HEADERS_ROOT}:${PY_HEADERS_PYTHON312}:${PY_HEADERS_MULTIARCH}:${CPATH:-}"
  export CFLAGS="-I${PY_HEADERS_ROOT} -I${PY_HEADERS_PYTHON312} -I${PY_HEADERS_MULTIARCH} ${CFLAGS:-}"
  export CPPFLAGS="-I${PY_HEADERS_ROOT} -I${PY_HEADERS_PYTHON312} -I${PY_HEADERS_MULTIARCH} ${CPPFLAGS:-}"
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "vLLM already running (pid=${OLD_PID})."
    echo "Stop first: ${SCRIPT_DIR}/stop_vllm_untouched_openai.sh"
    exit 0
  fi
fi

HF_OVERRIDES='{"quantization_config":{"quant_method":"bitsandbytes","load_in_8bit":true,"load_in_4bit":false}}'
CHAT_KWARGS='{"enable_thinking":false}'

CMD=(
  "${VLLM_BIN}" serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --dtype bfloat16
  --quantization bitsandbytes
  --load-format bitsandbytes
  --hf-overrides "${HF_OVERRIDES}"
  --default-chat-template-kwargs "${CHAT_KWARGS}"
  --enforce-eager
  --disable-frontend-multiprocessing
  --language-model-only
  --trust-remote-code
)

if [[ -n "${API_KEY}" ]]; then
  CMD+=(--api-key "${API_KEY}")
fi

echo "Starting vLLM int8 server..."
echo "Model path: ${MODEL_PATH}"
echo "Model name: ${SERVED_MODEL_NAME}"
echo "Endpoint: http://${HOST}:${PORT}/v1"
echo "Log file: ${LOG_FILE}"

nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "PID: ${PID}"
echo "Waiting for server readiness..."
for _ in $(seq 1 "${READY_WAIT_SEC}"); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    LAN_IP="$(hostname -I | awk '{print $1}')"
    echo "READY"
    echo "Local: http://127.0.0.1:${PORT}/v1"
    if [[ -n "${LAN_IP}" ]]; then
      echo "LAN:   http://${LAN_IP}:${PORT}/v1"
    fi
    echo "Model id: ${SERVED_MODEL_NAME}"
    exit 0
  fi
  sleep 1
done

echo "Server did not become ready in time. Check logs:"
echo "  ${LOG_FILE}"
exit 1

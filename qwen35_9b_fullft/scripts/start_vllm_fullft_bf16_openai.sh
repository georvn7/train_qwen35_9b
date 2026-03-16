#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

VLLM_BIN="${WORKSPACE_ROOT}/.venv/bin/vllm"
PY_BIN="${WORKSPACE_ROOT}/.venv/bin/python"
MODEL_PATH_DEFAULT="${PROJECT_DIR}/runs/20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1/artifacts/full_model"
MODEL_PATH="${MODEL_PATH:-$MODEL_PATH_DEFAULT}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-9b-fullft-bf16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
API_KEY="${API_KEY:-}"
READY_WAIT_SEC="${READY_WAIT_SEC:-900}"
PY_HEADERS_ROOT="${PY_HEADERS_ROOT:-${WORKSPACE_ROOT}/.local_py312dev/usr/include}"

LOG_DIR="${WORKSPACE_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/vllm_fullft_bf16_${TS}.log"
PID_FILE="${LOG_DIR}/vllm_fullft_bf16.pid"

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

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model path does not exist: ${MODEL_PATH}"
  exit 1
fi

if [[ -f "${MODEL_PATH}/config.json" ]]; then
  MODEL_TYPE="$(
    "${PY_BIN}" - <<PY
import json
cfg=json.load(open("${MODEL_PATH}/config.json"))
print(cfg.get("model_type",""))
PY
  )"
  if [[ "${MODEL_TYPE}" == "qwen3_5_text" ]]; then
    COMPAT_DIR="${MODEL_PATH}_vllm_compat"
    "${PY_BIN}" "${PROJECT_DIR}/scripts/make_vllm_compat_fullft_model.py" \
      --full-model-dir "${MODEL_PATH}" \
      --out-dir "${COMPAT_DIR}" \
      --base-model-id "Qwen/Qwen3.5-9B"
    MODEL_PATH="${COMPAT_DIR}"
  fi
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "vLLM already running (pid=${OLD_PID})."
    echo "Stop first: ${SCRIPT_DIR}/stop_vllm_fullft_bf16_openai.sh"
    exit 0
  fi
fi

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
  --default-chat-template-kwargs "${CHAT_KWARGS}"
  --enforce-eager
  --disable-frontend-multiprocessing
  --language-model-only
)

if [[ -n "${API_KEY}" ]]; then
  CMD+=(--api-key "${API_KEY}")
fi

echo "Starting vLLM bf16 server..."
echo "Model path: ${MODEL_PATH}"
echo "Model name: ${SERVED_MODEL_NAME}"
echo "Endpoint: http://${HOST}:${PORT}/v1"
echo "Log file: ${LOG_FILE}"

# Hard-detach from parent session so serving survives command-wrapper exit.
nohup setsid "${CMD[@]}" </dev/null >"${LOG_FILE}" 2>&1 &
PID=$!
disown "${PID}" 2>/dev/null || true
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

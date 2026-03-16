#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-/home/georvn/train_qwen35_9b}"
STATE_DIR="$WORK/.state"
LOG_DIR="$WORK/logs"

SESSION_PTR="${SESSION_PTR:-$STATE_DIR/session_qwen35_9b_full1109_32k_v1.txt}"
RUN_SCRIPT="${RUN_SCRIPT:-$WORK/qwen35_9b_fullft/scripts/run_train_qwen35_9b_full1109_resume_safe.sh}"
DAEMON_LOG="${DAEMON_LOG:-$LOG_DIR/train_qwen35_9b_watchdog.log}"
LAUNCHER_STDOUT_LOG="${LAUNCHER_STDOUT_LOG:-$LOG_DIR/train_qwen35_9b_watchdog_stdout.log}"

POLL_SEC="${POLL_SEC:-20}"
MAX_LAUNCHES="${MAX_LAUNCHES:-16}"
launch_count=0

mkdir -p "$STATE_DIR" "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$DAEMON_LOG"
}

resolve_session_dir() {
  if [[ -f "$SESSION_PTR" ]]; then
    local from_ptr
    from_ptr="$(tr -d '\r\n' < "$SESSION_PTR" || true)"
    if [[ -n "${from_ptr:-}" && -d "${from_ptr:-}" ]]; then
      printf '%s\n' "$from_ptr"
      return 0
    fi
  fi

  local latest
  latest="$(ls -1dt "$WORK"/qwen35_9b_fullft/runs/*qwen35_9b_instruct_full1109_32k_recipe_v1 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest:-}" && -d "${latest:-}" ]]; then
    printf '%s\n' "$latest"
    return 0
  fi

  return 1
}

session_completed() {
  local session_dir="$1"
  [[ -f "$session_dir/artifacts/full_model/config.json" ]]
}

trainer_pid() {
  local session_dir="$1"
  pgrep -f -- "train_session.py --session-dir ${session_dir}" | head -n 1 || true
}

launcher_pid() {
  pgrep -f -- "run_train_qwen35_9b_full1109_resume_safe.sh" | head -n 1 || true
}

launch_resume_safe() {
  local env_cmd=(
    env
    SESSION_PTR="$SESSION_PTR"
    MIN_MEM_AVAIL_MIB="${MIN_MEM_AVAIL_MIB:-1536}"
    MAX_STAGNANT_FAILURES="${MAX_STAGNANT_FAILURES:-3}"
    MAX_TOTAL_ATTEMPTS="${MAX_TOTAL_ATTEMPTS:-24}"
    RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-8}"
    SAVE_STEPS="${SAVE_STEPS:-50}"
    LOGGING_STEPS="${LOGGING_STEPS:-5}"
    MAX_GPU_MEMORY_GIB="${MAX_GPU_MEMORY_GIB:-110}"
    CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION:-0.88}"
    CUDA_ALLOC_CONF="${CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
    CAUSAL_LOSS_MODE="${CAUSAL_LOSS_MODE:-active_chunked_no_upcast}"
    CAUSAL_LOSS_CHUNK_TOKENS="${CAUSAL_LOSS_CHUNK_TOKENS:-2048}"
    CHECKPOINT_MAX_SHARD_SIZE="${CHECKPOINT_MAX_SHARD_SIZE:-512MB}"
    CHECKPOINT_SAFE_SERIALIZATION="${CHECKPOINT_SAFE_SERIALIZATION:-true}"
    CHECKPOINT_PRESAVE_GC="${CHECKPOINT_PRESAVE_GC:-1}"
    CHECKPOINT_PRESAVE_EMPTY_CACHE="${CHECKPOINT_PRESAVE_EMPTY_CACHE:-1}"
    CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY="${CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY:-1}"
    RESUME_TORCH_LOAD_MMAP="${RESUME_TORCH_LOAD_MMAP:-1}"
    RUN_LOG="${RUN_LOG:-$LOG_DIR/train_qwen35_9b_full1109_32k_v1.log}"
    FAIL_LEDGER="${FAIL_LEDGER:-$LOG_DIR/train_qwen35_9b_failed_steps.tsv}"
    "$RUN_SCRIPT"
  )

  nohup "${env_cmd[@]}" >> "$LAUNCHER_STDOUT_LOG" 2>&1 &
  printf '%s\n' "$!"
}

SESSION_DIR="$(resolve_session_dir || true)"
if [[ -z "${SESSION_DIR:-}" ]]; then
  log "watchdog abort: could not resolve session dir from SESSION_PTR=$SESSION_PTR"
  exit 1
fi
printf '%s\n' "$SESSION_DIR" > "$SESSION_PTR"
log "watchdog start session_dir=$SESSION_DIR poll_sec=$POLL_SEC max_launches=$MAX_LAUNCHES"

while true; do
  if session_completed "$SESSION_DIR"; then
    log "watchdog stop: training artifacts detected, session complete"
    exit 0
  fi

  tpid="$(trainer_pid "$SESSION_DIR")"
  if [[ -n "${tpid:-}" ]]; then
    mem_avail_mib="$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)"
    gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
    gpu_total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
    gpu_used_mib="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{gsub(/ /, "", $1); if ($1+0 > m) m=$1+0} END {print m+0}' || true)"
    if [[ -z "${gpu_util:-}" ]]; then gpu_util="unknown"; fi
    if [[ -z "${gpu_total_mib:-}" || "${gpu_total_mib:-}" == "[N/A]" ]]; then gpu_total_mib="unknown"; fi
    if [[ -z "${gpu_used_mib:-}" ]]; then gpu_used_mib="unknown"; fi
    log "heartbeat trainer_pid=$tpid mem_avail_mib=$mem_avail_mib gpu_util_pct=$gpu_util gpu_mem_used_mib=$gpu_used_mib gpu_mem_total_mib=$gpu_total_mib"
    sleep "$POLL_SEC"
    continue
  fi

  lpid="$(launcher_pid)"
  if [[ -n "${lpid:-}" ]]; then
    log "launcher present without active trainer (pid=$lpid), waiting"
    sleep "$POLL_SEC"
    continue
  fi

  launch_count=$((launch_count + 1))
  if (( launch_count > MAX_LAUNCHES )); then
    log "watchdog abort: launch budget exceeded (max_launches=$MAX_LAUNCHES)"
    exit 1
  fi

  new_pid="$(launch_resume_safe)"
  log "launched resume-safe runner pid=$new_pid launch_count=$launch_count"
  sleep "$POLL_SEC"
done

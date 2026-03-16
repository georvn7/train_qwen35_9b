#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-/home/georvn/train_qwen35_9b}"
VENV_PY="${VENV_PY:-$WORK/.venv/bin/python}"
STATE_DIR="$WORK/.state"
LOG_DIR="$WORK/logs"
RUN_LOG="${RUN_LOG:-$LOG_DIR/train_qwen35_9b_full1109_resume_safe.log}"
FAIL_LEDGER="${FAIL_LEDGER:-$LOG_DIR/train_qwen35_9b_failed_steps.tsv}"
FAILED_STEPS_TODO="${FAILED_STEPS_TODO:-$STATE_DIR/train_qwen35_9b_failed_steps_todo.txt}"
SESSION_PTR="${SESSION_PTR:-$STATE_DIR/session_qwen35_9b_full1109_e1.txt}"

SAVE_STEPS="${SAVE_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
MAX_GPU_MEMORY_GIB="${MAX_GPU_MEMORY_GIB:-110}"
MAX_STAGNANT_FAILURES="${MAX_STAGNANT_FAILURES:-3}"
MAX_TOTAL_ATTEMPTS="${MAX_TOTAL_ATTEMPTS:-24}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-8}"
MIN_MEM_AVAIL_MIB="${MIN_MEM_AVAIL_MIB:-4096}"
EXTERNAL_GPU_GUARD_GIB="${EXTERNAL_GPU_GUARD_GIB:-0}"
CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION:-0.88}"
CUDA_ALLOC_CONF="${CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
CAUSAL_LOSS_MODE="${CAUSAL_LOSS_MODE:-default}"
CAUSAL_LOSS_CHUNK_TOKENS="${CAUSAL_LOSS_CHUNK_TOKENS:-2048}"
CHECKPOINT_MAX_SHARD_SIZE="${CHECKPOINT_MAX_SHARD_SIZE:-512MB}"
CHECKPOINT_SAFE_SERIALIZATION="${CHECKPOINT_SAFE_SERIALIZATION:-true}"
CHECKPOINT_PRESAVE_GC="${CHECKPOINT_PRESAVE_GC:-1}"
CHECKPOINT_PRESAVE_EMPTY_CACHE="${CHECKPOINT_PRESAVE_EMPTY_CACHE:-1}"
CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY="${CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY:-1}"
RESUME_TORCH_LOAD_MMAP="${RESUME_TORCH_LOAD_MMAP:-1}"

EXTERNAL_GPU_GUARD_MIB=$((EXTERNAL_GPU_GUARD_GIB * 1024))

DATASET_ROOT="${DATASET_ROOT:-$WORK/qwen35_9b_fullft/data/all_1109_rows_no_assistant_thinking.jsonl}"
LABEL="${LABEL:-candidate_qwen35_9b_full_all1109_seed3413_e1}"

mkdir -p "$STATE_DIR" "$LOG_DIR"
if [[ ! -f "$FAIL_LEDGER" ]]; then
  printf 'timestamp\tsession_dir\tphase\tattempt\tstagnant_failures\trc\tfrom_checkpoint\tto_checkpoint\tnote\n' >"$FAIL_LEDGER"
fi

log() {
  # Keep logs out of command substitution captures (e.g., SESSION_DIR="$(...)").
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$RUN_LOG" >&2
}

record_failure() {
  local phase="$1"
  local attempt="$2"
  local stagnant="$3"
  local rc="$4"
  local from_ckpt="$5"
  local to_ckpt="$6"
  local note="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Iseconds)" \
    "$SESSION_DIR" \
    "$phase" \
    "$attempt" \
    "$stagnant" \
    "$rc" \
    "${from_ckpt:-none}" \
    "${to_ckpt:-none}" \
    "$note" >>"$FAIL_LEDGER"
}

record_failed_step_todo() {
  local phase="$1"
  local from_ckpt="$2"
  local to_ckpt="$3"
  local note="$4"
  local key="phase=$phase|from=${from_ckpt:-none}|to=${to_ckpt:-none}|note=$note"
  touch "$FAILED_STEPS_TODO"
  if ! grep -Fxq "$key" "$FAILED_STEPS_TODO"; then
    printf '%s\n' "$key" >>"$FAILED_STEPS_TODO"
  fi
}

resolve_session_dir() {
  if [[ -f "$SESSION_PTR" ]]; then
    local from_ptr
    from_ptr="$(cat "$SESSION_PTR" | tr -d '\r\n' || true)"
    if [[ -n "${from_ptr}" && -d "${from_ptr}" ]]; then
      printf '%s\n' "$from_ptr"
      return 0
    fi
  fi

  log "creating new session label=$LABEL"
  "$VENV_PY" "$WORK/qwen35_9b_fullft/scripts/create_session.py" \
    --workspace-root "$WORK" \
    --dataset-root "$DATASET_ROOT" \
    --label "$LABEL" \
    --notes "Qwen3.5-9B full-finetuning on local all_1109_rows_no_assistant_thinking.jsonl (resume-safe)" \
    >> "$RUN_LOG" 2>&1

  local created
  created="$(ls -1dt "$WORK"/qwen35_9b_fullft/runs/*"${LABEL}" 2>/dev/null | head -n 1)"
  if [[ -z "$created" || ! -d "$created" ]]; then
    log "failed to create/resolve session dir for label=$LABEL"
    exit 1
  fi
  printf '%s\n' "$created"
}

latest_checkpoint() {
  local session_dir="$1"
  ls -1dt "$session_dir"/checkpoints/checkpoint-* 2>/dev/null | head -n 1 || true
}

SESSION_DIR="$(resolve_session_dir)"
printf '%s\n' "$SESSION_DIR" > "$SESSION_PTR"

CKPT="$(latest_checkpoint "$SESSION_DIR")"
log "session_dir=$SESSION_DIR"
log "latest_checkpoint=${CKPT:-none}"
log "python_interpreter=$VENV_PY"
log "dataset_root=$DATASET_ROOT"
log "retry_policy: max_stagnant_failures=$MAX_STAGNANT_FAILURES max_total_attempts=$MAX_TOTAL_ATTEMPTS retry_sleep_sec=$RETRY_SLEEP_SEC"
log "failure_tracking: ledger=$FAIL_LEDGER todo=$FAILED_STEPS_TODO"
log "memory_safety: internal_gpu_guard_gib=$MAX_GPU_MEMORY_GIB external_gpu_guard_gib=$EXTERNAL_GPU_GUARD_GIB min_mem_avail_mib=$MIN_MEM_AVAIL_MIB"
log "memory_tuning: cuda_memory_fraction=$CUDA_MEMORY_FRACTION cuda_alloc_conf=$CUDA_ALLOC_CONF causal_loss_mode=$CAUSAL_LOSS_MODE causal_loss_chunk_tokens=$CAUSAL_LOSS_CHUNK_TOKENS checkpoint_max_shard_size=$CHECKPOINT_MAX_SHARD_SIZE checkpoint_safe_serialization=$CHECKPOINT_SAFE_SERIALIZATION checkpoint_presave_gc=$CHECKPOINT_PRESAVE_GC checkpoint_presave_empty_cache=$CHECKPOINT_PRESAVE_EMPTY_CACHE checkpoint_presave_disable_cuda_history=$CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY resume_torch_load_mmap=$RESUME_TORCH_LOAD_MMAP"

build_train_cmd() {
  local checkpoint_arg="$1"
  CMD=(
    "$VENV_PY" "$WORK/qwen35_9b_fullft/scripts/train_session.py"
    --session-dir "$SESSION_DIR"
    --model-name "${MODEL_NAME:-Qwen/Qwen3.5-9B}"
    --max-seq-length "${MAX_SEQ_LENGTH:-32768}"
    --num-train-epochs "${NUM_TRAIN_EPOCHS:-1.0}"
    --truncation-side "${TRUNCATION_SIDE:-left}"
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
    --device-map "${DEVICE_MAP:-cuda:0}"
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --dataset-num-proc "${DATASET_NUM_PROC:-1}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
    --gradient-checkpointing "${GRADIENT_CHECKPOINTING:-unsloth}"
    --precision "${PRECISION:-auto}"
    --torch-dtype "${TORCH_DTYPE:-bfloat16}"
    --unsloth-mixed-precision "${UNSLOTH_MIXED_PRECISION:-auto}"
    --learning-rate "${LEARNING_RATE:-1e-5}"
    --warmup-steps "${WARMUP_STEPS:-50}"
    --seed "${SEED:-3413}"
    --logging-steps "$LOGGING_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "${SAVE_TOTAL_LIMIT:-4}"
    --max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB"
    --cuda-memory-fraction "$CUDA_MEMORY_FRACTION"
    --cuda-alloc-conf "$CUDA_ALLOC_CONF"
    --causal-loss-mode "$CAUSAL_LOSS_MODE"
    --causal-loss-chunk-tokens "$CAUSAL_LOSS_CHUNK_TOKENS"
    --checkpoint-max-shard-size "$CHECKPOINT_MAX_SHARD_SIZE"
    --checkpoint-safe-serialization "$CHECKPOINT_SAFE_SERIALIZATION"
    --full-finetuning
    --no-load-in-4bit
    --disable-unsloth-compile
    --disable-moe-triton
    --disable-flex-attention
    --disable-cce
    --no-packing
    --assistant-only-loss
    --group-by-length
    --skip-merged-export
    --skip-gguf-export
  )

  if [[ "$CHECKPOINT_PRESAVE_GC" == "1" || "$CHECKPOINT_PRESAVE_GC" == "true" ]]; then
    CMD+=(--checkpoint-presave-gc)
  else
    CMD+=(--no-checkpoint-presave-gc)
  fi
  if [[ "$CHECKPOINT_PRESAVE_EMPTY_CACHE" == "1" || "$CHECKPOINT_PRESAVE_EMPTY_CACHE" == "true" ]]; then
    CMD+=(--checkpoint-presave-empty-cache)
  else
    CMD+=(--no-checkpoint-presave-empty-cache)
  fi
  if [[ "$CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY" == "1" || "$CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY" == "true" ]]; then
    CMD+=(--checkpoint-presave-disable-cuda-history)
  else
    CMD+=(--no-checkpoint-presave-disable-cuda-history)
  fi
  if [[ "$RESUME_TORCH_LOAD_MMAP" == "1" || "$RESUME_TORCH_LOAD_MMAP" == "true" ]]; then
    CMD+=(--resume-torch-load-mmap)
  else
    CMD+=(--no-resume-torch-load-mmap)
  fi

  if [[ -n "${MAX_STEPS:-}" ]]; then
    CMD+=(--max-steps "$MAX_STEPS")
  fi
  if [[ -n "${checkpoint_arg}" ]]; then
    CMD+=(--resume-from-checkpoint "$checkpoint_arg")
  fi
}

stagnant_failures=0
total_attempts=0
last_ckpt="${CKPT:-}"

while true; do
  total_attempts=$((total_attempts + 1))
  if (( total_attempts > MAX_TOTAL_ATTEMPTS )); then
    log "training failed: total attempt budget exceeded (max_total_attempts=$MAX_TOTAL_ATTEMPTS)"
    record_failure "train_session" "$total_attempts" "$stagnant_failures" "1" "$CKPT" "$CKPT" "max_total_attempts_exceeded"
    record_failed_step_todo "train_session" "$CKPT" "$CKPT" "max_total_attempts_exceeded"
    exit 1
  fi

  CKPT="$(latest_checkpoint "$SESSION_DIR")"
  if [[ "${CKPT:-}" != "${last_ckpt:-}" ]]; then
    log "checkpoint advanced: ${last_ckpt:-none} -> ${CKPT:-none}; reset stagnant_failures"
    stagnant_failures=0
    last_ckpt="${CKPT:-}"
  fi

  build_train_cmd "$CKPT"
  log "RUN attempt=$total_attempts stagnant_failures=$stagnant_failures cmd=${CMD[*]}"

  guard_triggered="no"
  guard_reason=""
  set +e
  "${CMD[@]}" > >(tee -a "$RUN_LOG") 2> >(tee -a "$RUN_LOG" >&2) &
  TRAIN_PID=$!
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    MEM_AVAIL="$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)"
    GPU_USED=0
    if (( EXTERNAL_GPU_GUARD_MIB > 0 )); then
      GPU_USED="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk '{gsub(/ /, "", $1); if ($1+0 > m) m=$1+0} END {print m+0}')"
      if [[ -z "${GPU_USED:-}" ]]; then GPU_USED=0; fi
    fi
    if (( MIN_MEM_AVAIL_MIB > 0 && MEM_AVAIL <= MIN_MEM_AVAIL_MIB )); then
      guard_triggered="yes"
      guard_reason="mem_avail_mib=${MEM_AVAIL} <= min_mem_avail_mib=${MIN_MEM_AVAIL_MIB}"
    elif (( EXTERNAL_GPU_GUARD_MIB > 0 && GPU_USED >= EXTERNAL_GPU_GUARD_MIB )); then
      guard_triggered="yes"
      guard_reason="gpu_used_mib=${GPU_USED} >= external_gpu_guard_mib=${EXTERNAL_GPU_GUARD_MIB}"
    fi
    if [[ "$guard_triggered" == "yes" ]]; then
      log "external guard triggered: $guard_reason"
      kill -TERM "$TRAIN_PID" 2>/dev/null || true
      sleep 2
      kill -KILL "$TRAIN_PID" 2>/dev/null || true
      break
    fi
    sleep 1
  done
  wait "$TRAIN_PID"
  rc=$?
  set -e

  if [[ "$guard_triggered" == "yes" && "$rc" -eq 0 ]]; then
    rc=137
  fi

  if (( rc == 0 )); then
    log "training command exited successfully"
    break
  fi

  new_ckpt="$(latest_checkpoint "$SESSION_DIR")"
  if [[ "$guard_triggered" == "yes" ]]; then
    failure_note="external_guard_triggered:$guard_reason"
  else
    failure_note="failed_run"
  fi
  if [[ "${new_ckpt:-}" != "${CKPT:-}" ]]; then
    log "attempt failed rc=$rc but checkpoint progressed: ${CKPT:-none} -> ${new_ckpt:-none}; retrying"
    record_failure "train_session" "$total_attempts" "$stagnant_failures" "$rc" "$CKPT" "$new_ckpt" "failed_with_progress_retrying|$failure_note"
    stagnant_failures=0
    last_ckpt="${new_ckpt:-}"
  else
    stagnant_failures=$((stagnant_failures + 1))
    log "attempt failed rc=$rc with no checkpoint progress (stagnant_failures=$stagnant_failures/$MAX_STAGNANT_FAILURES)"
    record_failure "train_session" "$total_attempts" "$stagnant_failures" "$rc" "$CKPT" "$new_ckpt" "failed_no_progress|$failure_note"
  fi

  if (( stagnant_failures >= MAX_STAGNANT_FAILURES )); then
    log "training stopped: stagnant failure budget exceeded at checkpoint=${new_ckpt:-none}"
    record_failure "train_session" "$total_attempts" "$stagnant_failures" "$rc" "$CKPT" "$new_ckpt" "stagnant_failure_budget_exceeded"
    record_failed_step_todo "train_session" "$CKPT" "$new_ckpt" "stagnant_failure_budget_exceeded"
    exit "$rc"
  fi

  sleep "$RETRY_SLEEP_SEC"
done

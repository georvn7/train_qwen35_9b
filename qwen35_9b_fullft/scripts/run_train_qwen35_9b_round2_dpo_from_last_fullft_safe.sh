#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-/home/georvn/train_qwen35_9b}"
VENV_PY="${VENV_PY:-$WORK/.venv/bin/python}"
STATE_DIR="$WORK/.state"
LOG_DIR="$WORK/logs"
RUN_LOG="${RUN_LOG:-$LOG_DIR/train_qwen35_9b_round2_dpo_resume_safe.log}"
FAIL_LEDGER="${FAIL_LEDGER:-$LOG_DIR/train_qwen35_9b_round2_dpo_failed_steps.tsv}"
FAILED_STEPS_TODO="${FAILED_STEPS_TODO:-$STATE_DIR/train_qwen35_9b_round2_dpo_failed_steps_todo.txt}"

LAST_FULL_MODEL="${LAST_FULL_MODEL:-$WORK/qwen35_9b_fullft/runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1/artifacts/full_model}"
RAW_DPO_DATASET="${RAW_DPO_DATASET:-$WORK/qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean.jsonl}"
PREPARED_DPO_DATASET="${PREPARED_DPO_DATASET:-$WORK/qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean_prepared.jsonl}"
SESSION_PTR="${SESSION_PTR:-$STATE_DIR/session_qwen35_9b_round2_dpo_702_clean_16k_v1.txt}"
LABEL="${LABEL:-qwen35_9b_round2_dpo_702_clean_16k_v1}"

SAVE_STEPS="${SAVE_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
EXTRA_SAVE_STEPS="${EXTRA_SAVE_STEPS-10,20,30,40}"
DISABLE_INTERMEDIATE_CHECKPOINTS="${DISABLE_INTERMEDIATE_CHECKPOINTS:-0}"
MAX_GPU_MEMORY_GIB="${MAX_GPU_MEMORY_GIB:-110}"
MAX_STAGNANT_FAILURES="${MAX_STAGNANT_FAILURES:-3}"
MAX_TOTAL_ATTEMPTS="${MAX_TOTAL_ATTEMPTS:-24}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-8}"
MIN_MEM_AVAIL_MIB="${MIN_MEM_AVAIL_MIB:-3072}"
RESUME_MIN_MEM_AVAIL_MIB="${RESUME_MIN_MEM_AVAIL_MIB:-1024}"
CHECKPOINT_SAVE_MIN_MEM_AVAIL_MIB="${CHECKPOINT_SAVE_MIN_MEM_AVAIL_MIB:-1536}"
EXTERNAL_GPU_GUARD_GIB="${EXTERNAL_GPU_GUARD_GIB:-0}"
CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION:-0.88}"
CUDA_ALLOC_CONF="${CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
CHECKPOINT_MAX_SHARD_SIZE="${CHECKPOINT_MAX_SHARD_SIZE:-512MB}"
CHECKPOINT_SAFE_SERIALIZATION="${CHECKPOINT_SAFE_SERIALIZATION:-true}"
CHECKPOINT_PRESAVE_GC="${CHECKPOINT_PRESAVE_GC:-1}"
CHECKPOINT_PRESAVE_EMPTY_CACHE="${CHECKPOINT_PRESAVE_EMPTY_CACHE:-1}"
RESUME_TORCH_LOAD_MMAP="${RESUME_TORCH_LOAD_MMAP:-1}"
EXTERNAL_GPU_GUARD_MIB=$((EXTERNAL_GPU_GUARD_GIB * 1024))

mkdir -p "$STATE_DIR" "$LOG_DIR"
if [[ ! -f "$FAIL_LEDGER" ]]; then
  printf 'timestamp\tsession_dir\tphase\tattempt\tstagnant_failures\trc\tfrom_checkpoint\tto_checkpoint\tnote\n' >"$FAIL_LEDGER"
fi

log() {
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

  log "creating new DPO session label=$LABEL"
  "$VENV_PY" "$WORK/qwen35_9b_fullft/scripts/create_session.py" \
    --workspace-root "$WORK" \
    --dataset-root "$PREPARED_DPO_DATASET" \
    --label "$LABEL" \
    --notes "Round-2 DPO continuation from the latest round-2 full-FT model" \
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
  local checkpoint
  for checkpoint in $(ls -1dt "$session_dir"/checkpoints/checkpoint-* 2>/dev/null || true); do
    if [[ "$(basename "$checkpoint")" == *.incomplete* ]]; then
      continue
    fi
    if checkpoint_is_valid "$checkpoint"; then
      printf '%s\n' "$checkpoint"
      return 0
    fi
  done
  return 0
}

checkpoint_is_valid() {
  local checkpoint_dir="$1"
  [[ -d "$checkpoint_dir" ]] || return 1
  [[ -f "$checkpoint_dir/config.json" ]] || return 1
  [[ -f "$checkpoint_dir/trainer_state.json" ]] || return 1
  [[ -f "$checkpoint_dir/training_args.bin" ]] || return 1
  if [[ -f "$checkpoint_dir/checkpoint_complete.json" ]]; then
    return 0
  fi
  if [[ -f "$checkpoint_dir/model.safetensors.index.json" || -f "$checkpoint_dir/model.safetensors" || -f "$checkpoint_dir/pytorch_model.bin" || -f "$checkpoint_dir/pytorch_model.bin.index.json" ]]; then
    return 0
  fi
  return 1
}

quarantine_invalid_checkpoints() {
  local session_dir="$1"
  local checkpoint
  local quarantine_path
  for checkpoint in $(ls -1dt "$session_dir"/checkpoints/checkpoint-* 2>/dev/null || true); do
    if [[ "$(basename "$checkpoint")" == *.incomplete* ]]; then
      continue
    fi
    if checkpoint_is_valid "$checkpoint"; then
      continue
    fi
    quarantine_path="${checkpoint}.incomplete.$(date +%s)"
    mv "$checkpoint" "$quarantine_path"
    log "quarantined invalid checkpoint: $checkpoint -> $quarantine_path"
  done
}

prepare_dataset_if_needed() {
  if [[ -f "$PREPARED_DPO_DATASET" ]]; then
    return 0
  fi
  if [[ ! -d "$LAST_FULL_MODEL" ]]; then
    log "ERROR: previous full model not found: $LAST_FULL_MODEL"
    exit 1
  fi
  if [[ ! -f "$RAW_DPO_DATASET" ]]; then
    log "ERROR: raw DPO dataset not found: $RAW_DPO_DATASET"
    exit 1
  fi
  log "preparing DPO dataset view: $PREPARED_DPO_DATASET"
  "$VENV_PY" "$WORK/qwen35_9b_fullft/scripts/prepare_round2_dpo_dataset_view.py" \
    --input-jsonl "$RAW_DPO_DATASET" \
    --output-jsonl "$PREPARED_DPO_DATASET" \
    --tokenizer-model "$LAST_FULL_MODEL" \
    --max-prompt-length "${MAX_PROMPT_LENGTH:-14848}" \
    --max-completion-length "${MAX_COMPLETION_LENGTH:-1536}" \
    --max-length "${MAX_LENGTH:-16384}" \
    >> "$RUN_LOG" 2>&1
}

prepare_dataset_if_needed

SESSION_DIR="$(resolve_session_dir)"
printf '%s\n' "$SESSION_DIR" > "$SESSION_PTR"
quarantine_invalid_checkpoints "$SESSION_DIR"
CKPT="$(latest_checkpoint "$SESSION_DIR")"
log "session_dir=$SESSION_DIR"
log "latest_checkpoint=${CKPT:-none}"
log "python_interpreter=$VENV_PY"
log "prepared_dpo_dataset=$PREPARED_DPO_DATASET"
log "last_full_model=$LAST_FULL_MODEL"
log "retry_policy: max_stagnant_failures=$MAX_STAGNANT_FAILURES max_total_attempts=$MAX_TOTAL_ATTEMPTS retry_sleep_sec=$RETRY_SLEEP_SEC"
log "memory_safety: internal_gpu_guard_gib=$MAX_GPU_MEMORY_GIB external_gpu_guard_gib=$EXTERNAL_GPU_GUARD_GIB min_mem_avail_mib=$MIN_MEM_AVAIL_MIB resume_min_mem_avail_mib=$RESUME_MIN_MEM_AVAIL_MIB checkpoint_save_min_mem_avail_mib=$CHECKPOINT_SAVE_MIN_MEM_AVAIL_MIB resume_torch_load_mmap=$RESUME_TORCH_LOAD_MMAP"
if [[ "$DISABLE_INTERMEDIATE_CHECKPOINTS" == "1" || "$DISABLE_INTERMEDIATE_CHECKPOINTS" == "true" ]]; then
  log "checkpoint_policy: disable_intermediate_checkpoints=1 (resume anchor remains latest valid checkpoint; only final model export will persist new weights)"
else
  log "checkpoint_policy: save_steps=$SAVE_STEPS extra_save_steps=${EXTRA_SAVE_STEPS:-<none>}"
fi

build_train_cmd() {
  local checkpoint_arg="$1"
  local resume_warm_marker_arg="$2"
  local checkpoint_save_marker_arg="$3"
  local effective_save_steps="$SAVE_STEPS"
  local effective_extra_save_steps="$EXTRA_SAVE_STEPS"
  if [[ "$DISABLE_INTERMEDIATE_CHECKPOINTS" == "1" || "$DISABLE_INTERMEDIATE_CHECKPOINTS" == "true" ]]; then
    effective_save_steps="100000000"
    effective_extra_save_steps=""
  fi
  CMD=(
    "$VENV_PY" "$WORK/qwen35_9b_fullft/scripts/train_dpo_session.py"
    --session-dir "$SESSION_DIR"
    --model-name "${MODEL_NAME:-$LAST_FULL_MODEL}"
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
    --device-map "${DEVICE_MAP:-cuda:0}"
    --max-prompt-length "${MAX_PROMPT_LENGTH:-14848}"
    --max-completion-length "${MAX_COMPLETION_LENGTH:-1536}"
    --max-length "${MAX_LENGTH:-16384}"
    --truncation-mode "${TRUNCATION_MODE:-keep_end}"
    --num-train-epochs "${NUM_TRAIN_EPOCHS:-1.0}"
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
    --learning-rate "${LEARNING_RATE:-1e-6}"
    --warmup-steps "${WARMUP_STEPS:-50}"
    --weight-decay "${WEIGHT_DECAY:-0.01}"
    --logging-steps "$LOGGING_STEPS"
    --save-steps "$effective_save_steps"
    --extra-save-steps "$effective_extra_save_steps"
    --save-total-limit "${SAVE_TOTAL_LIMIT:-4}"
    --optim "${OPTIM:-adamw_8bit}"
    --seed "${SEED:-3413}"
    --beta "${BETA:-0.05}"
    --loss-type "${LOSS_TYPE:-sigmoid}"
    --precompute-ref-batch-size "${PRECOMPUTE_REF_BATCH_SIZE:-1}"
    --dataset-num-proc "${DATASET_NUM_PROC:-1}"
    --precision "${PRECISION:-auto}"
    --torch-dtype "${TORCH_DTYPE:-bfloat16}"
    --max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB"
    --cuda-memory-fraction "$CUDA_MEMORY_FRACTION"
    --cuda-alloc-conf "$CUDA_ALLOC_CONF"
    --checkpoint-max-shard-size "$CHECKPOINT_MAX_SHARD_SIZE"
    --checkpoint-safe-serialization "$CHECKPOINT_SAFE_SERIALIZATION"
    --precompute-ref-log-probs
    --use-logits-to-keep
    --resume-warm-marker-path "$resume_warm_marker_arg"
    --checkpoint-save-marker-path "$checkpoint_save_marker_arg"
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
  if [[ -n "${MAX_STEPS:-}" ]]; then
    CMD+=(--max-steps "$MAX_STEPS")
  fi
  if [[ -n "${MAX_SAMPLES:-}" ]]; then
    CMD+=(--max-samples "$MAX_SAMPLES")
  fi
  if [[ -n "${checkpoint_arg}" ]]; then
    CMD+=(--resume-from-checkpoint "$checkpoint_arg")
    if [[ "$RESUME_TORCH_LOAD_MMAP" == "1" || "$RESUME_TORCH_LOAD_MMAP" == "true" ]]; then
      CMD+=(--resume-torch-load-mmap)
    else
      CMD+=(--no-resume-torch-load-mmap)
    fi
  fi
}

stagnant_failures=0
total_attempts=0
last_ckpt="${CKPT:-}"

while true; do
  total_attempts=$((total_attempts + 1))
  if (( total_attempts > MAX_TOTAL_ATTEMPTS )); then
    log "training failed: total attempt budget exceeded"
    record_failure "train_dpo_session" "$total_attempts" "$stagnant_failures" "1" "$CKPT" "$CKPT" "max_total_attempts_exceeded"
    record_failed_step_todo "train_dpo_session" "$CKPT" "$CKPT" "max_total_attempts_exceeded"
    exit 1
  fi

  quarantine_invalid_checkpoints "$SESSION_DIR"
  CKPT="$(latest_checkpoint "$SESSION_DIR")"
  if [[ "${CKPT:-}" != "${last_ckpt:-}" ]]; then
    log "checkpoint advanced: ${last_ckpt:-none} -> ${CKPT:-none}; reset stagnant_failures"
    stagnant_failures=0
    last_ckpt="${CKPT:-}"
  fi

  RESUME_WARM_MARKER="$SESSION_DIR/metadata/resume_warm_marker.json"
  CHECKPOINT_SAVE_MARKER="$SESSION_DIR/metadata/checkpoint_save_marker.json"
  rm -f "$RESUME_WARM_MARKER"
  rm -f "$CHECKPOINT_SAVE_MARKER"
  build_train_cmd "$CKPT" "$RESUME_WARM_MARKER" "$CHECKPOINT_SAVE_MARKER"
  if [[ -n "${CKPT:-}" ]]; then
    log "resume warm guard active: checkpoint=${CKPT} min_mem_avail_mib=${RESUME_MIN_MEM_AVAIL_MIB} until marker=${RESUME_WARM_MARKER}; checkpoint_save_min_mem_avail_mib=${CHECKPOINT_SAVE_MIN_MEM_AVAIL_MIB}; steady_min_mem_avail_mib=${MIN_MEM_AVAIL_MIB}"
  fi
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
    ACTIVE_MIN_MEM_AVAIL_MIB="$MIN_MEM_AVAIL_MIB"
    ACTIVE_GUARD_PHASE="steady"
    if [[ -f "$CHECKPOINT_SAVE_MARKER" ]]; then
      ACTIVE_MIN_MEM_AVAIL_MIB="$CHECKPOINT_SAVE_MIN_MEM_AVAIL_MIB"
      ACTIVE_GUARD_PHASE="checkpoint_save"
    elif [[ -n "${CKPT:-}" && ! -f "$RESUME_WARM_MARKER" ]]; then
      ACTIVE_MIN_MEM_AVAIL_MIB="$RESUME_MIN_MEM_AVAIL_MIB"
      ACTIVE_GUARD_PHASE="resume_warm"
    fi
    if (( ACTIVE_MIN_MEM_AVAIL_MIB > 0 && MEM_AVAIL <= ACTIVE_MIN_MEM_AVAIL_MIB )); then
      guard_triggered="yes"
      guard_reason="phase=${ACTIVE_GUARD_PHASE} mem_avail_mib=${MEM_AVAIL} <= active_min_mem_avail_mib=${ACTIVE_MIN_MEM_AVAIL_MIB}"
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

  quarantine_invalid_checkpoints "$SESSION_DIR"
  new_ckpt="$(latest_checkpoint "$SESSION_DIR")"
  if [[ "$guard_triggered" == "yes" ]]; then
    failure_note="external_guard_triggered:$guard_reason"
  else
    failure_note="failed_run"
  fi
  if [[ "${new_ckpt:-}" != "${CKPT:-}" ]]; then
    log "attempt failed rc=$rc but checkpoint progressed: ${CKPT:-none} -> ${new_ckpt:-none}; retrying"
    record_failure "train_dpo_session" "$total_attempts" "$stagnant_failures" "$rc" "$CKPT" "$new_ckpt" "failed_with_progress_retrying|$failure_note"
    stagnant_failures=0
    last_ckpt="${new_ckpt:-}"
  else
    stagnant_failures=$((stagnant_failures + 1))
    log "attempt failed rc=$rc with no checkpoint progress (stagnant_failures=$stagnant_failures/$MAX_STAGNANT_FAILURES)"
    record_failure "train_dpo_session" "$total_attempts" "$stagnant_failures" "$rc" "$CKPT" "$new_ckpt" "failed_no_progress|$failure_note"
  fi

  if (( stagnant_failures >= MAX_STAGNANT_FAILURES )); then
    log "training stopped: stagnant failure budget exceeded at checkpoint=${new_ckpt:-none}"
    record_failure "train_dpo_session" "$total_attempts" "$stagnant_failures" "$rc" "$CKPT" "$new_ckpt" "stagnant_failure_budget_exceeded"
    record_failed_step_todo "train_dpo_session" "$CKPT" "$new_ckpt" "stagnant_failure_budget_exceeded"
    exit "$rc"
  fi

  sleep "$RETRY_SLEEP_SEC"
done

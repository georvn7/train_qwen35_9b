#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  BASE_MODEL=<baseline_tag> CAND_MODEL=<candidate_tag> qwen35_9b_fullft/scripts/run_ab_schema20_ollama.sh

Optional env vars:
  WORK, VENV_PY, BASE_URL, EVAL_FILE, REPORT_ROOT, LABEL
EOF
  exit 0
fi

WORK="${WORK:-/home/georvn/train_qwen35_9b}"
VENV_PY="${VENV_PY:-$WORK/.venv/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:11434}"
EVAL_FILE="${EVAL_FILE:-$WORK/qwen35_9b_fullft/evals/agent_cases_20_schema_final_v1.json}"
BASE_MODEL="${BASE_MODEL:-qwen3.5:9b_instruct}"
CAND_MODEL="${CAND_MODEL:-qwen35-9b-ft:candidate}"
REPORT_ROOT="${REPORT_ROOT:-$WORK/qwen35_9b_fullft/reports}"
LABEL="${LABEL:-ab_schema20_$(date +%Y%m%d_%H%M%S)_${BASE_MODEL//[:\/]/_}_vs_${CAND_MODEL//[:\/]/_}}"
REPORT_DIR="$REPORT_ROOT/$LABEL"

mkdir -p "$REPORT_DIR"

BASE_JSON="$REPORT_DIR/baseline_${BASE_MODEL//[:\/]/_}.json"
CAND_JSON="$REPORT_DIR/candidate_${CAND_MODEL//[:\/]/_}.json"
COMPARE_JSON="$REPORT_DIR/compare.json"
SUMMARY_MD="$REPORT_DIR/summary.md"

"$VENV_PY" "$WORK/qwen35_9b_fullft/scripts/benchmark_ollama_chat.py" \
  --endpoint-mode v1 \
  --model "$BASE_MODEL" \
  --base-url "$BASE_URL" \
  --messages-file "$EVAL_FILE" \
  --max-tokens 4096 \
  --temperature 0.0 \
  --top-p 1.0 \
  --num-ctx 32768 \
  --reasoning-effort medium \
  --verbosity medium \
  --save-json "$BASE_JSON" \
  --progress-log-file "$REPORT_DIR/baseline.progress.log"

"$VENV_PY" "$WORK/qwen35_9b_fullft/scripts/benchmark_ollama_chat.py" \
  --endpoint-mode v1 \
  --model "$CAND_MODEL" \
  --base-url "$BASE_URL" \
  --messages-file "$EVAL_FILE" \
  --max-tokens 4096 \
  --temperature 0.0 \
  --top-p 1.0 \
  --num-ctx 32768 \
  --reasoning-effort medium \
  --verbosity medium \
  --save-json "$CAND_JSON" \
  --progress-log-file "$REPORT_DIR/candidate.progress.log"

"$VENV_PY" "$WORK/qwen35_9b_fullft/scripts/compare_schema20_ab.py" \
  --baseline-json "$BASE_JSON" \
  --candidate-json "$CAND_JSON" \
  --baseline-model "$BASE_MODEL" \
  --candidate-model "$CAND_MODEL" \
  --compare-json "$COMPARE_JSON" \
  --summary-md "$SUMMARY_MD"

echo "report_dir=$REPORT_DIR"
echo "summary=$SUMMARY_MD"
echo "compare_json=$COMPARE_JSON"

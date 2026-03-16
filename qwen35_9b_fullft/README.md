# Qwen3.5 9B Full-Weight Fine-Tuning

This folder adapts your proven `gpt-oss-120b` session workflow to **Qwen3.5 9B full-finetuning**.

## Reproducibility First

- Canonical reproduction runbook:
  - `docs/REPRODUCE_FULLFT_20260307.md`
- Git vs external artifact policy:
  - `docs/REPO_KEEP_AND_DRIVE_POLICY.md`
- Canonical run snapshots for agents:
  - `docs/repro/README.md`

## Dataset (fixed target)

Default training dataset is copied locally to:

`/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_1109_rows_no_assistant_thinking.jsonl`

This is wired as the default in:

- `scripts/run_train_qwen35_9b_full1109_resume_safe.sh`

## Workflow

1. Create a session (manifest + reproducibility metadata):

```bash
python qwen35_9b_fullft/scripts/create_session.py \
  --workspace-root /home/georvn/train_qwen35_9b \
  --dataset-root /home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_1109_rows_no_assistant_thinking.jsonl \
  --label qwen35_9b_full_all1109 \
  --notes "Qwen3.5-9B full-finetuning on no-assistant-thinking 1109 rows"
```

2. Dry-run validation (manifest + dataset parse only):

```bash
python qwen35_9b_fullft/scripts/train_session.py \
  --session-dir qwen35_9b_fullft/runs/<session_id> \
  --model-name Qwen/Qwen3.5-9B \
  --full-finetuning \
  --no-load-in-4bit \
  --dry-run
```

3. Resume-safe real training:

```bash
qwen35_9b_fullft/scripts/run_train_qwen35_9b_full1109_resume_safe.sh
```

## Notes

- Session manifests now support dataset entries both inside and outside workspace (`relative_path` or `absolute_path`).
- Full-weight mode is the default in this stack:
  - `--full-finetuning` default: enabled
  - `--load-in-4bit` default: disabled
- Training artifacts:
  - `artifacts/full_model/` for full finetune checkpoints
  - `artifacts/merged_16bit/` and `artifacts/gguf/` optional exports
- Recommended full-FT defaults (validated on DGX Spark):
  - `model=Qwen/Qwen3.5-9B`
  - `max_seq_length=12288`, `truncation_side=left`
  - `max_gpu_memory_gib=110`
  - `dataset_num_proc=1`
  - `gradient_checkpointing=unsloth`
  - `learning_rate=1e-5`, `warmup_steps=50`
  - `gradient_accumulation_steps=1`
  - `save_steps=50`, `save_total_limit=4` (large full-FT checkpoints)
  - `precision=bf16`, `optim=adamw_8bit`

## Schema20 Eval Gate

- Eval cases: `qwen35_9b_fullft/evals/agent_cases_20_schema_final_v1.json`
- Benchmark scorer: `qwen35_9b_fullft/scripts/benchmark_ollama_chat.py`
- A/B comparator + strict gate: `qwen35_9b_fullft/scripts/compare_schema20_ab.py`
- One-command A/B runner: `qwen35_9b_fullft/scripts/run_ab_schema20_ollama.sh`

## LAN Serving (`/v1/chat/completions`)

- Canonical always-on single-user profile (same endpoint/model attrs each run):
  - `qwen35_9b_fullft/scripts/start_vllm_fullft_bf16_resident.sh`
- Start OpenAI-compatible vLLM server (full-FT model, BitsAndBytes 8-bit, default port `8000`):
  - `qwen35_9b_fullft/scripts/start_vllm_fullft_int8_openai.sh`
- Start OpenAI-compatible vLLM server (full-FT model, bf16, default port `8002`):
  - `qwen35_9b_fullft/scripts/start_vllm_fullft_bf16_openai.sh`
- Start untouched Qwen3.5-9B server (BitsAndBytes 8-bit, default port `8001`):
  - `qwen35_9b_fullft/scripts/start_vllm_untouched_int8_openai.sh`
- Stop server:
  - `qwen35_9b_fullft/scripts/stop_vllm_fullft_openai.sh`
  - `qwen35_9b_fullft/scripts/stop_vllm_fullft_bf16_openai.sh`
  - `qwen35_9b_fullft/scripts/stop_vllm_untouched_openai.sh`
- Full serving instructions:
  - `qwen35_9b_fullft/docs/SERVING_VLLM_INT8.md`

## Documentation

- Canonical repro runbook: `docs/REPRODUCE_FULLFT_20260307.md`
- Repo keep vs external storage policy: `docs/REPO_KEEP_AND_DRIVE_POLICY.md`
- External artifact manifest template: `docs/EXTERNAL_ARTIFACTS_MANIFEST_TEMPLATE.md`
- Decision and technical log: `docs/PROJECT_LOG.md`
- Run records and metrics snapshots: `docs/RUN_HISTORY.md`
- Current recommended training config: `docs/QUALITY_RECIPE.md`
- Strict schema benchmark gate: `docs/EVAL_SCHEMA20_GATE.md`
- Inference stack recommendation: `docs/INFERENCE_ENGINE_RECOMMENDATION.md`

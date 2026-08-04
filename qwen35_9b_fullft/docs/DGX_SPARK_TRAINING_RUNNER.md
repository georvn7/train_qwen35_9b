# DGX Spark Training Job Runner

This implements the DGX Spark side of Hen sequential training-job contract for the Hayabusa/Qwen 3.5 9B full-finetuning stack.

## Runner Command

From `/home/georvn/train_qwen35_9b`:

```bash
./.venv/bin/python qwen35_9b_fullft/scripts/train_job_runner.py \
  --jobs-root /home/georvn/train_qwen35_9b/jobs \
  --once
```

Fixture mode, no GPU and no vLLM restart:

```bash
./.venv/bin/python qwen35_9b_fullft/scripts/train_job_runner.py \
  --jobs-root /tmp/hayabusa_runner_fixture_jobs \
  --workspace-root /tmp/hayabusa_runner_fixture_workspace \
  --once \
  --fixture-mode
```

## Jobs Root

Hen should transfer bundles here:

```text
/home/georvn/train_qwen35_9b/jobs/incoming/<job_id>/
  job.json
  train_sft.jsonl
  train_dpo.jsonl
  READY
```

`READY` must be written last. The runner ignores directories without it.

The runner owns these state directories:

```text
/home/georvn/train_qwen35_9b/jobs/running
/home/georvn/train_qwen35_9b/jobs/completed
/home/georvn/train_qwen35_9b/jobs/failed
/home/georvn/train_qwen35_9b/jobs/runner.lock
```

## SSH Target

Observed DGX Spark host details:

```text
spark-08c2
10.0.0.34
```

Hen can use the configured SSH alias or `georvn@10.0.0.34`, depending on Mac-side SSH config.

## Required Base Checkpoint

For the next continuation job, use the latest local Hayabusa checkpoint:

```text
/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260713_055248_hayabusa-9b_dataset2_step2_simplec_s0_2_first_byte_dpo24_prepared_from_dataset2_sft11_16k_v1/artifacts/full_model
```

## Supported Training Profiles

V1 supports exactly one profile:

```text
micro_contract_validation
```

V1 rejects non-empty `stages.sft.overrides` or `stages.dpo.overrides`. This is deliberate because silently ignoring overrides would make the contract unsafe.

Supported stage combinations:

- bootstrap: `sft.enabled=true`, `dpo.enabled=true`; DPO starts from the SFT output.
- CDPO iteration: `sft.enabled=false`, `dpo.enabled=true`; DPO starts directly from `base_checkpoint`.

SFT-only jobs are rejected because curriculum convergence is orchestrated from DPO outputs.

Thinking-enabled jobs declare this manifest contract:

```json
"assistant_reasoning": {
  "mode": "required",
  "field": "thinking",
  "thinking_max_chars": 1800,
  "semantic_judging": "final_content_only"
}
```

The runner rejects missing or oversized final-assistant thinking in SFT and in
both DPO completions. Older manifests without this object retain answer-only
compatibility.

## SFT Recipe

The runner substitutes only `--session-dir`, `--model-name`, and `--max-seq-length` from the job manifest. The rest matches the known-good micro SFT recipe:

```bash
./.venv/bin/python qwen35_9b_fullft/scripts/train_session.py \
  --session-dir <sft_session_dir> \
  --model-name <job.base_checkpoint> \
  --max-seq-length <min(job.max_sequence_length,32768)> \
  --num-train-epochs 1.0 \
  --truncation-side left \
  --attn-implementation sdpa \
  --device-map cuda:0 \
  --per-device-train-batch-size 1 \
  --dataset-num-proc 1 \
  --gradient-accumulation-steps 1 \
  --gradient-checkpointing unsloth \
  --precision auto \
  --torch-dtype bfloat16 \
  --learning-rate 1e-5 \
  --save-steps 20 \
  --save-total-limit 4 \
  --max-gpu-memory-gib 110 \
  --cuda-memory-fraction 0.88 \
  --causal-loss-mode active_chunked_no_upcast \
  --causal-loss-chunk-tokens 2048 \
  --full-finetuning \
  --no-load-in-4bit \
  --assistant-only-loss \
  --loss-target final_assistant \
  --group-by-length
```

The actual log contains the full exact command including checkpoint, allocator, and export flags.
Full checkpoints are saved every 20 optimizer steps plus the mandatory final model export. This bounds recovery loss without repeatedly writing roughly 40 GiB checkpoints during short curriculum jobs.
`train_session.py` maps Hen `thinking` to Qwen `reasoning_content`; final-
assistant loss therefore includes both reasoning and final-answer tokens.

## DPO Recipe

The runner substitutes `--session-dir` and `--model-name <sft_checkpoint>`. The DPO profile remains capped at 16K because that is the known-good DGX Spark DPO recipe:

```bash
./.venv/bin/python qwen35_9b_fullft/scripts/train_dpo_session.py \
  --session-dir <dpo_session_dir> \
  --model-name <sft_checkpoint> \
  --attn-implementation sdpa \
  --device-map cuda:0 \
  --max-prompt-length 14848 \
  --max-completion-length 1536 \
  --max-length 16384 \
  --truncation-mode keep_end \
  --num-train-epochs 1.0 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-6 \
  --save-steps 20 \
  --save-total-limit 4 \
  --optim adamw_8bit \
  --beta 0.05 \
  --loss-type sigmoid \
  --precompute-ref-log-probs \
  --use-logits-to-keep
```

If `job.max_sequence_length` is below 16K, the runner lowers the DPO max length accordingly.
Conversational DPO rows remain unchanged on disk. Immediately before
tokenization, the trainer maps Hen `thinking` to Qwen `reasoning_content` in
chosen and rejected messages. The standard DPO objective then covers each
complete reasoning-plus-answer completion; no DPO hyperparameter changes are
required.

## Output Mapping

For `output_checkpoint=<name>`, the runner creates:

```text
qwen35_9b_fullft/runs/<timestamp>_contract_<name>_sft/
qwen35_9b_fullft/runs/<timestamp>_contract_<name>_dpo/
```

The final checkpoint is:

```text
<dpo_session_dir>/artifacts/full_model
```

The job directory records exact paths in `stage_sessions.json` and `result.json`.

## Validation Rules

The runner rejects before training on missing required fields, unsupported `format_version`, unsafe relative paths, checksum mismatch, row-count mismatch, malformed JSONL, invalid SFT/DPO schemas, `max_sequence_length > 32768`, unsupported profile, disabled SFT/DPO stages, and non-empty stage overrides.

## Deployment

Real mode stops the current vLLM server immediately after job validation and
before loading either the SFT or DPO training model. This gives the training
process exclusive GPU ownership. After successful training,
`deployment.enabled=true` restarts vLLM with thinking enabled when the job's
assistant-reasoning mode is `required`:

```bash
MODEL_PATH=<final_checkpoint> \
SERVED_MODEL_NAME=<served_model_name> \
MAX_MODEL_LEN=65536 \
GPU_MEMORY_UTILIZATION=0.70 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=32768 \
PORT=8002 \
READY_WAIT_SEC=900 \
ENABLE_THINKING=true \
qwen35_9b_fullft/scripts/start_vllm_fullft_bf16_openai.sh
```

Endpoint rule:

```text
local: http://127.0.0.1:8002/v1
LAN:   http://10.0.0.34:8002/v1
model: deployment.served_model_name, else output_checkpoint
```

For thinking-enabled jobs, the health check requires both separate reasoning
and non-empty final content before `status=complete`.

## Tests

Run:

```bash
./.venv/bin/python -m unittest qwen35_9b_fullft.tests.test_train_job_runner -v
```

Latest result on DGX Spark:

```text
Ran 20 tests
OK
```

Covered cases include thinking-aware SFT/DPO validation and Qwen rendering,
valid tiny bundle completion, invalid checksum, malformed SFT/DPO schemas,
concurrent runner lock, SFT failure preventing DPO, DPO failure preventing
deployment, health failure preventing completion, valid JSON status/result
after failure, and no repeat of completed jobs.

## Real Micro-Job Status

The real six-row SFT/thirteen-pair DPO micro-job has not been launched by this implementation step.

Reason: a resident vLLM server may be active, and the real job needs GPU ownership. The runner is implemented so `deployment.enabled=true` will stop and restart vLLM. Run the real micro-job only when it is acceptable for the runner to own the GPU and reload serving.

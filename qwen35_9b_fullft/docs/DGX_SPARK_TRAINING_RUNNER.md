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
  train_rl.jsonl
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

## Transport

The runner is host-independent. Transfer each immutable bundle to the selected
training host using SSH, shared storage, or another authenticated transport. Do
not commit host addresses, user-specific SSH material, API keys, or private
checkpoint paths. The producer must write `READY` only after `job.json` and all
input files have been durably transferred.

## Required Base Checkpoint

`job.json.base_checkpoint` is the authoritative starting checkpoint and must be
an absolute path available on the training host. The runner does not discover or
hardcode a "latest" model. A bootstrap job starts SFT from this checkpoint and
then DPO from the SFT output. A continuation job starts DPO directly from the
previous successful job's checkpoint recorded in `result.json`.

## Supported Training Profiles

V1 supports exactly one profile:

```text
micro_contract_validation
```

V1 rejects non-empty `stages.sft.overrides` or `stages.dpo.overrides`. This is deliberate because silently ignoring overrides would make the contract unsafe. RL accepts only the validated clip/KL/one-epoch overrides documented below.

Supported stage combinations:

- bootstrap: `sft.enabled=true`, `dpo.enabled=true`; DPO starts from the SFT output.
- CDPO iteration: `sft.enabled=false`, `dpo.enabled=true`; DPO starts directly from `base_checkpoint`.
- checkpointed RL: `sft.enabled=false`, `dpo.enabled=false`, `rl.enabled=true`; RL starts from the verified DPO checkpoint.
- repair-distance AWR: SFT, DPO, and RL are disabled; AWR starts from the immediately preceding verified checkpoint.

Exactly one of DPO, RL, or AWR must be enabled. SFT-only jobs and mixed objective jobs
are rejected because each curriculum phase has an independently auditable model
boundary.

Thinking-enabled jobs declare this manifest contract:

```json
"assistant_reasoning": {
  "mode": "required",
  "field": "thinking",
  "thinking_max_chars": 1800,
  "semantic_judging": "final_content_only"
}
```

SFT, DPO, and checkpointed RL use `semantic_judging=final_content_only`.
Repair-distance AWR uses `semantic_judging=not_used` because its deterministic
weights are computed before the Spark job and no judge participates in training.

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

The job manifest selects the branch execution strategy:

```json
"dpo_execution_mode": "batched | split_backward | auto"
```

- `batched` is faster and preserves the original 16K effective cap.
- `split_backward` computes the same sigmoid DPO objective while evaluating and
  backpropagating chosen and rejected branches serially; it supports the
  requested length through 32K with lower branch-dependent activation pressure.
- `auto` selects split backward above 16K and batched execution otherwise.

A 32K `auto` job resolves to:

```bash
./.venv/bin/python qwen35_9b_fullft/scripts/train_dpo_session.py \
  --session-dir <dpo_session_dir> \
  --model-name <sft_checkpoint> \
  --attn-implementation sdpa \
  --device-map cuda:0 \
  --max-prompt-length 31232 \
  --max-completion-length 1536 \
  --max-length 32768 \
  --truncation-mode keep_end \
  --dpo-execution-mode split_backward \
  --requested-dpo-execution-mode auto \
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

Split backward first obtains serial no-gradient branch log probabilities,
derives the exact scalar DPO gradients, then performs independent chosen and
rejected gradient passes. Reference log-probability precomputation is also
serialized. Unsupported objective combinations fail validation rather than
silently changing the mathematics.

The longest retained curriculum pair had a 23,538-token prompt. One full 32K
split-backward optimizer step completed untruncated on the DGX Spark without
OOM. `run_config.json` and `result.json` record requested/effective modes and
lengths; no OOM path silently reduces context.

Conversational DPO rows remain unchanged on disk. Immediately before
tokenization, the trainer maps Hen `thinking` to Qwen `reasoning_content` in
chosen and rejected messages. The standard DPO objective then covers each
complete reasoning-plus-answer completion; no DPO hyperparameter changes are
required.

The tokenizer keeps the single EOS marker already emitted by the Qwen chat
template instead of appending a duplicate marker. For over-length conversational
prompts, `keep_end` remains the policy, but truncation preserves chat structure:
the system message is retained, complete old turns are dropped at user
boundaries, and only an individually oversized newest user message is trimmed
from the start. `dpo_tokenization_stats.json` records the resulting
`prompt_truncation_modes`. Plain-string DPO rows keep the legacy raw token-level
`keep_end` behavior.

Every DPO session selects a stable, group-diverse frozen preference subset
before training. It records base, configured intermediate checkpoint, and final
loss/reward-margin/accuracy metrics against that same subset. This comparison
is diagnostic; the curriculum's fresh endpoint and student run remain the
authoritative outcome tests.

## Checkpointed RL Recipe

An RL-only job consumes one immutable `train_rl.jsonl`. Rows are grouped by a
historical debugger checkpoint and contain raw reward, normalized group
advantage, rollout-normalized policy-step weight, and one or more exact
Hayabusa prompt/completion pairs. Every group must contain at least two
rollouts and centered non-constant advantages. Normal policy responses require
non-empty thinking of at most 1,800 characters. The only exception is an exact
single-response reward-`-1` `reasoning_structure_negative` whose metadata and
completion prove `missing_thinking` or `thinking_too_long`; this teaches the
format failure without penalizing earlier valid responses. The malformed row
must still fit the 32,768-token sequence limit.

The serialized trainer uses the exact input checkpoint as the frozen old
policy. Before the first optimizer update it caches completion-token log
probabilities for every policy response. Each training forward handles one
sequence, masks prompt tokens, and applies the scalar rollout advantage to all
Qwen thinking and final-answer tokens. The objective is token-level clipped
policy optimization with an approximate KL penalty to that frozen policy.

The fixed V1 recipe is:

```bash
./.venv/bin/python qwen35_9b_fullft/scripts/train_rl_session.py \
  --session-dir <rl_session_dir> \
  --model-name <verified_dpo_checkpoint> \
  --max-length 32768 \
  --num-train-epochs 1 \
  --learning-rate 5e-7 \
  --clip-epsilon 0.20 \
  --kl-beta 0.01 \
  --optim adamw_8bit \
  --save-steps 10,20,40,60 \
  --save-total-limit 4
```

The trainer performs full fine-tuning; no base-model language parameters are
frozen. It trades additional serialized forwards for bounded 32K activation
memory. Resume from an optimizer checkpoint is deliberately unsupported in V1:
the old-policy cache and one-epoch update contract make ambiguous partial
resume unsafe, so interrupted immutable jobs fail and must be replaced.

RL frozen evaluation is selected before training and recorded at base,
configured intermediate checkpoints, and final. Metrics include policy loss,
approximate KL, clip fraction, ratio mean, gradient norm, tokenization and
truncation counts, memory, frozen-subset hash, and checkpoint identity.

The semantic teacher is not part of Spark training. macOS has already scored
only each terminal fix answer. Thinking is retained in `train_rl.jsonl` for
structure validation and model loss but is excluded from semantic judging.

Before the first production RL job for a model/length recipe, run the trainer
directly with `--smoke-optimizer-steps 1` against a representative immutable
RL bundle. Smoke mode performs tokenization, frozen old-policy precomputation,
base/final evaluation, and one real optimizer step. It records loss and peak
memory but deliberately does not create `artifacts/full_model`, so it cannot be
deployed or mistaken for a production lineage checkpoint.

These advantages are on/near-policy only for the exact recorded checkpoint
lineage. Reusing the rows with another model is off-policy and is not equivalent
to this trainer's objective.

## Repair-Distance AWR Recipe

An AWR-only job consumes one immutable `train_awr.jsonl`. Each row contains a
positive finite `sample_weight`, the exact student prompt, and one final
assistant completion with Qwen thinking plus visible content. Prompt and prior
assistant tokens are masked; only that final thinking-and-answer completion is
trained. The objective is `sample_weight * mean completion NLL`.

AWR uses serialized 32K full fine-tuning, one epoch, `learning_rate=5e-7`, and
the same prompt-left-truncation rule as checkpointed RL. Checkpoints 10/20/40/60
and final are evaluated against one frozen subset and recorded in
`frozen_checkpoint_metrics.jsonl`. Checkpoint retention is bounded by
`save_total_limit`. The runner requires non-empty AWR metrics before deployment.

## Output Mapping

For `output_checkpoint=<name>`, the runner creates:

```text
qwen35_9b_fullft/runs/<timestamp>_contract_<name>_sft/
qwen35_9b_fullft/runs/<timestamp>_contract_<name>_dpo/
qwen35_9b_fullft/runs/<timestamp>_contract_<name>_rl/
qwen35_9b_fullft/runs/<timestamp>_contract_<name>_awr/
```

The final checkpoint is:

```text
<dpo_or_rl_session_dir>/artifacts/full_model
```

The job directory records exact paths in `stage_sessions.json` and `result.json`.

## Per-Stage Training Observability

Future SFT, DPO, serialized RL, and repair-distance AWR sessions persist an
`execution_observability` object in `metadata/train_metrics.json`. The runner
copies that object unchanged into the stage metrics in immutable `result.json`.
It records:

- configured maximum sequence length, packing mode, and observed token lengths;
- instantiated optimizer class, configured optimizer name, and materialized
  optimizer-state tensor dtypes, element counts, and byte counts;
- trainable/model parameter dtypes and counts;
- gradient-checkpointing state and the active loss implementation;
- peak CUDA allocator memory and process high-water RSS;
- a clearly labeled, non-packed token-exposure throughput estimate.

DPO additionally records whether reference log probabilities were precomputed,
whether a durable cache was reused, and the devices/bytes of any reference-model
object still resident immediately before optimization. This distinguishes a
precomputed-reference run from one that retained reference weights in memory.
Throughput is labeled `estimated_tokens_per_second`: it is computed from the
final tokenized rows and completed epoch fraction, rather than claimed as a
hardware token counter.

## Validation Rules

The runner rejects before training on missing required fields, unsupported
`format_version`, unsafe relative paths, checksum mismatch, row-count mismatch,
malformed JSONL, invalid SFT/DPO/RL/AWR schemas, `max_sequence_length > 32768`,
unsupported profiles/stage combinations, ambiguous RL resume, and unsupported
overrides.

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
./.venv/bin/python -m unittest \
  qwen35_9b_fullft.tests.test_train_job_runner \
  qwen35_9b_fullft.tests.test_train_dpo_session \
  qwen35_9b_fullft.tests.test_train_rl_session -v
```

Latest result on DGX Spark:

```text
Ran 43 tests
OK
```

Covered cases include thinking-aware SFT/DPO/RL/AWR validation and Qwen rendering,
serial-versus-batched loss/gradient/update equivalence, accumulation scaling,
automatic execution-mode selection and result metadata,
RL prompt masking, reasoning-plus-answer targets, old-policy precomputation,
clipped-objective behavior, and frozen-subset selection,
valid tiny bundle completion, invalid checksum, malformed SFT/DPO schemas,
concurrent runner lock, SFT failure preventing DPO, DPO failure preventing
deployment, health failure preventing completion, valid JSON status/result
after failure, and no repeat of completed jobs.

## Operational Validation

The runner is exercised by real reasoning-aware bootstrap and DPO-only
continuation jobs. Runtime job bundles, receipts, datasets, checkpoints, and
endpoint credentials are external artifacts and are intentionally not stored in
Git. Fixture tests cover the orchestration contract without requiring a GPU.

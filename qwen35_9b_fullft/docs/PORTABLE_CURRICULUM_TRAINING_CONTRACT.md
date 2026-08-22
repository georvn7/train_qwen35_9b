# Portable Curriculum Training Contract

This document defines the semantic behavior a second training backend, including
MLX, must preserve. CUDA-, Unsloth-, TRL-, and vLLM-specific implementation
details may change; the supervised tokens, preference pairs, checkpoint chain,
and validation gates may not.

## Inputs

A job bundle contains `job.json`, optional `train_sft.jsonl`, exactly one of
`train_dpo.jsonl`, `train_rl.jsonl`, or `train_awr.jsonl`, and a `READY` marker written last. The
manifest records input row counts and SHA-256 hashes. The runner must validate
these before claiming GPU resources. DPO, RL, and AWR are separate immutable
jobs and must not be enabled together.

Reasoning-aware jobs declare:

```json
{
  "assistant_reasoning": {
    "mode": "required",
    "field": "thinking",
    "thinking_max_chars": 1800,
    "semantic_judging": "final_content_only"
  }
}
```

SFT, DPO, and checkpointed RL use `final_content_only`; AWR uses `not_used`
because its weights are deterministic upstream labels rather than judge output.

The dataset remains in Hen's conversational schema on disk. The training backend
maps final-assistant `thinking` to Qwen's `reasoning_content` only while applying
the model chat template.

## SFT Semantics

- Perform full-weight language-model fine-tuning, not LoRA or QLoRA.
- Train for one epoch from `job.json.base_checkpoint`.
- Render messages with the checkpoint's Qwen chat template.
- Supervise only the final assistant turn.
- The supervised span includes both final-assistant reasoning and final content.
- Earlier assistant messages remain context and receive no loss.
- Preserve the end of over-length context; do not silently remove or truncate the
  final assistant target.

Current production defaults are batch size 1, gradient accumulation 1, BF16,
`learning_rate=1e-5`, no packing, and a maximum SFT sequence length supplied by
the job up to 32K.

## DPO Semantics

Each row contains conversational `prompt`, `chosen`, and `rejected` fields.
Normalize `thinking` to `reasoning_content`, then render every field through the
same Qwen chat template used for inference. Do not flatten, reword, or
chat-template a field twice.

- A bootstrap job starts DPO from its SFT checkpoint.
- A continuation job starts DPO from the immediately preceding successful DPO
  checkpoint.
- Preference loss covers each complete reasoning-plus-answer completion.
- Prompt tokens are context, not completion targets.
- Use one epoch, `learning_rate=1e-6`, `beta=0.05`, sigmoid DPO loss, batch size
  1, and gradient accumulation 1.
- The manifest may request `batched`, `split_backward`, or `auto` execution.
  Batched execution retains a 16,384-token cap; split backward supports the
  requested total length through 32,768 by serializing chosen/rejected policy
  forwards and backwards without changing the preference objective. `auto`
  selects split backward above 16K.
- Current completion capacity is 1,536 tokens; remaining sequence capacity is
  assigned to the prompt with `keep_end` truncation.
- Record requested and effective execution mode and sequence lengths. Never
  silently reduce context after an OOM.
- Chat-templated chosen and rejected completions must contain exactly the EOS
  marker emitted by the template. Do not append another EOS when the rendered
  completion already ends with one (ignoring trailing whitespace).
- `keep_end` truncation for conversational prompts must preserve chat framing:
  retain the system message, drop complete old turns at user boundaries first,
  and, only when the newest user turn still exceeds the limit, trim the start of
  that user's content while preserving its role header and newest evidence.
  Plain-string prompts retain raw token-level `keep_end` compatibility.
- A completion that exceeds its bound must be reported; it must not be silently
  relabeled or lose only its reasoning section.

## Sequential Checkpoint Invariant

The training chain is ordered:

```text
base -> cycle 1 SFT -> cycle 1 DPO -> cycle 1 RL -> cycle 2 DPO -> cycle 2 RL -> ...
```

For an eligible recovery round, one optional repair-distance AWR job may run
after Cycle 1's DPO and checkpointed RL. It is a round-level objective and must
not repeat in later cycles of that round.

A failed job must not advance the active checkpoint. Every successful result
must identify the exact output checkpoint and input dataset hashes. The next job
must name that output as its base checkpoint; filesystem recency is not an
acceptable substitute.

## Checkpointed RL Semantics

Each RL row belongs to a historical debugger-checkpoint group and contains a
raw terminal reward, a group-normalized advantage, a rollout-normalized
policy-step weight, and ordered exact prompt/completion pairs from one
Hayabusa rollout.

- Start from the immediately preceding verified DPO checkpoint.
- Compute and freeze old-policy completion-token log probabilities before the
  first optimizer update.
- Mask prompt tokens. Train every sampled policy response in the rollout,
  including both assistant reasoning and final-answer tokens.
- Apply one scalar rollout advantage to each response, normalized by the number
  of policy responses so long rollouts do not receive more total weight merely
  for being long.
- Use one epoch, serialized one-sequence forwards, `learning_rate=5e-7`, clip
  epsilon `0.20`, KL beta `0.01`, and maximum total length 32,768.
- Reject groups without at least two rollouts or centered non-constant
  advantages. Normal policy responses require non-empty reasoning of at most
  1,800 characters.
- A reasoning-format violation is trainable only as one isolated policy
  response with reward `-1`, `reasoning_structure_valid=false`, and consistent
  `reasoning_structure_negative` evidence naming `missing_thinking` or
  `thinking_too_long`. Never admit this exception for a positive/graded row or
  propagate its negative reward to preceding valid responses. The complete
  prompt and malformed completion must still fit the 32,768-token limit.
- Preserve the full completion and truncate only old prompt context from the
  start. Reject a row if its completion cannot fit.
- Do not resume a partial optimizer checkpoint in V1; replacement jobs must
  restart from the immutable input checkpoint and old-policy contract.

The semantic teacher score is an upstream data-generation artifact. It sees
only the terminal fix answer, not thinking. Thinking remains supervised by the
RL objective once the scalar rollout reward has been assigned.

Grouped advantages are on/near-policy only for their recorded model checkpoint.
Using them directly with another model is off-policy and is not equivalent to
this contract.

## Repair-Distance AWR Semantics

Repair-distance AWR trains one final assistant response per recorded failed
student policy decision, including thinking and visible content. Prompt and all
prior responses are masked. A positive deterministic sample weight combines
the failed trajectory's unhinted-teacher repair distance with local fix/blocker
shaping. The serialized objective is weighted mean completion NLL; it is not a
preference pair or an on-policy rollout objective. AWR runs at most once per
eligible recovery round.

## Frozen Checkpoint Evaluation

DPO, RL, and AWR select one deterministic frozen subset before training and evaluate
the base, configured intermediate checkpoints, and final checkpoint against
that same subset. Record subset identity/hash, checkpoint identity, loss, and
phase-specific quality metrics. These comparisons diagnose over-training and
checkpoint quality; fresh deployment and student execution remain authoritative.

## Inference Contract

Reasoning-aware checkpoints are served with the Qwen reasoning parser and
thinking enabled. A deployment health check must require both:

- non-empty separate reasoning; and
- non-empty final content conforming to the requested response schema.

Training and inference must use the same tokenizer, special tokens, and chat
template behavior.

## MLX Equivalence Gate

Before replacing the reference backend, compare a fixed micro-bundle on both stacks:

1. Identical rendered token IDs for representative SFT, DPO, and RL rows.
2. Identical supervised SFT span boundaries.
3. Identical prompt/chosen/rejected DPO boundaries after truncation.
   Truncated conversational prompts must still start at a valid chat-role
   boundary, and rendered completions must not contain a duplicated EOS.
4. Identical RL prompt/completion boundaries, including reasoning-plus-answer
   completion masks and frozen old-policy identity.
5. The same base checkpoint and ordered continuation checkpoint.
6. Finite loss and gradients with all intended language-model parameters
   trainable.
7. A loadable checkpoint that produces separate reasoning and final content.
8. No mutation of source JSONL files.

Exact floating-point losses need not match across frameworks, but tokenization,
loss masks, preference direction, and checkpoint ancestry must match. Keep the
reference path available as the reference implementation until this gate passes.

# Round 2 DPO Plan

- date: `2026-04-20`
- start model: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1/artifacts/full_model`
- raw DPO dataset: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_706_rows_dbg_dpo_round2.jsonl`
- canonical cleaned DPO dataset: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean.jsonl`
- canonical prepared DPO dataset: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean_prepared.jsonl`

## Non-Negotiables

- Start from the latest finished round-2 full-FT model, not from scratch.
- Keep Spark safety conventions close to the March/April full-FT recipe:
  - `bf16`
  - `adamw_8bit`
  - `save_steps=50`
  - `save_total_limit=4`
  - `max_gpu_memory_gib=110`
  - `cuda_memory_fraction=0.88`
  - checkpoint pre-save cleanup

## Why DPO Now

- The remaining schema20 misses are mostly next-step preference errors:
  - over-predicting `function_info`
  - under-predicting `debug_function`, `log_info`, and `file_info`
  - sometimes jumping to `fix_function` too early
- That is a policy/preference problem, not a JSON-format problem, so DPO is the right next training stage.

## Dataset Shape

- `706` rows total
- raw format is structurally valid conversational preference data:
  - `prompt`
  - `chosen`
  - `rejected`
- each `chosen` and `rejected` side is an assistant continuation

## Spark-Safe DPO Recipe

- trainer: `trl==0.24.0` `DPOTrainer`
- attention: `sdpa`
- precision: `bf16`
- optimizer: `adamw_8bit`
- epochs: `1`
- per-device batch size: `1`
- grad accumulation: `1`
- learning rate: `1e-6`
- warmup steps: `50`
- beta: `0.05`
- loss: `sigmoid`
- precompute reference log-probs: `true`
- use logits-to-keep: `true`
- padding-free: `false`
- steady host-RAM guard: `3072 MiB`
- resume warmup host-RAM guard: `1024 MiB`
- checkpoint-save host-RAM guard: `1536 MiB`
- early durable checkpoints: `10,20,30,40` in addition to regular `save_steps=50`
- resume checkpoint loads use `torch.load(..., mmap=True)` under the checkpoint tree with automatic fallback

## Length Budget

- `max_prompt_length=14848`
- `max_completion_length=1536`
- `max_length=16384`
- `truncation_mode=keep_end`

Reason:

- DPO carries both `chosen` and `rejected` continuations, so `32K` DPO is not comparable to `32K` SFT on Spark.
- The measured raw DPO distribution supports a `16K` side budget with low truncation pressure.

## Prepared Dataset View

- script: `scripts/prepare_round2_dpo_dataset_view.py`
- cleaning script: `scripts/clean_round2_dpo_dataset.py`
- output: `qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean_prepared.jsonl`

This prepared view freezes the Qwen chat template into plain `prompt/chosen/rejected` strings and writes explicit token-budget stats beside it.

## Dataset Repair

- `4` source DPO rows had empty chosen outputs and cannot be repaired from the source trajectory files.
- `3` rows were missing the `breakpoints` field on the rejected side and are normalized to `[]`.
- The canonical DPO training set therefore uses `702` rows, not the raw `706`.

## Local Environment Notes

- `Unsloth` compatibility patch from the earlier SFT work remains local in the venv:
  - `/home/georvn/train_qwen35_9b/.venv/lib/python3.12/site-packages/unsloth/models/_utils.py`
- For this DPO stage, local `trl` also needed lazy-import fixes so `DPOTrainer` can import without unrelated extras:
  - `trl/trainer/judges.py`
  - `trl/trainer/callbacks.py`

These are environment-local compatibility patches, not repo-tracked upstream fixes.

## Resume Hardening

- Full DPO checkpoints are large because they include optimizer state. On this run family, `checkpoint-10/optimizer.pt` is about `23 GiB`.
- Exact resume is therefore kept, but the recovery path is hardened in two ways:
  - durable reference-logprob cache is written to `metadata/train_ref_logprobs_cache.npz` after precompute completes, so retries do not repeat precompute
  - resume uses a two-phase external host-RAM guard:
    - looser threshold during checkpoint restore and optimizer rehydration
    - tighter threshold after the first resumed training step completes
- The training process writes `metadata/resume_warm_marker.json` after its first completed step so the launcher can switch from resume-warmup guard back to steady-state guard.
- The training process also writes `metadata/checkpoint_save_marker.json` immediately before checkpoint save and clears it after save completes, so the launcher can apply a lower host-RAM floor only during checkpoint writes.
- Completed DPO checkpoints now write `checkpoint_complete.json`; launcher-side checkpoint selection prefers fully completed checkpoints and quarantines partial `checkpoint-*` directories.
- Resume also forces mmap-backed `torch.load` under the checkpoint path, with automatic fallback if mmap is unsupported.

## Execution Order

1. Build the prepared DPO dataset view.
2. Dry-run the DPO session script against the prepared dataset.
3. Run a `1-step` smoke.
4. Run a `20-step` checkpoint smoke.
5. If stable, launch the full `706`-row DPO epoch.

# Visual Tower Footprint (Qwen3.5-9B)

Date: `2026-03-06`

## Why this note exists

User requested text-only full FT ("we aren't going to train visual"), so we quantified how much `model.visual.*` costs and what it changes for memory.

## Measured parameter split

From `Qwen3_5ForConditionalGeneration` config instantiation:

- Total params: `9,409,813,744`
- Visual params (`model.visual.*`): `456,010,480` (`4.846%`)
- Text params: `8,953,803,264`

## Memory impact (approx)

- Visual weights in BF16: `~0.849 GiB`
- If visual also receives gradients: `+~0.849 GiB`
- If optimizer states are FP32 Adam moments: `+~3.398 GiB`
- Worst-case visual footprint with full gradients+FP32 states: `~5.10 GiB`

In text-only batches (no image/video inputs), visual gradients and optimizer states are typically not materialized, but visual weights are still resident unless the loader avoids the vision tower.

## Implemented change

`scripts/train_session.py` now defaults to forcing text-only loading via:

- `--force-causal-lm-loader` (default enabled)

This routes Unsloth through `AutoModelForCausalLM`, which loads `Qwen3_5ForCausalLM` (no `model.visual` module).

Validation snippet result:

- Loaded class: `Qwen3_5ForCausalLM`
- `has model.visual = False`
- Trainable params: `8,953,803,264`


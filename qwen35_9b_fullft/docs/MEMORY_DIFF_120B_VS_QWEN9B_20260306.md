# Why 120B 32K Worked But 9B Initially Failed

Date: `2026-03-06`

## Short Answer

The key difference is **training mode**, not headline model size:

- `gpt-oss-120b` run used `4-bit QLoRA` (`load_in_4bit=true`, `full_finetuning=false`), so base weights were frozen and optimizer state was tiny (LoRA-only trainable set).
- `Qwen3.5-9B` run used `full_finetuning=true` with BF16 dense updates, so it carried full weights + full gradients + full optimizer states, then long-context activation spikes on top.

## Evidence Snapshot

## `gpt-oss-120b` (archived promoted runs)

- Source manifests:
  - `/home/georvn/final_archive-20260305T073201Z-3-001/final_archive/20260304_gptoss_debugger_adapters_only_bundle.tar.zst`
- Run config facts (from archived `run_config.json`):
  - `model_name=unsloth/gpt-oss-120b-unsloth-bnb-4bit`
  - `max_length=32768`
  - `optim=adamw_8bit`
  - `load_in_4bit=true`
  - `full_finetuning=false` (LoRA `r=8`, `alpha=16`, `dropout=0.05`)
  - `max_gpu_memory_gib_guard=103.8`
  - `rows_truncated=2/1109`, `max_original_tokens=33408` (thinking-preserved) / `33269` (no-thinking)
- Outcome:
  - Full epoch completed (`278` steps) under that recipe.

## `Qwen3.5-9B` (current workspace)

- Baseline failing full-FT config:
  - Session: `20260305_165312_qwen35_9b_instruct_full1109_quality_v2`
  - `max_length=32768`, `optim=adamw_8bit`, `load_in_4bit=false`, `full_finetuning=true`
  - `rows_truncated=6/1109`, `max_original_tokens=36458`
  - Failed around late steps (`~8-9`) with OOM behavior in repeated 32K probes.

## Why This Produces Different Memory Curves

1. `120B QLoRA` stores many weights (quantized), but trains only adapters.
2. `9B Full-FT` trains all parameters, so gradients + optimizer states become large fixed overhead.
3. Long-context dynamic memory spikes (activations/workspaces) stack on top of that static overhead.
4. Tokenization/truncation tails differ by model tokenizer and dataset formatting (`120B` archive max ~33.4K vs `Qwen` max ~36.5K before truncation).

## New Result That Changes the Picture

Optimizer A/B on `Qwen3.5-9B` full-FT at `16K` (same no-save setup):

- `adamw_8bit`: peak `114449 MiB`
- `paged_adamw_8bit`: peak `87777 MiB`
- Delta: `-26672 MiB` (`-26.05 GiB`, `-23.3%`)

Current `32K` paged-optimizer result:

- Session: `20260306_053909_qwen35_9b_instruct_memprobe_32k_nosave_pagedadamw_v1`
- Completed `10/10` with observed peak `96903 MiB` and no OOM.

This indicates the original 32K blocker was heavily tied to optimizer-state placement, not only sequence length.

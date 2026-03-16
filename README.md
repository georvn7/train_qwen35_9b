# Train Qwen3.5 9B (Full FT)

This repo contains the reproducible training/serving workflow for the full-weight Qwen3.5-9B run.

## Start Here

- Repro runbook: `qwen35_9b_fullft/docs/REPRODUCE_FULLFT_20260307.md`
- Git vs external artifacts policy: `qwen35_9b_fullft/docs/REPO_KEEP_AND_DRIVE_POLICY.md`
- Main project README: `qwen35_9b_fullft/README.md`

## Canonical Repro Target

- Run ID: `20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1`
- Model: `Qwen/Qwen3.5-9B`
- Context: `32768`, truncation side `left`
- Training mode: full fine-tuning (no LoRA / no 4-bit)

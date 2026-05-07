# Round 3 Rare-Actions Archive

Date: `2026-05-06`

Run id:

- `20260506_045018_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1`

Session dir:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260506_045018_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1`

## Archive Externally

Archive these outside Git, for example Google Drive or Hugging Face:

### Required For Inference / Sharing

Archive the complete final full-model directory:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260506_045018_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1/artifacts/full_model`

Observed size:

- `17G`

This directory contains the deployable bf16 model:

- `config.json`
- `generation_config.json`
- `chat_template.jinja`
- tokenizer files
- `model.safetensors.index.json`
- `model-00001-of-00032.safetensors` through `model-00032-of-00032.safetensors`

### Required To Resume Training

Archive the final checkpoint directory:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260506_045018_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1/checkpoints/checkpoint-841`

Observed size:

- `40G`

This is larger than `artifacts/full_model` because it includes trainer state needed for continuation/resume.

### Required For Reproducibility

Archive the lightweight metadata directory:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260506_045018_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1/metadata`

Archive the launcher log:

- `/home/georvn/train_qwen35_9b/logs/train_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1.log`

Archive the training dataset used for this stage:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/super-debug-v2-rare-actions-no-assistant-thinking.jsonl`

Optional paired dataset, useful for public dataset documentation:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/super-debug-v2-rare-actions-with-assistant-thinking.jsonl`

## Optional Backup

If storage is cheap, also archive the previous retained checkpoint as an emergency rollback point:

- `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260506_045018_qwen35_9b_rare_actions_sft_from_round2_dpo_841_32k_v1/checkpoints/checkpoint-800`

This is not required if `checkpoint-841` is verified, but it protects against a later discovered final-checkpoint issue.

## Git Policy

Do not commit model weights, optimizer state, checkpoints, or raw JSONL datasets to Git.

Git should keep:

- launcher scripts
- run documentation
- repro snapshots under `qwen35_9b_fullft/docs/repro/`
- metadata summaries

External storage should keep:

- full model
- final checkpoint
- raw datasets
- large logs if desired

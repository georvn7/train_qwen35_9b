# External Artifacts Manifest Template

Use this file format to track large files stored outside Git (Google Drive / HF).

## Required Columns

- `run_id`
- `artifact_type`
- `local_relative_path`
- `external_url`
- `sha256`
- `size_bytes`
- `created_at_utc`
- `notes`

## Example Rows

| run_id | artifact_type | local_relative_path | external_url | sha256 | size_bytes | created_at_utc | notes |
|---|---|---|---|---|---:|---|---|
| 20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1 | full_model | qwen35_9b_fullft/runs/.../artifacts/full_model | https://drive.google.com/... | `<sha256>` | `<bytes>` | 2026-03-07T23:07:24Z | canonical publish |
| 20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1 | checkpoint | qwen35_9b_fullft/runs/.../checkpoints/checkpoint-1109 | https://drive.google.com/... | `<sha256>` | `<bytes>` | 2026-03-07T23:07:24Z | final ckpt |

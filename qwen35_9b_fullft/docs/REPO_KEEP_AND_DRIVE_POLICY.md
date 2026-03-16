# Repo Keep vs External Artifact Policy

This policy is optimized for collaboration: small, searchable repo + reproducible training + large artifacts offloaded (Google Drive/HF).

## Keep In Git

- `qwen35_9b_fullft/scripts/`
- `qwen35_9b_fullft/docs/`
- `qwen35_9b_fullft/evals/`
- `qwen35_9b_fullft/reports/`
- `qwen35_9b_fullft/requirements.txt`
- `qwen35_9b_fullft/requirements.lock.txt`
- `qwen35_9b_fullft/docs/repro/*.json`
- repo root `README.md` and `.gitignore`

## Store Externally (Google Drive or HF)

- `qwen35_9b_fullft/runs/**/checkpoints/**`
- `qwen35_9b_fullft/runs/**/artifacts/**`
- `qwen35_9b_fullft/.cache/**`
- local runtime logs under `logs/`
- local venv/toolchain dirs (`.venv/`, `.local_py312dev/`, etc.)

## Minimum Metadata To Preserve For Repro

For each published run, keep these JSON files in repo:

1. `run_config.json`
2. `environment.json`
3. `dataset_manifest.json`
4. `train_metrics.json`

Current canonical set is committed under:

- `qwen35_9b_fullft/docs/repro/`

## Recommended External Manifest (manual)

For each external artifact folder, keep a manifest entry:

- `run_id`
- `artifact_type` (`checkpoint`, `full_model`, `vllm_compat_model`)
- `external_url` (Drive/HF)
- `sha256` (folder archive or per-file checksums)
- `created_at_utc`

This keeps collaboration deterministic without storing large binaries in Git.

Template:

- `qwen35_9b_fullft/docs/EXTERNAL_ARTIFACTS_MANIFEST_TEMPLATE.md`

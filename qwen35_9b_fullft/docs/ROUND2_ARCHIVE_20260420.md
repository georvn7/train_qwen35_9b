# Round 2 Archive Manifest

Date: `2026-04-20`
Run id: `20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1`

## Purpose

Preserve the finished round-2 continuation run without committing large binaries to Git.

Git keeps:

- launch scripts
- dataset prep scripts
- docs
- small dataset metadata
- extracted run snapshots under `docs/repro/`

External storage should keep:

- final full model
- final checkpoint
- merged SFT dataset
- merged DPO dataset

## Git-Tracked Repro Snapshot Set

These files are now stored in-repo and are the canonical small archive for humans/agents:

- `qwen35_9b_fullft/docs/repro/run_config_qwen35_9b_round2_cont_sft_1869_32k_v1.json`
- `qwen35_9b_fullft/docs/repro/environment_qwen35_9b_round2_cont_sft_1869_32k_v1.json`
- `qwen35_9b_fullft/docs/repro/dataset_manifest_qwen35_9b_round2_cont_sft_1869_32k_v1.json`
- `qwen35_9b_fullft/docs/repro/train_metrics_qwen35_9b_round2_cont_sft_1869_32k_v1.json`
- `qwen35_9b_fullft/docs/repro/session_qwen35_9b_round2_cont_sft_1869_32k_v1.json`
- `qwen35_9b_fullft/docs/repro/truncation_stats_qwen35_9b_round2_cont_sft_1869_32k_v1.json`
- `qwen35_9b_fullft/docs/repro/train_log_history_qwen35_9b_round2_cont_sft_1869_32k_v1.json`
- `qwen35_9b_fullft/docs/repro/export_report_qwen35_9b_round2_cont_sft_1869_32k_v1.json`
- `qwen35_9b_fullft/docs/repro/log_qwen35_9b_round2_cont_sft_1869_32k_v1.txt`

## External Artifacts To Preserve

`external_url` is intentionally left as `PENDING_UPLOAD` until the files are copied to Drive/HF.

For directory artifacts, `sha256` below is the hash of the sorted per-file `sha256sum` manifest for that directory.

| artifact_type | local_relative_path | external_url | sha256 | size_bytes | notes |
|---|---|---|---|---:|---|
| full_model | `qwen35_9b_fullft/runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1/artifacts/full_model` | `PENDING_UPLOAD` | `8277756950ccf7cb98f21ebd5f0ed0879462b818d1d4ee20061e480888cce4ee` | 17927710966 | canonical served model |
| checkpoint | `qwen35_9b_fullft/runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1/checkpoints/checkpoint-1869` | `PENDING_UPLOAD` | `de8d9ab1c23a3e4595d917019cdb43f83b9507fbbec1d3e7daf5cab8b1a4847b` | 42187543609 | final resumable checkpoint |
| sft_dataset | `qwen35_9b_fullft/data/all_1869_rows_no_assistant_thinking_round2.jsonl` | `PENDING_UPLOAD` | `b0d3c21f330d8a73f3cf96487508efe4f2710360acf397bc085eb3e70b3ca018` | 82237149 | merged round-2 SFT dataset |
| dpo_dataset | `qwen35_9b_fullft/data/all_706_rows_dbg_dpo_round2.jsonl` | `PENDING_UPLOAD` | `82b945af6c86a51bc56eccf1a7035d2cd7d53ea4dd097f8525801f8a35039b4e` | 16571378 | merged round-2 DPO dataset |
| live_log | `logs/qwen35_9b_round2_cont_sft_1869_32k_v1.live.out` | `PENDING_UPLOAD` | `817c9478fb5a75e4f37e0de0a7ab25ca38364aeea256cf39dc2a8f3fffdf667d` | 593293 | raw launcher/training log |

## Keep / Drop Guidance

Must keep:

- final full model
- final checkpoint `1869`
- merged SFT dataset
- merged DPO dataset
- checked-in repro snapshots

Safe to defer or remove after verified external copy:

- intermediate checkpoints (`checkpoint-1650`, `1700`, `1750`, `1800`, `1850`)
- old live shell state

Conservative rule:

- do not delete the final checkpoint or final full model until the external copy is verified against the recorded hash.

# Round 2 DPO Archive Manifest

Date: `2026-04-22`
Run id: `20260421_032308_qwen35_9b_round2_dpo_702_clean_16k_v1`

## Purpose

Preserve the finished round-2 DPO run without committing large binaries to Git.

Git keeps:

- launch scripts
- training docs
- archive manifest
- small repro snapshots under `docs/repro/`
- benchmark reports under `reports/`

External storage should keep:

- final DPO full model
- final DPO checkpoint
- canonical cleaned DPO dataset
- canonical prepared DPO dataset
- full live launcher log

## Git-Tracked Repro Snapshot Set

These files are now stored in-repo and are the canonical small archive for humans/agents:

- `qwen35_9b_fullft/docs/repro/run_config_qwen35_9b_round2_dpo_702_clean_16k_v1.json`
- `qwen35_9b_fullft/docs/repro/environment_qwen35_9b_round2_dpo_702_clean_16k_v1.json`
- `qwen35_9b_fullft/docs/repro/dataset_manifest_qwen35_9b_round2_dpo_702_clean_16k_v1.json`
- `qwen35_9b_fullft/docs/repro/dpo_tokenization_stats_qwen35_9b_round2_dpo_702_clean_16k_v1.json`
- `qwen35_9b_fullft/docs/repro/train_metrics_qwen35_9b_round2_dpo_702_clean_16k_v1.json`
- `qwen35_9b_fullft/docs/repro/session_qwen35_9b_round2_dpo_702_clean_16k_v1.json`
- `qwen35_9b_fullft/docs/repro/train_log_history_qwen35_9b_round2_dpo_702_clean_16k_v1.json`
- `qwen35_9b_fullft/reports/schema20_local_hf_20260423_024134.json`
- `qwen35_9b_fullft/reports/schema20_local_hf_20260423_024134.md`

## External Artifacts To Preserve

`external_url` is intentionally left as `PENDING_UPLOAD` until the files are copied to Drive/HF.

For directory artifacts, `sha256` below is the hash of the sorted per-file `sha256sum` manifest for that directory.

| artifact_type | local_relative_path | external_url | sha256 | size_bytes | notes |
|---|---|---|---|---:|---|
| full_model | `qwen35_9b_fullft/runs/20260421_032308_qwen35_9b_round2_dpo_702_clean_16k_v1/artifacts/full_model` | `PENDING_UPLOAD` | `535c603559c3bbb3d9a33a0260370f5d1e34d009ff8b736d071b1e0486cb8526` | 17927711478 | canonical DPO served model |
| checkpoint | `qwen35_9b_fullft/runs/20260421_032308_qwen35_9b_round2_dpo_702_clean_16k_v1/checkpoints/checkpoint-702` | `PENDING_UPLOAD` | `ed529e43ecaeb5b9e5983a223860a39a5931b9a8519104177505875e8e1ace3b` | 42187554694 | final resumable DPO checkpoint |
| dpo_dataset_clean | `qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean.jsonl` | `PENDING_UPLOAD` | `a8637d64c5c29c3edb17d30dae8951cfd75004a2a23621b8c2a03344c8eb9e22` | 16563120 | canonical cleaned DPO source set |
| dpo_dataset_prepared | `qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean_prepared.jsonl` | `PENDING_UPLOAD` | `a1cffa42f265bec3ed3ab10c3d4b7f667c30cf993c88ade5e362a2b55503dc3c` | 16563141 | canonical prepared prompt/chosen/rejected view |
| live_log | `logs/train_qwen35_9b_round2_dpo_resume_safe.log` | `PENDING_UPLOAD` | `e7c531a6060fb8332c6d5a07a80b32df26914041333e04395608fbbe35d81823` | 1405815 | full launcher and retry history |

## Evaluation Snapshot

Post-DPO local-HF schema20 report:

- report json:
  - `qwen35_9b_fullft/reports/schema20_local_hf_20260423_024134.json`
  - `sha256 = 08473e688d9e8b3b94e865c8f6e9c5329abbd29c1178ea2537ed24543c6b4837`
  - `size_bytes = 15049`
- report md:
  - `qwen35_9b_fullft/reports/schema20_local_hf_20260423_024134.md`
  - `sha256 = 95c8d6feb222706e19114ec97e2790b497b1eeea322d81aa16d3e1616a402f23`
  - `size_bytes = 2249`
- result summary:
  - `avg_structured_score = 0.7725`
  - previous round-2 SFT reference:
    - `qwen35_9b_fullft/reports/schema20_local_hf_20260420_235146.json`
    - `avg_structured_score = 0.7675`
  - observed delta:
    - `+0.0050`

## Keep / Drop Guidance

Must keep:

- final DPO full model
- final DPO checkpoint `702`
- cleaned and prepared DPO datasets
- checked-in repro snapshots
- schema20 evaluation report generated against this DPO model

Safe to defer or remove after verified external copy:

- intermediate DPO checkpoints (`checkpoint-450`, `500`, `550`, `600`, `650`, `700`)
- stale attempt-only files under `metadata/` that are superseded by final `status=trained`

Conservative rule:

- do not delete the final checkpoint or final full model until the external copy is verified against the recorded hash.

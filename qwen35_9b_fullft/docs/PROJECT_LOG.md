# Qwen3.5 9B Full FT Project Log

This is the running project log for decisions, technical changes, and discussion outcomes.

## Logging Rules

- Add new items at the top of each section.
- Record exact dates in `YYYY-MM-DD` format.
- Include concrete file paths and session ids for reproducibility.

## Objective

Train `Qwen3.5 9B` with full-weight fine-tuning using the same session/manifest/dataset conventions used in the prior `gpt-oss-120b` workflow.

## Fixed Dataset Convention

- Public HF dataset:
  - `https://huggingface.co/datasets/georvn7/super-debug-v1`
  - Note: this HF dataset contains assistant `thinking` fields.
- Source dataset selected by user:
  - `/home/georvn/final_archive-20260305T073201Z-3-001/final_archive/datasets_views/all_1109_rows_no_assistant_thinking.jsonl`
- Local copied dataset used for training:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/data/all_1109_rows_no_assistant_thinking.jsonl`
  - Note: this is the no-thinking training variant used for full-FT.

## Key Decisions

- `2026-04-22`: Post-DPO local-HF schema20 evaluation on `runs/20260421_032308_qwen35_9b_round2_dpo_702_clean_16k_v1/artifacts/full_model` scored `avg_structured_score=0.7725`, slightly above the pre-DPO round-2 SFT reference `0.7675`.
- `2026-04-22`: Round-2 DPO run `20260421_032308_qwen35_9b_round2_dpo_702_clean_16k_v1` completed successfully with final checkpoint `checkpoint-702`, exported model under `artifacts/full_model`, and final `train_loss=0.09880231095854713`.
- `2026-04-22`: Round-2 DPO archive policy fixed: preserve external copies of final DPO full model, final DPO checkpoint, canonical cleaned DPO dataset, canonical prepared DPO dataset, and full launcher log; preserve small DPO repro snapshots in Git under `docs/repro/`.
- `2026-04-22`: Added dedicated DPO bf16 serving wrappers `scripts/start_vllm_round2_dpo_bf16_openai.sh` and `scripts/stop_vllm_round2_dpo_bf16_openai.sh` so manual probing points at the DPO `full_model` by default.
- `2026-04-21`: Round-2 DPO resume hardening v2: after full precompute and `checkpoint-10`, the remaining blocker was host-RAM pressure during exact resume of the `~23 GiB` `optimizer.pt`. The launcher now uses a two-phase host-RAM guard (`resume_min_mem_avail_mib=1024`, steady `min_mem_avail_mib=3072` by default), writes a `resume_warm_marker.json` after the first resumed step, and densifies early durable checkpoints to `10,20,30,40` before the regular `save_steps=50`.
- `2026-04-21`: Round-2 DPO resume path now applies mmap-backed `torch.load` for files under the resume checkpoint tree, with automatic fallback to non-mmap if unsupported. This mirrors the earlier SFT resume hardening and is intended to reduce CPU-memory duplication during checkpoint restore.
- `2026-04-21`: Round-2 DPO checkpoint-save hardening v1: save-phase host-RAM guard is now split from steady-state (`checkpoint_save_min_mem_avail_mib=1536` by default), training writes `metadata/checkpoint_save_marker.json` before checkpoint writes, and completed checkpoints write `checkpoint_complete.json`. Launcher-side checkpoint selection now skips `.incomplete*` directories and quarantines only raw invalid `checkpoint-*` dirs.
- `2026-04-20`: Canonical round-2 DPO training set is the cleaned `702`-row variant, not the raw `706` rows. Reason: `4` source rows had empty chosen outputs and are not repairable; `3` rows had missing `breakpoints` and were normalized.
- `2026-04-20`: Round-2 DPO policy is fixed to continue from the finished round-2 full model `runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1/artifacts/full_model`; do not start DPO from base or from scratch.
- `2026-04-20`: Round-2 DPO recipe is pinned conservatively for Spark: `DPOTrainer`, `bf16`, `adamw_8bit`, `lr=1e-6`, `beta=0.05`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=1`, `precompute_ref_log_probs=true`, `use_logits_to_keep=true`.
- `2026-04-20`: Round-2 DPO length budget fixed to `max_prompt_length=14848`, `max_completion_length=1536`, `max_length=16384`, `truncation_mode=keep_end`.
- `2026-04-20`: Canonical round-2 DPO input is a prepared chat-templated view built from `all_706_rows_dbg_dpo_round2.jsonl`; this freezes the prompt/chosen/rejected string rendering used by training.
- `2026-04-20`: Round-2 continuation run `20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1` completed successfully at `1869/1869` with final `train_loss=0.23181603004614768`, final checkpoint `checkpoint-1869`, and exported full model under `artifacts/full_model`.
- `2026-04-20`: Round-2 archive policy fixed: preserve external copies of final full model, final checkpoint, merged round-2 SFT dataset, and merged round-2 DPO dataset; preserve small repro snapshots and logs in Git under `docs/repro/`.
- `2026-04-20`: Round-2 checked-in reproducibility snapshot set added for the finished run: run config, environment, dataset manifest, train metrics, session, truncation stats, train-log history, export report, and copied live log.
- `2026-04-18`: Round-2 continuation smoke `runs/20260418_215311_qwen35_9b_round2_cont_sft_1869_smoke1_fix1` proved the previous March 32K full-FT recipe still works when starting from the prior full model: one `32768` full-weight step completed and checkpoint save finished successfully.
- `2026-04-18`: Checkpoint policy for round-2 continuation is unchanged from the last stable run: `save_steps=50`, `save_total_limit=4`, safe sharded save enabled, with pre-save `gc` + `torch.cuda.empty_cache()`.
- `2026-04-18`: Round-2 continuation policy set: start from previous full-FT model `runs/20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1/artifacts/full_model`, not from `Qwen/Qwen3.5-9B`.
- `2026-04-18`: New dataset prep for round 2 is split into two canonical merged files under `qwen35_9b_fullft/data/`: one SFT continuation file built from all `train_dbg_sft.jsonl` + `train_run_sft.jsonl`, and one separate DPO file built from all `train_dbg_dpo.jsonl`.
- `2026-04-18`: For round-2 SFT, keep overlength rows and preserve the existing left-truncation rule. Reason: this dataset family puts the latest debugging state and supervised answer at the tail; dropping all `>32K` rows would discard too much new signal.
- `2026-04-18`: DPO is explicitly staged as a second-phase experiment after continuation SFT, even though the new `train_dbg_dpo.jsonl` files are already in conversational `prompt/chosen/rejected` format.
- `2026-03-15`: Documentation clarified that HF dataset `georvn7/super-debug-v1` contains thinking, while full-FT training here uses the derived no-thinking file `all_1109_rows_no_assistant_thinking.jsonl`.
- `2026-03-15`: Public-collaboration reproducibility policy set: keep code/docs/evals/snapshots in Git; keep model/checkpoint binaries and caches in external storage (Drive/HF) with manifest-style references.
- `2026-03-15`: Canonical reproducibility target for collaborators/agents is explicitly pinned to run `20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1` with checked-in run/env/dataset/metrics snapshots under `docs/repro/`.
- `2026-03-09`: Canonical inference launch profile is now `scripts/start_vllm_fullft_bf16_resident.sh` (fixed attrs: host `0.0.0.0`, port `8002`, model id `qwen35-9b-fullft-bf16`, `max_model_len=32768`, `gpu_memory_utilization=0.90`, `max_num_seqs=1`) for single-user always-on serving.
- `2026-03-09`: vLLM bf16 launcher detachment hardened with `setsid` + `disown` so serving survives non-interactive command-wrapper exit and keeps model residency stable.
- `2026-03-07`: LAN inference is served through OpenAI-compatible vLLM endpoints (`/v1/models`, `/v1/chat/completions`) with separate ports for full-FT and untouched models.
- `2026-03-07`: Serving quantization policy for deployment is BitsAndBytes 8-bit (`load_in_8bit=true`, `load_in_4bit=false`) with `enable_thinking=false` default chat kwargs.
- `2026-03-07`: Full-FT export compatibility shim added for vLLM: when checkpoint config is `qwen3_5_text`, build a config-only `*_vllm_compat` serving copy (`qwen3_5` wrapper config + preserved weights).
- `2026-03-07`: Repeated-save 32K checkpoint path is now validated on `step18` (`20260307_035417_step18_ckpt11_sdpa_chunked_shard512`): run completed `11/11` with successful saves at steps `10` and `11` (`32/32` shards each), confirming the shard+pre-save-cleanup recipe holds beyond one save event.
- `2026-03-07`: Checkpoint-save mitigation validated on 32K single-step run `step17b`: `checkpoint_max_shard_size=512MB` + pre-save cleanup completed `Writing model shards: 32/32` with no external guard trip.
- `2026-03-07`: `step17` confirmed a separate failure mode: `attn_implementation=eager` + default loss can OOM before checkpoint write (`Tried to allocate 64.00 GiB`), so checkpoint tuning must be tested on a stable training path (`sdpa` + chunked loss).
- `2026-03-07`: Resume-memory root cause isolated with phase probes: `torch.load(optimizer.pt)` host-RSS spike is removed by `mmap=True` (from ~`26 GiB` RSS to ~`2.1 GiB` at load boundary in matched `checkpoint-900` probes).
- `2026-03-07`: Resume path hardened: `train_session.py` now supports `--resume-torch-load-mmap` (default on) with safe fallback to non-mmap if mmap load is unsupported.
- `2026-03-07`: Full 32K resume from `checkpoint-900` was relaunched with unchanged training recipe and mmap-enabled resume I/O; resumed stepping beyond `900` (`901+`) without immediate startup guard trips.
- `2026-03-07`: `step16` (`save_only_model=true`) still guard-stopped at checkpoint write (`10/11`, `mem_avail_mib` floor `1846`), so model-only checkpointing alone does not solve the host-RAM save spike.
- `2026-03-07`: Started checkpoint-focused 32K retry `step16` with `save_only_model=true` and `max_steps=11` to test whether model-only checkpoint writes avoid the host-RAM guard trip seen in `step15`.
- `2026-03-06`: Long 32K proof `step15` reached `10/30` and was stopped by external host-memory guard at checkpoint save time (`mem_avail_mib=3206 <= 4096`), confirming a checkpoint-phase host RAM spike.
- `2026-03-06`: 32K stress run `step14d` validated that `cuda_memory_fraction=0.88` is sufficient even with `causal_loss_mode=default` (`gpu_peak_mib=110097`, `rc=0`).
- `2026-03-06`: Long proof session `20260306_194048_step15_proof32k30_default_adamw8_cudafrac088` ran with default loss, 32K context, and external host guard (`min_mem_avail_mib=4096`) to avoid desktop freeze.
- `2026-03-06`: Plain-language training semantics locked: `assistant_only_loss=true` means loss is computed only on assistant tokens (labels not equal to `-100`), not on user/system tokens.
- `2026-03-06`: Text-only full FT now explicitly excludes multimodal tower loading by default (`--force-causal-lm-loader`), so Qwen3.5-9B runs through `Qwen3_5ForCausalLM` (no `model.visual` module).
- `2026-03-06`: Added an external spike guard profile for desktop safety (`gpu_guard_gib=100`, `min_mem_avail_mib=8192`) plus allocator default (`expandable_segments:True,max_split_size_mb:256`) in smoke runner defaults.
- `2026-03-06`: For full-FT memory relief, optimizer strategy was switched/tested to `paged_adamw_8bit` and validated by A/B at 16K (`114449 MiB -> 87777 MiB`, `-26.05 GiB` peak).
- `2026-03-06`: 32K full-FT feasibility was re-opened after optimizer A/B; paged-optimizer probe (`20260306_053909_qwen35_9b_instruct_memprobe_32k_nosave_pagedadamw_v1`) completed (`10/10`) with no OOM and peak `96903 MiB`.
- `2026-03-06`: Added explicit comparison note for the user question "why 120B 32K worked while 9B failed" in `docs/MEMORY_DIFF_120B_VS_QWEN9B_20260306.md`.
- `2026-03-05`: Added explicit memory-root-cause note (`docs/MEMORY_ANALYSIS_20260305.md`): dynamic activation/workspace memory is dominant beyond static model+optimizer footprint.
- `2026-03-05`: Production full run launched as session `20260305_184211_qwen35_9b_instruct_full1109_quality_v3_12k_guard110` using the validated Spark-safe recipe (`12K`, guard `110`, `unsloth` checkpointing).
- `2026-03-05`: On DGX Spark, full-weight instruct runs at `32K/24K/16K` consistently hit the same late-step memory cliff; production recipe is pinned to `max_seq_length=12288` with `max_gpu_memory_gib=110`.
- `2026-03-05`: Per user direction, do not continue full training on `Qwen/Qwen3.5-9B-Base`; production full run switched to `Qwen/Qwen3.5-9B`.
- `2026-03-05`: Instruct full-run optimizer policy is conservative to preserve post-training behavior: `learning_rate=1e-5`, `warmup_steps=50`.
- `2026-03-05`: Reuse the exact schema20 evaluation criteria from the 120B workflow (action_type/action_subject-weighted structured scoring + strict A/B gate).
- `2026-03-05`: Full training launch for the current recipe is attached to a live shell session (`exec` session id) in this environment, since detached background children are reaped by the tooling runtime.
- `2026-03-05`: Preferred base checkpoint is `Qwen/Qwen3.5-9B-Base` for this dataset (much lower initial loss and saner grad norms than `Qwen/Qwen3.5-9B` in matched probes).
- `2026-03-05`: Learning rate target for full run is `2e-5` (conservative pick; `3e-5` was nearly tied in short probe).
- `2026-03-05`: Use two-stage tuning on single DGX Spark: fast short train-loss probes first, then a 32K long-subset confirmation run.
- `2026-03-05`: With available disk headroom (~2TB), checkpoint policy tightened to `save_steps=50`, `save_total_limit=4`.
- `2026-03-05`: Default inference engine for this project is `vLLM` (see `docs/INFERENCE_ENGINE_RECOMMENDATION.md`).
- `2026-03-05`: Keep session and manifest conventions from the previous project. Do not redesign experiment bookkeeping.
- `2026-03-05`: Full-weight fine-tuning is the default training mode in this stack (`--full-finetuning` enabled, `--load-in-4bit` disabled).
- `2026-03-05`: Use left truncation for over-length samples at 32K context (drop tokens from the start of the sample).
- `2026-03-05`: Stable smoke validation target is one training step first, then scale.
- `2026-03-05`: Not required to use the exact same environment/stack as `gpt-oss-120b`; prioritize a stable recipe for Qwen3.5 9B.

## Technical Changes Implemented

- `scripts/clean_round2_dpo_dataset.py`
  - Added canonical DPO cleaning step:
    - drops rows with unusable chosen/rejected action JSON
    - normalizes missing `breakpoints` to `[]`
    - writes a manifest-style cleanup report beside the output JSONL
- Local venv compatibility fix (not committed to repo code)
  - Patched `/home/georvn/train_qwen35_9b/.venv/lib/python3.12/site-packages/trl/trainer/judges.py`.
  - Moved `llm_blender` import inside `PairRMJudge.__init__` so `DPOTrainer` can import without the judge-only extra installed.
- Local venv compatibility fix (not committed to repo code)
  - Patched `/home/georvn/train_qwen35_9b/.venv/lib/python3.12/site-packages/trl/trainer/callbacks.py`.
  - Moved `weave` and `mergekit` imports off the module import path and into the specific callback codepaths that actually use them.
  - Reason: local `trl==0.24.0` eagerly imported optional extras and blocked `DPOTrainer` import for unrelated training flows.
- `scripts/prepare_round2_dpo_dataset_view.py`
  - Added canonical round-2 DPO prepared-view builder that freezes the Qwen chat template into plain `prompt/chosen/rejected` strings and writes explicit token-budget stats.
- `scripts/train_dpo_session.py`
  - Added manifest-driven DPO trainer entrypoint with:
    - explicit local tokenization to `prompt_input_ids/chosen_input_ids/rejected_input_ids`
    - `DPOTrainer` on top of the latest round-2 full model
    - checkpoint save cleanup and internal GPU memory guard
    - `run_config.json`, `environment.json`, `dpo_tokenization_stats.json`, `train_metrics.json`, and `train_log_history.json` outputs
- `scripts/run_train_qwen35_9b_round2_dpo_from_last_fullft_safe.sh`
  - Added resume-safe DPO launcher with the same external guard and checkpoint retry conventions used for the SFT run family.
- `docs/ROUND2_DPO_PLAN_20260420.md`
  - Added the round-2 DPO design note: model lineage, length budget, optimizer/precision choices, environment patches, and launch order.
- Local venv compatibility fix (not committed to repo code)
  - Patched `/home/georvn/train_qwen35_9b/.venv/lib/python3.12/site-packages/unsloth/models/_utils.py`.
  - Added `from transformers.utils import auto_docstring` so `Unsloth 2026.3.3` can re-exec current `transformers` Qwen config classes without startup failure.
  - This fixes `NameError: name 'auto_docstring' is not defined`.
  - This is an environment patch only; it is not a tracked code change in this repository.
- `scripts/prepare_round2_continuation_datasets.py`
  - Added round-2 dataset merger that builds canonical continuation SFT and DPO files from all new dataset bundles under `/home/georvn/new_datasets`.
- `scripts/run_train_qwen35_9b_round2_from_last_fullft_safe.sh`
  - Added round-2 continuation launcher that starts from the previous full-FT model and keeps the March 32K Spark-safe recipe and checkpoint policy.
- `docs/ROUND2_ARCHIVE_20260420.md`
  - Added concrete round-2 archive manifest with exact local paths, sizes, checksums, and keep/drop guidance.
- `docs/repro/`
  - Added checked-in round-2 reproducibility snapshots and copied live training log for the finished run.
- `.gitignore`
  - Added repository-wide ignore policy to prevent accidental commits of local env/toolchain dirs, runtime logs, HF cache, run artifacts, and large model binary formats.
- `requirements.lock.txt`
  - Added pinned dependency lock (derived from canonical run environment snapshot) for reproducible setup by collaborators.
- `docs/REPRODUCE_FULLFT_20260307.md`
  - Added single-file canonical runbook with exact command/env knobs, dataset checksum contract, and expected metrics.
- `docs/REPO_KEEP_AND_DRIVE_POLICY.md`
  - Added explicit keep-in-git vs external-artifact policy for collaboration at scale.
- `docs/repro/`
  - Added checked-in canonical run snapshots:
    - `run_config_qwen35_9b_instruct_full1109_32k_recipe_v1.json`
    - `environment_qwen35_9b_instruct_full1109_32k_recipe_v1.json`
    - `dataset_manifest_qwen35_9b_instruct_full1109_32k_recipe_v1.json`
    - `train_metrics_qwen35_9b_instruct_full1109_32k_recipe_v1.json`
- `README.md` (repo root) and `qwen35_9b_fullft/README.md`
  - Added prominent “start here” references to repro runbook and artifact policy for humans/agents.
- `scripts/start_vllm_fullft_bf16_resident.sh`
  - Added canonical resident serving profile wrapper for repeatable single-user always-on launches.
- `scripts/start_vllm_fullft_bf16_openai.sh`
  - Launch detachment updated to `nohup setsid ... &` + `disown` to prevent process reaping when parent wrapper exits.
- `scripts/create_session.py`
  - Default runs path set to `qwen35_9b_fullft/runs`.
  - Added `artifacts/full_model`.
  - Manifest entries now support both `relative_path` and `absolute_path`.
- `scripts/train_session.py`
  - Added checkpoint-save controls: `--checkpoint-max-shard-size`, `--checkpoint-safe-serialization`, and `--no-save-only-model`.
  - Added checkpoint pre-save housekeeping controls and callback telemetry:
    - `--checkpoint-presave-gc`
    - `--checkpoint-presave-empty-cache`
    - `--checkpoint-presave-disable-cuda-history`
    - pre/post-save memory logging (`rss`, `MemAvailable`, process GPU usage)
  - Added `disable_cuda_memory_history()` helper to release allocator-history buffers before save when enabled.
  - Full finetune defaults enabled.
  - Default recipe now aligned to instruct-first Spark quality run (`model default=Qwen/Qwen3.5-9B`, `max_seq_length=12288`, `truncation_side=left`, `learning_rate=1e-5`, `warmup_steps=50`).
  - Added advanced causal-loss modes for memory probes: `forward_chunked_fp32`, `forward_chunked_no_upcast`, `forward_active_chunked_fp32`, `forward_active_chunked_no_upcast`.
  - Added optional `cuda_memory_fraction` allocator cap hook (uses `torch.cuda.set_per_process_memory_fraction`).
  - Added `--force-causal-lm-loader` (default enabled) to force text-only `AutoModelForCausalLM` load path and skip multimodal vision-tower weights.
  - Added `--freeze-visual-modules` (default enabled) to freeze any remaining `model.visual.*` params in text-only training.
  - LoRA wrapping is conditional and skipped for full finetuning mode.
  - Added full-model save support under `artifacts/full_model`.
  - Added robust chat-template/processor fallback handling.
  - Added compatibility filtering for `trl` `SFTConfig` kwargs.
  - Default stability flags set to disable `flex_attention` and `cce`.
  - Auto GPU memory guard for Qwen3.5-9B full-FT tuned to `110 GiB`.
  - Added optional holdout-eval controls and `train_log_history.json` export.
  - Added `--skip-final-save` and configurable `--save-strategy` for fast probe runs.
  - Added optimizer/memory knobs for controlled probes: `--optim`, `--save-only-model`, `--torch-empty-cache-steps`, `--cuda-alloc-conf`.
  - Fixed optimizer wiring so `--optim` is honored in `SFTConfig` (instead of hardcoded `adamw_8bit`).
  - Fixed `torch_empty_cache_steps` compatibility by passing it only when `>0` (current TRL/Transformers rejects zero).
  - Added resume-specific checkpoint-load control: `--resume-torch-load-mmap` / `--no-resume-torch-load-mmap`.
  - Added resume-time `torch.load` hook for checkpoint files so optimizer/scheduler/rng state loads can use mmap on resume and fall back automatically if mmap fails.
- `scripts/run_pipeline.py`
  - Paths and defaults aligned to `qwen35_9b_fullft`.
  - Default model now `Qwen/Qwen3.5-9B`; default learning rate `1e-5`.
  - Default `max_seq_length` tuned to `12288` for Spark stability.
  - Supports dataset-root as a direct file path.
- `scripts/analyze_context_lengths.py`
  - Supports manifest entries with `absolute_path`.
- `scripts/run_quality_probe.py`
  - Added short-run LR probing workflow with auto summaries in `docs/quality_probe_*.json|md`.
- `scripts/run_train_qwen35_9b_full1109_resume_safe.sh`
  - Added pass-through env knobs for checkpoint memory tuning:
    - `CHECKPOINT_MAX_SHARD_SIZE`
    - `CHECKPOINT_SAFE_SERIALIZATION`
    - `CHECKPOINT_PRESAVE_GC`
    - `CHECKPOINT_PRESAVE_EMPTY_CACHE`
    - `CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY`
  - Fixed `SESSION_DIR` corruption bug by routing launcher `log()` output to `stderr`, so command substitution captures only the session path.
  - Default model switched to `Qwen/Qwen3.5-9B` (user-directed instruct run).
  - Default LR set to `1e-5`; warmup set to `50` steps.
  - Default `gradient_accumulation_steps=1`.
  - 32K defaults set for current recipe (`max_seq_length=32768`) with allocator tuning (`cuda_memory_fraction=0.88`, `cuda_alloc_conf=expandable_segments:True,max_split_size_mb:256`, `causal_loss_mode=default`).
  - Added external safety guard loop for long runs (`min_mem_avail_mib` floor and optional `external_gpu_guard_gib`) so the launcher can stop attempts before host hard-freeze.
  - Internal trainer guard remains enabled via `max_gpu_memory_gib=110`, with `dataset_num_proc=1`, `gradient_checkpointing=unsloth`.
  - Checkpoint policy updated for full-FT artifact sizes (`save_steps=50`, `save_total_limit=4`).
  - Added explicit resume-load mode passthrough (`RESUME_TORCH_LOAD_MMAP`, default enabled) so resume behavior is deterministic in launcher-managed runs.
- `scripts/make_vllm_compat_fullft_model.py`
  - Added config-only converter for full-FT Qwen3.5 exports so vLLM can load text-trained checkpoints under `qwen3_5` wrapper config.
- `scripts/start_vllm_fullft_int8_openai.sh`
  - Added OpenAI-compatible vLLM launcher for full-FT serving (`/v1/chat/completions`).
  - Adds BitsAndBytes 8-bit loading via `hf_overrides` (`load_in_8bit=true`).
  - Auto-builds and uses `*_vllm_compat` model dir when source checkpoint is `qwen3_5_text`.
- `scripts/start_vllm_untouched_int8_openai.sh`
  - Added OpenAI-compatible vLLM launcher for untouched `Qwen/Qwen3.5-9B` serving on separate port.
- `scripts/stop_vllm_fullft_openai.sh`, `scripts/stop_vllm_untouched_openai.sh`
  - Added explicit PID-based shutdown helpers for both serving processes.
- `scripts/watch_resume_full1109_daemon.sh`
  - Added a dedicated 32K full-FT watchdog daemon for the `full1109` run family.
  - Watches the active trainer heartbeat (`train_session.py --session-dir ...`) and relaunches `run_train_qwen35_9b_full1109_resume_safe.sh` only if training exits before completion.
  - Uses `session_qwen35_9b_full1109_32k_v1.txt` as the default session pointer to avoid accidental fallback to legacy 12K session state.
  - Passes the same production memory-safety knobs on relaunch (`MIN_MEM_AVAIL_MIB`, causal-loss chunking mode, checkpoint presave cleanup, resume mmap).
- `scripts/run_container_smoke_ab.sh`
  - Added checkpoint tuning env knobs and flags:
    - `CHECKPOINT_MAX_SHARD_SIZE`
    - `CHECKPOINT_SAFE_SERIALIZATION`
    - `CHECKPOINT_PRESAVE_GC`
    - `CHECKPOINT_PRESAVE_EMPTY_CACHE`
    - `CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY`
  - Added debug toggle knobs so memory-history capture is opt-in for checkpoint tests:
    - `ENABLE_CUDA_DEBUG_HISTORY`
    - `DEBUG_CUDA_HISTORY_MAX_ENTRIES`
    - `DEBUG_CUDA_SNAPSHOT_ON_ERROR`
  - Added desktop-safety guard defaults: `gpu_guard_gib=100`, `min_mem_avail_mib=8192`.
  - Added optional internal guard and allocator args: `internal_gpu_guard_gib`, `cuda_alloc_conf`.
  - Added optional `save_only_model` pass-through for checkpoint stress testing.
  - Default smoke runner allocator now uses `expandable_segments:True,max_split_size_mb:256`.
  - Explicitly uses `--force-causal-lm-loader` and `--freeze-visual-modules` for text-only 32K stress runs.
- `scripts/benchmark_ollama_chat.py`
  - Imported schema20 scoring evaluator from the 120B workflow to keep exact structured-output metrics.
- `scripts/compare_schema20_ab.py`
  - Added strict A/B comparator producing `compare.json` + markdown summary with promote/no-promote gate.
- `scripts/run_ab_schema20_ollama.sh`
  - Added one-command baseline-vs-candidate schema20 run wrapper for Ollama `/v1/chat/completions`.
- `evals/agent_cases_20_schema_final_v1.json`
  - Imported fixed 20-case benchmark set used in prior 120B gating.

## Model Choice Notes

- Current smoke tests were executed with `Qwen/Qwen3.5-9B-Base` to establish stable end-to-end full FT behavior.
- Additional direct probe on `Qwen/Qwen3.5-9B` showed higher initial loss and very high grad norm on matched settings; keep `Qwen/Qwen3.5-9B-Base` as the known-good baseline.
- Current production training direction is user-selected instruct model (`Qwen/Qwen3.5-9B`) with reduced LR for retention.

## Open Items

- Re-run longer 32K checkpoint cadence proof (for example `max_steps>=11`, `save_steps=10`) with the new shard + pre-save cleanup settings to confirm repeated-save stability.
- Add an interactive safety profile for desktop use (`max_gpu_memory_gib` around `100-105`) to reduce UI freeze risk during late-step spikes.
- Add periodic holdout-eval checkpoints during long run (sparse cadence) once first full epoch finishes.

## 2026-03-08 Inference Findings

- Local schema20 HF evaluation is highly sensitive to thinking mode.
- With thinking-like behavior on, structured benchmark score collapsed (`avg_structured_score=0.085` on full 20-case run).
- With `enable_thinking=False` and same scorer/settings:
  - full-FT candidate (`artifacts/full_model`): `0.7175`
  - untouched `Qwen/Qwen3.5-9B` instruct baseline: `0.6425`
  - net gain from full-FT: `+0.075` absolute (`~+11.7%` relative).

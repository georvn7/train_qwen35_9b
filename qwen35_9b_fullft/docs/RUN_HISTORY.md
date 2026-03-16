# Run History

Chronological record of meaningful validation and smoke runs.

## 2026-03-06

### Session group: `step14*` 32K stress (memory-cap diagnostics)

- Purpose:
  - Validate whether CUDA allocator capping can stabilize the same 32K full-FT stress rows that previously hit hard guard/kill.
- Common setup:
  - Dataset: `stress_top6_over32k_rows.jsonl` (`6/6` rows truncated to 32768)
  - `model=Qwen/Qwen3.5-9B`
  - `optim=adamw_8bit`
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=1`
  - `max_steps=1`
  - `cuda_alloc_conf=expandable_segments:True,max_split_size_mb:256`

#### Session: `20260306_190922_step14a_forwardactivechunkednoupcast_adamw8_guard110_mem4g` (failed)

- Result:
  - Guarded run failed (`rc=137`, `session_status=created`).
- Evidence:
  - `gpu_peak_mib=114906`
  - `guard_triggered=yes`
  - `guard_reason=gpu_used_mib=114906 >= gpu_guard_mib=112640`

#### Session: `20260306_191210_step14b_forwardactivechunkednoupcast_adamw8_cudafrac088_diag` (completed)

- Additional config:
  - `cuda_memory_fraction=0.88`
  - `gpu_guard_gib=120`, `internal_gpu_guard_gib=0`
  - `causal_loss_mode=forward_active_chunked_no_upcast`
- Result:
  - Completed (`rc=0`, `session_status=trained`).
- Metrics:
  - `gpu_peak_mib=110311`
  - `train_runtime=309.36s`
  - `train_loss=2.8750`

#### Session: `20260306_191944_step14c_activechunkednoupcast_adamw8_cudafrac088_diag` (completed)

- Additional config:
  - `cuda_memory_fraction=0.88`
  - `gpu_guard_gib=120`, `internal_gpu_guard_gib=0`
  - `causal_loss_mode=active_chunked_no_upcast`
- Result:
  - Completed (`rc=0`, `session_status=trained`).
- Metrics:
  - `gpu_peak_mib=110223`
  - `train_runtime=330.28s`
  - `train_loss=2.8594`

#### Session: `20260306_192835_step14d_default_adamw8_cudafrac088_diag` (completed)

- Additional config:
  - `cuda_memory_fraction=0.88`
  - `gpu_guard_gib=120`, `internal_gpu_guard_gib=0`
  - `causal_loss_mode=default`
- Result:
  - Completed (`rc=0`, `session_status=trained`).
- Metrics:
  - `gpu_peak_mib=110097`
  - `train_runtime=309.46s`
  - `train_loss=2.8577`
  - `guard_triggered=no`

### Session: `20260306_194048_step15_proof32k30_default_adamw8_cudafrac088` (guard-stopped)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260306_194048_step15_proof32k30_default_adamw8_cudafrac088`
- Purpose:
  - Longer 32K stress proof (`30` steps) before launching full 1109-row 32K production run.
- Core config:
  - `model=Qwen/Qwen3.5-9B`
  - `max_seq_length=32768`
  - `truncation_side=left`
  - `optim=adamw_8bit`
  - `max_steps=30`
  - `save_strategy=steps`, `save_steps=10`
  - `cuda_memory_fraction=0.88`
  - `gpu_guard_gib=120`, `min_mem_avail_mib=4096`
  - `causal_loss_mode=default`
- Result:
  - Reached `10/30` steps, then stopped by external host-memory guard (`rc=137`, `session_status=created`).
- Metrics / evidence:
  - `gpu_peak_mib=110453`
  - `guard_triggered=yes`
  - `guard_reason=mem_avail_mib=3206 <= min_mem_avail_mib=4096`
  - Last completed progress line: `10/30 [40:19 ...]`
  - A partial `checkpoint-10` directory exists; save was interrupted during `Writing model shards`.

### Session: `20260307_024450_step16_ckpt11_saveonlymodel_default_adamw8_cudafrac088` (guard-stopped)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260307_024450_step16_ckpt11_saveonlymodel_default_adamw8_cudafrac088`
- Purpose:
  - Checkpoint-focused 32K retry with model-only checkpoint writes to reduce host-RAM pressure at save time.
- Core config:
  - `model=Qwen/Qwen3.5-9B`
  - `max_seq_length=32768`
  - `optim=adamw_8bit`
  - `max_steps=11`
  - `save_strategy=steps`, `save_steps=10`
  - `save_only_model=true`
  - `cuda_memory_fraction=0.88`
  - external guards: `gpu_guard_gib=120`, `min_mem_avail_mib=4096`
- Result:
  - Reached `10/11` steps, then stopped by external host-memory guard during checkpoint save (`rc=137`, `session_status=created`).
- Metrics / evidence:
  - `gpu_peak_mib=110451`
  - `guard_triggered=yes`
  - `guard_reason=mem_avail_mib=1846 <= min_mem_avail_mib=4096`
  - Progress markers present through `10/11`.
  - `save_only_model=true` reduced partial checkpoint file size (about `2.03 GB`), but did not prevent host-memory collapse during `Writing model shards`.

### Session: `20260307_034033_step17_ckpt1_32k_shard512_nodbg` (failed before checkpoint save)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260307_034033_step17_ckpt1_32k_shard512_nodbg`
- Purpose:
  - First checkpoint-tuning probe with explicit shard control and pre-save cleanup enabled.
- Core config:
  - `max_seq_length=32768`
  - `max_steps=1`, `save_strategy=steps`, `save_steps=1`
  - `save_only_model=true`
  - `checkpoint_max_shard_size=512MB`
  - `checkpoint_safe_serialization=true`
  - `checkpoint_presave_gc=true`
  - `checkpoint_presave_empty_cache=true`
  - `checkpoint_presave_disable_cuda_history=true`
  - `enable_cuda_debug_history=false`
  - `attn_implementation=eager`, `causal_loss_mode=default`
- Result:
  - Failed before checkpoint save with CUDA OOM in attention (`rc=1`, `session_status=failed`).
- Metrics / evidence:
  - `gpu_peak_mib=87664`
  - Error: `torch.OutOfMemoryError ... Tried to allocate 64.00 GiB`

### Session: `20260307_034334_step17b_ckpt1_32k_sdpa_chunked_shard512` (completed)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260307_034334_step17b_ckpt1_32k_sdpa_chunked_shard512`
- Purpose:
  - Validate checkpoint-write memory mitigation with stable 32K training path and forced shard splitting.
- Core config:
  - `max_seq_length=32768`
  - `max_steps=1`, `save_strategy=steps`, `save_steps=1`
  - `save_only_model=true`
  - `checkpoint_max_shard_size=512MB`
  - `checkpoint_safe_serialization=true`
  - `checkpoint_presave_gc=true`
  - `checkpoint_presave_empty_cache=true`
  - `checkpoint_presave_disable_cuda_history=true`
  - `enable_cuda_debug_history=false`
  - `attn_implementation=sdpa`
  - `causal_loss_mode=active_chunked_no_upcast`
  - `cuda_memory_fraction=0.88`
- Result:
  - Completed (`rc=0`, `session_status=trained`), including full checkpoint write.
- Metrics / evidence:
  - `gpu_peak_mib=110223`
  - `guard_triggered=no`
  - Trainer log shows `Writing model shards: ... 32/32` completed.
  - Pre-save callback telemetry:
    - Before cleanup: `mem_avail_mib=46726.4`, `nvidia_used_mib=70421.0`
    - After cleanup: `mem_avail_mib=75883.2`, `nvidia_used_mib=41835.0`
    - Post-save: `mem_avail_mib=75001.7`, `nvidia_used_mib=41835.0`
  - Checkpoint output contains `model-00001-of-00032.safetensors ... model-00032-of-00032.safetensors`.

### Session: `20260307_035417_step18_ckpt11_sdpa_chunked_shard512` (completed)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260307_035417_step18_ckpt11_sdpa_chunked_shard512`
- Purpose:
  - Validate repeated 32K checkpoint-save stability (`save_steps=10`) with the shard + pre-save-cleanup recipe, beyond a single-step save.
- Core config:
  - `model=Qwen/Qwen3.5-9B`
  - `max_seq_length=32768`
  - `max_steps=11`, `save_strategy=steps`, `save_steps=10`
  - `save_only_model=true`
  - `checkpoint_max_shard_size=512MB`
  - `checkpoint_safe_serialization=true`
  - `checkpoint_presave_gc=true`
  - `checkpoint_presave_empty_cache=true`
  - `checkpoint_presave_disable_cuda_history=true`
  - `attn_implementation=sdpa`
  - `causal_loss_mode=active_chunked_no_upcast`
  - `cuda_memory_fraction=0.88`
- Result:
  - Completed (`rc=0`, `session_status=trained`).
  - Both scheduled save points succeeded:
    - `checkpoints/checkpoint-10`
    - `checkpoints/checkpoint-11`
- Metrics / evidence:
  - `train_runtime=2960.7115s`
  - `train_loss=0.4176`
  - `epoch=1.8333`
  - Trainer log shows two full save passes with `Writing model shards: ... 32/32`.
  - Pre-save telemetry (save at step 10): `mem_avail_mib=13595.3`, `nvidia_used_mib=102583.0`.
  - Pre-save telemetry (final save at step 11): `mem_avail_mib=25035.2`, `nvidia_used_mib=91703.0`.
  - Post-cleanup before each save recovered to roughly `~75 GiB` host free memory and `~41.8 GiB` GPU used.

### Session: `20260306_053909_qwen35_9b_instruct_memprobe_32k_nosave_pagedadamw_v1` (completed)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260306_053909_qwen35_9b_instruct_memprobe_32k_nosave_pagedadamw_v1`
- Purpose:
  - Direct 32K feasibility probe after optimizer memory A/B.
- Core config:
  - `model=Qwen/Qwen3.5-9B`
  - `max_seq_length=32768`
  - `optim=paged_adamw_8bit`
  - `save_strategy=no`
  - `max_steps=10`
  - `max_gpu_memory_gib=120`
- Result:
  - Completed (`10/10`) with `train_loss=4.3841`, `train_runtime=1150.5s`.
- Peak memory evidence:
  - Crossed prior failure boundary (`step 8`) without OOM.
  - External sampler peak: `96903 MiB` (at `2026-03-05T21:50:23-08:00`).
  - Sampler file:
    - `/home/georvn/train_qwen35_9b/logs/mem_sample_32k_nosave_pagedadamw_v1.tsv`

### Session: `20260306_051846_qwen35_9b_instruct_memprobe_16k_nosave_pagedadamw_v2` (completed)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260306_051846_qwen35_9b_instruct_memprobe_16k_nosave_pagedadamw_v2`
- Purpose:
  - Isolate optimizer impact on peak VRAM at 16K using no-save probe.
- Core config:
  - `model=Qwen/Qwen3.5-9B`
  - `max_seq_length=16384`
  - `optim=paged_adamw_8bit`
  - `save_strategy=no`
  - `max_steps=10`
  - `max_gpu_memory_gib=120`
- Result:
  - Completed (`10/10`) with `train_loss=3.9006`, `train_runtime=947.3s`.
- Peak memory evidence:
  - External sampler:
    - `/home/georvn/train_qwen35_9b/logs/mem_sample_16k_nosave_pagedadamw_v2.tsv`
  - `max=87777 MiB`
- A/B delta versus prior 16K no-save baseline (`adamw_8bit`, max `114449 MiB`):
  - `-26672 MiB` (`-26.05 GiB`, `-23.3%`)

## 2026-03-05

### Session: `20260305_184211_qwen35_9b_instruct_full1109_quality_v3_12k_guard110` (full run in progress)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_184211_qwen35_9b_instruct_full1109_quality_v3_12k_guard110`
- Purpose:
  - Production full-epoch full-weight run after stability gating.
- Core config:
  - `model=Qwen/Qwen3.5-9B`
  - `max_seq_length=12288`
  - `truncation_side=left`
  - `max_gpu_memory_gib=110`
  - `dataset_num_proc=1`
  - `gradient_checkpointing=unsloth`
  - `learning_rate=1e-5`
  - `warmup_steps=50`
  - `save_steps=50`, `save_total_limit=4`

### Session: `20260305_182218_qwen35_9b_instruct_memprobe_12k_guard110` (stability proof, completed)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_182218_qwen35_9b_instruct_memprobe_12k_guard110`
- Purpose:
  - Validate Spark-safe full-weight recipe after repeated 32K/24K/16K failures.
- Core config:
  - `model=Qwen/Qwen3.5-9B`
  - `max_seq_length=12288`
  - `truncation_side=left`
  - `max_gpu_memory_gib=110`
  - `dataset_num_proc=1`
  - `gradient_checkpointing=unsloth`
  - `learning_rate=1e-5`
  - `max_steps=12`
- Result:
  - Completed successfully (`checkpoint-12` + `artifacts/full_model` written).
  - `train_loss=2.3061`, `train_runtime=862.8s`.

### Session: `20260305_175553_qwen35_9b_instruct_memprobe_16k` (failed)

- Purpose:
  - Test if 16K context avoids OOM boundary.
- Outcome:
  - Guard trip at step boundary: `measured=114449 MiB`, `limit=96 GiB`.
  - Exit with `RuntimeError` from memory guard (non-kernel kill).

### Session: `20260305_174338_qwen35_9b_instruct_memprobe_24k` (failed)

- Purpose:
  - Test if 24K context avoids OOM boundary.
- Outcome:
  - OOM-killed (`rc=137`) near the same late-step boundary.

### Session: `20260305_173245_qwen35_9b_instruct_memprobe_eager_32k` (failed)

- Purpose:
  - Test `attn_implementation=eager` at 32K.
- Outcome:
  - Reached step 8/9 and was OOM-killed (`rc=137`).

### Session: `20260305_171211_qwen35_9b_instruct_memprobe_unslothgc_32k` (failed)

- Purpose:
  - Test `gradient_checkpointing=unsloth` at 32K.
- Outcome:
  - OOM-killed (`rc=137`) around the same boundary.

### Session: `20260305_165312_qwen35_9b_instruct_full1109_quality_v2` (stopped/failed)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_165312_qwen35_9b_instruct_full1109_quality_v2`
- Purpose:
  - Full-epoch instruct-model (`Qwen3.5-9B`) full-weight run after user decision to avoid base-only training.
- Core config:
  - `model=Qwen/Qwen3.5-9B`
  - `max_seq_length=32768`
  - `truncation_side=left`
  - `learning_rate=1e-5`
  - `warmup_steps=50`
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=1`
  - `save_steps=50`
  - `save_total_limit=4`
- Launch notes:
  - New session created intentionally (old base-session pointer cleared).
  - Initial launch used resume-safe wrapper, but run repeatedly failed around step 8 with kernel OOM (`rc=137`).

### Session: `20260305_163434_qwen35_9b_full1109_quality_v1` (full run in progress)

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_163434_qwen35_9b_full1109_quality_v1`
- Purpose:
  - Start full-epoch, full-weight finetuning with the selected quality recipe on the 1109-row dataset.
- Core config:
  - `model=Qwen/Qwen3.5-9B-Base`
  - `max_seq_length=32768`
  - `truncation_side=left`
  - `learning_rate=2e-5`
  - `warmup_steps=30`
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=1`
  - `save_steps=50`
  - `save_total_limit=4`
- Launch notes:
  - Resume-safe wrapper bug fixed (stdout contamination in session-dir resolution).
  - Current active training process is launched in a live shell session for this environment.
- Observed startup stats:
  - `rows_truncated=6/1109`
  - `max_original_tokens=36448`
  - `max_final_tokens=32768`
  - `gpu_memory_guard=112 GiB`
  - `trainable_parameters=9,409,813,744 (100%)`
- Stop condition:
  - User requested switch away from base-model full training; run was intentionally stopped after early startup steps.

### Sweep: `quality_probe_20260305_090939`

- Summary files:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/docs/quality_probe_20260305_090939.json`
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/docs/quality_probe_20260305_090939.md`
- Purpose:
  - Fast LR ranking on 32K setup under single-DGX-Spark constraints.
- Common config:
  - `model=Qwen/Qwen3.5-9B-Base`
  - `max_seq_length=32768`
  - `max_samples=64`
  - `max_steps=2`
  - `warmup_steps=0`
  - `gradient_accumulation_steps=1`
  - `save_strategy=no`
  - `skip_final_save=true`
- LR results (last logged train loss):
  - `1e-5 -> 1.2335`
  - `2e-5 -> 1.1534`
  - `3e-5 -> 1.1523`
  - `5e-5 -> 1.3502`
- Outcome:
  - `2e-5` and `3e-5` are best; `5e-5` is clearly worse.
  - Operational default set to conservative `2e-5`.

### Session: `20260305_093103_confirm_long320_lr2e5_2step`

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_093103_confirm_long320_lr2e5_2step`
- Purpose:
  - Confirm selected LR on longer subset with wider token lengths.
- Core config:
  - `model=Qwen/Qwen3.5-9B-Base`
  - `max_seq_length=32768`
  - `max_samples=320`
  - `max_steps=2`
  - `learning_rate=2e-5`
  - `gradient_accumulation_steps=1`
  - `save_strategy=no`, `skip_final_save=true`
- Result:
  - Completed successfully without memory-guard trip.
- Metrics:
  - `train_runtime=335.99s`
  - `train_loss=1.3011`
  - `rows_truncated=0/320`
  - `max_original_tokens=20918`

### Session: `20260305_093920_probe_instruct1step_64rows_lr2e5`

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_093920_probe_instruct1step_64rows_lr2e5`
- Purpose:
  - Directly compare `Qwen/Qwen3.5-9B` (instruct) against base-model behavior.
- Core config:
  - `model=Qwen/Qwen3.5-9B`
  - `max_seq_length=32768`
  - `max_samples=64`
  - `max_steps=1`
  - `learning_rate=2e-5`
- Result:
  - Completed, but with much higher initial loss/grad norm than base probe.
- Metrics:
  - `train_loss=3.3253`
  - `grad_norm(step1)=193.0`
- Outcome:
  - Keep `Qwen/Qwen3.5-9B-Base` as default for this training recipe.

### Session: `20260305_081954_smoke32k_fullft_step1`

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_081954_smoke32k_fullft_step1`
- Model:
  - `Qwen/Qwen3.5-9B-Base`
- Mode:
  - full finetuning (`full_finetuning=true`, `load_in_4bit=false`)
- Core config:
  - `max_seq_length=32768`
  - `truncation_side=left`
  - `max_samples=8`
  - `max_steps=1`
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=1`
- Result:
  - Completed successfully.
  - Checkpoint written to `checkpoints/checkpoint-1/`.
  - Full model written to `artifacts/full_model/`.
- Metrics (`metadata/train_metrics.json`):
  - `train_runtime=151.6194`
  - `train_loss=2.0975`
- Truncation stats (`metadata/truncation_stats.json`):
  - `rows_truncated=0/8`
  - `max_original_tokens=5307`
  - `max_final_tokens=5307`

### Session: `20260305_075229_smoke_localcopy_manifest`

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_075229_smoke_localcopy_manifest`
- Purpose:
  - Validate local copied dataset + manifest compatibility.
- Result:
  - 1-step smoke completed and full model artifacts produced.

### Session: `20260305_075114_smoke_manifest_all1109_nothinking`

- Path:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260305_075114_smoke_manifest_all1109_nothinking`
- Purpose:
  - Verify session creation/manifest pathing against selected 1109-row dataset.
- Result:
  - Session created successfully; dataset counted as 1109 rows.

### Resume Probe: `checkpoint-900` host-memory phase instrumentation (2026-03-07)

- Logs:
  - Baseline (no mmap): `/home/georvn/train_qwen35_9b/logs/resume_probe_ckpt900_phase_20260307_112249.log`
  - Mmap-enabled: `/home/georvn/train_qwen35_9b/logs/resume_probe_ckpt900_phase_mmap_20260307_113537.log`
- Goal:
  - Explain repeated resume/startup failures from `checkpoint-900` with concrete memory deltas.
- Key measured deltas at `torch.load(optimizer.pt)`:
  - Baseline:
    - RSS: `~2100 MiB -> ~26078 MiB`
    - MemAvailable: `~101178 MiB -> ~77274 MiB`
  - Mmap-enabled:
    - RSS: `~2111 MiB` (no material jump)
    - MemAvailable: `~100985 MiB -> ~100977 MiB` (near-flat at load boundary)
- Interpretation:
  - Resume load spike was dominated by `optimizer.pt` deserialization mode.
  - `mmap=True` removes the largest host-RAM jump at load time.
  - Remaining resume pressure is mostly optimizer state materialization (GPU/host transient), not raw file deserialization.

### Session: `20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1` (resume continuation)

- Resume source:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1/checkpoints/checkpoint-900`
- Resume recipe:
  - Same 32K training hyperparameters as first 900 steps.
  - Operational change only: `--resume-torch-load-mmap` enabled.
  - Guard profile: external `min_mem_avail_mib=1536`, internal GPU guard `110 GiB`.
- Current status (live continuation at time of write):
  - Run has resumed past checkpoint and advanced through `901+` steps in the same session lineage.
  - No immediate resume-startup guard stop after mmap-enabled path.

### Training continuity watchdog (2026-03-07)

- Script:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/scripts/watch_resume_full1109_daemon.sh`
- Purpose:
  - Keep autonomous progress while operator is away by monitoring liveness and auto-relaunching only on unexpected pre-completion exits.
- Session routing rule:
  - Default pointer pinned to `/home/georvn/train_qwen35_9b/.state/session_qwen35_9b_full1109_32k_v1.txt` to guarantee resume targets the live 32K run lineage.
- Relaunch policy:
  - Uses `run_train_qwen35_9b_full1109_resume_safe.sh` with production memory safeguards (`MIN_MEM_AVAIL_MIB`, 110GiB internal guard, chunked active-token CE, checkpoint pre-save cleanup, `resume_torch_load_mmap=1`).

### Schema20 local HF eval (2026-03-08, no-thinking)

- Evaluator:
  - `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/scripts/run_schema20_local_hf.py`
- Inference controls:
  - `enable_thinking=False` via chat template
  - `max_new_tokens=192`
  - same 20-case set: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/evals/agent_cases_20_schema_final_v1.json`
- Candidate (full-FT model artifacts):
  - model: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1/artifacts/full_model`
  - score: `avg_structured_score=0.7175`
  - report: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/reports/schema20_local_hf_20260308_054524.json`
- Baseline (untouched instruct):
  - model: `Qwen/Qwen3.5-9B`
  - score: `avg_structured_score=0.6425`
  - report: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/reports/schema20_local_hf_20260308_055348.json`
- Baseline (untouched instruct, thinking enabled):
  - model: `Qwen/Qwen3.5-9B`
  - score: `avg_structured_score=0.0500`
  - report: `/home/georvn/train_qwen35_9b/qwen35_9b_fullft/reports/schema20_local_hf_20260308_061650.json`
- Delta:
  - absolute: `+0.0750`
  - relative over baseline: `~+11.7%`

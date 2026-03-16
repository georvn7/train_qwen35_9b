# Memory Analysis (DGX Spark, Qwen3.5-9B Full FT)

Date: `2026-03-05`

## Goal

Identify what does not fit memory for full-weight training and estimate biggest contributors.

## Dataset Length Tail (post chat-template, pre truncation)

- Total rows: `1109`
- Max tokens: `36458`
- `p99=28901`, `p95=19742`, `p90=16494`, `p50=5444`
- Rows above thresholds:
  - `>32768`: `6`
  - `>24576`: `17`
  - `>16384`: `114`
  - `>12288`: `200`
- Longest rows (index, tokens):
  - `(1089, 36458)`, `(1096, 36101)`, `(1091, 33701)`, `(1090, 33353)`, `(1099, 33234)`, `(1098, 32943)`

## Failure Evidence

- Repeated full-FT runs at `32K`, `24K`, and `16K` hit the same early failure boundary (around step `8-9`).
- Unguarded runs: kernel OOM kill (`rc=137`).
- Guarded runs:
  - `16K`: `measured=114449 MiB` (`torch_peak_reserved=114164 MiB`)
  - `12K` with `96 GiB` guard: `measured=100615 MiB` (`torch_peak_reserved=100330 MiB`)
- Stable proof:
  - `12K` with `110 GiB` guard completed (`12/12` steps) and wrote full artifacts.

### Independent Cross-Check (outside trainer code)

- Run: `20260306_043458_qwen35_9b_instruct_full1109_quality_v4_16k_guard120`
- External sampler: `nvidia-smi --query-compute-apps=pid,used_memory` every 1s (no trainer internals).
- Evidence file:
  - `/home/georvn/train_qwen35_9b/logs/mem_sample_16k_guard120_pid38430.tsv`
- Observed range:
  - `min=71883 MiB`
  - `max=114437 MiB`

This confirms late-step high-water marks are real and not an artifact of in-trainer memory accounting.

## Optimizer A/B at 16K (same recipe, no checkpoint save)

Compared runs:

- Baseline:
  - Session: `20260306_045646_qwen35_9b_instruct_memprobe_16k_nosave`
  - Optimizer: `adamw_8bit`
  - External sampler: `/home/georvn/train_qwen35_9b/logs/mem_sample_16k_nosave_long.tsv`
  - Peak: `114449 MiB`
- Tuned:
  - Session: `20260306_051846_qwen35_9b_instruct_memprobe_16k_nosave_pagedadamw_v2`
  - Optimizer: `paged_adamw_8bit`
  - External sampler: `/home/georvn/train_qwen35_9b/logs/mem_sample_16k_nosave_pagedadamw_v2.tsv`
  - Peak: `87777 MiB`

Delta:

- `-26672 MiB` (`-26.05 GiB`, `-23.3%` peak reduction)

Interpretation:

- The dominant failure driver was not checkpoint writing.
- Optimizer memory placement/paging materially changes total envelope and can unlock higher context on Spark.

## 32K Feasibility Probe (paged optimizer, completed)

- Session: `20260306_053909_qwen35_9b_instruct_memprobe_32k_nosave_pagedadamw_v1`
- Core config:
  - `max_seq_length=32768`
  - `optim=paged_adamw_8bit`
  - `save_strategy=no`
  - `max_steps=10`
  - `max_gpu_memory_gib=120`
- External sampler:
  - `/home/georvn/train_qwen35_9b/logs/mem_sample_32k_nosave_pagedadamw_v1.tsv`
- Final observed high-water mark:
  - `96903 MiB` at `2026-03-05T21:50:23-08:00`
- Final outcome:
  - Completed `10/10` steps (`rc=0`), no OOM/guard trip.
  - `train_runtime=1150.5s`, `train_loss=4.3841`
- Behavior:
  - Crossed prior failure zone (`step 8`) without OOM.
  - Spike pattern: fast climb (`~55.9 GiB -> ~96.9 GiB`) then drop and settle (`~82-84 GiB`), then release to ~`18.6 GiB` after process end.

## Approximate Memory Contributor Split

Model params: `9,409,813,744`

- BF16 weights: ~`17.53 GiB`
- BF16 gradients: ~`17.53 GiB`
- AdamW 8-bit states (rough): ~`19.3 GiB`
- Static subtotal (rough): ~`54.3 GiB`

Observed peaks near failure are `~100-114 GiB`, so dynamic memory (`~46-60 GiB`) is the dominant extra load.  
That dynamic portion is primarily activations + temporary workspaces (attention/linear-attn kernels, checkpointing recompute buffers, allocator growth/fragmentation).

## Practical Conclusion

- Biggest contributor beyond static model+optimizer footprint is activation/workspace memory under long context.
- With `adamw_8bit`, stable envelope was `12K` and `16K+` commonly crossed late-step limits.
- With `paged_adamw_8bit`, measured peak dropped by `~26 GiB` at `16K`, and a `32K` probe completed (`10/10`) with peak `96.9 GiB`.

## 2026-03-06 Addendum (32K stress rows, full-FT, adamw_8bit)

Three targeted 1-step stress runs (`6` longest rows, all truncated to `32768`) were used to isolate a stability lever:

- No allocator cap (`step14a`):
  - Failed (`rc=137`) at `gpu_peak_mib=114906`.
- With `cuda_memory_fraction=0.88`, `gpu_guard_gib=120`, `internal_gpu_guard_gib=0`:
  - `step14b` (`forward_active_chunked_no_upcast`) completed, `gpu_peak_mib=110311`, `train_runtime=309.36s`.
  - `step14c` (`active_chunked_no_upcast`) completed, `gpu_peak_mib=110223`, `train_runtime=330.28s`.
  - `step14d` (`default`) completed, `gpu_peak_mib=110097`, `train_runtime=309.46s`.

Interpretation:

- The allocator cap is a decisive stabilizer for the late-step spike on Spark in this stress pattern.
- Peak remains high (~`110 GiB`) but stays under the failure boundary seen in uncapped runs.
- Default loss mode is not worse than custom chunked variants in this 1-step stress check.
- Runtime cost is significant (single worst-case 32K step is ~`5+` minutes in these probes).

## Recommended Next 32K Diagnostic

1. Repeat `32K` with a stricter interactive guard (`100-105 GiB`) when desktop responsiveness matters.
2. Run the same `32K` recipe with checkpoint saves enabled to measure save-time overhead at long context.
3. Add one longer `32K` proof (`>=30` steps) before promoting as production default.
4. If needed, tune allocator/pacing (`PYTORCH_CUDA_ALLOC_CONF`, optional empty-cache cadence) after the longer proof run.

## 2026-03-06 Addendum (long 32K proof with checkpointing)

Session: `20260306_194048_step15_proof32k30_default_adamw8_cudafrac088`

- Config:
  - `max_seq_length=32768`, `optim=adamw_8bit`, `cuda_memory_fraction=0.88`
  - `save_strategy=steps`, `save_steps=10`
  - external guards: `gpu_guard_gib=120`, `min_mem_avail_mib=4096`
- Outcome:
  - Training progressed to `10/30`.
  - Stop reason was external host-memory guard during checkpoint save (`Writing model shards`), not CUDA OOM.
- Measured:
  - `gpu_peak_mib=110453`
  - host free memory floor before stop: `3206 MiB` (below the 4096 MiB floor)

Interpretation:

- The 32K training step path is stable under the allocator cap.
- Checkpoint-save phase introduces an extra host RAM wave that can breach desktop-safe memory floor.

## 2026-03-07 Addendum (model-only checkpoint retry)

Session: `20260307_024450_step16_ckpt11_saveonlymodel_default_adamw8_cudafrac088`

- Config delta vs `step15`:
  - `save_only_model=true`
  - `max_steps=11`, `save_steps=10`
- Outcome:
  - Progressed through `10/11`, then guard stop during checkpoint save (`rc=137`).
- Measured:
  - `gpu_peak_mib=110451`
  - host free memory floor: `1846 MiB` (guard floor `4096 MiB`)
  - guard reason: `mem_avail_mib=1846 <= min_mem_avail_mib=4096`

Interpretation:

- `save_only_model=true` did not remove the save-time host RAM cliff.
- The bottleneck remains host-memory pressure during shard writing, not forward/backward training.

## 2026-03-07 Addendum (checkpoint write tuning: shard size + pre-save cleanup)

Sessions:

- `20260307_034033_step17_ckpt1_32k_shard512_nodbg` (control failure)
- `20260307_034334_step17b_ckpt1_32k_sdpa_chunked_shard512` (successful save)

Config deltas introduced:

- `checkpoint_max_shard_size=512MB` (forces multi-shard save path instead of one large shard)
- `checkpoint_safe_serialization=true`
- pre-save housekeeping:
  - `checkpoint_presave_gc=true`
  - `checkpoint_presave_empty_cache=true`
  - `checkpoint_presave_disable_cuda_history=true`
- diagnostics off by default for this check:
  - `enable_cuda_debug_history=false`

Observed behavior:

- `step17` did not reach save phase due forward OOM with `attn_implementation=eager` + `causal_loss_mode=default`.
- `step17b` used stable 32K path (`sdpa` + `active_chunked_no_upcast`) and completed the full checkpoint save.
- Trainer log confirms `Writing model shards: 32/32` completed.
- Pre-save callback telemetry in `step17b`:
  - Before cleanup: `mem_avail_mib=46726.4`, `nvidia_used_mib=70421.0`
  - After cleanup: `mem_avail_mib=75883.2`, `nvidia_used_mib=41835.0`
  - Post-save: `mem_avail_mib=75001.7`, `nvidia_used_mib=41835.0`
- External sampler for `step17b` stayed above guard floor and never triggered guard:
  - minimum sampled `mem_avail_mib` during run: `~6434`
  - `guard_triggered=no`

Interpretation:

- Checkpoint-write memory pressure is configurable and can be reduced materially by:
  - forcing smaller checkpoint shards, and
  - reclaiming cache/garbage right before save.
- This resolves the immediate single-step save failure mode at 32K.
- Next validation should be a longer checkpoint cadence run (for example `>=11` steps with save at step `10`) to confirm the mitigation holds over repeated save cycles.

## 2026-03-07 Addendum (longer checkpoint cadence validation)

Session: `20260307_035417_step18_ckpt11_sdpa_chunked_shard512`

- Config:
  - same mitigation as `step17b` (`checkpoint_max_shard_size=512MB`, `checkpoint_safe_serialization=true`, pre-save cleanup enabled)
  - stable 32K training path (`attn_implementation=sdpa`, `causal_loss_mode=active_chunked_no_upcast`)
  - `max_steps=11`, `save_steps=10`, `save_only_model=true`
- Outcome:
  - Completed (`rc=0`, `session_status=trained`), including both save points.
  - Checkpoints present: `checkpoint-10` and `checkpoint-11`.
- Save-time memory telemetry:
  - Before step-10 save cleanup: `mem_avail_mib=13595.3`, `nvidia_used_mib=102583.0`
  - Before final save cleanup: `mem_avail_mib=25035.2`, `nvidia_used_mib=91703.0`
  - After cleanup before each save: host free memory recovered to about `~75 GiB`, GPU used dropped to about `~41835 MiB`.
  - Post-save remained stable (`mem_avail_mib` about `75 GiB`, `nvidia_used_mib` about `41835 MiB`).

Interpretation:

- The checkpoint-write mitigation is now validated across repeated save events, not only a single save.
- The dominant failure mode for this recipe has shifted away from checkpoint write pressure under this configuration.

## 2026-03-07 Addendum (resume deserialization memory)

Problem:

- Resume from `checkpoint-900` repeatedly failed at startup windows with host-memory pressure and intermittent external guard stops.

Measured with phase probes:

- Baseline resume probe log:
  - `/home/georvn/train_qwen35_9b/logs/resume_probe_ckpt900_phase_20260307_112249.log`
- Mmap-enabled resume probe log:
  - `/home/georvn/train_qwen35_9b/logs/resume_probe_ckpt900_phase_mmap_20260307_113537.log`

Key boundary (`torch.load(optimizer.pt)`):

- Baseline:
  - RSS jumped from about `2.1 GiB` to about `26.1 GiB`
  - `MemAvailable` dropped from about `101.2 GiB` to about `77.3 GiB`
- Mmap-enabled:
  - RSS stayed about `2.1 GiB`
  - `MemAvailable` stayed about `101.0 GiB` at the load boundary

Interpretation:

- Large host-memory surge at resume was caused by optimizer file deserialization mode, not model forward/backward compute.
- Enabling mmap for resume loads removes the largest startup host-RAM jump and improves resume stability.
- Remaining transient pressure after this point is optimizer state materialization; this is smaller than the previous deserialization spike and is compatible with the current guard profile in successful probes.

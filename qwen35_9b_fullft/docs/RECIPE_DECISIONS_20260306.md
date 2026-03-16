# Recipe Decisions (2026-03-06)

## Plain-language notes

- `-100` labels mean: "ignore this token in loss".
- With `assistant_only_loss=true`, the trainer computes loss on assistant tokens, not on user/system tokens.
- This is usually "all assistant spans in the conversation", not automatically "only the last assistant reply".

## Why 32K was unstable before

- In uncapped runs, one late training wave repeatedly pushed GPU memory above the practical limit.
- On Spark this caused either guard kill or kernel OOM kill (`rc=137`).

## What changed today

- Added optional forward-chunked loss mode in `train_session.py`.
- Ran stress probes on the 6 longest rows at full 32K.
- Main successful lever was setting `cuda_memory_fraction=0.88`.

## Observed outcomes

- Without cap:
  - `step14a` failed, peak `114906 MiB`.
- With cap (`cuda_memory_fraction=0.88`):
  - `step14b` succeeded, peak `110311 MiB`, runtime `309.36s`.
  - `step14c` succeeded, peak `110223 MiB`, runtime `330.28s`.
  - `step14d` succeeded in `default` loss mode, peak `110097 MiB`, runtime `309.46s`.

## Current recommendation

- For 32K full-FT on Spark, keep allocator cap enabled (`cuda_memory_fraction=0.88`) for stability.
- Prefer `causal_loss_mode=default` for quality fidelity unless a future stress case proves otherwise.
- Keep external guard (`gpu_guard_gib=120`) and host-memory floor guard.
- Continue with a longer 32K proof run (>=30 steps) before full-epoch launch.

# Reproduce Full-FT Run (32K, 1109 rows)

This is the canonical, explicit reproduction document for:

- `run_id`: `20260307_050331_qwen35_9b_instruct_full1109_32k_recipe_v1`
- `model`: `Qwen/Qwen3.5-9B`
- `mode`: full fine-tuning (not QLoRA), bf16
- `dataset_rows`: `1109`

## 1) Fast Path For Agents

Read these files first:

1. `qwen35_9b_fullft/scripts/run_train_qwen35_9b_full1109_resume_safe.sh`
2. `qwen35_9b_fullft/docs/repro/run_config_qwen35_9b_instruct_full1109_32k_recipe_v1.json`
3. `qwen35_9b_fullft/docs/repro/environment_qwen35_9b_instruct_full1109_32k_recipe_v1.json`
4. `qwen35_9b_fullft/docs/repro/dataset_manifest_qwen35_9b_instruct_full1109_32k_recipe_v1.json`

Then run the command block in section `4) Exact Launch Command`.

## 2) Environment (Pinned)

Reference environment snapshot:

- `qwen35_9b_fullft/docs/repro/environment_qwen35_9b_instruct_full1109_32k_recipe_v1.json`

Pinned Python deps (from the snapshot):

- `qwen35_9b_fullft/requirements.lock.txt`

Install:

```bash
cd /path/to/train_qwen35_9b
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r qwen35_9b_fullft/requirements.lock.txt
```

Expected platform class for closest parity:

- Linux aarch64
- CUDA driver in the same generation as snapshot (`580.126.09`, CUDA `13.0`)

## 3) Dataset Contract (HF-ready)

Expected local file path during training:

- `qwen35_9b_fullft/data/all_1109_rows_no_assistant_thinking.jsonl`

Expected checksum:

- `sha256=3c6ecbe7ae3ac997ee9396f71263b56b94637bbc6846e0229b2eb55686901ee2`

If you host the dataset on Hugging Face, download/export it to the path above, then verify:

```bash
cd /path/to/train_qwen35_9b
sha256sum qwen35_9b_fullft/data/all_1109_rows_no_assistant_thinking.jsonl
```

The hash must match before training.

## 4) Exact Launch Command

This uses the canonical resume-safe launcher with explicit knobs (so there is no ambiguity from defaults):

```bash
cd /path/to/train_qwen35_9b
source .venv/bin/activate

WORK=/path/to/train_qwen35_9b \
DATASET_ROOT=/path/to/train_qwen35_9b/qwen35_9b_fullft/data/all_1109_rows_no_assistant_thinking.jsonl \
MODEL_NAME=Qwen/Qwen3.5-9B \
MAX_SEQ_LENGTH=32768 \
TRUNCATION_SIDE=left \
ATTN_IMPLEMENTATION=sdpa \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
DATASET_NUM_PROC=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
GRADIENT_CHECKPOINTING=unsloth \
LEARNING_RATE=1e-5 \
WARMUP_STEPS=50 \
SAVE_STEPS=50 \
SAVE_TOTAL_LIMIT=4 \
MAX_GPU_MEMORY_GIB=110 \
CUDA_MEMORY_FRACTION=0.88 \
CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256' \
CAUSAL_LOSS_MODE=active_chunked_no_upcast \
CAUSAL_LOSS_CHUNK_TOKENS=2048 \
CHECKPOINT_MAX_SHARD_SIZE=512MB \
CHECKPOINT_SAFE_SERIALIZATION=true \
CHECKPOINT_PRESAVE_GC=1 \
CHECKPOINT_PRESAVE_EMPTY_CACHE=1 \
CHECKPOINT_PRESAVE_DISABLE_CUDA_HISTORY=1 \
RESUME_TORCH_LOAD_MMAP=1 \
NUM_TRAIN_EPOCHS=1.0 \
SEED=3413 \
LABEL=qwen35_9b_instruct_full1109_32k_recipe_v1_repro \
SESSION_PTR=/path/to/train_qwen35_9b/.state/session_qwen35_9b_full1109_32k_repro_e1.txt \
qwen35_9b_fullft/scripts/run_train_qwen35_9b_full1109_resume_safe.sh
```

## 5) Expected Outcome

Reference metrics from canonical run:

- `qwen35_9b_fullft/docs/repro/train_metrics_qwen35_9b_instruct_full1109_32k_recipe_v1.json`
  - `train_loss: 0.25022190599639316`
  - `epoch: 1.0`

Minor drift across hosts is expected, but training should complete and produce:

- `checkpoints/checkpoint-*` shards
- `artifacts/full_model/` shards
- metadata JSON files under `metadata/`

## 6) Serving After Training (OpenAI-compatible API)

Single-user resident serving profile:

```bash
qwen35_9b_fullft/scripts/start_vllm_fullft_bf16_resident.sh
```

Expected:

- endpoint: `http://<host>:8002/v1`
- model id: `qwen35-9b-fullft-bf16`

## 7) Canonical Repro Files (Checked In)

- `qwen35_9b_fullft/docs/repro/run_config_qwen35_9b_instruct_full1109_32k_recipe_v1.json`
- `qwen35_9b_fullft/docs/repro/environment_qwen35_9b_instruct_full1109_32k_recipe_v1.json`
- `qwen35_9b_fullft/docs/repro/dataset_manifest_qwen35_9b_instruct_full1109_32k_recipe_v1.json`
- `qwen35_9b_fullft/docs/repro/train_metrics_qwen35_9b_instruct_full1109_32k_recipe_v1.json`

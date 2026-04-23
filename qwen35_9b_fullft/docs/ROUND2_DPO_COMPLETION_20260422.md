# Round 2 DPO Completion

Date: `2026-04-22`
Run id: `20260421_032308_qwen35_9b_round2_dpo_702_clean_16k_v1`

## Outcome

The round-2 DPO run completed successfully from the latest round-2 continuation full model.

- final checkpoint:
  - `qwen35_9b_fullft/runs/20260421_032308_qwen35_9b_round2_dpo_702_clean_16k_v1/checkpoints/checkpoint-702`
- final exported model:
  - `qwen35_9b_fullft/runs/20260421_032308_qwen35_9b_round2_dpo_702_clean_16k_v1/artifacts/full_model`
- trainer completion time:
  - `2026-04-22T17:13:14-07:00`

## Final Trainer Metrics

- `train_runtime = 7816.4941`
- `train_samples_per_second = 0.09`
- `train_steps_per_second = 0.09`
- `train_loss = 0.09880231095854713`
- `epoch = 1.0`

## Canonical Inputs

- start model:
  - `qwen35_9b_fullft/runs/20260418_220025_qwen35_9b_round2_cont_sft_1869_32k_v1/artifacts/full_model`
- cleaned DPO dataset:
  - `qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean.jsonl`
- prepared DPO dataset:
  - `qwen35_9b_fullft/data/all_702_rows_dbg_dpo_round2_clean_prepared.jsonl`

## Operational Notes

- The run used durable cached reference log-probs and exact checkpoint resume.
- Early resume/load instability was solved before the final successful attempt.
- During the successful late stage, periodic `50`-step checkpoints continued to be written and completed cleanly.
- Final durable checkpoints reached:
  - `550`
  - `600`
  - `650`
  - `700`
  - `702`

## Serving

Dedicated bf16 DPO-serving launchers were added:

- start:
  - `qwen35_9b_fullft/scripts/start_vllm_round2_dpo_bf16_openai.sh`
- stop:
  - `qwen35_9b_fullft/scripts/stop_vllm_round2_dpo_bf16_openai.sh`

Default served model identity:

- model path:
  - `qwen35_9b_fullft/runs/20260421_032308_qwen35_9b_round2_dpo_702_clean_16k_v1/artifacts/full_model`
- model id:
  - `qwen35-9b-round2-dpo-bf16`
- endpoint:
  - `http://<host>:8002/v1`

## Evaluation

Post-DPO local-HF schema20 report:

- json:
  - `qwen35_9b_fullft/reports/schema20_local_hf_20260423_024134.json`
- markdown:
  - `qwen35_9b_fullft/reports/schema20_local_hf_20260423_024134.md`
- score:
  - `avg_structured_score = 0.7725`
  - `median_structured_score = 1.0`
  - `min_structured_score = 0.25`
  - `max_structured_score = 1.0`

Reference comparison against the pre-DPO round-2 SFT model:

- previous report:
  - `qwen35_9b_fullft/reports/schema20_local_hf_20260420_235146.json`
- previous `avg_structured_score = 0.7675`
- observed delta:
  - `+0.0050`

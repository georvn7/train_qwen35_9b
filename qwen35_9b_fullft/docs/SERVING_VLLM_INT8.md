# vLLM OpenAI-Compatible Serving (Full-FT 8-bit)

This exposes your full-FT Qwen3.5-9B model via OpenAI-style API endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`

You can run three variants:

- Full-FT int8 (`port 8000`, model `qwen35-9b-fullft-int8`)
- Untouched int8 (`port 8001`, model `qwen35-9b-untouched-int8`)
- Full-FT bf16 (`port 8002`, model `qwen35-9b-fullft-bf16`)

## Start

```bash
cd /home/georvn/train_qwen35_9b
chmod +x qwen35_9b_fullft/scripts/start_vllm_fullft_int8_openai.sh \
         qwen35_9b_fullft/scripts/stop_vllm_fullft_openai.sh

# Optional:
# export API_KEY='your-key'
# export PORT=8000
# export HOST=0.0.0.0

qwen35_9b_fullft/scripts/start_vllm_fullft_int8_openai.sh
```

For untouched int8:

```bash
cd /home/georvn/train_qwen35_9b
qwen35_9b_fullft/scripts/start_vllm_untouched_int8_openai.sh
```

For full-FT bf16:

```bash
cd /home/georvn/train_qwen35_9b
qwen35_9b_fullft/scripts/start_vllm_fullft_bf16_openai.sh
```

Default served model id:

- `qwen35-9b-fullft-int8`

## Test `/v1/models`

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .
```

If using an API key:

```bash
curl -s -H "Authorization: Bearer $API_KEY" \
  http://127.0.0.1:8000/v1/models | jq .
```

## Test `/v1/chat/completions`

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-9b-fullft-int8",
    "messages": [
      {"role":"system","content":"You are concise."},
      {"role":"user","content":"Return JSON with keys action_type and action_subject."}
    ],
    "temperature": 0.0,
    "max_tokens": 128
  }' | jq .
```

With API key:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-9b-fullft-int8",
    "messages": [{"role":"user","content":"Hello"}]
  }' | jq .
```

## Stop

```bash
cd /home/georvn/train_qwen35_9b
qwen35_9b_fullft/scripts/stop_vllm_fullft_openai.sh
qwen35_9b_fullft/scripts/stop_vllm_untouched_openai.sh
qwen35_9b_fullft/scripts/stop_vllm_fullft_bf16_openai.sh
```

## Notes

- Quantization is set to BitsAndBytes 8-bit (`load_in_8bit=true`).
- `enable_thinking` is disabled by default at server level (`default_chat_template_kwargs`), matching your training/eval direction.
- For full-FT exports saved as `qwen3_5_text`, startup script auto-builds a config-compatible serving copy at:
  - `<full_model_dir>_vllm_compat`
  - This rewrites config only and reuses the same weights.
- Stability defaults:
  - `max_num_seqs=1`
  - `max_num_batched_tokens=32768`
  - `gpu_memory_utilization=0.88`

# Inference Engine Recommendation (Qwen3.5 9B)

Date: `2026-03-05`

## Decision

Use **vLLM** as the default inference engine for this project.

## Why vLLM First

- Official Qwen3.5 model card directly lists vLLM support and provides vLLM launch examples.
- vLLM explicitly includes `qwen3_5` model support.
- vLLM provides OpenAI-compatible serving (`vllm serve`) with broad ecosystem/tool compatibility.
- For Qwen3.5 hybrid models, vLLM documents `--language-model-only` to skip multimodal modules and free GPU memory for KV cache.

## Ranked Options

1. **vLLM (default)**
   - Best balance of compatibility, throughput, and operational simplicity.
2. **SGLang (strong alternative)**
   - Good if we prioritize SGLang-specific parsers/features and its serving stack.
3. **TensorRT-LLM (specialized max-perf path)**
   - Strong option when we can invest in NVIDIA-specific optimization and deployment complexity.
4. **TGI (fallback only for this model family)**
   - Current public support list does not explicitly list Qwen3.5, so it is not first choice here.

## Starter Command (vLLM)

```bash
vllm serve /home/georvn/train_qwen35_9b/qwen35_9b_fullft/runs/<session_id>/artifacts/full_model \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 32768 \
  --language-model-only
```

Adjust `--max-model-len` to deployment needs and available VRAM.

## Sources

- Qwen3.5-9B-Base model card: https://huggingface.co/Qwen/Qwen3.5-9B-Base
- vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models.html
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- SGLang supported generative models: https://docs.sglang.io/supported_models/generative_models.html
- TensorRT-LLM support matrix: https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html
- TGI model support list: https://huggingface.co/docs/text-generation-inference/supported_models

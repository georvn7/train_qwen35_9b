# Serving Numerical Health

## Qwen Hybrid Cache Fix

Do not use unpatched vLLM 0.17.0 for Qwen3.5 hybrid-model sampling. Version
0.17.1 includes upstream [PR 35219](https://github.com/vllm-project/vllm/pull/35219),
which clears newly allocated attention cache blocks before reuse. Stale SSM
bytes can otherwise be interpreted as NaN values by attention kernels.

The 2026-09-04 incident produced ten 1,400-token punctuation-only completions
during CDPO. Ordinary requests returned HTTP 200; requesting log probabilities
exposed a server-side `Out of range float values are not JSON compliant: nan`.
Do not classify these as model-policy negatives or report the affected pass@K
as a valid measurement. Quarantine the sampling artifacts and resample after
validating the inference engine. Teacher API distillation and model weights
are not modified by this recovery.

Test the patch release separately before upgrading the deployed runtime. Keep
Torch, Transformers, FlashInfer, training dependencies, weights, BF16, chat
template, sampling settings, and CUDA-graph execution unchanged. In this
incident, all 1,466 installed vLLM Python files matched their original package
hashes, so no local model-support patches needed preservation. Check this
again before replacing another installation.

The BF16 launcher enables `VLLM_COMPUTE_NANS_IN_LOGITS=1` by default. Monitor
`vllm:corrupted_requests_total` on `/metrics`, recording the server process and
counter before and after sampling. A positive increase invalidates that sampling
window and must hold training for investigation. The launcher exposes the
counter; it does not itself stop a running curriculum. Missing telemetry is not
proof of zero corruption. An explicit `VLLM_COMPUTE_NANS_IN_LOGITS=0` disables
this observation and should not be used for audited sampling.

Release checks must include repeated short and full-budget requests across
different saved context lengths, finite returned log probabilities, a zero
corruption counter, normal structured completions, and a throughput comparison.
Do not use a single health-check response as the numerical validation gate.

# GLM-5 family

Zhipu AI / Tsinghua's MoE family. GLM-4.5, GLM-4.7, GLM-5.0, GLM-5.1 (2025–2026). The model that's most commonly cited as **the easiest large MoE to run on workstation Blackwell**, because of how its dependency stack is structured.

## The model

| | GLM-5.0 | GLM-5.1 (full) | GLM-5.1 (REAP-160) |
| --- | --- | --- | --- |
| Total parameters | ~620 B | 744 B | 478 B |
| Active per token | ~36 B | 42 B | ~36 B |
| Number of experts | 160 routed | 256 routed | 160 routed (after pruning) |
| Top-k | 6 | 8 | 8 |
| Hidden dim | 6144 | 7168 | 7168 |
| Attention | MHA + DSA option | MHA + DSA option | MHA + NSA option |
| Native quantization | FP8 | NVFP4 | NVFP4 |

GLM-5.1 introduced two attention variants: standard MHA, and DSA (Differential Sparse Attention) for long-context. DSA has a known kernel issue on SM120 (see below); MHA works fine.

REAP-160 is Zhipu's official pruning recipe that brings GLM-5.1 from 256 to 160 experts with minimal quality loss (~0.3 perplexity drop). The resulting 478B-parameter model is **specifically sized to fit 4× 96 GB workstation Blackwell**.

## What the reference deployment assumes

Zhipu's deployment guidance for GLM-5 targets:

- **Hardware**: H200 / B200, with NVLink ideally
- **GEMM**: stock CUTLASS NVFP4 templates
- **Attention**: FlashAttention-2 (MHA) or custom DSA kernel
- **MoE**: sglang or vLLM, with the engine choosing between FlashInfer-MoE and CUTLASS-MoE backends
- **Parallelism**: EP=N for the experts, TP within each expert

The dependency stack is **the lightest of the three families covered here**: stock CUTLASS, stock FlashAttention, no custom GEMM library. The only `tcgen05`-specific code path is the optional FlashInfer-MoE one-shot a2a, which is opt-in.

## What breaks on workstation Blackwell

### 1. EP plan (default for some configs)

If the inference engine defaults to EP, you hit the all-to-all bandwidth wall. Switch to TP.

### 2. DSA path on SM120

GLM-5.1's DSA kernel has a path that fails to compile on SM120 in some inference engine versions. This is a known issue; vLLM 0.7.x and sglang 0.5.10+ have patched it.

For long-context inference, you can either:

- Enable DSA with the patched engines
- Fall back to MHA (slower at long context, but correct)

### 3. CUTLASS SMEM cliff

Same as elsewhere — CUTLASS NVFP4 templates may overflow SMEM. Use SM120 templates with smaller tiles.

### 4. fp8_e4m3 KV on certain sglang versions

Some intermediate sglang versions hit a `tcgen05.cp`-based dequant path for FP8 KV. Pin to **sglang 0.5.10.post1 or later** which uses an `mma.sync`-based dequant path.

## What works on workstation Blackwell

A working configuration for GLM-5.1 REAP-160 NVFP4:

```yaml
weights: NVFP4 (REAP-160 prune of GLM-5.1, fits 4× 96GB)
kv_cache: FP8 E4M3
attention: Triton-based with kv_splits=64
parallelism:
  tensor_parallel: 4
  pipeline_parallel: 1
  expert_parallel: 1
gemm_backend: cutlass (SM120 templates)
attention_backend: triton (auto-selected on SM120)
```

GLM-5.1 REAP-160 + NVFP4 quantization fits 4× 96 GB workstation Blackwell with substantial KV-cache headroom — enough for **200k-token context** with FP8 KV. This is the model class for which workstation Blackwell is genuinely viable as a deployment target.

## Performance expectations

On 4× workstation Blackwell:

| Variant | Context | Decode tok/s |
| --- | ---: | ---: |
| GLM-5.1 REAP-160 | 256 | 46 |
| GLM-5.1 REAP-160 | 4 K | 42 |
| GLM-5.1 REAP-160 | 16 K | 38 |
| GLM-5.1 REAP-160 | 150 K | 22 |

These are from a TP=4, NVFP4-weights, FP8-KV, kv_splits=64 configuration. Without the kv_splits=64 setting, the long-context number drops to ~5 tok/s — that one knob is the largest single perf lever on this hardware class.

For comparison: the same model on a 4× B100 (datacenter Blackwell) deployment hits ~120–150 tok/s in the same configuration. The ~5× gap is consistent across all the case studies.

## What makes GLM-5 the "easy" case

- **Lighter custom-kernel surface**: stock CUTLASS, stock FlashAttention. No DeepGEMM-equivalent.
- **Pruning recipe sized for consumer hardware**: REAP-160 specifically targets 4× 96 GB.
- **MoE plan flexibility**: model architecture works fine under TP-only, no EP requirement
- **Active community**: workstation Blackwell deployment is documented and tested

If you're new to running MoE on workstation Blackwell and want a model that "just works," GLM-5.1 REAP-160 NVFP4 is the recommended starting point.

## What MTP and NSA add

- **MTP (Multi-Token Prediction)**: speculative decoding for higher throughput. Adds an extra prediction head; requires `page_size=64` and BF16 KV. Optional; opt-in via inference engine flags.
- **NSA (Native Sparse Attention)**: sparsity in the attention computation for long-context. Has SM120 kernel issues in some versions; safest to leave disabled.

For initial deployments, leave both disabled. Enable MTP if you need throughput beyond what the baseline configuration provides.

## See also

- [`deepseek-v3-v4`](deepseek-v3-v4.md) — heavier dependency stack
- [`kimi-k2`](kimi-k2.md) — comparable size, more memory pressure
- [`compatibility/`](../compatibility/index.md) — the patterns
- Zhipu AI's GLM-5 release notes and HuggingFace model cards
- The REAP pruning paper

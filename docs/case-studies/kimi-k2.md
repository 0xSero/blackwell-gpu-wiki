# Kimi-K2 family

Moonshot AI's MoE family. Released across 2025–2026 (K2.0 → K2.6+). Similar profile to DeepSeek-V3/V4 in that it's MoE with NVFP4 quantization, but with a few distinguishing factors that affect how it runs on workstation Blackwell.

## The model

| | K2.0 | K2.6 |
| --- | --- | --- |
| Total parameters | ~600 B | ~700 B |
| Active per token | ~32 B | ~40 B |
| Number of experts | 384 | 384 (with refined routing) |
| Top-k | 8 | 6 |
| Hidden dim | 6144 | 7168 |
| Attention | GQA (Grouped-Query Attention) | GQA |
| Native quantization | FP8 → NVFP4 (later release) | NVFP4 |

Kimi uses standard GQA rather than DeepSeek's MLA. The KV cache is larger per token than DeepSeek MLA, but the attention kernels are universally available (FlashAttention-2 supports GQA out-of-the-box).

## What the reference deployment assumes

Moonshot's deployment guidance for K2 targets:

- **Hardware**: H200 / B100 / B200, ideally with NVLink
- **GEMM**: their fork of CUTLASS with custom NVFP4 templates targeting `sm_100a`
- **Attention**: FlashAttention-2 or FlashInfer (GQA paths)
- **MoE**: vLLM with FlashInfer-MoE for the all-to-all
- **Parallelism**: EP for the experts, TP within each expert, PP across model layers

The dependency surface is **less aggressive than DeepSeek's**: Kimi doesn't ship a custom GEMM library; they use CUTLASS. They use FlashAttention-2 (portable) rather than a custom MLA kernel. The MoE all-to-all is the single significant SM100-only assumption.

## What breaks on workstation Blackwell

### 1. CUTLASS NVFP4 paths hit the SMEM cliff

Moonshot's CUTLASS templates inherit the same SMEM-budget assumptions as upstream CUTLASS. On SM120, the auto-carveout request exceeds 99 KiB and corrupts SMEM banks.

Fix: either patch the CUTLASS templates to set explicit smaller `StageCount`, or use the upstream SM120-targeted templates (with slightly different tile shapes than Moonshot's defaults).

### 2. EP-with-FlashInfer-a2a breaks on PCIe atomics

Same as DeepSeek. Use NCCL fallback or switch to TP-only.

### 3. The 384-expert count makes TP-only memory-tight

With 384 experts each ~1.5 GB at NVFP4, total expert weight memory is ~570 GB. On a 4× 96 GB rig (384 GB total), TP=4 with replicated experts (each GPU holds all 384 experts' TP-slices) needs 570/4 ≈ 143 GB per GPU — **doesn't fit** in 96 GB.

This is the case where pure TP-only doesn't work and you need a hybrid plan: TP=4 + PP=2 (split layers across GPU pairs), or accept EP-with-NCCL despite the bandwidth cost.

For K2.6, you typically need either:

- Pruning down to ~256 active experts (REAP-style) → fits TP-only
- Hybrid TP × PP plan (slower per-token decode, lower memory pressure)
- Acceptance of EP-with-NCCL (slow but works)

### 4. Routing kernel quirks

K2 uses a custom top-k routing kernel that assumes `tcgen05`-style asynchronous Tensor Core execution for the routing softmax. On SM120, this falls back to a slower path. Not a correctness issue, just a performance hit.

## Working configuration

```yaml
weights: NVFP4 (Moonshot's K2.6 release, 256 experts after REAP pruning)
kv_cache: FP8 E4M3
attention: FlashAttention-2 (GQA path) or Triton fallback
parallelism:
  tensor_parallel: 4
  pipeline_parallel: 1 (or 2 if memory tight)
  expert_parallel: 1
gemm_backend: cutlass with explicit StageCount=2 or 3
```

Notice: with FlashAttention-2 working out-of-the-box for GQA, Kimi-K2 is in some ways **easier** than DeepSeek-V4 to run on workstation Blackwell, despite being a comparable-scale model.

## Performance expectations

On 4× workstation Blackwell:

| Variant | Decode tok/s |
| --- | ---: |
| K2.6 with REAP pruning to 256 experts | 30–50 |
| K2.6 full 384 experts via TP × PP=2 | 15–30 |
| K2.6 via EP-NCCL | 5–10 |

A datacenter B100 deployment hits 100–200 tok/s. The gap is similar to DeepSeek-V4: ~5×.

## What's distinctive about Kimi

- **No custom GEMM library**: depends on CUTLASS, which has SM120 support. Easier to port.
- **Standard GQA attention**: works on every kernel library.
- **High expert count**: stresses memory more than DeepSeek (256 → 384).
- **Routing kernel uses `tcgen05`-style**: a specific kernel-level dependency, not a model-architecture one.

If you can run DeepSeek-V4 on workstation Blackwell, you can definitely run Kimi-K2 — except possibly for the memory pressure from 384 experts.

## See also

- [`deepseek-v3-v4`](deepseek-v3-v4.md) — comparable profile, MLA attention
- [`generic-moe-on-consumer-blackwell`](generic-moe-on-consumer-blackwell.md) — the synthesis
- [`compatibility/ep-to-tp-rewriting`](../compatibility/ep-to-tp-rewriting.md) — patterns for the parallelism rewrite
- Moonshot AI's K2 release blog posts and HuggingFace model cards

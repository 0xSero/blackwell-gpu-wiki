# MoE parallelism

How a Mixture-of-Experts model is split across GPUs, and why the choice of plan dominates throughput on workstation Blackwell.

## The plans, briefly

| Plan | What's split | What's communicated per layer | Bandwidth need |
| --- | --- | --- | --- |
| **TP (Tensor Parallel)** | Each weight matrix split across N GPUs | `all_reduce` after attention output and FFN output | low–moderate |
| **PP (Pipeline Parallel)** | Layers split into N stages, microbatch through | P2P send/recv between adjacent stages | low |
| **EP (Expert Parallel)** | Each MoE layer's experts split across N GPUs | `all_to_all` for token dispatch and combine | **high** |
| **DP (Data Parallel)** | Replicate model, split batch | `all_reduce` per layer (training); rare in inference | low |

For dense models (Llama, Mistral), only TP and PP apply. For MoE models (DeepSeek, Mixtral, Kimi, GLM-5), EP is an additional option that can be combined with TP and PP.

## How EP works

A MoE layer has N experts, each a feed-forward network. Routing assigns each token to top-k of the N experts (typically k=2 or 8).

With EP=N, expert *i* lives on GPU *i*. For each token:

1. **Routing**: small computation (a softmax), shared across GPUs
2. **Dispatch**: send the token to its top-k experts (cross-GPU)
3. **Compute**: each expert runs FFN on its assigned tokens
4. **Combine**: send expert outputs back to the originating GPU (cross-GPU)

Steps 2 and 4 are **all-to-all** operations. They're the dominant cost.

## How TP works in MoE

With TP=N (no EP), each expert's weight matrices are split across all N GPUs. **All N GPUs hold a slice of every expert.** When token routing assigns tokens to experts, each GPU computes its slice of the FFN; an `all_reduce` at the end sums slices.

Steps:

1. **Routing**: same
2. **Dispatch**: a permutation of tokens within each GPU (no cross-GPU traffic for the dispatch)
3. **Compute**: each GPU computes its slice of every expert's FFN, multiplied by the routing weights
4. **Reduce**: `all_reduce` across the N GPUs (one per layer)

The communication is `all_reduce` (volume = hidden_dim × tokens), not `all_to_all` (volume = N × hidden_dim × tokens). Roughly **N× less data** moves.

## Why the bandwidth difference matters

For a typical MoE inference step:

| Quantity | Value |
| --- | --- |
| Hidden dim | 7168 |
| Tokens per step | 64 |
| Number of GPUs | 4 |
| Number of experts | 256 |
| Top-k | 8 |

EP all-to-all volume per layer: `4 × 7168 × 64 × 2 bytes (BF16) ≈ 3.7 MB` for dispatch alone. Combine doubles it. Roughly **8 MB per layer**.

TP all_reduce volume per layer: `7168 × 64 × 2 bytes ≈ 920 KB`. Roughly **1 MB per layer**.

Per-step communication ratio: ~8× more data for EP.

But that's just the volume. The bigger issue is **bandwidth utilization**:

- NVLink 5: ~1.8 TB/s. 8 MB completes in **~5 µs**. EP is fine.
- PCIe Gen4: ~32 GB/s. 8 MB completes in **~250 µs**. EP is the dominant cost.

For a model with 100 layers, that's 25 ms per token of pure communication on PCIe — the model can't decode faster than ~40 tok/s regardless of compute speed. With TP, the same model decodes in single-digit-millisecond per token.

## The empirical collapse

Measured on a 4-GPU workstation Blackwell setup running a 478B-parameter MoE model:

| Plan | Decode tok/s |
| --- | ---: |
| TP=4 (no EP) | ~49 |
| EP=4 (PCIe all-to-all via NCCL) | ~1.4 |

A **35× slowdown** from one configuration choice. The whole "make this model run on workstation Blackwell" project amounts to: configure the inference engine to use TP=4 and avoid EP entirely.

## When EP is unavoidable

Some MoE models are designed for EP and don't have a clean TP fallback. Two common patterns:

### "Shared experts" plus "routed experts"

DeepSeek-V2/V3 splits experts into:
- **Shared experts**: small, every token uses them (TP-friendly)
- **Routed experts**: large, sparsely activated, EP-natural

For these models, the routed experts are best served with EP if NVLink is available, with TP if not.

### Very high N (large expert count)

Models with N > 64 experts may not fit a TP-only plan because each GPU must replicate every expert's weights — for very large N this is memory-prohibitive.

A 512-expert model with experts each ~5 GB would need 2.5 TB of expert weights per GPU under TP. EP is forced.

For the MoE models that fit on workstation Blackwell (typically 100B–700B with 64–256 experts), TP-only plans **do** fit because of the broader memory savings from NVFP4 quantization.

## Hybrid plans

A typical production deployment isn't pure TP or pure EP — it's hybrid:

- **TP × EP**: split experts across some GPUs (EP) and split each expert's weights across the rest (TP). Common in large datacenter deployments.
- **TP × PP**: tensor-parallel within a stage, pipeline-parallel across stages. Standard for very large dense models.
- **TP × EP × PP**: all three. Standard for trillion-parameter MoE.

For workstation Blackwell with no NVLink:

- **TP-only** is the default
- **TP × PP** if memory is tight (more model fits via PP staging, at slight latency cost)
- **No EP** unless you've solved the atomics issue (and even then, NCCL fallback is the limit)

## Reading an inference engine's parallelism flags

In sglang:

```bash
--tensor-parallel-size 4         # TP=4
--pipeline-parallel-size 1       # PP=1
# (no flag for EP; sglang chooses based on model + topology)
```

In vLLM:

```bash
--tensor-parallel-size 4
--pipeline-parallel-size 1
--enable-expert-parallel false   # disable EP explicitly (vLLM 0.7+)
```

The "EP flag" is sometimes implicit in the model config. For DeepSeek-V3 the reference config assumes EP; you may need to override it explicitly.

## Detecting which plan is active

Looking at the boot log:

```bash
grep -iE 'tensor.parallel|pipeline.parallel|expert.parallel' /tmp/sglang.log
```

Or running an instrumented forward and checking NCCL traffic:

```bash
NCCL_DEBUG=INFO bash launcher.sh 2>&1 | grep -i 'all_to_all\|all_reduce\|nccl'
```

If you see `all_to_all` calls in the per-layer hot path, EP is active. If only `all_reduce`, TP-only.

## Memory implications

The plans differ in memory footprint:

| Plan | Per-GPU model memory |
| --- | --- |
| TP=N | total_model / N (model is split) |
| EP=N (no TP) | total_model / N (experts are split) |
| TP=N × EP=M | total_model / (N × M) |
| Replicated experts (per-GPU TP) | total_model / N **but with all experts on each GPU** |

In the "replicated experts" pattern (each GPU has all expert weights, but each expert's weights are TP-split), the per-GPU memory is `(non_expert_weights / N) + (sum_of_all_experts / N)`. Same total, just a different way of slicing.

For a MoE model where experts dominate (e.g., 90 % of weight memory), the TP-replicated-experts plan and the EP plan use roughly the same per-GPU memory. The trade is communication style: TP all_reduce vs EP all-to-all.

## Summary

The MoE parallelism choice for workstation Blackwell:

- **Default**: TP, no EP, no PP
- **If memory tight**: add PP (small latency cost, more total memory)
- **If you really need EP**: only with atomics enabled, and accept that performance is bounded by PCIe
- **Never**: NVSHMEM-based EP (DeepEP intranode), unless you have NVLink

The single largest performance lever on this hardware class is "don't use EP."

## See also

- [`nvlink-vs-pcie`](nvlink-vs-pcie.md) — the bandwidth numbers behind this choice
- [`p2p-and-atomics`](p2p-and-atomics.md) — what blocks EP-with-atomics-based-kernels
- [`kernels/nvshmem-and-deepep`](../kernels/nvshmem-and-deepep.md) — the libraries that fail
- [`compatibility/ep-to-tp-rewriting`](../compatibility/ep-to-tp-rewriting.md) — patterns for the rewrite
- *DeepSeek-V3 Technical Report* (the original EP-on-NVL72 design)

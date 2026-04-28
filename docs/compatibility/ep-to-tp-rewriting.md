# EP-to-TP plan rewriting

The single highest-impact compatibility pattern: restructuring an expert-parallel deployment plan into a tensor-parallel one.

## Why this matters

Expert parallelism, as discussed in [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md), demands a token all-to-all every layer. On a system without NVLink and without P2P atomics, that all-to-all goes through the host bridge and tanks performance.

Tensor parallelism only requires per-layer all-reduce. NCCL ring all-reduce works tolerably over PCIe and the host bridge, especially with NCCL_P2P_LEVEL=PIX.

Result: rewriting an EP plan as a TP plan often eliminates 80%+ of communication overhead with no kernel-level work required.

## When the rewrite is feasible

Three conditions:

1. **Aggregate VRAM is sufficient.** Sharding all experts on every GPU in TP requires the full expert weight to be N-way split. For an MoE model with E experts of weight `W_e` each, total weight is `E × W_e`. TP needs this to fit across `N` GPUs at `(E × W_e) / N` per GPU.

2. **The model isn't engineered around EP.** Some models (DeepSeek-V4 with NSA) have routing tightly coupled to a specific expert distribution. Rewriting becomes harder.

3. **The inference engine supports the alternative.** vLLM, sglang, and TRT-LLM all support TP-only configurations for MoE models, but the flag names and exact semantics differ.

## The mechanical rewrite

Conceptually:

| EP plan | TP plan |
| --- | --- |
| Each GPU holds a disjoint subset of experts | Each GPU holds a TP-shard of every expert |
| Per-token routing → all-to-all | Per-layer all-reduce (no token routing) |
| Per-token bandwidth: high | Per-layer bandwidth: lower |
| Memory: less per GPU (only some experts) | Memory: more per GPU (all experts, sharded) |

In an inference-engine config:

```yaml
# Before (EP)
tensor_parallel_size: 1
expert_parallel_size: 4
moe_routing: standard
moe_all_to_all_backend: deepep   # or pplx, or nvshmem-based

# After (TP)
tensor_parallel_size: 4
expert_parallel_size: 1            # or omit; defaults to 1
disable_expert_parallelism: true
```

This usually works as a drop-in for models where the expert weights split cleanly along the hidden dimension (MLP up_proj, gate_proj, down_proj). For models with shared experts that cross expert boundaries, additional engine support may be needed.

## Memory accounting

Assume an MoE model with:

- L layers
- E experts per layer
- Each expert: weight `W_e` (in NVFP4 ≈ 0.5 bytes per parameter)
- Plus shared (non-expert) weights `W_shared`

**EP=N:**

```
Per-GPU weight = (W_shared) + (E/N) × W_e × L
```

**TP=N:**

```
Per-GPU weight = (W_shared / N) + E × (W_e / N) × L
                = (W_shared + E × W_e × L) / N
                = total_weight / N
```

For most models, **TP=N uses *less* memory per GPU** than EP=N, because shared weights also shard. The memory advantage of EP is illusory once you account for replicated shared weights.

## Bandwidth accounting

Per token, per layer:

**EP**: each token's hidden state (H bytes) goes from current GPU → expert's GPU → back. Two all-to-alls per layer. Total bytes per token per layer: ~2 × H × (number of expert hops needed to cover top-k experts).

**TP**: each layer does one all-reduce of the activation `(B × T × H)`. For batch B = 1, sequence T = 1 (decode), this is `H` bytes per layer, but **distributed across the ring** in pieces. Effective per-layer comm: `H × 2(N-1)/N` bytes through the slowest link.

For typical numbers (H = 8192, N = 4, top-k = 8):

- EP: ~16 × 8 × 8192 = ~1 MB per layer of cross-GPU traffic
- TP: ~12 KB per layer of cross-GPU traffic

A **~80×** difference. This is why TP is dramatically faster on consumer Blackwell.

## Throughput consequences

A model that achieves 100 tok/s in optimal EP on B200 + NVLink might achieve only 5 tok/s in EP on workstation Blackwell (PCIe + host bridge). Same model in TP on workstation Blackwell often hits 50–70 tok/s — a ~10× recovery.

## Some EP-to-TP gotchas

- **Routing kernel changes.** EP launches a routing kernel that picks experts per token. TP doesn't need this — every GPU has every expert, so routing is local. Some engines hardcode the routing kernel; verify the engine actually skips it in TP mode.

- **Activation memory.** TP sometimes increases peak activation memory (each GPU computes the full hidden activation before sharding the next layer). Watch for OOM.

- **Numerical precision.** TP all-reduce is associative-ish; EP is exact (no cross-GPU reduction). For some models, TP introduces subtle numerical differences. Test for output equivalence, not just non-NaN.

- **Microbatching.** EP encourages large per-token batches (to amortize all-to-all). TP doesn't. You may want to revisit microbatch size after the rewrite.

## A pseudocode plan rewriter

```python
def rewrite_ep_to_tp(model_config, num_gpus):
    """
    Take a model config that uses EP and rewrite it for TP-only deployment.
    """
    if model_config.parallelism.ep_size <= 1:
        return model_config    # already TP-only

    new = copy.deepcopy(model_config)
    new.parallelism.tp_size = num_gpus
    new.parallelism.ep_size = 1
    new.parallelism.disable_expert_parallelism = True

    # Some engines have separate flags for the routing path
    new.engine.moe_routing_kernel = "local"   # not "all_to_all"
    new.engine.moe_all_to_all_backend = None

    # Memory budget check
    total_weight = compute_total_weight(model_config)
    per_gpu_after_tp = total_weight / num_gpus
    if per_gpu_after_tp > GPU_MEMORY * 0.94:
        warn("TP may not fit; consider reducing context or using PP")

    return new
```

## When TP-only doesn't fit

If the model is too large for full TP across your GPU count, fall back to:

- **TP × PP hybrid.** Split layers across pairs of GPUs (PP), with TP within each pair.
- **A pruned variant.** REAP-160-style pruning eliminates roughly 1/3 of experts with minimal quality loss.
- **Lower precision.** Mixed-precision (some layers W4A16 via Marlin, others NVFP4) can save memory.

These are graceful degradations, not failures.

## See also

- [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md) — why EP is bandwidth-hungry
- [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md) — why all-to-all is hard on workstation Blackwell
- [`kernels/inference-engines`](../kernels/inference-engines.md) — engine-specific knobs for the rewrite
- [`case-studies/glm-5`](../case-studies/glm-5.md) — a working example of EP-to-TP applied

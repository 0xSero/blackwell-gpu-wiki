# DeepSeek-V3 and V4

The canonical "designed for NVL72, breaks on consumer Blackwell" model family. DeepSeek-V3 (Dec 2024), V3.1, V4 (Q1 2026), V4-Flash (Q2 2026) — all share a similar profile.

## The model

| | DeepSeek-V3 | DeepSeek-V4 | DeepSeek-V4-Flash |
| --- | --- | --- | --- |
| Total parameters | 671 B | ~700 B | ~250 B |
| Active per token | 37 B | ~40 B | ~25 B |
| Architecture | MoE with shared + routed experts | MoE, refined routing | MoE, latency-optimized |
| Number of experts | 256 routed + 1 shared | 256 routed + 1 shared | 64 routed |
| Top-k routing | 8 | 8 | 4 |
| Hidden dim | 7168 | 7168 | 5120 |
| Layers | 61 | ~80 | ~40 |
| Native quantization | FP8 | NVFP4 | NVFP4 |
| Attention | MLA (Multi-head Latent Attention) | MLA | MLA |

DeepSeek's signature is **MLA** (Multi-head Latent Attention) — a low-rank attention variant that reduces KV cache size by ~6× compared to standard MHA. Plus their MoE: aggressive top-k=8 routing into 256 experts.

## What the reference deployment assumes

DeepSeek's published reference deployment for V3/V4 targets:

- **Hardware**: 8× H100 / H200 (DGX) or 16+ × B100/B200 (HGX or NVL72)
- **GEMM**: DeepGEMM (their own library), targeting `sm_100a` for Blackwell
- **Attention**: FlashInfer with custom MLA-aware kernels
- **MoE all-to-all**: DeepEP intranode (NVSHMEM over NVLink) for in-chassis, DeepEP internode (RDMA) for cross-node
- **Parallelism**: EP-heavy (each expert lives on a small group of GPUs), with TP within each expert and PP across model layers
- **Inference engine**: vLLM or sglang (their own forks)

The whole stack is engineered for **EP across NVLink-class fabrics**. Three of the four major dependencies (DeepGEMM `tcgen05`, DeepEP NVSHMEM, FlashInfer NVFP4 GEMM) assume datacenter Blackwell; one (DeepEP NVSHMEM) assumes NVLink.

## What breaks on workstation Blackwell

In rough order of severity:

### 1. DeepGEMM doesn't run

DeepGEMM cubins are `sm_100a` only. Loading them on SM120 fails with "no kernel image."

Fix: substitute CUTLASS NVFP4 GEMM. Inference engines have a `disable_deepgemm` flag.

### 2. DeepEP intranode requires NVLink, internode requires RDMA

Neither is present on a typical workstation Blackwell rig. DeepEP refuses to initialize.

Fix: don't use DeepEP. Configure the inference engine for TP-only.

### 3. EP plan over PCIe is unviable

Even with NCCL fallback all-to-all, EP=N over PCIe Gen4 is bandwidth-bound. For a model with 80 layers and 8-way EP, decode at PCIe speed is on the order of 1–2 tok/s.

Fix: use TP=4 (or whatever your GPU count is) instead of EP. This means each GPU holds **all expert weights** but only processes its TP-slice. Memory cost: experts must fit in `total_VRAM / N` per GPU.

### 4. FlashInfer one-shot MoE a2a needs P2P atomics

If you do try EP with FlashInfer's optimized a2a kernel, it busy-polls on completion atomics that aren't enabled.

Fix: same as 3.

### 5. NVFP4 scale layout

DeepSeek's V4 NVFP4 weights use one specific scale layout. Loading them through a kernel expecting MX-FP4 or differently-laid-out NVFP4 produces silent garbage.

Fix: ensure inference engine's quantization config matches the artifact (check `quantization_config.json` in the model directory and the engine's loader).

### 6. MLA's KV cache quirks

MLA stores keys and values as low-rank latent vectors that get expanded just-in-time. The expansion kernel has `tcgen05` paths in the reference implementation.

Fix: most inference engines have a `mma.sync`-based MLA expansion kernel (the original DeepSeek-V2 paper's reference C++ implementation). It works on SM120, just slower.

## What works on workstation Blackwell

A working configuration:

```yaml
# Conceptually — translate to your engine's flag syntax
weights: NVFP4 (DeepSeek's published artifacts)
kv_cache: FP8 E4M3
attention: Triton-based with kv_splits=64
parallelism:
  tensor_parallel: 4 (= number of GPUs)
  pipeline_parallel: 1
  expert_parallel: 1     # disable EP
gemm_backend: cutlass    # not deepgemm
moe_runner: cutlass      # not flashinfer one-shot
```

For the V4-Flash variant (250 B parameters, 64 experts), this fits comfortably in 4× 96 GB workstation Blackwell with room for ~32k context KV cache. For full V4 (700 B), you'd need pruning (REAP-style) to bring the parameter count down to ~478 B before NVFP4-quantization makes it fit.

## Performance expectations

On a 4× workstation Blackwell rig, with the workaround configuration:

| Model | Expected decode tok/s |
| --- | ---: |
| V4-Flash | 50–80 (smaller model, fits comfortably) |
| V4 (after REAP pruning) | 20–50 (varies with context length) |
| V3 (FP8 weights, no pruning) | 10–30 (FP8 weights are larger than V4's NVFP4) |

Compared to a B100 deployment of V4 at ~200 tok/s on similar hardware count, this is roughly **5–10× slower**. The gap is partly hardware (memory bandwidth, SM count) and partly the lack of optimal kernels (`tcgen05`-based vs `mma.sync`-based GEMM).

## What's special about DeepSeek

A few things that make DeepSeek the canonical "doesn't run on consumer Blackwell" example:

1. **They wrote their own GEMM library** (DeepGEMM) and made it `sm_100a`-only. Most labs use CUTLASS, which has both targets.
2. **MLA is unusual.** Most attention kernels in the wild are standard MHA / GQA; MLA needs special kernel paths that not every library provides.
3. **The model is huge.** Even with NVFP4 you're squeezing every byte.
4. **Reference docs assume datacenter.** Their setup guides have step-by-step instructions for DGX H100; nothing for workstation cards.

If you can run DeepSeek V4 on workstation Blackwell, you can probably run anything.

## See also

- [`kimi-k2`](kimi-k2.md) — similar profile, different team
- [`glm-5`](glm-5.md) — less reliant on `tcgen05`, easier port
- [`compatibility/`](../compatibility/index.md) — general patterns for the workarounds
- DeepSeek-V3 and V4 technical reports
- `deepseek-ai/DeepGEMM`, `deepseek-ai/DeepEP` on GitHub

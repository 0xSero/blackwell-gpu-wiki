# Glossary

Project-specific and Blackwell-specific vocabulary. Cross-linked to the page where each term is explained in depth.

## Compute capability

**Compute capability** — NVIDIA's versioning scheme for SM ISAs, written as `<major>.<minor>` (e.g., `7.0` Volta, `8.0` Ampere, `9.0` Hopper, `10.0` datacenter Blackwell, `12.0` workstation/consumer Blackwell). Same major version generally implies forward-compatible PTX; different major version means features may not exist. See [`fundamentals/cuda-pipeline`](../fundamentals/cuda-pipeline.md).

**`sm_NN`** — the lowercase compiler-flag form of compute capability. `sm_100`, `sm_120`. Bare form: portable subset of the architecture.

**`sm_NNa`** — "architecture-specific accelerated." Allows non-portable instructions (e.g., `sm_100a` enables `tcgen05.*`). Code compiled with the `a` suffix runs only on that exact compute capability — not earlier, not later.

**`sm_NNf`** — "forward-compatible." Restricts the code to instructions that will run on `sm_NN` and any later same-major architecture. Useful for code that needs to work across `sm_120` workstation parts and any future `sm_12N` parts.

## Architectures and codenames

**Blackwell** — NVIDIA's GPU generation, 2024–2026.
- **GB100**: B100/B200 datacenter chips, SM 10.0, HBM3e, NVLink 5.
- **GB202**: workstation/consumer chips (RTX PRO 6000 Workstation, RTX 5090), SM 12.0, GDDR7, no NVLink.

**Hopper** — preceding NVIDIA generation. SM 9.0 (H100/H200). Introduced TMA, async tensor cores (`wgmma.async`), thread block clusters.

**Ampere** — generation before Hopper. SM 8.0/8.6/8.9 (A100, RTX 30/40 series).

**SXM** — NVIDIA's datacenter card form factor. Implies presence of NVLink. (RTX cards are PCIe form factor; the "PRO" line spans both.)

## Memory

**Global memory (HBM/GDDR)** — off-chip device memory. HBM3e on datacenter Blackwell (~3–8 TB/s), GDDR7 on workstation (~1.6 TB/s).

**Shared memory (SMEM)** — on-chip per-block scratchpad. Programmer-managed. Capacities: 99 KiB/block on SM120, 228 KiB/block on SM100. The 99 vs 228 split is one of the most consequential numbers in this wiki. See [`fundamentals/memory-hierarchy`](../fundamentals/memory-hierarchy.md).

**Tensor Memory (TMEM)** — a new on-chip memory class introduced with SM100. Holds Tensor Core accumulators decoupled from registers. **Does not exist on SM120.** See [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md).

**Registers** — per-thread storage. Limit: 255 32-bit registers per thread on most current arches.

**Constant memory** — small, cached, read-only memory class.

**L1 / L2 cache** — on-chip caches in the standard memory hierarchy.

## Tensor Core ISA

**`mma.sync`** — universal Tensor Core MMA instruction, available since Volta. Synchronous: warp blocks until result lands in registers. Operates on small tiles (m16n8k16 / m16n8k32). Available on **both** SM100 and SM120.

**`wgmma.async`** — Hopper's warp-group async MMA. Larger tiles, asynchronous. Mostly superseded by `tcgen05.mma` on Blackwell datacenter; still supported on Hopper.

**`tcgen05.mma`** — Blackwell datacenter MMA family. Asynchronous, large-tile (up to m128n128k64 single-CTA, m256n128k64 CTA-pair), accumulator in TMEM. **Datacenter only.** See [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md).

**`tcgen05.alloc` / `tcgen05.commit` / `tcgen05.cp`** — companion instructions for managing TMEM allocation, completion, and copy-out.

## Number formats

**FP16** — half-precision IEEE float. 1 sign + 5 exponent + 10 mantissa.

**BF16** — brain-float-16. 1 sign + 8 exponent + 7 mantissa. Same range as FP32; less precision.

**FP8 E4M3** — 8-bit float, 4 exponent bits, 3 mantissa. Better precision, smaller range.

**FP8 E5M2** — 8-bit float, 5 exponent, 2 mantissa. Larger range, less precision. Often used for gradients in training.

**FP6** — 6-bit float (E2M3 or E3M2). Less common; appears in some quantization recipes.

**FP4 (E2M1)** — 4-bit float, 1 sign + 2 exponent + 1 mantissa. Tiny range; only useful in **block-quantized** form with a per-block scale.

**MX-FP4** — Open Compute Project Microscaling spec for FP4. 32-element block, per-block FP6 (E3M2) scale.

**NVFP4** — NVIDIA's variant of MX-FP4. **16-element block** (smaller → better dynamic-range tracking), per-block **FP8 (E4M3) scale** (more scale precision than FP6). Native on both SM100 and SM120 Tensor Cores. See [`fundamentals/number-formats`](../fundamentals/number-formats.md).

**TF32** — 19-bit Tensor Core internal format on Ampere+. 1 sign + 8 exponent + 10 mantissa. Used for FP32 matmuls accelerated through Tensor Cores.

## Parallelism plans

**TP (Tensor Parallelism)** — split each weight matrix across N GPUs, all-reduce per layer. Good for GEMM-bound models on any topology.

**PP (Pipeline Parallelism)** — split layers across N GPUs, microbatch through. Good for memory-bound models. Some bubble overhead.

**EP (Expert Parallelism)** — for MoE: each GPU owns a subset of experts, route tokens via all-to-all. Bandwidth-hungry; works well only on NVLink-class fabrics. See [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md).

**DP (Data Parallelism)** — replicate the model, split batch. Common for training, less for inference.

**Hybrid plans** — TP + PP, EP + TP, etc. The pragmatic choice on most rigs.

## Interconnect

**NVLink** — NVIDIA's proprietary high-bandwidth GPU-to-GPU interconnect. Generation 5 (Blackwell): 1.8 TB/s/GPU on NVL72. **Datacenter only.**

**NVSwitch** — NVLink-based switch fabric. Connects 8+ GPUs in a single chassis (DGX, HGX). Gives uniform bandwidth across all pairs.

**MNNVL (Multi-Node NVLink)** — NVL72-class fabric extending NVLink across racks (up to 72 GPUs).

**PCIe** — universal host-side interconnect. Gen4 (16 GB/s/lane), Gen5 (32 GB/s/lane). x16 → 32 GB/s or 64 GB/s per direction.

**P2P (peer-to-peer)** — direct GPU-to-GPU memory access without staging through host RAM. Enabled if GPUs share a switch or root complex.

**Atomics** — atomic memory operations across the interconnect. On consumer GPUs, P2P atomics are software-gated off by default; require BIOS (ACS Disabled) + driver (`RMDisableFeatureDisablement=1`) to enable. See [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md).

**ACS (Access Control Services)** — PCIe feature that, when **enabled**, isolates devices behind separate IOMMU groups and **blocks** P2P atomics. Counterintuitively, disabling ACS is what allows atomics.

**RDMA** — remote-DMA over network (InfiniBand, RoCE). Used for multi-node GPU-to-GPU transfers in datacenters. Not relevant to single-node consumer setups.

## Kernel libraries

**CUTLASS** — NVIDIA's CUDA Templates library. The reference implementation of high-performance GEMM. Templates compile per-architecture; SM100 templates default to `sm_100a`. See [`kernels/cutlass`](../kernels/cutlass.md).

**FlashAttention (FA-2, FA-3)** — Tri Dao's attention kernel. FA-2 portable; FA-3 Hopper-only (a Blackwell port is in development). See [`kernels/flashattention`](../kernels/flashattention.md).

**FlashInfer** — kernel library for serving (attention + MoE). NVFP4 paths exist; some MoE all-to-all kernels need P2P atomics. See [`kernels/flashinfer`](../kernels/flashinfer.md).

**DeepGEMM** — DeepSeek's high-throughput FP8/FP4 GEMM. SM100-only as shipped. See [`kernels/deepgemm`](../kernels/deepgemm.md).

**Marlin** — INT4 GEMM with FP16 activations. Older arch; works on SM120.

**Triton** — DSL compiler for custom kernels. Works on SM120.

**TransformerEngine** — NVIDIA's mixed-precision wrapper library. SM120 support evolving.

**NVSHMEM** — one-sided GPU memory primitives over NVLink. **Requires NVLink** for performance; PCIe fallback exists but is unusably slow.

**DeepEP** — DeepSeek's expert-parallel a2a kernels. Intranode requires NVLink + NVSHMEM; internode requires RDMA.

**vLLM, sglang, TensorRT-LLM** — high-level inference engines. They compose the above libraries. See [`kernels/inference-engines`](../kernels/inference-engines.md).

## Models referenced

**DeepSeek-V3 / V4 / V4-Flash** — DeepSeek's frontier MoE family (671B, with V4 evolutions). Heavy users of `tcgen05`, DeepGEMM, NVSHMEM, EP.

**Kimi-K2 / K2.6** — Moonshot's MoE model family. Similar dependencies.

**GLM-5.0 / 5.1** — Zhipu's MoE family (~478B–744B). Less aggressive on `tcgen05`, but still reference-deployed on SM100.

**Qwen-3 (MoE variants)** — Alibaba's open MoE family.

**REAP** — "REbalanced Activation Pruning," a pruning technique that removes whole experts from a MoE model with minimal quality loss.

## Compilation pipeline

**`nvcc`** — NVIDIA CUDA Compiler. Front-end that drives host C++ compilation and produces PTX/cubin.

**`ptxas`** — PTX assembler. Lowers PTX to SASS (cubin).

**`cuobjdump`** — inspector for compiled CUDA binaries.

**`nvdisasm`** — disassembler for SASS.

**SASS** — NVIDIA's per-architecture machine code. Not portable across SM versions.

**Cubin** — compiled CUDA binary. Contains SASS for one or more architectures, plus optional PTX for JIT.

**JIT** — just-in-time. The driver can compile PTX to SASS at load time if no matching cubin section exists.

## Other

**KV cache** — key-value cache in attention. Stores past tokens' K and V projections so attention is O(N) per new token rather than O(N²).

**Page-attention / paged KV** — block-based KV cache management (vLLM-style).

**MTP (Multi-Token Prediction)** — speculative-decoding scheme that predicts multiple tokens in parallel.

**NSA (Native Sparse Attention)** — DeepSeek's sparse-attention variant.

**REAP-NN** — pruning denoting how many experts (per layer) survive (e.g., REAP-160 = 160 experts kept out of 256).

**Watchdog** — sglang/vLLM background thread that kills the server if a forward pass takes too long.

**xid** — NVIDIA driver error code (e.g., Xid 79 = GPU has fallen off the bus).

**AER (Advanced Error Reporting)** — PCIe link-layer error reporting. RxErr counters on stressed Gen4 links.

**Bus / function / device IDs** — PCIe addressing, e.g., `01:00.0`. May change after BIOS settings change.

## See also

- [`reference/bibliography`](../reference/bibliography.md) for primary sources
- [`reference/abbreviations`](../reference/abbreviations.md) for the alphabet-soup version

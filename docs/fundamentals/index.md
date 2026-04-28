# Fundamentals

The minimum context needed before the rest of the wiki makes sense. If you've written CUDA kernels and you're comfortable reading PTX listings, skim these pages and move on. If you haven't, read carefully — the rest of the wiki assumes the vocabulary developed here.

## Pages in this section

- [`gpu-execution-model`](gpu-execution-model.md) — warps, blocks, SMs, grid hierarchy, the SIMT abstraction
- [`memory-hierarchy`](memory-hierarchy.md) — registers, SMEM, L1/L2, global memory, the bandwidth pyramid
- [`cuda-pipeline`](cuda-pipeline.md) — `nvcc` → PTX → `ptxas` → cubin → driver → SASS
- [`tensor-cores`](tensor-cores.md) — what Tensor Cores are, the generations, `mma.sync` and `wgmma.async` before Blackwell
- [`number-formats`](number-formats.md) — FP16/BF16, FP8 (E4M3/E5M2), FP6, FP4 and the MX-FP4/NVFP4 family

## What's not here

- **CUDA C++ language details.** This wiki uses CUDA C++ as input to the compilation pipeline; it doesn't teach the language itself. NVIDIA's *CUDA C++ Programming Guide* is the canonical source.
- **General GPU programming patterns.** Tiling, occupancy, register pressure — taught indirectly through Blackwell-specific examples in later sections.
- **Performance tooling.** Nsight Systems, Nsight Compute. Mentioned in passing in [`overview/getting-started`](../overview/getting-started.md), not taught here.
- **History before Volta.** The wiki begins with Volta because that's when Tensor Cores became a first-class concept and `mma.sync` entered the ISA.

## Why this section exists

The Blackwell-specific story (the next section) only makes sense if you know what's *not* changing across the SM100/SM120 split. Tensor Cores still execute MMA. Shared memory still has banks and bank conflicts. The CUDA compilation pipeline still produces PTX-then-SASS. What changes is *which* PTX instructions exist, *how much* shared memory is available, *which* tile shapes work — all changes that would be invisible if you didn't know what was being changed.

These pages establish the baseline. Later sections describe deltas from that baseline.

## Suggested order

1. [`gpu-execution-model`](gpu-execution-model.md) — the abstract machine
2. [`memory-hierarchy`](memory-hierarchy.md) — where data lives
3. [`tensor-cores`](tensor-cores.md) — what the matrix-multiply hardware does
4. [`number-formats`](number-formats.md) — what the matrix-multiply hardware operates *on*
5. [`cuda-pipeline`](cuda-pipeline.md) — how source code becomes hardware instructions

Most readers find pages 1–4 to be review and read 5 carefully. If that's you, jump straight to [`cuda-pipeline`](cuda-pipeline.md).

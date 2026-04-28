# Getting started

Prerequisites and reading paths for the wiki itself. This page is meta — it tells you what you need before each section makes sense.

## Reading prerequisites

| If you already know... | You can skip... |
| --- | --- |
| GPU programming basics: warps, blocks, SMs, shared memory | most of [`fundamentals/`](../fundamentals/index.md), but skim the memory-hierarchy page for Blackwell-specific numbers |
| CUDA C++ + PTX: how `nvcc` produces a binary, what `ptxas` does | [`fundamentals/cuda-pipeline`](../fundamentals/cuda-pipeline.md) |
| Tensor Core programming through Hopper (`wgmma.async`, TMA) | [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md), but read [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) carefully — it's all new |
| Number formats (FP16, BF16, FP8 E4M3/E5M2, FP4) | [`fundamentals/number-formats`](../fundamentals/number-formats.md) |
| MoE inference at production scale (DeepEP, NVSHMEM, EP/TP/PP) | [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md) |

If none of those apply, read top-to-bottom in sidebar order. The sidebar is built so each page introduces only what the next page assumes.

## What you don't need to set up

You can read this entire wiki without touching a GPU. Every concept is illustrated through PTX listings, ISA tables, and reasoning about hardware — no code is run, no kernels are launched.

That's intentional. The wiki teaches the **shape** of the problem (ISA mismatches, SMEM ceilings, atomic-bus topology) — not how to operate any specific tool. Tools change every two months; the shape doesn't.

## What you need if you want to go beyond reading

If you want to **verify** the claims in this wiki on real hardware, you'll want one of:

- **An SM 10.0 GPU**: B100, B200, or B300 (datacenter Blackwell). Available on cloud providers (Lambda, RunPod, Together, AWS p5e, etc.). For the case studies and kernel-library deep-dives.
- **An SM 12.0 GPU**: RTX PRO 6000 Workstation, RTX 5090, or RTX 5080. Available for purchase, or rentable on some workstation-cloud providers. For verifying that the things this wiki says fail, actually fail.

You don't need both, but having both side-by-side is the only way to truly internalize what changes between them. Most concrete numbers and observations in this wiki were derived by comparing the two.

For tooling: any recent CUDA toolkit (12.5+) on either card will let you reproduce the PTX generation and inspection examples. Specifically you'll use:

- `nvcc` for compilation
- `ptxas` for the PTX-to-cubin step
- `cuobjdump` for inspecting compiled binaries
- `nvdisasm` for SASS disassembly
- `nvidia-smi` for topology and bus-level information
- `nsys` and `ncu` (Nsight Systems/Compute) for performance attribution

None of these are required to follow the wiki. They're optional companions if you want to leave the page and verify something in your own console.

## Suggested learning paths

Three paths through the material, depending on your goal:

### "I want to read top-to-bottom"

Read in sidebar order. Total time: ~3–4 hours for a careful first read, ~2 hours on a re-read. Each section ends with a "checkpoint" question — if you can answer it, move on; if not, re-read.

### "I have a specific model that won't run"

1. Read [`overview/index`](index.md) and [`overview/architecture`](architecture.md) (the two pages you're already near).
2. Find your model in [`case-studies/`](../case-studies/index.md) (DeepSeek-V3/V4, Kimi-K2, GLM-5, generic).
3. The case study lists the exact features the model assumes. For each, follow the link to the page that explains why it doesn't work on consumer Blackwell.
4. Read [`compatibility/`](../compatibility/index.md) for what the alternatives look like.

Time: ~1 hour for a focused read, plus deep-dives as you follow links.

### "I'm a CUDA/kernel developer porting work between SM100 and SM120"

1. [`blackwell/sm100-vs-sm120`](../blackwell/sm100-vs-sm120.md) — the diff, in detail.
2. [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) — the new datacenter-only ISA family.
3. [`kernels/cutlass`](../kernels/cutlass.md) — the most common library you'll port.
4. [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) — patterns for rewriting `tcgen05` as `mma.sync`.
5. [`compatibility/smem-budget-management`](../compatibility/smem-budget-management.md) — patterns for fitting the 99 KiB ceiling.

Time: ~2 hours, plus the source-code reading the wiki points you to.

## Checkpoints

You'll know you've gotten value from the wiki when you can answer the following without re-reading:

- Why doesn't `tcgen05.mma` work on a workstation Blackwell card?
- What's the per-block shared-memory ceiling on workstation Blackwell, and what's it on datacenter Blackwell?
- Why does an MoE inference plan that runs at 49 tok/s on NVL72 collapse to 1.4 tok/s on a 4-way PCIe topology?
- What two settings do you need to enable PCIe atomics on a consumer Blackwell rig?
- What does `sm_120f` mean, and how is it different from `sm_120a`?

If those are mysteries, the wiki has work to do for you.

## Where to ask questions

This wiki is a static reference; it doesn't have a discussion forum. If you find an error or have a question:

- For NVIDIA-specific questions: NVIDIA Developer Forums, the CUTLASS / FlashInfer GitHub issue trackers
- For ML-systems questions: the inference-engine project's issue tracker (sglang, vLLM, TRT-LLM)
- For broader architecture questions: the Frontier Forge, GPU MODE, or Discord communities for ML systems

The bibliography in [`reference/bibliography`](../reference/bibliography.md) lists every primary source the wiki draws from; consult those before assuming the wiki itself is the authoritative source on any specific claim.

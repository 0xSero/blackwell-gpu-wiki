# tcgen05 and Tensor Memory

The new datacenter-Blackwell-only Tensor Core ISA family, and the on-chip memory class it depends on.

## What `tcgen05` is

`tcgen05` is a family of PTX instructions introduced in PTX ISA 8.4 (with refinements in 8.5), targeting compute capability 10.0 only (i.e., SM100). The `5` denotes Tensor Core generation 5; the `gen05` denotes generation-5-specific. Its design goals:

1. **Decouple Tensor Core execution from warp execution.** The warp issues an MMA and continues; the Tensor Core runs to completion in parallel.
2. **Support larger MMA tiles than `wgmma.async`.** Up to 128×128 single-CTA, 256×128 CTA-pair.
3. **Reduce register-file bandwidth pressure.** Accumulators land in TMEM, not registers.

These goals together produce roughly **2–3× peak FP4/FP6/FP8 throughput** relative to a `wgmma.async`-based kernel on the same SM.

## The instructions

| Instruction | Role |
| --- | --- |
| `tcgen05.alloc.cta_group::N %dst, N` | Allocate N bytes of TMEM, return base address in `%dst` |
| `tcgen05.dealloc %addr, N` | Free N bytes at `%addr` |
| `tcgen05.relinquish_alloc_permit` | Inform the runtime that this CTA won't allocate more TMEM |
| `tcgen05.cp.shared::cta::tmem.b64 [%tmem], [%smem]` | Copy from SMEM to TMEM |
| `tcgen05.cp.tmem.shared::cta.b64 [%smem], [%tmem]` | Copy from TMEM to SMEM |
| `tcgen05.shift [%tmem], shift_amount` | Logical shift within a TMEM allocation (used for layout transforms) |
| `tcgen05.mma.cta_group::1.kind::<dtype>` | Single-CTA MMA |
| `tcgen05.mma.cta_group::2.kind::<dtype>` | CTA-pair MMA |
| `tcgen05.commit.cta_group::N %sema` | Commit a barrier to wait on all outstanding MMAs |
| `tcgen05.wait.cta_group::N %sema` | Wait on a previously-committed barrier |

`<dtype>` enumerates supported MMA kinds: `f4`, `mxf4`, `nvf4`, `f6`, `f8f6f4`, `f8`, `f16`, `bf16`, `tf32`, etc.

## A complete `tcgen05` MMA in PTX

A datacenter-Blackwell GEMM tile, simplified:

```ptx
.reg .b64 %tmem_base;
.reg .b32 %sema;

// 1. Allocate 16 KB of TMEM for accumulator
tcgen05.alloc.cta_group::1 %tmem_base, 16384;

// 2. (Operands A and B already staged in SMEM via TMA)

// 3. Issue MMA: D = A * B (FP4 inputs, FP32 accumulator in TMEM)
tcgen05.mma.cta_group::1.kind::nvf4
    [%tmem_base],          // accumulator
    [%smem_a],             // operand A (in SMEM)
    [%smem_b],             // operand B (in SMEM)
    %scale_a, %scale_b;    // NVFP4 scale registers

// 4. Commit barrier
tcgen05.commit.cta_group::1 %sema;

// 5. Continue doing other work while MMA runs
// ... more code ...

// 6. Wait for MMA completion
tcgen05.wait.cta_group::1 %sema;

// 7. Copy result from TMEM to SMEM for downstream consumption
tcgen05.cp.tmem.shared::cta.b64 [%smem_out], [%tmem_base];

// 8. Free TMEM
tcgen05.dealloc %tmem_base, 16384;
tcgen05.relinquish_alloc_permit;
```

The contrast with `wgmma.async`:

- `wgmma` accumulators live in **registers**; `tcgen05` accumulators live in **TMEM**
- `wgmma` is issued by a **warp group** (4 warps); `tcgen05` is issued by a **single warp** (or a CTA pair)
- `wgmma` tiles max out at m64n256k16; `tcgen05` tiles go to m128n128k64 (single CTA) or m256n128k64 (CTA pair)
- Both are async; both have commit/wait barriers

## Tensor Memory (TMEM)

A new on-chip memory class. Properties:

- **Capacity**: 256 KB per SM
- **Allocation granularity**: 128 bytes
- **Allocator**: `tcgen05.alloc` returns a TMEM base address, `tcgen05.dealloc` frees it
- **Addressing**: TMEM addresses are **separate** from SMEM and global addresses — they're 32-bit logical addresses within the per-SM TMEM region
- **Bandwidth**: high enough to feed `tcgen05.mma` at peak rates
- **Visibility**: TMEM is per-SM; the issuing CTA (or CTA pair) can address it; other CTAs can't

TMEM exists for one specific reason: at the throughput levels of FP4/FP6 MMA, **register file bandwidth becomes the bottleneck** for a `wgmma.async`-style kernel. The Tensor Core wants to consume operands faster than a warp's worth of register reads can supply. By moving accumulators out of registers (and operand-staging out of registers, into SMEM and then TMEM), the warp's register file is free to serve only the **issue and finalize** stages, not the running MMA.

A useful mental model: TMEM is to Tensor Cores what L1 cache is to ALUs.

### TMEM layouts

`tcgen05` supports several TMEM layouts:

- **Default**: row-major within each 32-element band
- **Strided**: configurable stride for transposed access
- **Replicated**: one logical operand replicated across multiple TMEM regions for CTA-pair MMA
- **Compressed**: NVFP4-aware layout that interleaves values and scales

The `tcgen05.shift` instruction transforms between layouts in place.

## CTA-pair / `cta_group::2` mode

The largest `tcgen05.mma` tile (m256n128k64) is too big to fit a single CTA's TMEM budget. `cta_group::2` mode uses two CTAs cooperating:

- Both CTAs are launched as part of the same **cluster** (`.cluster_dim 2,1,1`)
- They share TMEM allocations across the cluster's SMEM-link
- A single `tcgen05.mma.cta_group::2` instruction is issued (by both CTAs in lock-step) and produces a tile twice the size of single-CTA mode

This is one of the reasons SM100 supports thread block clusters > 1: `tcgen05` CTA-pair mode requires it.

**Workstation Blackwell does not support clusters > 1, and therefore cannot use `tcgen05` CTA-pair MMA at all.** A kernel compiled for SM120 must use single-CTA tile shapes only — or, more typically, must not use `tcgen05` at all.

## Why workstation Blackwell doesn't have `tcgen05`

NVIDIA's likely reasoning (inferred from the architecture):

- TMEM costs significant die area (256 KB/SM is real silicon)
- Cluster execution requires extra SM-to-SM linkage (more silicon)
- Consumer workloads (gaming, content creation, light ML) get little benefit from m128n128k64 GEMMs
- Differentiating datacenter from consumer is a deliberate product strategy

The result: workstation Blackwell has the **same Tensor Core hardware** (gen 5, native FP4/FP6/FP8) but accesses it only through `mma.sync` and `wgmma.async`, both of which are register-bound. So peak FP4 throughput per SM is similar to Hopper-FP8 throughput per SM — useful, but not the 2–3× generational jump that SM100 sees.

## What runs on what, with examples

Concrete examples of code that does and does not run:

```ptx
// ✓ Runs on SM 9.0, 10.0, 12.0 — universal
mma.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 ...;

// ✓ Runs on SM 9.0, 10.0, 12.0 — but lower throughput on 12.0
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 ...;

// ✓ Runs on SM 10.0 only
tcgen05.mma.cta_group::1.kind::nvf4 ...;

// ✓ Runs on SM 10.0 only
tcgen05.mma.cta_group::2.kind::nvf4 ...;

// ✓ Runs on SM 10.0 only
tcgen05.cp.shared::cta::tmem.b64 ...;
```

If you compile any of the last three for `--gpu-name=sm_120`, `ptxas` errors:

```
ptxas fatal: Internal error: instruction 'tcgen05.mma' not supported in this PTX version
```

## How libraries handle this

CUTLASS exposes the choice:

```cpp
// CUTLASS Blackwell datacenter template — uses tcgen05
using GemmKernel = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,                    // ← architecture choice
    ...,
    cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename CollectiveOp::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;
```

Switching to `cutlass::arch::Sm120` selects a parallel template tree that uses `mma.sync` and `wgmma.async`, with smaller tile shapes that fit the 99 KiB SMEM ceiling.

DeepGEMM, by contrast, currently has only an `Sm100`-targeted code path (as of early 2026); a `Sm120` port is in progress but not landed. Loading DeepGEMM kernels on workstation Blackwell fails at runtime.

FlashInfer has separate Triton-based and CUTLASS-based attention kernels; the CUTLASS-Blackwell path uses `tcgen05`, the Triton path doesn't, so workstation Blackwell falls back to the Triton path with reduced throughput.

## Translating `tcgen05` to `mma.sync`

If you have an SM100-only kernel and need it on SM120, the conceptual translation:

| SM100 op | SM120 equivalent |
| --- | --- |
| `tcgen05.alloc N` | `__shared__` allocation of N bytes (counts against 99 KiB) |
| `tcgen05.cp.shared::cta::tmem` | `cp.async.bulk` or simple SMEM staging |
| `tcgen05.mma.cta_group::1.kind::nvf4 m128n128k64` | ~256 sequential `mma.sync m16n8k32` instructions, accumulating in registers |
| `tcgen05.commit` | `bar.sync` or just the last `mma.sync`'s completion |
| `tcgen05.cp.tmem.shared::cta` | direct register-to-SMEM store |
| `tcgen05.dealloc` | scope end |

The translation is **mechanical** but produces **substantially more PTX** — roughly 256× as many instructions for the largest tile. The achieved Tensor Core throughput per SM ends up around 40–70 % of optimal SM120 throughput (which is itself a fraction of optimal SM100 throughput). See [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) for the detailed pattern.

## Checkpoint

You should be able to answer:

- What's TMEM and how is it different from SMEM and registers?
- Why does `tcgen05.mma` exist when `wgmma.async` already provided async MMA?
- What does `cta_group::2` mean?
- Why does SM120 not have `tcgen05`?
- Roughly how does SM120 throughput compare to SM100 for FP4 GEMM?

## See also

- [`sm100-vs-sm120`](sm100-vs-sm120.md) — the full architectural diff
- [`thread-block-clusters`](thread-block-clusters.md) — clusters and CTA-pair MMA
- [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md) — `mma.sync` and `wgmma.async` background
- [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) — porting patterns
- *NVIDIA PTX ISA 8.5*, "TensorCore instructions" → "tcgen05 family"
- *NVIDIA Blackwell Architecture Whitepaper*, "Fifth-Generation Tensor Cores"

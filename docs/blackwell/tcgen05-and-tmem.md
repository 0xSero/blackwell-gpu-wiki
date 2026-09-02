# tcgen05 and Tensor Memory

The new datacenter-Blackwell-only Tensor Core ISA family, and the on-chip memory class it depends on.

## What `tcgen05` is

`tcgen05` is a family of PTX instructions introduced in PTX ISA 8.6 (shipped with CUDA 12.8, extended in later releases), targeting datacenter Blackwell's architecture-specific targets only (`sm_100a` and family members such as `sm_103a`). The `5` denotes Tensor Core generation 5; the `gen05` denotes generation-5-specific. Its design goals:

1. **Decouple Tensor Core execution from warp execution.** The warp issues an MMA and continues; the Tensor Core runs to completion in parallel.
2. **Support larger MMA tiles than `wgmma.async`.** Up to M=128, N=256 single-CTA; M=256, N=256 CTA-pair.
3. **Reduce register-file bandwidth pressure.** Accumulators land in TMEM, not registers.

These goals together produce roughly **2–3× peak FP4/FP6/FP8 throughput** relative to a `wgmma.async`-based kernel on the same SM.

## The instructions

Some qualifiers are omitted for readability; `N` is `1` or `2`.

| Instruction | Role |
| --- | --- |
| `tcgen05.alloc.cta_group::N.sync.aligned.shared::cta.b32 [dst], nCols` | Allocate `nCols` **columns** of TMEM (a power of 2 from 32 to 512) and write the TMEM address to SMEM at `[dst]`; the whole warp executes it |
| `tcgen05.dealloc.cta_group::N.sync.aligned.b32 taddr, nCols` | Free the allocation |
| `tcgen05.relinquish_alloc_permit.cta_group::N.sync.aligned` | Declare that this CTA won't allocate again, so other CTAs on the SM can get TMEM |
| `tcgen05.ld.sync.aligned.<shape>.x<n>.b32 {regs}, [taddr]` | TMEM → registers (warp-collective) |
| `tcgen05.st.sync.aligned.<shape>.x<n>.b32 [taddr], {regs}` | Registers → TMEM |
| `tcgen05.wait::ld.sync.aligned` / `tcgen05.wait::st.sync.aligned` | Wait for this thread's earlier `ld` / `st` to complete |
| `tcgen05.cp.cta_group::N.<shape> [taddr], sdesc` | SMEM → TMEM (the only direction; typically used to stage block scales into TMEM) |
| `tcgen05.shift.cta_group::N.down [taddr]` | Shift TMEM data down by 32 lanes (used with the weight-stationary `tcgen05.mma.ws`) |
| `tcgen05.mma.cta_group::N.kind::<kind> [dtmem], adesc, bdesc, idesc, enable_input_d` | MMA: D(TMEM) = A×B (+D). A and B are given by SMEM matrix descriptors (A may also live in TMEM); `idesc` encodes shape and data types; **issued by a single thread** |
| `tcgen05.mma.cta_group::N.kind::mxf4nvf4.block_scale.scale_vec::4X … [scale_a], [scale_b]` | Block-scaled MMA; the scale factors live in TMEM |
| `tcgen05.commit.cta_group::N.mbarrier::arrive::one.shared::cluster.b64 [mbar]` | Batch all `tcgen05` async ops this thread issued so far, and arrive once on the mbarrier when they all complete |
| `tcgen05.fence::before_thread_sync` / `tcgen05.fence::after_thread_sync` | Order `tcgen05` async ops against ordinary thread synchronization (`bar.sync`, mbarriers) |

`<kind>` enumerates the supported MMA kinds: `f16` (FP16/BF16 inputs), `tf32`, `f8f6f4` (mixed FP8/FP6/FP4), `i8`, `mxf8f6f4`, `mxf4`, `mxf4nvf4` (MXFP4 and NVFP4, block-scaled).

## A complete `tcgen05` MMA in PTX

A datacenter-Blackwell GEMM tile, simplified. This is illustrative: descriptor construction and mbarrier initialization are elided.

```ptx
.shared .b32 tmem_slot;          // alloc writes the TMEM address here
.shared .b64 mma_bar;            // mbarrier
.reg .b32 %taddr, %idesc, %r<32>;
.reg .b64 %adesc, %bdesc;
.reg .pred %acc;

// 1. Allocate 128 TMEM columns: 128 lanes × 128 columns × 4 B = 64 KB, one m128n128 FP32 accumulator.
//    The whole warp executes this. Relinquish right away so other CTAs can take the rest of TMEM.
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [tmem_slot], 128;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
tcgen05.fence::before_thread_sync;
bar.sync 0;                                   // make the address visible to the other warps
tcgen05.fence::after_thread_sync;
ld.shared.b32 %taddr, [tmem_slot];

// 2. (Operands A and B already staged in SMEM via TMA; %adesc / %bdesc are their matrix
//     descriptors, %idesc encodes M/N/K, data types and layout)

// 3. A single thread issues the MMA: D(TMEM) = A × B (+ D). %acc is false on the first
//    K-step so the old D is not read in.
tcgen05.mma.cta_group::1.kind::f16 [%taddr], %adesc, %bdesc, %idesc, %acc;

// 4. Batch the MMAs issued so far; arrive once on mma_bar when they have all completed
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mma_bar];

// 5. The warp is free to do other work while the MMA runs (e.g. kick off the next TMA load)

// 6. Wait on the mbarrier (an ordinary mbarrier.try_wait loop, elided)

// 7. Read the accumulator from TMEM into registers (each warp can only read its own 32 lanes),
//    then store to SMEM or global memory
tcgen05.ld.sync.aligned.32x32b.x32.b32 {%r0, ..., %r31}, [%taddr];
tcgen05.wait::ld.sync.aligned;

// 8. Free TMEM
tcgen05.dealloc.cta_group::1.sync.aligned.b32 %taddr, 128;
```

The contrast with `wgmma.async`:

- `wgmma` accumulators live in **registers**; `tcgen05` accumulators live in **TMEM**
- `wgmma` is issued by a **warp group** (4 warps); `tcgen05.mma` is issued by a **single thread** (`alloc` / `ld` / `st` are executed collectively by one warp)
- `wgmma` tiles max out at m64n256k16; `tcgen05` tiles go to M=128, N=256 (single CTA) or M=256, N=256 (CTA pair), with K set by the element width (16 for FP16, 32 for FP8, 64 for FP4)
- Both are async: `wgmma` uses `commit_group` / `wait_group`, `tcgen05` uses `commit` to signal completion on an mbarrier

## Tensor Memory (TMEM)

A new on-chip memory class. Properties:

- **Capacity**: 256 KB per SM, organized as 128 lanes × 512 columns of 32-bit cells
- **Allocation granularity**: by column, a power of 2 from 32 to 512 columns per allocation (32 columns = 16 KB)
- **Allocator**: `tcgen05.alloc` returns a TMEM base address, `tcgen05.dealloc` frees it
- **Addressing**: TMEM addresses are **separate** from SMEM and global addresses — they're 32-bit logical addresses with the lane in the high 16 bits and the column in the low 16 bits
- **Bandwidth**: high enough to feed `tcgen05.mma` at peak rates
- **Visibility**: TMEM is per-SM; the issuing CTA (or CTA pair) can address it; other CTAs can't

TMEM exists for one specific reason: at the throughput levels of FP4/FP6 MMA, **register file bandwidth becomes the bottleneck** for a `wgmma.async`-style kernel. The Tensor Core wants to consume operands faster than a warp's worth of register reads can supply. By moving accumulators out of registers (and operand-staging out of registers, into SMEM and then TMEM), the warp's register file is free to serve only the **issue and finalize** stages, not the running MMA.

A useful mental model: TMEM is to Tensor Cores what L1 cache is to ALUs.

### How TMEM is organized

TMEM is a two-dimensional array of 128 lanes × 512 columns. Row i of the accumulator D lives in lane i (M=128 fills every lane, M=64 fills half of them), and N is the number of columns it occupies (one value per column for an FP32 accumulator). So an m128n256 FP32 accumulator takes 256 columns — half of TMEM.

`tcgen05.ld` / `tcgen05.st` move data in a handful of fixed shapes (`32x32b`, `16x64b`, `16x128b`, `16x256b`), and a warp can only touch the 32 lanes that correspond to its position in the warpgroup (warp 0 owns lanes 0–31, warp 1 owns lanes 32–63, and so on). That is why epilogues are always done by four warps together.

`tcgen05.shift.down` shifts the contents of a TMEM region down by 32 lanes. It exists for the weight-stationary form of the MMA (`tcgen05.mma.ws`); an ordinary GEMM never uses it.

## CTA-pair / `cta_group::2` mode

An M=256 `tcgen05.mma` tile doesn't fit one CTA: its TMEM only has 128 lanes. `cta_group::2` mode uses two CTAs cooperating:

- Both CTAs are launched as part of the same **cluster** (cluster dimension 2)
- Both CTAs' SMEM and TMEM take part: A and B are read half from each CTA's SMEM, and the 256 rows of D land in the two CTAs' TMEM (128 lanes each)
- A single `tcgen05.mma.cta_group::2` instruction is issued once, by one thread in the leader CTA; the matching `commit` can arrive on mbarriers in both CTAs

This is one of the reasons SM100 supports thread block clusters > 1: `tcgen05` CTA-pair mode requires it.

**Workstation Blackwell does not support clusters > 1, and therefore cannot use `tcgen05` CTA-pair MMA at all.** A kernel compiled for SM120 must use single-CTA tile shapes only — or, more typically, must not use `tcgen05` at all.

## Why workstation Blackwell doesn't have `tcgen05`

NVIDIA's likely reasoning (inferred from the architecture):

- TMEM costs significant die area (256 KB/SM is real silicon)
- Cluster execution requires extra SM-to-SM linkage (more silicon)
- Consumer workloads (gaming, content creation, light ML) get little benefit from m128n256k64 GEMMs
- Differentiating datacenter from consumer is a deliberate product strategy

The result: workstation Blackwell has the **same Tensor Core hardware** (gen 5, native FP4/FP6/FP8) but accesses it only through `mma.sync` and `wgmma.async`, both of which are register-bound. So peak FP4 throughput per SM is similar to Hopper-FP8 throughput per SM — useful, but not the 2–3× generational jump that SM100 sees.

## What runs on what, with examples

Concrete examples of code that does and does not run:

```ptx
// ✓ Runs on SM 9.0, 10.0, 12.0 — universal
mma.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 ...;

// ✓ Runs on SM 9.0, 10.0, 12.0 — but lower throughput on 12.0
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 ...;

// ✓ Runs on SM 10.0 (sm_100a) only
tcgen05.mma.cta_group::1.kind::mxf4nvf4 ...;

// ✓ Runs on SM 10.0 (sm_100a) only
tcgen05.mma.cta_group::2.kind::mxf4nvf4 ...;

// ✓ Runs on SM 10.0 (sm_100a) only
tcgen05.ld.sync.aligned.32x32b.x32.b32 ...;
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
| `tcgen05.alloc nCols` | accumulator in registers, with whatever doesn't fit in `__shared__` (counts against 99 KiB) |
| `tcgen05.cp` (SMEM → TMEM, for block scales) | read the scales straight from SMEM into registers |
| `tcgen05.mma.cta_group::1.kind::mxf4nvf4 m128n128k64` | 128 `mma.sync m16n8k64` (`kind::mxf4nvf4.block_scale`) instructions, 32 per warp across 4 warps, accumulating in registers |
| `tcgen05.commit` + mbarrier | nothing — `mma.sync` is synchronous; when it returns, it's done |
| `tcgen05.ld` (TMEM → registers) | nothing — the result is already in registers |
| `tcgen05.dealloc` | scope end |

The translation is **mechanical** but produces **substantially more PTX** — the largest single-CTA tile (m128n256k64) becomes 256 `mma.sync` instructions. The achieved Tensor Core throughput per SM ends up around 40–70 % of optimal SM120 throughput (which is itself a fraction of optimal SM100 throughput). See [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) for the detailed pattern.

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
- *NVIDIA PTX ISA* (8.6 and later), "TensorCore 5th Generation Instructions"
- *NVIDIA Blackwell Architecture Whitepaper*, "Fifth-Generation Tensor Cores"

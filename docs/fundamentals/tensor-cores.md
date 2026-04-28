# Tensor Cores

Specialized execution units inside each SM that perform matrix-multiply-accumulate (MMA) operations on small tiles in a single instruction. The reason GPUs are useful for transformer inference.

## What they do, and why

A normal warp-level instruction operates on 32 32-bit elements in parallel — 32 SIMT lanes, one element each. A Tensor Core instruction operates on a **matrix tile**: a small block of 16×8 or 16×16 elements, computing the entire `D = A · B + C` for that tile in a single op.

Throughput improves substantially. For BF16 inputs and FP32 accumulator on Hopper:

- Plain FP32 FMA: ~30 TFLOPs/GPU
- BF16 Tensor Core MMA: ~990 TFLOPs/GPU

Roughly 30×. For FP4 on Blackwell datacenter the speedup is even greater (~165× over plain FP32).

The Tensor Core itself is invisible to the programmer beyond a special instruction. There's no separate "Tensor Core memory" (well, there is now — TMEM, but that's an SM100 thing — see below) — operands come from registers (or SMEM, or TMEM), results land in registers (or TMEM).

## The five generations

| Gen | Architecture | Compute capability | Key features |
| --- | --- | --- | --- |
| 1 | Volta | 7.0 (V100) | FP16 input, FP32 accum. `mma.sync` introduced. |
| 2 | Turing | 7.5 (T4, RTX 20) | + INT8, INT4, INT1 |
| 3 | Ampere | 8.0–8.9 (A100, RTX 30) | + BF16, TF32. `ldmatrix` for fast SMEM loads. |
| 4 | Hopper | 9.0 (H100/H200) | + FP8 (E4M3, E5M2). `wgmma.async` (warp-group async MMA). TMA. Thread block clusters. |
| 5 | Blackwell | 10.0 / 12.0 | + FP6, FP4, MX-FP4, NVFP4. `tcgen05` family (SM100 only). TMEM (SM100 only). |

This wiki is mostly about generation 5. The earlier generations are mentioned for context — you'll see references to `mma.sync` and `wgmma.async` in modern kernel code, especially as fallback paths.

## The MMA instruction families

Three relevant families. All are PTX-level constructs; you rarely write them directly in CUDA C++ but you'll see them in compiled PTX and in CUTLASS templates.

### `mma.sync` — universal, since Volta

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  %rd0, %rd1, %rd2, %rd3;
//                ^^^^^^^^^         ^^^^ ^^^^ ^^^^
//                tile shape         A     B    C/D
//                                  type type type
```

Parses as: "synchronously perform an MMA on a m16n8k16 tile (16 rows, 8 cols, 16 inner-product depth), with row-major A and col-major B, A and B in BF16, accumulator in FP32, output in registers `%rd0`, operands in `%rd1`, `%rd2`, `%rd3`."

Properties:

- **Synchronous**: warp issues the instruction and **all 32 threads in the warp must participate**. Result is in registers immediately.
- **Small tiles**: m16n8k16 (BF16/FP16), m16n8k32 (FP8/FP4). Larger logical tiles are issued as multiple `mma.sync` instructions in sequence.
- **Universal**: works on every NVIDIA GPU from Volta forward. **Works on both SM100 and SM120.**

Roughly the workhorse of all tensor-core code from Volta through Ampere, and the fallback path on every architecture. Most PTX you see in practice contains `mma.sync` instructions.

### `wgmma.async` — Hopper introduction

```ptx
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16  %rd0, %rd1, %rd2, %rd3;
```

Properties:

- **Asynchronous**: warp issues the instruction, continues executing other work; result lands later. Synchronization via `wgmma.commit_group.sync` and `wgmma.wait_group.sync`.
- **Warp-group**: a *warp group* is 4 warps (128 threads). The MMA is issued by the warp group, not a single warp.
- **Larger tiles**: m64n128k16 to m64n256k16, vs. mma.sync's m16n8k16. Fewer instructions issued for the same logical work.
- **Hopper and Hopper-aware Blackwell**: works on SM 9.0 and on SM 10.0 (datacenter Blackwell). **On SM 12.0 it works but Tensor Core throughput is lower than on datacenter parts.**

`wgmma.async` introduced async-everything-on-the-warp-group. Modern Hopper kernels (FA-3, CUTLASS Hopper templates) lean heavily on it.

### `tcgen05.mma` — Blackwell datacenter only

```ptx
tcgen05.mma.cta_group::1.kind::f4 [%tmem_d], [%tmem_a], [%tmem_b], %scale_a, %scale_b;
//          ^^^^^^^^^^^^ ^^^^^^^^
//          single CTA   FP4 inputs
```

Properties:

- **Asynchronous + decoupled**: even more so than `wgmma`. Operands and result are addressed by **TMEM addresses**, not registers. Issuing the instruction takes effectively zero time on the issuing warp.
- **Larger tiles**: up to m128n128k64 single-CTA, m256n128k64 CTA-pair (with `cta_group::2`).
- **CTA-pair / cluster-2 mode**: two CTAs cooperate via cluster-shared memory and TMEM to issue a single MMA over a larger tile. Requires `.cluster_dim 2,1,1`. **SM100 only.**
- **Companion ops**: `tcgen05.alloc` for TMEM allocation, `tcgen05.commit` for completion barrier, `tcgen05.cp` for TMEM-to-SMEM copy-out.
- **SM100 only.** **Does not work on SM120.**

The reason `tcgen05` exists is that at FP4/FP6 throughput levels, the warp-group MMA approach starts to bottleneck on register file bandwidth — the warp can't issue MMA instructions fast enough to keep the Tensor Core busy. By moving accumulators out of registers and into TMEM, the warp issues one `tcgen05.mma`, the Tensor Core runs to completion in TMEM, and the warp can do other work in parallel.

This deep coupling is why `tcgen05` is *not* a simple ISA addition — the entire kernel-design pattern around it (TMEM allocation, async commits, TMA-into-TMEM copies) is its own ecosystem.

## What runs on what

| Instruction family | SM 7.x | SM 8.x | SM 9.0 | SM 10.0 | SM 12.0 |
| --- | :---: | :---: | :---: | :---: | :---: |
| `mma.sync` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `wgmma.async` | | | ✓ | ✓ | ✓¹ |
| `tcgen05.mma` (single CTA) | | | | ✓ | |
| `tcgen05.mma` (CTA-pair) | | | | ✓ | |

¹ Available but Tensor Core throughput is lower; not the optimal path on consumer Blackwell.

## How CUTLASS chooses

CUTLASS, the reference high-performance GEMM library, has separate template hierarchies for each MMA family:

- `cutlass/include/cutlass/gemm/collective/sm80_*` — Ampere, `mma.sync`-based
- `cutlass/include/cutlass/gemm/collective/sm90_*` — Hopper, `wgmma`-based
- `cutlass/include/cutlass/gemm/collective/sm100_*` — Datacenter Blackwell, `tcgen05`-based
- `cutlass/include/cutlass/gemm/collective/sm120_*` — Workstation Blackwell, currently `mma.sync`-or-`wgmma`-based

When you instantiate a CUTLASS GEMM, you specify the target architecture; the template selects the appropriate MMA family. The SM100 templates target **`sm_100a`** because they need `tcgen05`. The SM120 templates target **`sm_120` or `sm_120f`** and use the older but still effective `mma.sync` / `wgmma` paths.

This is why a CUTLASS-built library shipped against one target won't run on the other: the actual instructions in the binary are different.

## Performance perspective

At FP4, a single Tensor Core can issue one m128n128k64 MMA per cycle on SM100 — that's 1,048,576 multiply-accumulates per cycle per Tensor Core. A B100 has 144 SMs each with multiple Tensor Cores, yielding ~5 PFLOPs at FP4.

On SM120, without `tcgen05`, you fall back to issuing many `mma.sync m16n8k32` ops to do the same work. The arithmetic throughput per Tensor Core is similar — the Tensor Core hardware is the same — but the *scheduling overhead* (more instructions issued, more register file traffic) reduces achievable throughput.

A rough rule of thumb: an SM120 GEMM kernel reaches **40–70 % of peak SM100 throughput per FLOP**, and SM120 hardware has fewer SMs and lower clock budget, so the absolute throughput is well below SM100. RTX PRO 6000 Workstation hits ~125 TFLOPs FP4; B100 hits ~5 PFLOPs. **A 40× absolute gap, roughly half hardware and half ISA.**

## Tile shapes commonly seen

The PTX ISA defines specific allowed tile shapes; libraries instantiate combinations of them. Common ones:

| Family | Tile shape | When |
| --- | --- | --- |
| `mma.sync` | m16n8k16 | FP16/BF16 |
| `mma.sync` | m16n8k32 | FP8/FP4 |
| `wgmma.async` | m64n{16…256}k{16,32} | Hopper FP16/FP8 |
| `tcgen05.mma` | m{64,128}n{64,128}k{16,32,64} | Datacenter Blackwell, single-CTA |
| `tcgen05.mma` | m{128,256}n{64,128}k{16,32,64} | Datacenter Blackwell, CTA-pair |

Tile shape selection trades off: larger tiles amortize instruction-issue overhead, smaller tiles fit more occupancy.

## Checkpoint

You should be able to answer:

- What does `m16n8k16` mean?
- What's the difference between `mma.sync` and `wgmma.async`?
- Why does `tcgen05.mma` use TMEM instead of registers for accumulators?
- What target architecture does CUTLASS use for its datacenter Blackwell templates?
- Roughly how big is the SM100-vs-SM120 throughput gap at FP4?

## See also

- [`memory-hierarchy`](memory-hierarchy.md) — where Tensor Memory fits
- [`number-formats`](number-formats.md) — FP4/FP6/FP8/BF16 in detail
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) — the full datacenter-Blackwell story
- NVIDIA *PTX ISA* spec, sections on MMA / WGMMA / TCGEN05
- *CUTLASS* documentation, particularly the "Blackwell architecture" section

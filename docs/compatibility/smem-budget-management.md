# SMEM budget management

How to fit kernels into SM120's tighter shared-memory budget.

## The numbers

| Architecture | Total SMEM/SM | Driver-reserved | Available to kernel | Headline difference |
| --- | --- | --- | --- | --- |
| SM90 (H100) | 228 KiB | ~1 KiB | **227 KiB** | baseline |
| SM100 (B100/B200) | 228 KiB | ~3 KiB | **225 KiB** | tiny |
| SM120 (RTX PRO 6000) | 100 KiB | ~1 KiB | **99 KiB** | **−128 KiB** |

A kernel that fits SM100 but uses 100–225 KiB of SMEM **cannot** be launched on SM120 without reducing its SMEM footprint.

## Where the SMEM goes

In a typical MoE GEMM kernel:

```
Total kernel SMEM usage =
    operand A staging buffer
  + operand B staging buffer
  + accumulator scratch (if not in registers / TMEM)
  + scales buffer (NVFP4: 1 byte per 16 elements)
  + barriers / mbarriers
  + per-warp scratch (if any)
  + epilogue scratch (e.g., for activation fusion)
```

For a m128n256k64 NVFP4 GEMM with 4-stage pipelining:

| Component | Size on SM100 | Size on SM120 |
| --- | --- | --- |
| Operand A: 128 × 64 × 0.5 B × 4 stages | 16 KiB | (same) |
| Operand B: 64 × 256 × 0.5 B × 4 stages | 32 KiB | (same) |
| Scales for A and B: roughly 4 KiB | 4 KiB | (same) |
| Accumulator (FP32): 128 × 256 × 4 B | (in TMEM, 0 KiB) | **128 KiB** |
| Barriers / mbarriers | 1 KiB | (same) |
| **Total SMEM** | 53 KiB | **181 KiB → exceeds 99 KiB** |

The accumulator is the cliff. On SM100 it lives in TMEM (free SMEM-wise); on SM120 it must live in SMEM or registers.

## Strategies to fit

### A. Reduce tile shape

Halve the tile in one dimension:

| Shape change | Accumulator drop | Throughput cost |
| --- | --- | --- |
| m128n256 → m128n128 | 128 → 64 KiB | ~half throughput per CTA, but 2× more CTAs cover the work |
| m128n256 → m64n256 | 128 → 64 KiB | similar |
| m128n256 → m64n128 | 128 → 32 KiB | quarter per-CTA, but 4× CTAs |

A m64n128 tile gives plenty of SMEM headroom and often performs nearly as well as m128n256 thanks to better SM occupancy (more concurrent CTAs).

### B. Move accumulator to registers

A m64n64 FP32 accumulator is 16 KB → 4096 32-bit registers across 128 threads = 32 registers/thread. Tight but feasible.

```cuda
// Per-thread register accumulator
float acc[8][4];   // each thread holds 8×4 = 32 FP32 values
```

For larger tiles, register pressure becomes the cliff: above ~96 registers/thread, occupancy drops sharply.

### C. Reduce pipeline depth

A 4-stage pipeline holds 4 copies of operand staging buffers. Reducing to 2 stages halves operand SMEM.

| Pipeline depth | Operand SMEM | Latency hiding |
| --- | --- | --- |
| 4 stages | 48 KiB (in our example) | excellent |
| 3 stages | 36 KiB | very good |
| 2 stages | 24 KiB | OK |
| 1 stage | 12 KiB | none, exposed memory latency |

3 stages is often a good compromise: most of the latency hiding, 25 % SMEM savings.

### D. Recompute instead of staging

For some kernels (e.g., FlashAttention), you can recompute intermediate values rather than staging them. Trades arithmetic for SMEM. On SM120 with proportionally more compute than SMEM, this is often a winning trade.

### E. Spill to L2 instead of SMEM

For values reused across iterations of the outer loop, write them to global memory (where they hit L2) instead of staging in SMEM. L2 is large (96 MB on workstation Blackwell, much more on B200) and acts as a slower-but-bigger SMEM. Latency is ~150 cycles vs ~30 for SMEM, so this only works for sufficiently long-reuse-distance values.

## A budget worksheet

For each kernel, compute the budget:

```
Budget = 99 KiB
  - 1 KiB driver reserved
  - 1 KiB barriers/mbarriers
  = 97 KiB available for data
```

Then categorize:

```
Operand A:        [size] × [pipeline_depth]
Operand B:        [size] × [pipeline_depth]
Scales:           [scale_size] × [pipeline_depth]
Accumulator:      [acc_size]   (if not in registers/TMEM)
Epilogue scratch: [epilogue_size]
                  ─────────────────────
Total:            [sum]    must be ≤ 97 KiB
```

If total > 97 KiB, apply strategies A–E and recompute.

## Pseudocode for a SMEM-aware kernel selector

```python
def select_kernel_for_sm120(M, N, K, dtype):
    """
    Pick a CUTLASS-style kernel template that fits the 99 KiB
    SMEM budget for the given GEMM shape and dtype.
    """
    candidates = enumerate_sm120_templates(M, N, K, dtype)
    feasible = []
    for tmpl in candidates:
        smem_use = compute_smem_use(tmpl)
        if smem_use <= 97 * 1024:
            feasible.append((tmpl, smem_use, estimate_throughput(tmpl)))

    if not feasible:
        # Fall back to even smaller tiles
        return fallback_small_tile_template(M, N, K, dtype)

    # Pick the one with highest estimated throughput
    return max(feasible, key=lambda x: x[2])
```

## A common pattern: "the kernel fits, but barely"

Many SM100 kernels translated to SM120 land in the 95–105 KiB range — *just* over budget. In these cases, the typical fix is to drop one pipeline stage (e.g., from 4 to 3), saving ~12 KiB. This is much less invasive than reducing the tile shape.

If even that doesn't fit, the next move is to halve one tile dimension. If even *that* doesn't fit, the kernel has to be substantially redesigned.

## Validation

After applying these strategies:

1. **Compile** the kernel with `nvcc --gpu-architecture=sm_120 --ptxas-options=-v` and read the SMEM number from ptxas.
2. **Confirm** it's ≤ 99 KiB. If ptxas reports > 99 KiB, the kernel will fail to launch with an "out of resources" error.
3. **Profile** to confirm the smaller tiles still achieve acceptable throughput. Often 60–75 % of the SM100 throughput is realistic.

## See also

- [`translating-tcgen05`](translating-tcgen05.md) — the partner pattern; tcgen05 translation often increases SMEM use
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) — why TMEM matters for the SMEM budget
- [`fundamentals/memory-hierarchy`](../fundamentals/memory-hierarchy.md) — the full memory picture
- [`kernels/cutlass`](../kernels/cutlass.md) — how CUTLASS exposes these tradeoffs

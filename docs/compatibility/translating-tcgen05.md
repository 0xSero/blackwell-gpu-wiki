# Translating tcgen05 to mma.sync

The pattern for rewriting datacenter-Blackwell-only PTX (using the `tcgen05` family) into workstation-Blackwell-compatible PTX (using `mma.sync` chains).

## The shape mapping

A single `tcgen05.mma` instruction does the work of many `mma.sync` instructions. The translation is a **shape decomposition**:

| `tcgen05.mma` shape | Equivalent `mma.sync` count | `mma.sync` shape |
| --- | --- | --- |
| `m64n64k16` (FP16) | 16 | m16n8k16 (4×4 grid in m,n; 1 in k) |
| `m64n64k32` (FP8) | 8 | m16n8k32 (4×4 in m,n; 0.5×scaling in k) |
| `m64n64k64` (FP4) | 4 | m16n8k32 (4×4 in m,n; 2-pass k accumulation) |
| `m128n128k64` (FP4, single-CTA) | 16 (×2 in n direction) | m16n8k32 |
| `m128n256k64` (FP4, single-CTA) | 32 | m16n8k32 |
| `m256n128k64` (FP4, **CTA-pair**) | (no SM120 single-CTA equivalent) | — |

The largest single-CTA `tcgen05.mma` (m128n256k64) decomposes to 32 `mma.sync m16n8k32` instructions per accumulator tile. With pipelining, this is feasible; without, it serializes.

The largest `tcgen05.mma.cta_group::2` shape (m256n128k64) **has no single-CTA equivalent**. To translate it, you must:

- Split the work into two halves
- Process each half as a single-CTA tile
- Glue the results together

This is more invasive than a simple shape decomposition.

## A worked translation

Original SM100 PTX:

```ptx
// Allocate 16 KB of TMEM for accumulator
.reg .b32 %tmem_d_addr;
tcgen05.alloc.cta_group::1 %tmem_d_addr, 16384;

// Issue MMA: D = A * B + D, NVFP4 inputs, FP32 accumulator
tcgen05.mma.cta_group::1.kind::nvf4
    [%tmem_d_addr],         // accumulator location (TMEM)
    [%smem_a_desc],         // A descriptor (SMEM)
    [%smem_b_desc],         // B descriptor (SMEM)
    %scale_a,
    %scale_b;

// Wait for completion
.reg .b64 %sema;
tcgen05.commit.cta_group::1 %sema;
tcgen05.wait.cta_group::1 %sema;

// Copy result from TMEM to SMEM for downstream
tcgen05.cp.tmem.shared::cta.b64 [%smem_out], [%tmem_d_addr];

// Free TMEM
tcgen05.dealloc %tmem_d_addr, 16384;
tcgen05.relinquish_alloc_permit;
```

Translated SM120 PTX (sketch):

```ptx
// Allocate equivalent space in SMEM (counts against 99 KiB budget)
.shared .align 16 .b32 smem_d_buf[4096];   // 16 KB / 4 bytes per FP32

// Load A and B operands from SMEM into registers
.reg .b32 %ra<8>;
.reg .b32 %rb<8>;
.reg .f32 %rd<32>;     // accumulator in registers (split across threads)

// Initialize accumulator (or load from previous accumulator)
mov.f32 %rd0, 0.0;
// ... %rd1 through %rd31 similarly ...

// Issue chain of mma.sync m16n8k32 (NVFP4 → FP32)
mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32
    {%rd0, %rd1, %rd2, %rd3},      // accumulator output
    {%ra0, %ra1},                   // operand A (NVFP4 packed)
    {%rb0, %rb1},                   // operand B (NVFP4 packed)
    {%rd0, %rd1, %rd2, %rd3};      // accumulator input

// ... 31 more similar mma.sync instructions for the other tiles ...

// Apply scales
.reg .f32 %scale_combined;
mul.f32 %scale_combined, %scale_a, %scale_b;
mul.f32 %rd0, %rd0, %scale_combined;
// ... apply to %rd1 through %rd31 ...

// Sync warp before SMEM store
bar.sync 0;

// Store accumulator to SMEM (across the warp)
st.shared.f32 [%smem_d_buf+0],   %rd0;
st.shared.f32 [%smem_d_buf+128], %rd1;     // each thread stores its tile
// ... etc.
```

The translated PTX is **substantially longer**: ~50 lines instead of ~20. The instruction count is much higher (32 mma.sync × per-thread + scaling logic).

## Performance implications

The `tcgen05.mma` is asynchronous — the warp issues it and continues. A chain of `mma.sync` is synchronous — each one blocks the warp until completion.

To recover overlap on SM120, the kernel must:

1. **Pipeline** the operand loads and MMAs across multiple iterations of an outer loop
2. **Software-prefetch** operands into SMEM/registers ahead of the MMA chain
3. **Distribute work** across multiple warps in the same CTA, each running its own MMA chain

This is what CUTLASS's SM120 templates do, and it's why they exist as a separate template tree from SM100. The two trees aren't different by trivial differences — they're different kernel designs.

A kernel translated naively (no pipelining) achieves perhaps **30–50 %** of optimal SM120 throughput. With careful pipelining, **60–75 %**. The remaining gap to optimal SM120 is just instruction-issue overhead — physically unavoidable without `tcgen05`.

## TMEM-to-SMEM-or-registers

The TMEM allocation is the second translation challenge. Three strategies:

### Strategy 1: registers

For small accumulators (m64-class), the FP32 accumulator fits in registers. A m64n64k32 tile produces 64×64 = 4096 FP32 values × 4 bytes = 16 KB. Across a 128-thread block, that's 128 bytes/thread = 32 32-bit registers — feasible.

This is the cleanest translation: TMEM allocations become register declarations. No SMEM impact.

### Strategy 2: SMEM staging

For larger accumulators (m128-class), 64 KB doesn't fit in registers. Stage in SMEM:

```ptx
.shared .align 16 .b32 smem_accumulator[16384];  // 64 KB

// Inside the inner loop:
ld.shared.f32 %rd0, [smem_accumulator + offset];
// ... mma.sync into %rd0 ...
st.shared.f32 [smem_accumulator + offset], %rd0;
```

This consumes 64 KB of the 99 KiB SMEM budget, leaving 35 KiB for operand staging. Tight, often forcing reduced pipeline depth.

### Strategy 3: chunking

Decompose the m128 tile into 4 m64 tiles, processed sequentially with register accumulators. Trades throughput for SMEM-budget headroom.

## Scale-related quirks

NVFP4 scales need special handling. The `tcgen05.mma.kind::nvf4` instruction takes scale registers as input and applies them inside the Tensor Core. The `mma.sync.m16n8k32.f32.e2m1.e2m1.f32` instruction takes raw FP4 inputs without integrated scaling.

Translation: apply scales as a post-MMA multiply:

```ptx
// After the mma.sync chain:
mul.f32 %rd_out, %rd_acc, %scale_a_combined;
mul.f32 %rd_out, %rd_out, %scale_b_combined;
```

Or pre-multiply: scale the operands before the MMA, sacrificing precision.

## Cluster-pair MMA: no clean translation

`tcgen05.mma.cta_group::2` issues a m256-class MMA across two cooperating CTAs. There is no single-CTA `mma.sync` equivalent that produces the same tile in one launch.

The realistic translation:

1. Split the m256 tile into two m128 tiles
2. Launch them as separate CTAs (no clustering needed; they're independent)
3. Combine their outputs at a higher level (e.g., write both to global memory and let the next layer read the combined result)

This is more invasive than the rest of the patterns: it changes the kernel's launch structure, not just its PTX.

## Pseudocode for a generic translator

```python
def translate_tcgen05(ptx_input, target_arch="sm_120"):
    instructions = parse_ptx(ptx_input)
    output = []
    tmem_to_smem = {}    # map TMEM addresses to SMEM allocations

    for instr in instructions:
        if instr.op == "tcgen05.alloc":
            smem_alloc = allocate_smem(instr.size)
            tmem_to_smem[instr.dst_reg] = smem_alloc
            output.append(decl_smem(smem_alloc))

        elif instr.op == "tcgen05.mma":
            shape = instr.tile_shape
            mma_chain = decompose_to_mma_sync(shape, instr.kind)
            output.extend(mma_chain)

        elif instr.op == "tcgen05.cp.tmem.shared":
            smem_src = tmem_to_smem[instr.src_addr]
            output.append(make_shared_to_shared_copy(smem_src, instr.dst_smem))

        elif instr.op in ("tcgen05.commit", "tcgen05.wait"):
            # The mma.sync chain is synchronous; no barrier needed beyond bar.sync
            output.append("bar.sync 0;")

        elif instr.op == "tcgen05.dealloc":
            pass    # SMEM allocations are scope-bound

        elif instr.op == "cluster_dim 2,1,1":
            output.append("cluster_dim 1,1,1")
            # WARN: kernel must be split if it relied on cluster cooperation

        else:
            output.append(instr)    # passthrough for non-tcgen05 ops

    return emit_ptx(output)
```

This is conceptual. Real implementations (whether in CUTLASS, in Triton's compiler, or in a standalone tool) deal with many more cases: pipeline mbarriers, async TMA, scale-format conversion, register pressure analysis, etc.

## When automatic translation fails

Cases where you can't mechanically translate:

- **The kernel uses `tcgen05.shift`** (TMEM layout transforms with no SMEM equivalent)
- **The kernel relies on `cta_group::2` cooperation** (no single-CTA translation)
- **The kernel uses cluster-shared TMA** (`cp.async.bulk.tensor.shared::cluster.global` — needs cluster split)

For these, a hand-rewrite at the source level is the only option.

## See also

- [`smem-budget-management`](smem-budget-management.md) — the SMEM-budget side of the translation
- [`cluster-rewriting`](cluster-rewriting.md) — for `cta_group::2` translation
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) — what `tcgen05` is
- [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md) — `mma.sync` background
- *NVIDIA PTX ISA 8.5*, Tensor Core instructions section

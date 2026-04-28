# CUTLASS

NVIDIA's CUDA Templates library. The reference C++ template library for high-performance GEMM (and now also convolutions and conv-related ops). Most other GPU kernel libraries are either built on CUTLASS or directly inspired by it.

## What it is

A header-only C++ library of templates that compile to optimized CUDA kernels for matrix multiplication. The user instantiates a template specifying:

- Operand types (BF16, FP8, NVFP4, etc.)
- Tile shape (e.g., m128n128k64)
- Pipeline stages
- Target architecture (e.g., `cutlass::arch::Sm100`, `cutlass::arch::Sm120`)
- Layout (row/col-major)

CUTLASS then generates a kernel tuned for that combination. Modern (CUTLASS 3.x) uses CUTE — a high-level layout/algebra library — under the hood for clean indexing.

GitHub: `NVIDIA/cutlass`. Maintained by NVIDIA. Used as the GEMM backend by FlashInfer, vLLM (in some paths), TensorRT-LLM, sglang, DeepSeek-AI's stack, and many others.

## What it depends on

- CUDA toolkit (CUTLASS 3.6 needs CUDA ≥ 12.4 for full Blackwell features)
- C++17
- A C++ compiler that nvcc can drive

CUTLASS itself depends on nothing else at runtime — it's header-only, compiled into whatever uses it.

## SM100 story

CUTLASS 3.5+ has dedicated Blackwell datacenter templates under `cutlass/include/cutlass/gemm/collective/sm100_*` and `cutlass/include/cutlass/gemm/kernel/sm100_*`. These templates:

- Target `sm_100a` (architecture-specific accelerated)
- Use `tcgen05.mma` for the inner MMA loop
- Allocate accumulators in TMEM via `tcgen05.alloc`
- Use **single-CTA mode** (`cta_group::1`) by default; CTA-pair mode (`cta_group::2`) is opt-in via tile-shape choice
- Request up to ~220 KiB of SMEM for operand-staging pipeline buffers
- Lean on cluster-shared TMA for cross-CTA data movement when CTA-pair mode is on

Compiling them with `nvcc -gencode arch=compute_100,code=sm_100a` produces a fatbin that runs only on SM 10.0 devices.

## SM120 story

CUTLASS 3.6+ has parallel templates under `sm120_*`. These templates:

- Target `sm_120` (or `sm_120f` for forward-compat)
- Use `mma.sync` and `wgmma.async` instead of `tcgen05.mma`
- Allocate accumulators in **registers** (smaller tiles to fit) or stage through SMEM (larger tiles, more SMEM pressure)
- Use **single-CTA only** (no `cluster_dim > 1`)
- Restrict pipeline stages to fit the 99 KiB SMEM ceiling
- Achieve ~40–70 % of optimal SM120 throughput vs ~95 % for SM100 templates on SM100

The SM120 templates are *separate trees* from the SM100 templates. They're not just a recompile.

## The SMEM cliff

The single most-encountered CUTLASS issue on workstation Blackwell. The story:

1. CUTLASS uses `StageCountAutoCarveout<sizeof(SharedStorage)>` to determine how many pipeline stages to fit in available SMEM.
2. `StageCountAutoCarveout` calculates remaining SMEM as `total_smem - other_uses` where `total_smem` is taken from the architecture's published max.
3. On SM100, max is 228 KiB. On SM120, max is **99 KiB**.
4. If a developer tests their template on SM100 (with 228 KiB headroom) and then runs the same code on SM120, the auto-carveout calculation believes there's 228 KiB available and requests pipeline buffers that overflow the actual 99 KiB.
5. The launch succeeds (CUDA driver doesn't validate the request precisely), but writes past the SMEM boundary into adjacent banks. Outputs are zeroed/scrambled. **No error.**

The canonical issue: `NVIDIA/cutlass#3096` ("SMEM size detection on Blackwell consumer parts"). The fix in flight: a runtime SMEM budget query that respects the actual device limits.

Workarounds in the meantime:

- Manually set `StageCount` to a small number (2 or 3) instead of auto-carveout
- Use the `sm120_*` templates rather than the `sm100_*` ones
- Choose smaller tile shapes that don't push SMEM hard

## Common failures

**Failure 1: `no kernel image is available`**

You built a CUTLASS-using library against `sm_100a` and ran it on SM 12.0. The fatbin contains only `sm_100a` cubins, no SASS for `sm_120`, and the embedded PTX (if any) targets `sm_100a` which can't JIT to `sm_120`.

Fix: rebuild with `-gencode arch=compute_120,code=sm_120` *and* the SM120-targeted templates, not just the SM100 templates with a different gencode flag.

**Failure 2: Silent output corruption (the SMEM cliff)**

Described above. The kernel runs, the API returns success, the output is zeros or garbage.

Detection: for a CUTLASS-based GEMM, manually run a small reference computation in BF16 and compare. If the tile shape was sized for SM100 SMEM, this fails.

Fix: use SM120 templates with smaller tile shapes, or set explicit `StageCount`.

**Failure 3: Cluster downgrade**

A CUTLASS template with `cta_group::2` (CTA-pair MMA) is launched on SM120. The cluster dim is silently set to (1,1,1); the kernel deadlocks at the first `cluster.sync` or produces wrong outputs.

Fix: only use CUTLASS templates that have `cta_group::1`. The SM120 template tree enforces this; the SM100 tree does not.

**Failure 4: NVFP4 scale layout mismatch**

CUTLASS expects NVFP4 scales in a specific layout (block-interleaved, FP8 E4M3). If a model artifact was saved in MX-FP4 layout (block-32 with FP6 E3M2 scales) and loaded into a CUTLASS NVFP4 template, the scales are misinterpreted.

Fix: requantize the artifact, or use a different kernel library whose layout matches.

## Detection

To check whether a `.so` uses CUTLASS:

```bash
nm -D mylib.so | grep -i cutlass | head
# or
strings mylib.so | grep -E 'cutlass::|CollectiveBuilder|StageCount' | head
```

To check which arch targets are present:

```bash
cuobjdump --list-elf mylib.so
```

Look for `arch = sm_100a` (datacenter only) or `arch = sm_120` (workstation friendly).

## Reading CUTLASS source

The library is large (~200K LOC) but well-organized:

```
include/cutlass/
├── gemm/
│   ├── collective/
│   │   ├── sm70_*       # Volta
│   │   ├── sm80_*       # Ampere
│   │   ├── sm90_*       # Hopper
│   │   ├── sm100_*      # Blackwell datacenter
│   │   └── sm120_*      # Blackwell workstation
│   ├── kernel/          # Top-level kernel composition
│   └── threadblock/     # Older (pre-3.x) tile-level code
├── arch/                # Architecture wrappers (cutlass::arch::Sm120 etc.)
├── conv/                # Convolutions (similar structure)
└── ...
```

To understand what's specific to one architecture, diff `sm100_*` against `sm120_*`. The differences will be in MMA-instruction wrappers, tile shapes, and pipeline depth.

## CUTLASS issues to watch

A few open / recent issues that capture the SM120 story:

- `#3096` — SMEM size detection on consumer Blackwell
- `#3045` — NVFP4 scale layout discrepancies
- `#2950` — sm120 template stagecount auto-carveout
- `#3120` — wgmma fallback path for sm120

These issues are the current edge of CUTLASS development; their resolution will affect what works on SM120 in subsequent releases.

## See also

- [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md) — `mma.sync`, `wgmma`, `tcgen05`
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) — what the `sm100_*` templates use
- [`blackwell/sm100-vs-sm120`](../blackwell/sm100-vs-sm120.md) — the SMEM cliff
- [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) — porting patterns
- *NVIDIA/cutlass* on GitHub
- *CUTLASS Programming Guide* (in the repo's `media/docs/` directory)

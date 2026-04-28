# DeepGEMM

DeepSeek-AI's high-throughput FP8/FP4 GEMM library. Designed specifically for the GEMM shapes that come up in DeepSeek-V2/V3/V4 inference. **As of early 2026, SM100-only**, with an SM120 port in progress.

GitHub: `deepseek-ai/DeepGEMM`. License: MIT. Maintained by DeepSeek-AI.

## What it is

A standalone GEMM library, conceptually similar to CUTLASS but smaller, more focused, and optimized for one specific use case: **MoE inference at FP4/FP8** with a particular emphasis on grouped-GEMM performance (where N or M dimensions vary across "groups" — e.g., per-expert GEMMs in a MoE layer).

The library generates CUDA kernels at JIT time from Python templates, with a small set of supported tile shapes and pipeline configurations.

## Why it exists separately from CUTLASS

CUTLASS is general-purpose; DeepGEMM is targeted. Specifically:

- **Grouped GEMM**: DeepGEMM handles per-expert GEMMs more efficiently than CUTLASS's `GroupedGemm` template, by amortizing kernel launch overhead more aggressively
- **NVFP4 specifics**: DeepSeek's NVFP4 layout (their pre-quantized weight format) is supported natively
- **Smaller tile inventory**: DeepGEMM ships only the shapes that DeepSeek's models use, simplifying tuning
- **`tcgen05` first**: written for SM100 from the start, with the tcgen05/TMEM-centric design baked in

## What it depends on

- CUDA toolkit (≥ 12.4 for SM100)
- A `nvcc` capable of `--gpu-architecture=compute_100`
- PyTorch (for Python bindings)

## SM100 story

Full support, with high optimization. DeepGEMM achieves close to peak FP4 throughput on B100 / B200. The kernels:

- Target `sm_100a`
- Use `tcgen05.mma.cta_group::1` and `cta_group::2`
- Allocate accumulators in TMEM
- Use cluster-shared TMA for operand staging
- Are compiled JIT at first use, cached at `~/.cache/deepgemm/`

DeepGEMM is one of the canonical examples of an SM100-native library. Reading its source is a good way to learn modern Blackwell datacenter kernel design.

## SM120 story

**As shipped: not supported.** DeepGEMM's `gemm_jit.py` defaults to:

```python
gencode_flags = [
    "-gencode", "arch=compute_100,code=sm_100a",
]
```

Loading DeepGEMM on a workstation Blackwell card produces:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**A port is in progress.** The work is non-trivial because every kernel uses `tcgen05` directly:

```cuda
// Excerpt from a DeepGEMM kernel:
asm volatile(
    "tcgen05.alloc.cta_group::1.b32 %0, 16384;\n"
    : "=r"(tmem_base)
);
asm volatile(
    "tcgen05.mma.cta_group::1.kind::nvf4 [%0], [%1], [%2], %3, %4;\n"
    : : "r"(tmem_base), "r"(smem_a), "r"(smem_b), "r"(scale_a), "r"(scale_b)
);
```

The port has to:

1. Replace each `tcgen05.mma` with a chain of `mma.sync` instructions
2. Replace TMEM allocations with register or SMEM allocations
3. Reduce tile shapes to fit within SM120's 99 KiB SMEM ceiling
4. Avoid `cluster_dim > 1` (no CTA-pair MMA on SM120)

This is **not a recompile** — it's a substantial rewrite of the kernel inner loop. Hence the port is taking time.

## What to do in the meantime

For workstation Blackwell users running models that reference DeepGEMM:

**Option 1**: substitute CUTLASS NVFP4 GEMM. CUTLASS's SM120 NVFP4 templates produce correct outputs (with the SMEM cliff caveat). Throughput is lower than DeepGEMM-on-SM100 but not much lower than what DeepGEMM-on-SM120 will achieve once ported.

Most inference engines that use DeepGEMM also have a CUTLASS fallback path:

```bash
# In sglang, for example:
SGLANG_DISABLE_DEEP_GEMM=1
SGLANG_ENABLE_DEEP_GEMM=0
```

These environment variables disable DeepGEMM dispatch and route MoE GEMMs through CUTLASS.

**Option 2**: substitute Marlin (INT4) for the heaviest GEMMs. Lower precision, lower memory bandwidth, but works fine on SM120.

**Option 3**: serve at FP8 instead of FP4. Larger weights (~2× memory), but doesn't depend on NVFP4 paths at all. FP8 GEMM kernels exist for SM120 in CUTLASS without the cliff.

## Common failures

**Failure 1: `no kernel image`** — DeepGEMM cubins are SM100-only. See above.

**Failure 2: scale layout mismatch** — DeepGEMM uses NVFP4 layout in a particular form (block-interleaved, FP8 E4M3 scale). If a model artifact was saved in MX-FP4 layout (the OCP standard, block-32 with FP6 E3M2 scale), loading it through DeepGEMM produces silent garbage. Some models ship with both layouts; pick the right one.

**Failure 3: JIT cache pollution after a partial port attempt**

If you've experimented with patches to make DeepGEMM target `sm_120`, the JIT cache may contain partially-compiled garbage. Clear it: `rm -rf ~/.cache/deepgemm/`.

## Detection

```bash
python -c "import deep_gemm; print(deep_gemm.__file__, deep_gemm.__version__)"
ls ~/.cache/deepgemm/
```

If you see only `100a/` subdirectories in the cache and no `120/` or `120a/`, you're on a non-ported version.

## Reading DeepGEMM source

```
deep_gemm/
├── csrc/                       # C++ kernel sources
├── deep_gemm/jit/              # Python JIT framework (gemm_jit.py et al.)
├── tests/                      # Per-kernel correctness tests
└── tools/                      # Benchmarks
```

Start with `deep_gemm/jit/gemm_jit.py` to see the architecture-target logic. Then `csrc/` for the actual kernels.

## The broader implication

DeepGEMM is the canonical example of "model release that depends on a kernel library that ships only `sm_100a`." DeepSeek's V3 and V4 model release notes point to DeepGEMM as the recommended GEMM backend; the recommendation works on B100/B200 but quietly fails on RTX PRO 6000 Workstation.

This is the pattern behind half the case studies in [`case-studies/`](../case-studies/index.md): a frontier-lab model + its reference deployment stack assumes datacenter Blackwell. The kernel-library ecosystem (DeepGEMM in particular, but FlashInfer-MoE and DeepEP too) hasn't caught up with consumer Blackwell, even though the hardware is similar in many ways.

## See also

- [`cutlass`](cutlass.md) — the alternative GEMM backend on SM120
- [`blackwell/tcgen05-and-tmem`](../blackwell/tcgen05-and-tmem.md) — what DeepGEMM relies on
- [`blackwell/nvfp4-deep-dive`](../blackwell/nvfp4-deep-dive.md) — the format itself
- [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) — the porting pattern
- `deepseek-ai/DeepGEMM` on GitHub
- DeepSeek-V3 and DeepSeek-V4 model release blog posts

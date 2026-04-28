# Triton and TransformerEngine

Two adjacent libraries: a DSL compiler (Triton) and a high-level wrapper for mixed-precision (TransformerEngine). Both relevant on workstation Blackwell; both with their own quirks.

## Triton

A Python-based DSL for writing CUDA kernels. The user writes high-level code that resembles NumPy operations on tiles; Triton's compiler lowers it to PTX/SASS for the target architecture.

GitHub: `triton-lang/triton`. Originally OpenAI; now community-maintained. Works on SM80 through SM120.

### What it does

You write something like:

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    pid = tl.program_id(axis=0)
    # ... tile-level computation ...
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.dot(a, b)
    tl.store(c_ptrs, c)
```

Triton's compiler:

1. Schedules the iteration space to map to GPU threads
2. Allocates registers and SMEM
3. Lowers `tl.dot` to `mma.sync` or `wgmma.async` (or `tcgen05.mma` on SM100, in recent versions)
4. Handles synchronization, vectorization, and other low-level concerns

The result is a kernel that's **typically 70–90 % as fast as a hand-tuned CUTLASS equivalent**, with **dramatically less code**.

### SM compatibility

Triton 3.0+ targets SM80 through SM120. The compiler emits architecture-appropriate PTX based on the device at compile time:

- On SM80–SM89: `mma.sync`
- On SM90: `wgmma.async` where beneficial
- On SM100: `tcgen05.mma` for large tiles, `wgmma` for smaller (this is in flight; not all paths use `tcgen05` yet)
- On SM120: `mma.sync` and `wgmma.async`, no `tcgen05`

The SM120 path is well-supported. Triton has had explicit SM120 testing since Triton 3.0.

### Common failures

- **Triton version mismatch with downstream library**. FlashInfer, sglang, and others pin specific Triton versions. Mixing produces import errors or kernel-compile failures.
- **Kernel that uses experimental Hopper features on SM120**. Some Triton kernels use `tl.async_copy` (TMA) or cluster features. On SM120 these silently downgrade or error. Most well-maintained Triton kernels guard arch-specific features with `arch.sm > 9` checks.
- **Register pressure on SM120**. SM120's smaller SMEM means kernels that worked on SM100 might spill to local memory on SM120. Detection: `nvcc --resource-usage` or `ncu` register-count metrics.

### When to use Triton

For workstation Blackwell users:

- **Custom kernels**: writing one in Triton is faster than writing the CUTLASS equivalent and produces respectable performance
- **Attention with KV-splits**: FlashInfer's Triton-based attention with `--triton-attention-num-kv-splits 64` outperforms FA-2 on long contexts
- **MoE expert dispatch**: Triton kernels for routing and combine logic are common in serving stacks

## TransformerEngine

NVIDIA's high-level library for mixed-precision training and inference. Wraps PyTorch with FP8/FP4-aware modules (Linear, LayerNorm, Attention) that handle the quantization, scaling, and unscaling automatically.

GitHub: `NVIDIA/TransformerEngine`. License: Apache-2.0. Maintained by NVIDIA.

### What it does

```python
import transformer_engine.pytorch as te

linear = te.Linear(input_dim, output_dim, params_dtype=torch.bfloat16)
with te.fp8_autocast(enabled=True, fp8_recipe=te.recipe.DelayedScaling()):
    y = linear(x)
```

Behind the scenes:

1. Quantizes the linear's weights from BF16 to FP8 on-the-fly
2. Tracks per-tensor scaling factors over a window of recent calls
3. Uses CUTLASS or cuBLAS-Lt for the actual GEMM
4. Handles dequantization back to BF16 for downstream ops

The "DelayedScaling" recipe is one of several; others target FP4, MXFP, or static-scaling deployments.

### SM compatibility

| Architecture | Support |
| --- | --- |
| Ampere | works (FP16/BF16 only; FP8 paths skip) |
| Hopper | full FP8 support |
| Blackwell DC (SM 10.0) | full FP4/FP8 support |
| Blackwell WS (SM 12.0) | partial — some FP8 paths assume Hopper-class semantics; FP4 path evolving |

TransformerEngine's SM120 support is being added incrementally. As of early 2026, basic Linear and LayerNorm modules work on SM120; attention and MoE-specific modules are less reliable.

### When to use it

For workstation Blackwell:

- **If you want to add FP8 training or inference to an existing PyTorch model** without writing kernels yourself, TransformerEngine is the simplest path
- **Don't use it for inference-only workloads** where you're loading pre-quantized models — for that, the inference engines (vLLM, sglang) call CUTLASS / FlashInfer directly and TE's overhead isn't justified
- **Watch the GitHub issues** before relying on a specific TE module on SM120

### Common failures

- **Module not yet ported to SM120**. Symptom: `RuntimeError: Unsupported architecture` or `no kernel image`. Fix: check `transformer_engine` issue tracker; some modules need the SM120 backend manually enabled.
- **FP8 scale staleness**. The DelayedScaling recipe maintains a rolling history of activation scales; in inference (where you don't have many calls), the scale can be stale. Use `MXFP8` or static recipes instead for inference.

## How they fit together

In a real inference stack, you might see:

```
Inference engine (sglang, vLLM)
    │
    ├─→ FlashInfer (attention + MoE)
    │      │
    │      ├─→ Triton kernels (attention with KV-splits, expert routing)
    │      └─→ CUTLASS kernels (NVFP4 GEMM, FP8 GEMM)
    │
    └─→ Custom Triton kernels (LayerNorm, RoPE, sampling)

Training stack might add:
    └─→ TransformerEngine (FP8 Linear, FP8 LayerNorm, FP8 attention)
            │
            └─→ CUTLASS (the underlying GEMMs)
```

Triton sits as an alternative to CUTLASS for kernels that are easier to express in DSL than to template-instantiate. TransformerEngine sits above CUTLASS as a PyTorch-friendly wrapper.

## See also

- [`cutlass`](cutlass.md) — what TransformerEngine wraps
- [`flashinfer`](flashinfer.md) — uses Triton for some attention paths
- *Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"* (2019)
- `triton-lang/triton` and `NVIDIA/TransformerEngine` on GitHub

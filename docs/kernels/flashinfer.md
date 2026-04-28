# FlashInfer

A kernel library for serving — attention and MoE both, with a focus on the irregular shapes encountered in production inference (variable sequence lengths, page-attention KV caches, MoE routing).

GitHub: `flashinfer-ai/flashinfer`. License: Apache-2.0. Maintained by an academic-industry consortium.

## What it is

Two distinct kernel families under one roof:

- **Attention kernels**: page-attention, prefix-attention, decode attention, KV-split attention. Triton-based and CUTLASS-based variants.
- **MoE kernels**: top-k routing, token dispatch, expert FFN execution, output combine. Both single-GPU and multi-GPU (with all-to-all).

Used by sglang, vLLM (in some paths), and several other inference engines as the attention backend.

## What it depends on

- CUDA toolkit
- PyTorch (for the Python bindings)
- Triton (for the Triton-based kernels)
- CUTLASS (for the CUTLASS-based kernels, including NVFP4 paths)

Has its own JIT cache (typically at `~/.cache/flashinfer/<version>/<arch>/cached_ops/`), which compiles kernel variants on first use and reuses them on subsequent runs. The cache directory name encodes the architecture, e.g., `120a` for SM120.

## SM100 story

Full support. FlashInfer's NVFP4 paths use CUTLASS templates targeting `sm_100a` with `tcgen05.mma`. The MoE one-shot all-to-all uses MNNVL fabric primitives where available, falling back to NCCL.

## SM120 story

Mixed:

- **Attention**: works. The Triton-based kernels are arch-portable; the CUTLASS-based variants compile for SM120 with reduced tile sizes.
- **NVFP4 GEMM**: works via FlashInfer's CUTLASS-NVFP4 path on SM120, with the SMEM-cliff caveat (use smaller tile shapes).
- **FP4 GEMM CuDNN backend**: works on SM120 (CuDNN itself supports both architectures).
- **MoE one-shot all-to-all**: **does not work on consumer Blackwell** without P2P atomics. See below.

## The MoE one-shot all-to-all problem

FlashInfer's "one-shot" MoE all-to-all kernel is a high-performance variant that:

1. Each rank has a buffer of activations to dispatch to other ranks
2. Each rank writes (using P2P writes) directly into peer ranks' destination buffers
3. Each rank busy-polls on **completion flags** stored in peer ranks' memory
4. Once flags indicate all peers have written, the kernel proceeds to combine

Step 3 is the problem on consumer Blackwell. The completion flags are updated using **atomic writes** — and PCIe atomics are software-gated off by default on consumer GPUs. The polling rank therefore never observes the flag turning to "completed."

The kernel has a 60-second internal timeout. After 60 seconds:

```
flashinfer error: Rank 0 timed out waiting for completion flag from rank 1
flashinfer error: Rank 1 timed out waiting for completion flag from rank 2
... (similar from all ranks)
cudaFuncSetAttribute ... unspecified launch failure at cutlass_fused_moe_kernels.cuh:417
```

The server then crashes.

**Fix path 1**: enable P2P atomics. Requires both:

- BIOS: ACS Enable → **Disabled**
- Driver: `NVreg_RegistryDwords="RMDisableFeatureDisablement=1"` set via `/etc/modprobe.d/`

Both are typically locked on workstation hardware; ACS Disable in particular is gated by motherboard BIOS that may or may not honor the request.

**Fix path 2**: use a different MoE all-to-all kernel. NCCL-based all-to-all (FlashInfer has a fallback path) doesn't depend on atomics, but is slower and sometimes deadlocks at warmup.

**Fix path 3**: avoid all-to-all entirely. Use a tensor-parallelism plan instead of expert-parallelism. See [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md).

## Common failures

**Failure 1: NVFP4 + SMEM cliff**

The CUTLASS-based NVFP4 GEMM in FlashInfer hits the same SMEM cliff as raw CUTLASS. Mitigation: ensure the kernel's tile shape is sized for SM120 SMEM. FlashInfer's JIT cache for `120a` should select correct tile shapes by default in recent versions.

**Failure 2: JIT cache corruption**

The JIT cache occasionally caches a bad compile (e.g., from a previous library version). Symptom: a kernel that previously worked starts producing wrong outputs.

Fix: `rm -rf ~/.cache/flashinfer/` and let it rebuild.

**Failure 3: Triton version mismatch**

FlashInfer's Triton-based kernels need a specific Triton range (currently 3.0–3.2 as of late 2025). Older or newer Triton causes import errors or kernel-compile failures.

Fix: pin Triton in your environment, or set `FLASHINFER_DISABLE_VERSION_CHECK=1` if you know what you're doing.

**Failure 4: page-size mismatch**

FlashInfer's KV-cache attention takes a `page_size` (typically 16, 64, or 128). Some kernel paths only support specific page sizes:

- Decode-NSA path: requires `page_size=64`
- Prefill-MTP path: requires `page_size=64`
- Standard decode: any of {16, 64, 128}

If a serving stack tries to use NSA or MTP with `page_size=128`, FlashInfer asserts at startup. The fix is upstream: configure the inference engine to use `page_size=64`.

## Detection

```bash
python -c "import flashinfer; print(flashinfer.__version__, flashinfer.__file__)"

# Find the cached kernels:
ls -la ~/.cache/flashinfer/<version>/<arch>/cached_ops/
# 120a means SM120-cached kernels exist
```

The presence of `120a/cached_ops/` confirms that FlashInfer has been used on this device and has compiled SM120 kernels. Inspect specific kernels:

```bash
ls ~/.cache/flashinfer/0.6.7/120a/cached_ops/
# fused_moe_120, decode_attention_120, prefill_attention_120, ...
```

## Reading FlashInfer source

```
flashinfer/
├── csrc/                       # C++ + CUDA kernel implementations
│   ├── attention/              # Attention kernels
│   ├── moe/                    # MoE all-to-all + expert kernels
│   ├── gemm/                   # NVFP4 GEMM via CUTLASS
│   └── jit/                    # JIT runtime
├── flashinfer/                 # Python bindings
└── ...
```

The MoE one-shot a2a kernel is in `csrc/moe/all_to_all.cuh`. The atomic-busy-poll is in the wait loop near the end.

## See also

- [`cutlass`](cutlass.md) — what FlashInfer's NVFP4 paths use under the hood
- [`flashattention`](flashattention.md) — for attention specifically
- [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md) — the atomics problem in detail
- [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md) — the EP-vs-TP choice
- `flashinfer-ai/flashinfer` on GitHub
- *Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving"* (2024)

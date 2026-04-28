# Marlin and friends — INT4 GEMM paths

Older but still relevant: INT4 quantization with FP16 (or BF16) activations. Pre-dates Blackwell; works fine on workstation Blackwell. The path of least resistance for many deployments.

## Marlin

Marlin is a CUDA kernel implementing **W4A16 GEMM** — INT4 weights, FP16 activations, FP16 or FP32 accumulator. Tuned for Ampere (SM80) and works through Blackwell.

GitHub: `IST-DASLab/marlin`. License: Apache-2.0.

### What it does

A specialized GEMM where:

- Weights are **4-bit integers** with a separate scale factor (typically per-output-channel or per-group)
- Activations are FP16
- Tensor Core MMA happens at FP16; weights are dequantized on-the-fly using the per-group scales
- Output is FP16 accumulator (with an option for FP32)

Throughput at small batch sizes (decode-style workloads) is competitive with FP8 GEMM, with **half the weight memory bandwidth**.

### SM compatibility

| Architecture | Support |
| --- | --- |
| Ampere (SM 8.0/8.6/8.9) | full, the design target |
| Hopper (SM 9.0) | works, ~10–20 % slower than wgmma-FP8 paths |
| Blackwell DC (SM 10.0) | works, slower than NVFP4 native |
| Blackwell WS (SM 12.0) | **works**, often the practical fast path |

On workstation Blackwell, when NVFP4 paths are blocked or buggy (DeepGEMM not ported, CUTLASS NVFP4 hitting SMEM cliff), Marlin is a viable alternative. You pay in slightly larger weights (~4 bits → ~4.25 bits with metadata vs NVFP4's 4.5 bits — actually slightly *smaller*) and slightly less accuracy on certain models.

### Common failures

- **Group-size mismatch**: Marlin is sensitive to the group size used during quantization (typically 128 or 64). Loading a model quantized at one group size with a kernel expecting a different group size produces wrong outputs.
- **Activation precision mismatch**: feeding BF16 activations into a Marlin kernel that expects FP16 produces NaN-filled output (FP16 max is ~65 504; BF16 has wider range).

## AWQ — Activation-aware Weight Quantization

A *quantization scheme*, not a kernel — but it ships with kernels. AWQ uses the magnitude of activations to choose which weight columns to quantize aggressively vs preserve in FP16. Empirically: better accuracy than naive INT4 quantization at the same bitrate.

The AWQ kernel uses similar techniques to Marlin (W4A16) but with the activation-aware quantization metadata.

GitHub: `mit-han-lab/llm-awq`. Works on SM80–SM120.

## GPTQ — Gradient-based Post-Training Quantization

Another quantization scheme. Uses second-order gradient information from a calibration dataset to optimize per-weight rounding. Typically better than naive RTN (round-to-nearest) but slower to compute.

The GPTQ format is a target for many kernels; both Marlin and ExLlama (another inference library) have GPTQ-format readers.

GitHub: `IST-DASLab/gptq` and downstream. Works on SM80–SM120.

## Where these fit in the stack

For workstation Blackwell users, the realistic options for serving a large model:

1. **NVFP4 + CUTLASS**: best throughput when it works. Hits SMEM cliff for some kernels.
2. **NVFP4 + DeepGEMM**: not yet ported to SM120 (as of early 2026).
3. **NVFP4 + FlashInfer**: works for non-MoE-a2a paths.
4. **W4A16 + Marlin/AWQ/GPTQ**: lower throughput than NVFP4 (Tensor Core is at FP16 peak, not FP4 peak), but works reliably.
5. **FP8 + CUTLASS**: 2× weight memory vs NVFP4, but no SMEM cliff and broader kernel support.

The choice is often dictated by model availability — if a model only ships in NVFP4 form, you can't trivially convert to W4A16. Conversely, many community-quantized models exist as GPTQ/AWQ artifacts and play nicely with Marlin.

## Other formats worth knowing

- **EXL2 / EXL3** (ExLlama): community-popular custom formats, mixed precision per layer
- **bitsandbytes (NF4)**: HuggingFace integration, used in LoRA training
- **OpenCompute MX-INT4 / MX-INT8**: structured formats analogous to MX-FP4

For most production inference on workstation Blackwell, **GPTQ-formatted W4A16 via Marlin** is the simplest viable path. It's not the fastest, but it works.

## See also

- [`fundamentals/number-formats`](../fundamentals/number-formats.md) — the broader format landscape
- [`blackwell/nvfp4-deep-dive`](../blackwell/nvfp4-deep-dive.md) — the NVFP4 alternative
- *Frantar et al., "GPTQ"* (2023)
- *Lin et al., "AWQ"* (2024)
- `IST-DASLab/marlin` on GitHub

# NVFP4 deep dive

NVIDIA's variant of OCP MX-FP4. The single Blackwell-specific format that genuinely works the same on both halves of the generation. The recipe behind 478B-parameter models that fit in 4× 96 GB of VRAM.

## The format, exactly

A NVFP4 tensor is encoded as **blocks of 16 4-bit values**, each block accompanied by **one FP8 (E4M3) scale**.

```
Block layout (in memory):
+-------------------------------------+----------+
| 16 × 4-bit values  (8 bytes packed) | scale    |
+-------------------------------------+----------+
                                     8 bits

Total: 9 bytes per 16 values = 4.5 bits per value
```

Each 4-bit value is **E2M1**: 1 sign + 2 exponent + 1 mantissa, representing values in {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} (with NaN for the all-1 pattern).

The actual numerical value of element `i` in block `b` is:

```
value(i, b) = decode_e2m1(packed[b][i]) × decode_e4m3(scale[b])
```

The block-level scale gives the format effective dynamic range: per-block scales of 10⁻⁴ to 10² × the E2M1 range produce a usable working range across many orders of magnitude.

## NVFP4 vs MX-FP4: the key differences

The OCP MX-FP4 standard has a similar structure but different parameters:

| | MX-FP4 (OCP) | NVFP4 (NVIDIA) |
| --- | --- | --- |
| Block size | 32 elements | **16 elements** |
| Scale type | FP6 (E3M2) | **FP8 (E4M3)** |
| Effective bits/element | ~4.19 | ~4.50 |
| Adopters | AMD, Intel, ARM, NVIDIA | NVIDIA |

NVFP4 trades **slightly more storage** (4.5 vs 4.19 bits/element, ~7 % overhead) for **two practical benefits**:

1. **Smaller blocks** mean tensors with non-uniform value distributions (e.g., per-channel outliers) get tighter scales. Empirically: ~0.3–0.5 perplexity improvement on most LLM benchmarks compared to MX-FP4 at the same effective bitrate.
2. **FP8 scale** has roughly 16× the precision of FP6 scale, reducing error from scale quantization. This matters for layers with extreme weight magnitudes.

Both formats run natively on Blackwell Tensor Cores (gen 5). Hopper hardware emulates them through FP8 Tensor Cores at reduced throughput.

## Memory savings, in numbers

For a 478-billion-parameter LLM:

| Format | Total weight storage |
| --- | ---: |
| FP32 | ~1.9 TB |
| BF16 / FP16 | ~960 GB |
| FP8 | ~480 GB |
| MX-FP4 (4.19 b/elem) | ~250 GB |
| **NVFP4 (4.5 b/elem)** | **~270 GB** |

In practice, NVFP4 reduces a BF16 model to about **28 % of its size**, fitting models that would otherwise require 8× the VRAM. For workstation Blackwell with 96 GB cards × 4 = 384 GB total VRAM, this enables serving weights up to ~700B parameters with room for KV cache.

## The Tensor Core path

On Blackwell, an NVFP4 GEMM compiles to MMA instructions that consume NVFP4 inputs and produce FP32 (or BF16, or FP8) outputs. The dequantization happens **inside the Tensor Core** — there's no separate "dequant kernel" before the MMA.

```ptx
// Datacenter Blackwell (SM100) — uses tcgen05
tcgen05.mma.cta_group::1.kind::nvf4
    [%tmem_d],              // FP32 accumulator in TMEM
    [%smem_a],              // NVFP4 operand A
    [%smem_b],              // NVFP4 operand B
    %scale_a, %scale_b;     // E4M3 scale registers

// Workstation Blackwell (SM120) — uses mma.sync chain
// (For each m16n8k32 sub-tile of the larger logical tile:)
mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32
    {%rd0, %rd1, %rd2, %rd3},     // FP32 accumulator
    {%ra0, %ra1},                  // NVFP4 operand A (with implicit scale handling)
    {%rb0, %rb1},                  // NVFP4 operand B
    {%rc0, %rc1, %rc2, %rc3};      // FP32 input accumulator
```

Both paths access the **same Tensor Core hardware**; the difference is the issuing instruction. SM100's `tcgen05.mma.kind::nvf4` issues larger tiles asynchronously into TMEM. SM120's `mma.sync.m16n8k32.f32.e2m1.e2m1.f32` issues smaller tiles synchronously into registers.

## The scale layout problem

Despite NVFP4 working natively on both halves of Blackwell, **scale layout incompatibilities** are a recurring source of bugs.

The format defines what an NVFP4 block "is" — values + scale — but doesn't fully specify how scales are interleaved with values in memory. There are several layouts in active use:

| Layout | Where scales sit |
| --- | --- |
| **Block-interleaved** | After every 16 values: `[16v, scale, 16v, scale, ...]` |
| **Scales-after-values** | All values first, then all scales: `[v0..vN, s0..sM]` |
| **Per-K scales** | One scale per K-dimension block, vectorized differently |
| **Tile-major scales** | Scales packed into a separate tensor with its own tile structure |

CUTLASS's NVFP4 GEMM templates assume one specific layout per template. DeepGEMM uses a different layout. FlashInfer uses yet another. **Loading a weight artifact stored in one layout into a kernel expecting a different layout produces silent garbage.**

The "MX-FP4 vs NVFP4" confusion further complicates this: a kernel might be coded as "MX-FP4" but actually expect NVFP4 layout (because the developer copied an NVFP4 reference and renamed it). Reading values as if scale were FP6 when it's actually FP8 corrupts every block.

When a model that "should work" produces gibberish on workstation Blackwell, scale-layout mismatch is one of the top three causes (alongside SMEM cliff and EP plan).

## Quantization recipes

A model is converted to NVFP4 through a process that:

1. **Profiles** weights and activations on calibration data to determine per-block scales
2. **Applies per-channel or per-tensor scaling** to balance ranges before quantization
3. **Rounds** to nearest representable E2M1 value within each block
4. **Optionally fine-tunes** ("quantization-aware training") to recover accuracy

Production recipes (NVIDIA's TransformerEngine, Hugging Face's transformers + bitsandbytes, vendor-specific tools) automate this. The output is a HuggingFace artifact with weights in NVFP4 layout plus per-block scales as a separate tensor.

A typical Hugging Face artifact directory:

```
my-model-NVFP4/
├── config.json
├── tokenizer.json
├── chat_template.jinja
├── model.safetensors           # NVFP4-packed weights
├── scales.safetensors          # FP8 E4M3 scales
├── quantization_config.json    # block_size: 16, scale_dtype: fp8_e4m3
└── ...
```

The `quantization_config.json` is the source-of-truth for which layout the artifact uses. A loader that ignores it and assumes the wrong layout misdecodes everything.

## Quality vs other low-bit formats

Empirically, on standard LLM benchmarks:

| Format | Perplexity drop vs BF16 |
| --- | ---: |
| FP8 (E4M3) | ~0.0 |
| MX-FP4 | ~0.5–1.0 |
| **NVFP4** | **~0.3–0.5** |
| INT4 (per-channel scaling, e.g. AWQ) | ~0.5–1.5 |
| INT4 (per-block, e.g. GPTQ) | ~0.3–0.7 |

NVFP4 is competitive with the best INT4 schemes and slightly better than MX-FP4. Where it shines is in **inference simplicity**: the format is natively supported by the Tensor Core, so dequantization is free; INT4 schemes require a separate dequant kernel that costs throughput.

## REAP and NVFP4 together

REAP (REbalanced Activation Pruning) is an MoE-specific pruning technique that removes whole experts from a model. Combined with NVFP4 quantization, REAP produces compact models with minimal quality loss:

- Original GLM-5.1: 744B parameters, BF16 ≈ 1.5 TB
- REAP-160 (160 of 256 experts retained): 478B parameters, BF16 ≈ 960 GB
- REAP-160 + NVFP4: 478B parameters, NVFP4 ≈ 270 GB

The 270 GB version fits in 4× 96 GB workstation Blackwell cards with room for ~50 GB of KV cache — enough for 200k-token context.

This is the recipe that motivates the entire workstation-Blackwell-as-MoE-platform conversation. Without NVFP4, you can't fit these models. Without MoE expert pruning, you also can't fit them. Together, they enable a class of deployment that was impossible 18 months earlier.

## Checkpoint

You should be able to answer:

- What's the block size and scale type of NVFP4?
- Why does NVFP4 use a smaller block size than MX-FP4?
- Does NVFP4 work natively on workstation Blackwell? On Hopper?
- What's the most common bug when NVFP4 weights produce silent garbage?
- Roughly how many bits per element does NVFP4 use?

## See also

- [`fundamentals/number-formats`](../fundamentals/number-formats.md) — number formats in general
- [`tcgen05-and-tmem`](tcgen05-and-tmem.md) — the SM100 NVFP4 path
- [`kernels/cutlass`](../kernels/cutlass.md) and [`kernels/deepgemm`](../kernels/deepgemm.md) — library-specific NVFP4 paths
- *Open Compute Project Microscaling Format Specification*
- *NVIDIA Blackwell Architecture Whitepaper*, "FP4 Tensor Cores"
- HuggingFace blog: "NVFP4 quantization for inference"

# FlashAttention

The attention kernel that made long-context transformers practical. Three generations: FA-1 (2022, basic), FA-2 (2023, Ampere/Hopper-tuned), FA-3 (2024, Hopper-async). FA-3 has no Blackwell port yet (early 2026); FA-2 does.

## What it is

A CUDA kernel that computes scaled dot-product attention `softmax(QKᵀ/√d)V` without materializing the full N×N attention matrix. The key insights:

1. **Tiling**: process Q, K, V in tiles small enough to fit in SMEM
2. **Online softmax**: compute softmax incrementally as tiles are processed, never materializing the full attention matrix
3. **Recomputation in backward**: don't store the attention matrix; recompute it during backward pass

The result is **O(N) memory** instead of O(N²), and **2–4× wall-clock speedup** over naive attention. For long-context inference, this is the difference between possible and impossible.

GitHub: `Dao-AILab/flash-attention`. Author: Tri Dao + collaborators. MIT license.

## What it depends on

- CUDA toolkit
- C++17
- For FA-2: works on Ampere through Blackwell
- For FA-3: requires Hopper (Hopper-extended Blackwell datacenter included)

The kernel is essentially self-contained — it doesn't depend on CUTLASS or other libraries at runtime, though FA-3 internally uses CUTLASS-style abstractions.

## SM100 story

**FA-2** runs on SM100 via its general kernel path, achieving reasonable but sub-optimal throughput (the `wgmma.async` and `mma.sync` paths are used; `tcgen05` is not).

**FA-3** has Hopper support (its main target) and works on SM100 because SM100 is largely Hopper-extended (clusters, TMA, FP8). However, FA-3 doesn't yet exploit `tcgen05` — the speedup of FA-3 vs FA-2 on SM100 is similar to its speedup on Hopper (~30 %), not the much larger speedup that would come from `tcgen05`-based reformulation.

A "FA-Blackwell" port that uses `tcgen05.mma` is in development. NVIDIA's TransformerEngine has a candidate implementation; community work is also active.

## SM120 story

**FA-2** runs on SM120 with reduced throughput (no async TC overlap; `mma.sync` only; smaller tile sizes due to 99 KiB SMEM cap).

**FA-3** does not target SM120 (its kernel structure assumes Hopper-class SM features that workstation Blackwell mostly has but not all). A port is feasible but not in flight.

For most workstation Blackwell users, the practical answer is **use FA-2** (or one of FlashInfer's Triton-based attention kernels — see [`flashinfer`](flashinfer.md)).

## Common failures

**Failure 1: `flash_attn` import error / build failure on SM120**

The `flash-attn` Python package is built against specific arch targets. The default wheel may not include `sm_120` cubins.

Fix: build from source with explicit `TORCH_CUDA_ARCH_LIST="12.0"` set, or use a pre-built wheel that includes `sm_120` (the project ships one as of 2025-Q4).

**Failure 2: Long-context decode is slower than expected**

FA-2 on SM120 uses smaller tiles than on SM100, so per-step throughput is lower. For long-context decode (where batch=1 and you're memory-bound on KV reads), this hurts.

Workaround: use a Triton-based attention with `--triton-attention-num-kv-splits 64` (or equivalent), which can outperform FA-2 at long context on SM120 by aggressively parallelizing the KV-loop. This is what some inference engines do by default on SM120.

**Failure 3: `cu_seqlens` indexing inconsistency**

A long-standing FA-2 footgun: variable-length attention (`flash_attn_varlen_func`) takes `cu_seqlens_q` and `cu_seqlens_k` arrays. When these are misaligned with `qo_indptr` / `kv_indptr` from the wider serving stack (vLLM, sglang), attention reads off-the-end of one sequence into the next.

Symptom: attention output is mostly correct but contaminated with leakage between concurrent requests.

Fix: ensure `cu_seqlens` exactly matches the request-batch layout. SGLang ships a patch (`qo_indptr` indexing fix) that came up in early SM120 deployments.

## Detection

To check whether `flash-attn` is installed and which arches its kernels target:

```bash
python -c "import flash_attn; print(flash_attn.__file__)"
# Then on the .so referenced from there:
cuobjdump --list-elf $(python -c "import flash_attn; ...") | grep arch
```

To check which kernels the Python API will dispatch to:

```python
import flash_attn
print(flash_attn.__version__)            # 2.7.x or 3.x
# 3.x means FA-3 path is active
```

## What's next

The Blackwell-native attention kernel (using `tcgen05.mma` for the QKᵀ and softmax-V GEMMs) is the most-anticipated addition to the FA family. When it lands, it'll likely be FA-3.x or FA-4. The expected speedup vs FA-2 on B100 is large (2–3×) for long sequences.

Workstation Blackwell will benefit indirectly: an FA-3-Blackwell-WS port (without `tcgen05`) would still be a meaningful improvement over FA-2 on SM120, by virtue of the better pipeline structure FA-3 introduced. Whether such a port is undertaken depends on community demand.

## See also

- [`flashinfer`](flashinfer.md) — alternative attention kernels for serving, including Triton-based long-context paths
- [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md) — `mma.sync` vs `wgmma.async` vs `tcgen05.mma`
- *Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"* (2022)
- *Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"* (2023)
- *Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"* (2024)
- `Dao-AILab/flash-attention` on GitHub

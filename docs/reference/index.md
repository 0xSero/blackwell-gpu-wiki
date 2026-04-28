# Reference

Lookup material — abbreviations, bibliography, and reading order suggestions.

## Pages

- [`abbreviations`](abbreviations.md) — quick lookup for acronyms and shorthand used throughout
- [`bibliography`](bibliography.md) — citations to papers, NVIDIA docs, kernel-library source, and notable blog posts

For a glossary of terms (full prose definitions), see [`overview/glossary`](../overview/glossary.md).

## Suggested reading orders

### "I want to understand why my MoE model is slow on workstation Blackwell"

1. [`overview/architecture`](../overview/architecture.md) — the central thesis
2. [`blackwell/sm100-vs-sm120`](../blackwell/sm100-vs-sm120.md) — the hardware difference
3. [`interconnect/nvlink-vs-pcie`](../interconnect/nvlink-vs-pcie.md) — the bandwidth difference
4. [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md) — how MoE expects to use that bandwidth
5. [`compatibility/ep-to-tp-rewriting`](../compatibility/ep-to-tp-rewriting.md) — the fix

### "I want to understand the GPU itself, end-to-end"

1. [`fundamentals/gpu-execution-model`](../fundamentals/gpu-execution-model.md)
2. [`fundamentals/memory-hierarchy`](../fundamentals/memory-hierarchy.md)
3. [`fundamentals/cuda-pipeline`](../fundamentals/cuda-pipeline.md)
4. [`fundamentals/tensor-cores`](../fundamentals/tensor-cores.md)
5. [`fundamentals/number-formats`](../fundamentals/number-formats.md)
6. [`blackwell/`](../blackwell/index.md) (entire section)

### "I want to learn about the kernel libraries"

1. [`fundamentals/`](../fundamentals/index.md) (entire section)
2. [`kernels/`](../kernels/index.md) (entire section, in order)
3. [`compatibility/translating-tcgen05`](../compatibility/translating-tcgen05.md) — when you need to port one
4. [`compatibility/smem-budget-management`](../compatibility/smem-budget-management.md) — the constraint side

### "I just want a TL;DR"

[`overview/architecture`](../overview/architecture.md). Read that page; it summarizes the whole wiki in 5 minutes.

# Cluster rewriting

How to handle kernels that assume cluster size > 1, when running on hardware where clusters are unusable.

## The situation

A thread-block cluster ([`blackwell/thread-block-clusters`](../blackwell/thread-block-clusters.md)) groups multiple CTAs into a unit that can share SMEM via the distributed-shared-memory abstraction. On SM100, clusters of size 2–8 are routine. On SM120, **the only safe cluster size is 1**: no cluster-shared SMEM, no cluster-pair MMA, no cluster-distributed TMA.

If a kernel was written assuming `cluster_dim > 1`, you must rewrite it. Three approaches.

## Approach 1: collapse to single-CTA

If the cluster was used for **convenience** (e.g., to gang up SMs for a larger virtual SMEM pool), you can often just set `cluster_dim = 1` and accept smaller tiles per CTA.

```cuda
// Original SM100
__cluster_dims__(2, 1, 1)
__global__ void kernel(...) {
    // each CTA handles half a tile,
    // accesses neighbor's SMEM via distributed shared
    auto neighbor_smem = cg::cluster_group::block_index({1, 0, 0});
    ...
}

// SM120 rewrite: each CTA does the whole half-tile independently
__global__ void kernel(...) {
    // cluster_dim implicit at 1
    // no cross-CTA SMEM access
    ...
}
```

The reduction in tile size per CTA may need to be compensated by:

- Launching more CTAs (the same total work, more concurrent CTAs)
- Using SM120's higher SM count to absorb the extra CTAs

This works when the cluster was **convenient but not essential**.

## Approach 2: split into independent kernels

If the cluster was used for **cooperative computation** — e.g., a `tcgen05.mma.cta_group::2` that issues a 2×-wide MMA across two CTAs — you can't just collapse. The two CTAs were doing genuinely different work that combined into one output tile.

The rewrite: split the work into two independent kernels (or two CTAs of the same kernel), each producing half the output. Combine outputs at a higher level.

```cuda
// Original: m256n128 cooperative MMA
__cluster_dims__(2, 1, 1)
__global__ void mma_cluster_pair(...) {
    if (cluster_block_id() == 0) {
        // produce upper half via tcgen05.mma.cta_group::2
    } else {
        // contribute to upper half from lower-half data
    }
}

// SM120 rewrite: two independent m128n128 MMAs
__global__ void mma_independent(int half_id, ...) {
    if (half_id == 0) {
        // produce upper m128n128 tile via single-CTA mma
    } else {
        // produce lower m128n128 tile via single-CTA mma
    }
    // Caller launches with half_id=0 and half_id=1
}
```

The output is now in two pieces; downstream consumers must read both. This is invasive but mechanically straightforward.

## Approach 3: emulate cluster-shared via global memory

If the cluster used distributed-shared-memory access patterns extensively (e.g., a stencil computation where each CTA reads from many neighbors), substituting global memory through L2 may be the cleanest answer.

```cuda
// Original SM100: read neighbor's SMEM
auto neighbor = cg::cluster_group::block_index({1, 0, 0});
float val = neighbor.shared_buffer[idx];

// SM120 rewrite: each CTA writes its share to global memory,
// then reads back from neighbor's global region
__shared__ float local_buf[N];
// ... compute local_buf ...
__syncthreads();
// Write local_buf to global region for this CTA
gmem_buf[my_block_id * N + threadIdx.x] = local_buf[threadIdx.x];
__threadfence();
// Read neighbor's region from global memory
float val = gmem_buf[neighbor_block_id * N + idx];
```

Performance cost: global memory access is much slower than distributed-shared-memory access. But L2 (96 MB on workstation Blackwell) is large enough that the data often stays cached, mitigating the penalty.

## Approach 4: don't rewrite — substitute

For some kernels (especially in libraries like CUTLASS), the cleanest fix is to use a **different template** that targets SM120 directly. CUTLASS has both SM100 templates (using clusters and tcgen05) and SM120 templates (single-CTA, mma.sync). If your kernel uses CUTLASS, configure it to dispatch to the SM120 template tree. No rewrite needed.

This is by far the cleanest path when it applies.

## Detection

How do you know a kernel uses clusters? Look for:

```cuda
__cluster_dims__(X, Y, Z)         // CUDA C++ attribute
.cluster_dim X, Y, Z;              // PTX directive
cooperative_groups::cluster_group  // C++ API
__cluster_size_in_blocks           // builtin
distributed_shared_memory_address  // distributed-shared-memory access
cg::cluster_barrier                // cluster-wide barrier
```

In a compiled binary, look for `cluster_dim` directives in the cubin's PTX section.

## Pseudocode for a cluster-collapsing translator

```python
def collapse_cluster_dims(ptx_input, target_arch="sm_120"):
    out = []
    cluster_was_active = False
    cluster_size = (1, 1, 1)

    for line in ptx_input:
        if line.startswith(".cluster_dim"):
            cluster_size = parse_cluster_dim(line)
            if cluster_size != (1, 1, 1):
                cluster_was_active = True
                out.append(".cluster_dim 1, 1, 1")
            else:
                out.append(line)

        elif "cluster_block_id" in line:
            if cluster_was_active:
                # This block expected to know its position in the cluster.
                # If we're collapsing, the answer is always 0.
                out.append("    mov.b32 %ret, 0;    // collapsed cluster")
            else:
                out.append(line)

        elif "shared::cluster" in line:
            # Distributed-shared-memory access — must emulate via global
            out.extend(emit_global_emulation(line))

        elif "tcgen05.mma.cta_group::2" in line:
            # Cooperative MMA — must split (cannot mechanically translate)
            out.append(f"// FATAL: cta_group::2 has no SM120 equivalent")
            out.append(f"// Original: {line}")
            raise NotMechanicallyTranslatable(line)

        else:
            out.append(line)

    return out
```

Mechanical translation works for collapses (Approach 1) and global-memory emulation (Approach 3). For cooperative computation (Approach 2), automatic translation is impractical: the kernel must be split at the source level.

## When clusters are essential, not convenient

A few kernels genuinely **require** clusters for correctness, not just performance. Examples:

- Some FlashAttention v3 variants that span 2 CTAs per attention tile to fit the keys/values
- Ring-attention implementations across cluster-shared SMEM
- Persistent kernels that use cluster barriers as a synchronization primitive

For these, no automatic rewrite is feasible. The kernel needs a hand-redesigned single-CTA variant.

## See also

- [`blackwell/thread-block-clusters`](../blackwell/thread-block-clusters.md) — what clusters are
- [`translating-tcgen05`](translating-tcgen05.md) — partner pattern for `cta_group::2` MMAs
- [`fundamentals/cuda-pipeline`](../fundamentals/cuda-pipeline.md) — cooperative groups context

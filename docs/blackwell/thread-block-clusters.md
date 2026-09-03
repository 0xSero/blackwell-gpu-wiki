# Thread block clusters

The unit between a CTA and a grid. Introduced with Hopper, expanded on datacenter Blackwell, **present on workstation Blackwell too** — what SM120 lacks is the two SM100-only features built on top of clusters: CTA-pair MMA and hardware-accelerated TMA multicast.

## What a cluster is

A **cluster** is a group of CTAs that:

- Are **co-scheduled** onto SMs that share a *cluster shared memory* address space
- Can **synchronize** via `cluster.sync`
- Can **address** each other's SMEM via cluster-shared addressing
- Can issue **cluster-wide TMA** that deposits a tensor tile across all participating CTAs' SMEMs in a single op

Cluster size is declared at kernel launch:

```cpp
__global__ __cluster_dims__(2, 1, 1) void my_kernel(...) { ... }
```

Or in PTX:

```ptx
.cluster_dim 2,1,1
```

The launch-time cluster shape determines how many CTAs participate. Each CTA in the cluster has a `clusterIdx` and can address other CTAs by their cluster-relative index.

## Why clusters exist

For modern Tensor Core kernels, a single CTA's SMEM (228 KiB on Hopper / SM100) is sometimes too small to hold:

- Operand A staging (multiple pipeline stages)
- Operand B staging (multiple pipeline stages)
- Accumulator (large for big tiles)
- Pipeline state (mailboxes, mbarriers)

A cluster lets two or more CTAs **pool their SMEM** for a single logical kernel tile. The cluster-shared addressing means CTA-0's TMA can deposit operand A directly into CTA-1's SMEM bank — no copy through global memory needed.

For `tcgen05.mma.cta_group::2`, clustering is **mandatory**: the largest MMA tile (m256n128k64) requires two CTAs cooperating, since neither alone has enough TMEM.

## Cluster sizes by architecture

| Architecture | Max cluster size |
| --- | --- |
| Volta (SM 7.0) – Ampere (SM 8.x) | 1 (no clusters) |
| Hopper (SM 9.0) | up to 8 (portable limit), 16 (with the `cudaFuncAttributeNonPortableClusterSizeAllowed` opt-in) |
| Blackwell datacenter (SM 10.0) | up to 16 |
| Blackwell workstation (SM 12.0) | up to 8 (portable limit; no 16 opt-in) |

What bites on SM120 is not the cluster itself but the SM100-only features layered on top of it. A kernel compiled for `sm_120` with `.cluster_dim 2,1,1` will:

1. Compile, load, and launch with 2 cooperating CTAs; `cluster.sync` and `shared::cluster` addressing work as on Hopper
2. Fail in `ptxas` if it issues `tcgen05.mma.cta_group::2` — there is no `tcgen05` on SM120
3. Run, but slowly, if it uses multicast TMA — functional, no hardware acceleration

So cluster-related failures on SM120 are either compile-time errors (`cta_group::2`) or performance problems (multicast), not silent wrong answers. See [`sm100-vs-sm120`](sm100-vs-sm120.md).

## Cluster-related PTX

### Synchronization

```ptx
cluster.sync.aligned;            // barrier across all CTAs in cluster
cluster.arrive.aligned %sema;     // arrive on a cluster mbarrier
cluster.wait.aligned %sema;       // wait on a cluster mbarrier
```

`cluster.sync` is a cluster-wide barrier. All CTAs in the cluster must reach it; once they all do, all proceed. It works the same on SM120 as on Hopper and SM100.

### Addressing

```ptx
ld.shared::cluster.b32 %r0, [%addr];   // load from another CTA's SMEM in same cluster
st.shared::cluster.b32 [%addr], %r0;   // store to another CTA's SMEM
```

These reach across SMEMs of co-located SMs. The address space (`shared::cluster`) is wider than per-CTA SMEM. SM120 has distributed shared memory too, so `shared::cluster` accesses work there as well.

### Cluster TMA

```ptx
cp.async.bulk.tensor.shared::cluster.global ...;
```

Single-instruction asynchronous tensor-tile copy from global memory into cluster-shared SMEM, distributing portions across the participating CTAs. SM100 has hardware multicast for this. On SM120 the `.multicast::cluster` qualifier is accepted, but the PTX ISA notes it is optimized only for the `sm_90a` / `sm_100a` families and "may have substantially reduced performance on other targets" — in practice, no hardware multicast.

## Detecting cluster use in a kernel

To check whether a precompiled kernel uses clusters > 1:

```bash
cuobjdump --dump-elf-symbols mylib.so | grep -i cluster

# Or in dumped PTX:
cuobjdump --dump-ptx mylib.so | grep -E 'cluster_dim|cluster\.sync|shared::cluster'
```

If you see `cluster_dim 2,1,1` or higher, or any `cluster.sync` instruction, the kernel relies on cluster cooperation. On SM120 that alone is fine; what to look for next is `tcgen05.mma.cta_group::2` (fails at compile time) and `.multicast::cluster` (works, but slowly).

## When kernels don't actually need their declared cluster

Some kernels declare `cluster_dim 2,1,1` for performance reasons (multicast TMA bandwidth) but don't logically require cluster cooperation. For these, a port to SM120 is feasible: replace the multicast TMA with per-CTA TMA loads (the cluster itself can stay). The kernel is slower but correct.

CUTLASS's SM100-targeted templates often fall into this category. The SM120-targeted templates exist precisely to provide the non-cluster equivalents.

## When kernels truly need their declared cluster

A `tcgen05.mma.cta_group::2` issuing CTA-pair MMA absolutely requires cluster size 2 — there's no single-CTA equivalent for the m256-class tiles. Kernels that depend on these need to be rewritten to use the smaller m128-class single-CTA tiles (or even smaller `mma.sync` tiles). The rewrite isn't mechanical; tile shape choice is intertwined with tiling strategy.

## A historical note

Thread block clusters were introduced with Hopper as a way to scale Tensor Core work beyond a single SM's resources. They're a relatively new programming abstraction (pre-2022 there was no equivalent). The Hopper API exposed them via `cooperative_groups::cluster_group`; CUDA C++ supports them via `__cluster_dims__`.

Consumer Blackwell kept them. What SM120 lacks relative to Hopper lies elsewhere: no TMEM / `tcgen05`, no hardware TMA multicast, and a 99 KiB rather than 228 KiB SMEM ceiling. The one real case of a higher compute capability lacking a lower one's feature is `wgmma`, which is `sm_90a`-only — neither 10.x nor 12.x has it.

## Checkpoint

You should be able to answer:

- What's a cluster, and how is it different from a grid or a CTA?
- What's the maximum cluster size on SM100? On SM120?
- What happens when you launch a kernel with `cluster_dim 2,1,1` on SM120?
- Why does `tcgen05.mma.cta_group::2` require clusters?
- How do you detect, from a compiled binary, whether a kernel uses clusters?

## See also

- [`tcgen05-and-tmem`](tcgen05-and-tmem.md) — `tcgen05.mma.cta_group::2` and CTA-pair execution
- [`sm100-vs-sm120`](sm100-vs-sm120.md) — the broader architecture diff
- [`compatibility/cluster-rewriting`](../compatibility/cluster-rewriting.md) — porting patterns
- *NVIDIA PTX ISA 8.5*, sections on `.cluster_dim`, `cluster.sync`, `shared::cluster`
- *CUDA C++ Programming Guide*, "Thread Block Clusters"

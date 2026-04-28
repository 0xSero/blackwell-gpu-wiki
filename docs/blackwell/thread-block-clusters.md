# Thread block clusters

The unit between a CTA and a grid. Introduced with Hopper, expanded on datacenter Blackwell, **absent on workstation Blackwell**. A surprisingly common cause of kernels that "compile and launch" but produce wrong outputs.

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
| Hopper (SM 9.0) | up to 8 (typical), 16 (with portable cluster size opt-in) |
| Blackwell datacenter (SM 10.0) | up to 16 |
| Blackwell workstation (SM 12.0) | **1 (no clusters)** |

The SM120 case is the one that bites: a kernel compiled for `sm_120` with `.cluster_dim 2,1,1` will:

1. Compile successfully (the `.cluster_dim` directive is accepted by `ptxas`)
2. Load successfully on the device
3. Launch with the cluster dim **silently downgraded to (1,1,1)**
4. If the kernel uses `cluster.sync` or cluster-shared SMEM addressing, **deadlock** or **read garbage**

This is one of the silent-failure classes from [`sm100-vs-sm120`](sm100-vs-sm120.md).

## Cluster-related PTX

### Synchronization

```ptx
cluster.sync.aligned;            // barrier across all CTAs in cluster
cluster.arrive.aligned %sema;     // arrive on a cluster mbarrier
cluster.wait.aligned %sema;       // wait on a cluster mbarrier
```

`cluster.sync` is a cluster-wide barrier. All CTAs in the cluster must reach it; once they all do, all proceed. On SM120 with cluster size 1, `cluster.sync` is a no-op (only one CTA, it just continues), so a kernel that uses it as a sync between CTAs is silently broken.

### Addressing

```ptx
ld.shared::cluster.b32 %r0, [%addr];   // load from another CTA's SMEM in same cluster
st.shared::cluster.b32 [%addr], %r0;   // store to another CTA's SMEM
```

These reach across SMEMs of co-located SMs. The address space (`shared::cluster`) is wider than per-CTA SMEM. On SM120, `shared::cluster` accesses fall back to local SMEM (because there's no other CTA to address) — accessing a "cluster-shared" address that maps to another CTA's SMEM produces garbage.

### Cluster TMA

```ptx
cp.async.bulk.tensor.shared::cluster.global ...;
```

Single-instruction asynchronous tensor-tile copy from global memory into cluster-shared SMEM, distributing portions across the participating CTAs. SM100 supports this; SM120 does not (`cp.async.bulk.tensor.shared::cta` only).

## Detecting cluster use in a kernel

To check whether a precompiled kernel uses clusters > 1:

```bash
cuobjdump --dump-elf-symbols mylib.so | grep -i cluster

# Or in dumped PTX:
cuobjdump --dump-ptx mylib.so | grep -E 'cluster_dim|cluster\.sync|shared::cluster'
```

If you see `cluster_dim 2,1,1` or higher, or any `cluster.sync` instruction, the kernel relies on cluster cooperation. Running it on SM120 will likely fail subtly.

## When kernels don't actually need their declared cluster

Some kernels declare `cluster_dim 2,1,1` for performance reasons (cluster-shared TMA bandwidth) but don't logically require cluster cooperation. For these, a port to SM120 is feasible: rewrite the kernel to use cluster size 1 and direct SMEM staging instead of cluster-shared TMA. The kernel is slower but correct.

CUTLASS's SM100-targeted templates often fall into this category. The SM120-targeted templates exist precisely to provide the non-cluster equivalents.

## When kernels truly need their declared cluster

A `tcgen05.mma.cta_group::2` issuing CTA-pair MMA absolutely requires cluster size 2 — there's no single-CTA equivalent for the m256-class tiles. Kernels that depend on these need to be rewritten to use the smaller m128-class single-CTA tiles (or even smaller `mma.sync` tiles). The rewrite isn't mechanical; tile shape choice is intertwined with tiling strategy.

## A historical note

Thread block clusters were introduced with Hopper as a way to scale Tensor Core work beyond a single SM's resources. They're a relatively new programming abstraction (pre-2022 there was no equivalent). The Hopper API exposed them via `cooperative_groups::cluster_group`; CUDA C++ supports them via `__cluster_dims__`.

The fact that consumer Blackwell *removed* clusters is unusual — typically NVIDIA preserves features once introduced. The likely reason: cluster cooperation requires extra SM-to-SM hardware linkage (the cluster-shared SMEM bus) that the GB202 die intentionally omits to save area.

The result: code written assuming Hopper-or-newer cluster support unexpectedly fails on SM120, even though its compute capability (12.0) is *newer* than Hopper (9.0). This is the rare case where a higher CC number doesn't strictly include all the features of a lower CC number — which violates a normally reliable assumption.

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

# NVSHMEM and DeepEP

GPU-to-GPU communication primitives for MoE at scale. Both designed assuming NVLink-class fabrics; both perform poorly or refuse to run on workstation Blackwell.

## NVSHMEM

NVIDIA's implementation of the OpenSHMEM standard for GPUs: **one-sided** GPU memory operations (put, get, atomic) across a partitioned global address space (PGAS).

GitHub: not public; ships with HPC SDK. License: NVIDIA proprietary.

### What it does

A typical NVSHMEM call:

```c
nvshmem_putmem(dst_ptr_on_peer, src_ptr_local, size, peer_pe);
nvshmem_signal_op(flag_ptr_on_peer, value, op, peer_pe);
nvshmem_signal_wait_until(flag_ptr_local, op, value);
```

This:

1. Initiates an asynchronous one-sided write from local GPU memory to peer GPU memory
2. Signals a flag on the peer (a remote atomic update)
3. Waits for the local flag to be signaled (busy-poll on the atomic)

Critically: the operation is **fire-and-forget from the issuing GPU's perspective** — the issuer continues immediately. Synchronization is via the signal/wait pair.

### Why MoE all-to-all uses it

In a MoE layer with expert parallelism (EP), each token must be routed to its assigned experts on (potentially) other GPUs. The dispatch is an all-to-all:

```
For each rank r:
    For each peer p:
        Send (tokens-routed-to-p) to peer p
```

The naive implementation is a stream of NCCL `send`/`recv` calls, but those are *two-sided* (both sender and receiver issue ops). NVSHMEM's one-sided model is more efficient: the sender issues all writes, the receiver just polls the completion flags. This **halves the communication overhead** in steady-state.

DeepEP, DeepSeek's expert-parallel a2a kernel, uses NVSHMEM for exactly this reason.

### SM compatibility

NVSHMEM works on any CUDA architecture in principle. The performance profile differs sharply by interconnect:

| Topology | NVSHMEM throughput |
| --- | --- |
| NVLink 5 (NVL72) | ~1.8 TB/s/GPU, latency ~1 µs |
| NVSwitch (DGX) | ~900 GB/s/GPU |
| PCIe Gen5 P2P (datacenter) | ~64 GB/s/GPU pair |
| **PCIe Gen4 P2P (consumer)** | **~32 GB/s/GPU pair, much higher latency** |

The PCIe paths exist but are an order of magnitude slower than NVLink. Worse: NVSHMEM's signal-wait depends on **P2P atomics**, which on consumer cards are software-gated off (see [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md)).

### What this means for workstation Blackwell

Even when NVSHMEM is technically available, the practical workstation-Blackwell experience is:

- **No NVLink → bandwidth is at PCIe levels (~50× slower)**
- **No P2P atomics → signal/wait deadlocks**

So NVSHMEM-based MoE all-to-all is effectively non-functional on workstation Blackwell. Workarounds: use NCCL all-to-all instead, or avoid the all-to-all by changing parallelism plan.

## DeepEP

DeepSeek's expert-parallel all-to-all kernel suite. Implements the dispatch and combine phases of MoE EP layers using NVSHMEM (intranode) and RDMA (internode).

GitHub: `deepseek-ai/DeepEP`. License: MIT.

### What it provides

Three transports:

- **Intranode**: NVSHMEM-over-NVLink, single chassis. The fast path. Targets DGX-class hardware.
- **Internode**: RDMA-over-InfiniBand or RoCE, multi-node. The scaling path. Targets datacenter clusters with RDMA NICs.
- **Hybrid-EP** (experimental): hybrid PCIe + NVSHMEM. Less mature; design intent is to work on consumer-ish topologies.

### SM compatibility (and topology compatibility)

| Configuration | DeepEP intranode | DeepEP internode |
| --- | --- | --- |
| NVL72 (NVLink + MNNVL) | ✓ optimal | ✓ optimal |
| DGX H100/H200 | ✓ | ✓ via RDMA NICs |
| 8× H100 PCIe + RDMA | partial (no NVLink, must use internode) | ✓ |
| **4× workstation Blackwell, no NVLink, no RDMA** | ✗ requires NVLink | ✗ requires RDMA NIC |

For the workstation Blackwell case, **neither DeepEP transport is usable**. The hybrid-EP path is the only theoretical option; it's experimental and not validated on consumer Blackwell.

### Common failures

**On workstation Blackwell**:

- DeepEP intranode init fails because NVSHMEM detects no NVLink endpoints
- DeepEP internode init fails because no RDMA NIC is present
- Hybrid-EP path may launch but produce wrong outputs or hang

The recommended action is to **not use DeepEP at all** on workstation Blackwell, and instead replace the EP plan with a TP+PP plan that doesn't require all-to-all.

## NCCL as the fallback

NCCL (NVIDIA Collective Communications Library) provides standard collective operations: `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`. Unlike NVSHMEM, NCCL is **two-sided** and works over any backend (NVLink, PCIe, IB, even TCP).

For MoE all-to-all on workstation Blackwell, NCCL's `all_to_all_single` is the practical implementation:

```python
torch.distributed.all_to_all_single(output, input, output_split_sizes, input_split_sizes)
```

It's slower than NVSHMEM-on-NVLink (no one-sided benefits, no SM-resident execution) but works correctly without atomics.

NCCL has its own quirks on workstation Blackwell:

- `NCCL_P2P_LEVEL=PIX` recommended (only attempt P2P at PCIe Internal Switch level; cross-root-complex transfers go through host-staging)
- `NCCL_IB_DISABLE=1` if no IB
- Watch for cross-root-complex deadlocks at TP=4 warmup; sometimes fixed by `SGLANG_PYNCCL_SKIP_WARMUP=1` (or equivalent in vLLM)

## Summary table

| Library | Topology assumption | Workstation Blackwell? |
| --- | --- | --- |
| NVSHMEM | NVLink for performance + atomics for sync | broken without atomics; even with, very slow |
| DeepEP intranode | NVSHMEM + NVLink | requires NVLink |
| DeepEP internode | RDMA NIC | requires RDMA |
| DeepEP hybrid-EP | PCIe (experimental) | unproven |
| NCCL | any backend, two-sided | works correctly, slower than NVLink-NVSHMEM |
| FlashInfer one-shot a2a | atomics over P2P | broken without atomics |

**The takeaway**: every "fast" MoE all-to-all kernel currently shipping assumes either NVLink or PCIe atomics. Workstation Blackwell has neither (by default). The NCCL fallback works but eliminates most of the EP-vs-TP performance advantage.

This is why MoE inference on consumer Blackwell ends up using TP+PP instead of EP — see [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md).

## See also

- [`interconnect/nvlink-vs-pcie`](../interconnect/nvlink-vs-pcie.md) — why bandwidth alone matters
- [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md) — the atomics blocker
- [`interconnect/moe-parallelism`](../interconnect/moe-parallelism.md) — EP vs TP
- *NVIDIA NVSHMEM Programming Guide*
- `deepseek-ai/DeepEP` on GitHub

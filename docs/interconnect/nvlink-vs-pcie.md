# NVLink vs PCIe

The two interconnects that GPUs use to talk to each other. NVLink is fast and proprietary; PCIe is universal and slow(er). Workstation Blackwell has only PCIe.

## NVLink — the high end

NVLink is NVIDIA's proprietary GPU-to-GPU interconnect. Generation 5 (Blackwell) is the current generation.

| Generation | Per-link bandwidth | Per-GPU bandwidth | Architecture |
| --- | ---: | ---: | --- |
| NVLink 1 | 20 GB/s | 80 GB/s | Pascal (P100) |
| NVLink 2 | 25 GB/s | 300 GB/s | Volta (V100) |
| NVLink 3 | 50 GB/s | 600 GB/s | Ampere (A100) |
| NVLink 4 | 100 GB/s | 900 GB/s | Hopper (H100) |
| **NVLink 5** | **200 GB/s** | **1.8 TB/s** | **Blackwell datacenter (B100/B200)** |

Per-GPU bandwidth is what matters for inference: it's the total data rate at which a GPU can exchange information with all peers simultaneously. 1.8 TB/s on B100 is enough to fully saturate a 200k-token attention's KV-bandwidth requirements with room to spare.

NVLink endpoints exist as physical connectors on the GPU board. Two ways to use them:

- **NVLink Bridge**: directly connect 2 GPUs via a passive bridge. Used in workstations (RTX A6000, prior generations); **not available** for RTX PRO 6000 Workstation Blackwell.
- **NVSwitch**: a switch chip that connects 8+ GPUs to each other in a single chassis. Used in DGX/HGX boxes.

A specialized form is **MNNVL** (Multi-Node NVLink), available on NVL72 systems (NVIDIA's rack-scale platform). MNNVL extends the NVLink fabric across multiple chassis, presenting up to 72 GPUs as a single NVLink domain.

## NVSwitch and topology variations

In a DGX H100 (8× H100):

```
  H100  H100  H100  H100  H100  H100  H100  H100
   |     |     |     |     |     |     |     |
   +---NVSwitch----+----NVSwitch--------+
        (4 NVSwitches total, mesh-connected)
```

Each GPU has full NVLink bandwidth to every other GPU through the switches. Bandwidth is uniform (no near/far distinction). This is the "ideal" topology for which most modern MoE algorithms are designed.

NVL72 extends this to 72 GPUs at rack scale via MNNVL.

## PCIe — the universal fallback

PCIe is the standard host-side interconnect. GPUs connect to the CPU via PCIe; in multi-GPU systems they can also use PCIe to talk directly to each other (peer-to-peer, P2P).

| Generation | Per-lane bandwidth | x16 unidirectional | x16 bidirectional |
| --- | ---: | ---: | ---: |
| PCIe 3.0 | 1.0 GB/s | 16 GB/s | 32 GB/s |
| PCIe 4.0 | 2.0 GB/s | 32 GB/s | 64 GB/s |
| PCIe 5.0 | 4.0 GB/s | 64 GB/s | 128 GB/s |
| PCIe 6.0 | 8.0 GB/s | 128 GB/s | 256 GB/s |

A single x16 slot at Gen4 unidirectional is **32 GB/s** — vs NVLink 5's per-GPU bandwidth of **1.8 TB/s**. Roughly **55× slower**.

PCIe Gen5 doubles this, but consumer workstation motherboards often de-rate to Gen4 when many cards are inserted, due to PCB / signal-integrity constraints. So an RTX 5090 in a 4-slot motherboard might run at PCIe 4.0 x16 even though both ends technically support Gen5.

## Topology constraints

PCIe systems have **root complexes** — CPUs typically have multiple PCIe root complexes, each driving a subset of slots. Within a root complex, GPUs can do P2P transfers directly. Across root complexes, transfers may need to be **staged through host RAM** (slower).

A typical 4-GPU workstation:

```
+-[0000:00] Root Complex 0     ← GPU0
+-[0000:40] BMC root           ← (no GPUs)
+-[0000:80] Root Complex 2     ← GPU1, GPU2
+-[0000:c0] Root Complex 3     ← GPU3, GPU4
```

P2P matrix:

- GPU1 ↔ GPU2: PHB (PCIe Host Bridge) — **fast P2P possible**
- GPU3 ↔ GPU4: PHB — **fast P2P possible**
- GPU0 ↔ anything else: NODE (Infinity Fabric hop on AMD, or QPI on Intel) — **slower, sometimes requires host staging**
- GPU1 ↔ GPU3 (across root complexes): NODE — **same as above**

You can inspect this with:

```bash
nvidia-smi topo -m
```

On a workstation Blackwell rig:

```
        GPU0  GPU1  GPU2  GPU3
GPU0    X     PHB   NODE  NODE
GPU1    PHB   X     NODE  NODE
GPU2    NODE  NODE  X     PHB
GPU3    NODE  NODE  PHB   X
```

For NCCL, this means cross-root-complex transfers go through a slower path. The `NCCL_P2P_LEVEL` env var controls which pairs even attempt P2P:

- `NCCL_P2P_LEVEL=PIX` — only same PCIe Internal Switch
- `NCCL_P2P_LEVEL=PXB` — same root complex (PCIe Bridge level)
- `NCCL_P2P_LEVEL=NODE` — same NUMA node (allow cross-RC, may stage)
- `NCCL_P2P_LEVEL=SYS` — system-wide

A common workstation-Blackwell trap: NCCL's default attempts P2P across root complexes, deadlocks, and the inference engine times out at warmup. Setting `NCCL_P2P_LEVEL=PIX` avoids this by forcing host-staging across root complexes.

## ReBAR — Resizable BAR

A PCIe feature that allows the host to map the entire GPU memory into BAR (Base Address Register) space, so the host (and other GPUs) can address all of GPU memory directly. Without ReBAR, only a small window (256 MiB) is exposed.

For MoE all-to-all over P2P, ReBAR is **required** — the kernel needs to write into peer GPU memory at addresses outside the small default window.

ReBAR enable steps:

1. **BIOS**: enable "Above 4G Decoding" and "Resizable BAR" in motherboard settings
2. **GPU firmware**: must be ReBAR-capable (most Blackwell cards are)
3. **Driver**: NVIDIA driver 470+ supports ReBAR

A consequence of enabling ReBAR: PCIe **bus numbers can shift** because the BAR1 region becomes much larger and the BIOS re-allocates bus numbers. Any script that hard-codes bus IDs (e.g., `01:00.0`) might need updating after enabling ReBAR.

## When PCIe is "fine"

PCIe Gen4 / Gen5 is fully sufficient for:

- **Tensor Parallelism**: small per-layer all_reduce volume
- **Pipeline Parallelism**: P2P sends between adjacent stages
- **Single-GPU inference**: no inter-GPU traffic at all

It's marginal for:

- **MoE with small batch sizes**: per-step all-to-all is small; latency-sensitive
- **Long-context decode with TP**: KV-related collectives are sensitive

It's broken for:

- **MoE with EP and large batch**: all-to-all volume saturates PCIe; throughput collapses
- **Anything depending on P2P atomics**: see [`p2p-and-atomics`](p2p-and-atomics.md)

## Summary numbers to remember

- **NVL72 NVLink 5**: 1.8 TB/s/GPU
- **DGX-class NVSwitch**: 900 GB/s/GPU
- **PCIe Gen5 x16**: 64 GB/s unidirectional
- **PCIe Gen4 x16**: 32 GB/s unidirectional
- **Cross-root-complex with host-staging**: 8–16 GB/s effective
- **Bandwidth ratio NVLink-5 to PCIe-Gen4**: ~55×

That last number is the order-of-magnitude that explains the EP-vs-TP collapse in [`moe-parallelism`](moe-parallelism.md).

## See also

- [`p2p-and-atomics`](p2p-and-atomics.md) — peer-to-peer, atomics, and ACS
- [`moe-parallelism`](moe-parallelism.md) — what to do about the bandwidth gap
- [`kernels/nvshmem-and-deepep`](../kernels/nvshmem-and-deepep.md) — the libraries that target NVLink-class fabrics
- *NVIDIA NVLink and NVSwitch documentation*
- *PCIe Specification 4.0 / 5.0*

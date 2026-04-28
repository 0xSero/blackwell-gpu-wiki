# Abbreviations

Short forms used throughout the wiki, alphabetical.

| Abbrev | Expansion |
| --- | --- |
| **A2A** | All-to-all (collective communication) |
| **AR** | All-reduce (collective communication) |
| **B100/B200/B300** | NVIDIA Blackwell datacenter GPUs (SM100) |
| **BW** | Bandwidth |
| **CGA** | Cooperative thread-block cluster (sometimes "CTA group", same as cluster) |
| **CTA** | Cooperative Thread Array (a thread block, in PTX terminology) |
| **CUTLASS** | CUDA Templates for Linear Algebra Subroutines |
| **DSA** | DeepSeek Sparse Attention |
| **EP** | Expert Parallelism |
| **FA** | FlashAttention |
| **FA2 / FA3** | FlashAttention v2 / v3 |
| **FP4** | 4-bit floating-point (E2M1) |
| **FP8** | 8-bit floating-point (E4M3 or E5M2) |
| **FP16** | 16-bit floating-point (IEEE half) |
| **FP32** | 32-bit floating-point (IEEE single) |
| **GB200/GB300** | Grace–Blackwell superchips |
| **GEMM** | General Matrix-Matrix Multiplication |
| **GPC** | Graphics Processing Cluster (GPU subdivision) |
| **GPU** | Graphics Processing Unit |
| **GQA** | Grouped-Query Attention |
| **HBM** | High-Bandwidth Memory (used in datacenter Blackwell) |
| **IB** | InfiniBand |
| **KV** | Key/Value (in transformer attention) |
| **L1 / L2** | Level-1 / Level-2 cache |
| **MHA** | Multi-Head Attention |
| **MIG** | Multi-Instance GPU |
| **MLA** | Multi-Latent Attention |
| **MMA** | Matrix-Multiply-Accumulate (Tensor Core operation) |
| **MoE** | Mixture-of-Experts |
| **NCCL** | NVIDIA Collective Communications Library |
| **NSA** | DeepSeek Native Sparse Attention |
| **NVFP4** | NVIDIA FP4 microscaled format |
| **NVL72** | NVLink-72 rack interconnect |
| **NVSHMEM** | NVIDIA Symmetric Hierarchical Memory (PGAS for GPUs) |
| **OOM** | Out of memory |
| **P2P** | Peer-to-Peer (GPU-to-GPU access) |
| **PCIe** | PCI Express |
| **PGAS** | Partitioned Global Address Space |
| **PP** | Pipeline Parallelism |
| **PTX** | Parallel Thread eXecution (NVIDIA's GPU IR) |
| **RDMA** | Remote Direct Memory Access |
| **REAP** | Router-weighted Expert Activation Pruning |
| **SASS** | Shader Assembly (NVIDIA's GPU machine code) |
| **SDPA** | Scaled Dot-Product Attention |
| **SM** | Streaming Multiprocessor |
| **SM90 / SM100 / SM120** | Compute capability tags (Hopper / Blackwell DC / Blackwell workstation) |
| **SMEM** | Shared Memory (per-SM, per-block) |
| **SXM** | NVIDIA's high-power form factor (used for datacenter cards) |
| **T** | Tera (10¹²); tok/s context: tokens per second |
| **TC** | Tensor Core |
| **TE** | TransformerEngine (NVIDIA library) |
| **TFLOPS** | Tera-FLoating-point OPerations per Second |
| **TMA** | Tensor Memory Accelerator |
| **TMEM** | Tensor Memory (SM100/SM101 only) |
| **TP** | Tensor Parallelism |
| **VRAM** | Video RAM (i.e., GPU device memory) |
| **W4A16** | 4-bit Weight, 16-bit Activation (a quantization scheme) |

For full glossary entries with prose definitions, see [`overview/glossary`](../overview/glossary.md).

# Bibliography

Selected sources for further reading. Grouped by topic.

## NVIDIA architecture & PTX

- **NVIDIA. *Blackwell Architecture Whitepaper*.** 2024–2025. The canonical reference for SM100 and SM120 design.
- **NVIDIA. *PTX ISA 8.5 Reference Manual*.** Current as of CUDA 12.x / 13.x. Contains specifications for `mma.sync`, `tcgen05.*`, `cp.async`, `cp.async.bulk` (TMA), thread-block clusters, distributed shared memory.
- **NVIDIA. *CUDA C++ Programming Guide* (current edition).** Section on Tensor Cores, async copies, cooperative groups, cluster launch.
- **NVIDIA. *CUDA C++ Best Practices Guide*.** Memory hierarchy guidance, occupancy tuning.

## Number formats

- **NVIDIA. *NVFP4 Block Format Specification*.** Part of CUTLASS docs. Defines the 16-element block size, FP8 (E4M3) per-block scale, and SM100 vs SM120 layouts.
- **OCP (Open Compute Project). *Microscaling (MX) Formats Specification*, v1.0**. October 2023. Defines MX-FP4, MX-FP6, MX-FP8 (the open-format counterparts to NVFP4).
- **NVIDIA. *FP8 Formats for Deep Learning*.** White paper, 2022. E4M3 and E5M2 details, scaling, and accuracy implications.

## Tensor Cores & matrix ops

- **CUTLASS GitHub repository.** [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) — primary source for understanding Tensor Core invocation patterns. The `include/cutlass/arch/mma_sm100.h` and `mma_sm120.h` files document `tcgen05` and `mma.sync` in code form.
- **NVIDIA. *Programming Tensor Cores in CUDA*.** Developer blog post, multiple iterations.

## FlashAttention

- **Dao, Tri, et al. "*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*."** NeurIPS 2022.
- **Dao, Tri. "*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*."** 2023.
- **Shah, Jay, et al. "*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision*."** 2024. Hopper-specific design; documents the warp-specialized 2-stage pipeline that doesn't extend cleanly to SM100, much less SM120.
- **FlashAttention GitHub repo:** [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). FA2 source is portable; FA3 is Hopper-only.

## DeepSeek architecture

- **DeepSeek-AI. *DeepSeek-V3 Technical Report*.** December 2024. Introduces MLA (Multi-Latent Attention).
- **DeepSeek-AI. *DeepSeek-V3.2 Technical Report*.** Introduces DSA (DeepSeek Sparse Attention).
- **DeepSeek-AI. *Native Sparse Attention*.** Paper introducing NSA, which V4 is widely speculated to use.
- **DeepGEMM GitHub repository:** [github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM). FP8/FP4 GEMM kernels, SM100-targeted.

## Kimi K2

- **Moonshot AI. *Kimi K2 Technical Report*.** 2024. Background on the K2 family, MuonClip optimizer, MoE structure.
- **Kimi K2 model card on Hugging Face**: hyperparameters and architectural details.

## GLM-5

- **Zhipu AI / Tsinghua KEG. *GLM-5.1 Technical Report* (forthcoming or released depending on version).** Architecture details for GLM-5.1 478B-A42B.
- **REAP Pruning paper.** Routes-weighted expert pruning method enabling REAP-160 and REAP-172.

## NCCL & collectives

- **NVIDIA. *NCCL User Guide*.** Documents `NCCL_P2P_LEVEL`, `NCCL_IB_DISABLE`, ring vs tree algorithms, environment variables.
- **NCCL GitHub repository:** [github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl). Source for understanding ring-allreduce mechanics.

## NVSHMEM & PGAS

- **NVIDIA. *NVSHMEM Documentation*.** PGAS abstraction for GPU clusters.
- **NVSHMEM GitHub repository:** [github.com/NVIDIA/nvshmem](https://github.com/NVIDIA/nvshmem) (or via the open-source mirror).

## DeepEP

- **DeepEP GitHub repository:** [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP). MoE expert-parallel communication library, datacenter Blackwell targeted.

## Inference engines

- **vLLM GitHub repository:** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm). PagedAttention, continuous batching.
- **sglang GitHub repository:** [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang). Structured-output-aware inference engine.
- **TensorRT-LLM GitHub repository:** [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). NVIDIA's production inference compiler.
- **FlashInfer GitHub repository:** [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer). Attention and MoE kernel library used by major engines.

## MoE all-to-all

- **PPLX library**: alternative MoE all-to-all implementation, sometimes used by sglang and vLLM as a non-NVSHMEM path.

## Marlin & weight-only quantization

- **Frantar, Elias, et al. "*Marlin: Fast 4-bit Inference Kernels*."** 2024. Mixed-precision GEMM design.
- **Marlin GitHub repository.**

## Triton

- **Tillet, Philippe, et al. "*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*."** MAPL 2019.
- **Triton GitHub repository:** [github.com/triton-lang/triton](https://github.com/triton-lang/triton).

## Mixture-of-Experts foundations

- **Shazeer, Noam, et al. "*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*."** ICLR 2017. Origin of modern transformer MoE.
- **Fedus, William, et al. "*Switch Transformers*."** JMLR 2022.
- **Lepikhin, Dmitry, et al. "*GShard*."** ICLR 2021.

## REAP pruning

- **(REAP paper, 2024–2025).** "Router-weighted Expert Activation Pruning" — describes REAP-160 / REAP-172 methodology.

---

This wiki itself is opinionated synthesis, not a primary source. When in doubt, the NVIDIA documentation and the kernel library source code are authoritative.

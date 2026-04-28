# Case studies

Why specific frontier models break on workstation Blackwell, and what each one teaches about the architecture split.

## Pages in this section

- [`deepseek-v3-v4`](deepseek-v3-v4.md) — the canonical EP-over-NVLink design
- [`kimi-k2`](kimi-k2.md) — Moonshot's MoE family, similar profile
- [`glm-5`](glm-5.md) — Zhipu's MoE family, less aggressive on `tcgen05`
- [`generic-moe-on-consumer-blackwell`](generic-moe-on-consumer-blackwell.md) — the synthesis: a checklist for any new MoE model

## Why these models in particular

Each of these models was released between mid-2024 and early 2026 by a frontier lab targeting datacenter Blackwell as the deployment platform. They're representative of what the modern open-weight MoE looks like:

- 100B–700B total parameters, 30B–60B active per token
- 64–256 experts, top-k=2 to top-k=8 routing
- NVFP4 quantization shipped as part of the release
- Reference deployment using DeepGEMM, FlashInfer, sglang or vLLM
- Performance numbers benchmarked on B100 or H200

The common pattern: **the reference deployment assumes datacenter Blackwell**. Running on workstation Blackwell uncovers the same subset of issues each time, with model-specific variations.

## How to use this section

Each case study answers four questions:

1. **What is the model?** (architecture, size, intended use)
2. **What does its reference deployment assume?** (hardware, kernels, parallelism)
3. **What breaks on workstation Blackwell?** (specific kernels, specific failure modes)
4. **What's the workaround?** (configuration, kernel substitution, plan rewriting)

If you're trying to deploy a specific model, find it here and read the relevant page. If your model isn't covered, [`generic-moe-on-consumer-blackwell`](generic-moe-on-consumer-blackwell.md) gives you the diagnostic procedure to apply yourself.

## A meta-observation

The case studies all converge to the same answer: **TP-only parallelism + NVFP4 (or W4A16) weights + FP8 KV + Triton attention with high kv-splits + DeepGEMM disabled.**

This is approximately the recipe that any workstation-Blackwell MoE deployment ends up using by mid-2026. It's not optimal — there's a 30–50× throughput gap to a datacenter deployment — but it's the configuration that **runs**, which is what most users need.

The rest of the deployment story (specific tile shapes, specific FP8 KV layouts, specific page sizes) varies between models, but the broad strokes are converging on this pattern.

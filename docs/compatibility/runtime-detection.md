# Runtime detection

How a system can probe the GPU and topology at startup to dispatch the right code paths automatically.

## What to detect

A complete probe answers these questions:

| Question | API |
| --- | --- |
| What architecture is each GPU? | `cudaDeviceGetAttribute(cudaDevAttrComputeCapabilityMajor/Minor)` → `(12, 0)` for SM120 |
| How much memory does each GPU have? | `cudaMemGetInfo` |
| How much SMEM/SM is available? | `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` |
| What's the max cluster size? | `cudaDeviceGetAttribute(cudaDevAttrClusterLaunch)` + experimental probe |
| Is TMEM available? | indirect: SM100/SM101 only |
| Is NVLink connected? | `nvmlDeviceGetNvLinkState`, `nvidia-smi nvlink --status` |
| Can GPU A access GPU B's memory directly? | `cudaDeviceCanAccessPeer(A, B)` |
| Can GPU A perform atomic operations on GPU B's memory? | **No direct API**. Probe by performing an atomic and checking. |
| What's the PCIe topology? | `nvidia-smi topo -m` |

The last item — atomics-over-P2P — is the trickiest because there's no clean API. The pragmatic approach: launch a tiny kernel that does `atomicAdd` on remote memory and checks for hardware completion vs falling through to host emulation. If it completes correctly and quickly, atomics work. If it errors or hangs, atomics don't.

## A pseudocode probe

```python
class GPUProbe:
    def detect(self):
        info = {}
        n = cuda.device_count()
        info["device_count"] = n
        info["devices"] = []

        for i in range(n):
            d = {}
            d["arch_major"], d["arch_minor"] = cuda.compute_capability(i)
            d["arch_string"] = f"sm_{d['arch_major']}{d['arch_minor']}"
            d["memory_total"] = cuda.mem_info(i)[1]
            d["smem_per_block_optin"] = cuda.attr(i, "MaxSharedMemoryPerBlockOptin")
            d["sm_count"] = cuda.attr(i, "MultiProcessorCount")
            d["max_cluster_size"] = self._probe_cluster_size(i)
            d["has_tmem"] = d["arch_string"] in {"sm_100", "sm_101"}
            info["devices"].append(d)

        info["nvlink_links"] = self._probe_nvlink()
        info["p2p_matrix"] = self._probe_p2p(n)
        info["p2p_atomics"] = self._probe_p2p_atomics(n)
        info["pcie_topology"] = self._probe_pcie_topology()

        return info

    def _probe_cluster_size(self, dev):
        # Try to launch a kernel with cluster_dim 2; see if it succeeds
        for size in [16, 8, 4, 2, 1]:
            if try_launch_with_cluster_dim(dev, size):
                return size
        return 1

    def _probe_p2p(self, n):
        m = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    m[i][j] = cuda.can_access_peer(i, j)
        return m

    def _probe_p2p_atomics(self, n):
        # Launch a kernel from GPU i that does atomicAdd on GPU j's memory
        # and verify the result. Time the operation.
        m = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    result = try_p2p_atomic_add(src=i, dst=j)
                    m[i][j] = result    # may be "hardware", "host_fallback", or "fail"
        return m
```

## Using the probe at startup

A typical inference engine consumes the probe like this:

```python
probe = GPUProbe().detect()

# Architecture-based dispatch
arch = probe["devices"][0]["arch_string"]
if arch == "sm_120":
    cutlass_template_tree = "sm120_optimized"
    use_deepgemm = False
    use_tcgen05 = False
elif arch == "sm_100":
    cutlass_template_tree = "sm100_optimized"
    use_deepgemm = True
    use_tcgen05 = True

# SMEM-budget-based template selection
smem_per_kernel = probe["devices"][0]["smem_per_block_optin"]
if smem_per_kernel < 102400:    # less than 100 KiB
    select_smaller_tile_templates()

# Topology-based parallelism plan
if not probe["nvlink_links"]:
    # No NVLink: avoid EP, prefer TP
    parallelism_plan.disable_ep = True
elif probe["p2p_atomics"][0][1] != "hardware":
    # No P2P atomics: avoid one-shot all-to-alls
    flashinfer_one_shot_a2a = False
```

The probe runs once at startup; the results are cached and used to configure all subsequent kernel launches.

## The hardest probe: P2P atomics

Atomics-over-P2P is the most consequential capability for FlashInfer's MoE one-shot all-to-all (see [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md)). And it's the one with no direct API.

The pragmatic probe:

```cuda
__global__ void probe_atomic_kernel(int *dst, int *flag, int expected) {
    atomicAdd(dst, 1);
    __threadfence_system();
    *flag = 1;
}

bool probe_p2p_atomics_works(int src_dev, int dst_dev) {
    // Allocate dst on dst_dev
    cudaSetDevice(dst_dev);
    int *d_counter, *d_flag;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    // Launch from src_dev
    cudaSetDevice(src_dev);
    auto start = clock_now();
    probe_atomic_kernel<<<1024, 256>>>(d_counter, d_flag, 0);
    cudaDeviceSynchronize();
    auto elapsed = clock_now() - start;

    // Read result back
    int counter_val;
    cudaMemcpy(&counter_val, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    if (counter_val == 1024 * 256) {
        // All atomics succeeded
        return elapsed < THRESHOLD_HARDWARE;
        // If elapsed > threshold, atomics likely went through host emulation
    }
    return false;    // some atomics dropped; not safe
}
```

The threshold separates "hardware atomics" (~µs) from "host-fallback atomics" (~ms). On workstation Blackwell, this probe typically returns "host fallback" or "fail" between GPUs that aren't on the same PCIe switch.

## Caching and stability

The probe results don't change at runtime under normal conditions, so cache them:

- To a file (`/tmp/gpu_probe.json` keyed by hostname + GPU UUIDs)
- For the lifetime of the process

Re-probe when:

- GPUs change (driver reload, hot-swap — rare)
- Kernel module version changes
- A user explicitly forces re-probe

## Reporting

A useful probe also produces a human-readable report:

```
GPU Probe Report
================
Hostname: workstation-1
Detected 4 GPUs:

  GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation
         arch: sm_120, memory: 96 GB, SMEM/block: 99 KiB
  GPU 1: same
  GPU 2: same
  GPU 3: same

NVLink: not detected on any pair
P2P matrix:
       0    1    2    3
   0   -    Y    Y    Y     (all PCIe Gen4)
   1   Y    -    Y    Y
   2   Y    Y    -    Y
   3   Y    Y    Y    -

P2P atomics: HOST FALLBACK on all pairs (avg latency 4.2 ms)

Recommendations:
  - Use TP-only parallelism (avoid EP)
  - Disable FlashInfer one-shot all-to-all
  - Disable DeepGEMM
  - Set NCCL_P2P_LEVEL=PIX
  - Use SM120-targeted CUTLASS templates
  - Use Triton attention with kv_splits=64
```

This report is the output a startup probe should hand to the user (or log) so they understand the auto-configuration decisions.

## See also

- [`compatibility/ep-to-tp-rewriting`](ep-to-tp-rewriting.md) — what to do once you've detected no NVLink
- [`compatibility/smem-budget-management`](smem-budget-management.md) — what to do once you've detected SM120
- [`interconnect/p2p-and-atomics`](../interconnect/p2p-and-atomics.md) — the hardware reason atomics matter
- [`kernels/inference-engines`](../kernels/inference-engines.md) — how engines consume probe results

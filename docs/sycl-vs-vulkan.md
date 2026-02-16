# SYCL vs Vulkan — Intel GPU Backends for Ollama

Ollama supports two GPU backends for Intel Arc/iGPU acceleration. This document explains the differences and helps you choose.

## Performance Comparison

Both backends run on Intel GPUs, but SYCL has a significant speed advantage:

| Intel GPU | Vulkan (tok/s) | SYCL (tok/s) | Gain |
|---|---|---|---|
| MTL iGPU (Core Ultra 155H) | ~8–11 | **~16** | +45–100% |
| ARL-H iGPU (Arrow Lake) | ~10–12 | **~17** | +40–70% |
| Arc A770 (16 GB) | ~30–35 | **~55** | +57–83% |
| Flex 170 | ~30–35 | **~50** | +43–67% |
| Data Center Max 1550 | — | **~73** | — |

*Benchmarks: Llama 2 7B Q4_0, llama.cpp, community-reported.*

## What Makes SYCL Faster

- **oneMKL / oneDNN** — Intel's optimized math and neural network libraries, hand-tuned per architecture
- **Level-Zero** — direct GPU communication with lower overhead than Vulkan's abstraction layer
- **Intel-tuned kernels** — `MUL_MAT` operations hand-optimized for each architecture (Meteor Lake, Arrow Lake, Arc, Flex, PVC)

## When to Use Vulkan Instead

- **No build step required** — upstream Ollama ships Vulkan support out of the box (`OLLAMA_VULKAN=1`)
- **Cross-vendor** — same backend works on AMD, NVIDIA, and Intel
- **Smaller image** — no oneAPI runtime libraries needed (~2 GB smaller)
- **Latest Ollama features** — upstream Ollama (v0.16.1) vs patched SYCL build
- **Kernel 6.18+ compatibility** — Vulkan avoids the known Level-Zero regression on kernel 6.18+

## Backend Options in This Repo

### Option 1: IPEX-LLM SYCL Bundle (current default)

Uses [`ipex-ollama/Dockerfile`](../ipex-ollama/Dockerfile) with the IPEX-LLM portable bundle.

| Attribute | Value |
|---|---|
| Ollama version | v0.9.3 |
| Backend | SYCL (IPEX-LLM patched llama.cpp) |
| Build time | ~2 min (download only) |
| Image size | ~1.5 GB |
| Status | Archived (Jan 2026), no further updates |

### Option 2: SYCL from Source (advanced)

Builds `ggml-sycl` from the exact llama.cpp commit that Ollama vendors, using Intel oneAPI. This unlocks a much newer Ollama version while keeping SYCL performance.

| Attribute | Value |
|---|---|
| Ollama version | v0.15.6+ |
| Backend | SYCL (ggml-sycl built from source) |
| Build time | ~10–15 min (compiles C++ with icpx) |
| Image size | ~2.5 GB |
| Status | Actively maintainable |

#### How the Source Build Works

Ollama ships the `ggml-sycl.h` header but intentionally excludes the SYCL implementation from its vendored ggml. The source build fills that gap:

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Build  (intel/oneapi-basekit)                 │
│                                                         │
│  ollama source ─────────┐                               │
│                         ├── cmake + icpx ── libggml-sycl.so
│  ggml-sycl @ <commit> ──┘                               │
│        ▲                                                │
│        └── patch-sycl.py (2 API fixes)                  │
├─────────────────────────────────────────────────────────┤
│  Stage 2: Runtime  (ubuntu:24.04)                       │
│                                                         │
│  ollama binary (official)                               │
│  + libggml-sycl.so + oneAPI runtime libs                │
│  + Intel GPU drivers (Level-Zero, IGC, compute-runtime) │
└─────────────────────────────────────────────────────────┘
```

The `ggml-sycl` source is fetched from the **exact llama.cpp commit** that Ollama vendors, ensuring ABI compatibility. Two small patches are applied:

1. **`graph_compute` signature** — Ollama adds an `int batch_size` parameter not present upstream
2. **`GGML_TENSOR_FLAG_COMPUTE` removal** — Ollama drops this enum; without the patch, every compute node gets skipped, producing garbage output

### Option 3: Upstream Ollama + Vulkan

Uses the official `ollama/ollama` Docker image with Vulkan enabled.

| Attribute | Value |
|---|---|
| Ollama version | v0.16.1 (latest) |
| Backend | Vulkan |
| Build time | None (pre-built image) |
| Image size | ~1 GB |
| Status | Actively maintained by Ollama team |

```bash
docker run -d \
  --device /dev/dri:/dev/dri \
  --shm-size 16G \
  -e OLLAMA_VULKAN=1 \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama:latest
```

## Troubleshooting

**SYCL device not detected** — Ensure `/dev/dri` is accessible inside the container. Check logs for `SYCL0` in the device list. Verify Intel GPU drivers are installed on the host.

**"failed to sample token"** — Usually an ABI mismatch between ggml-sycl and Ollama's vendored ggml. The ggml commit used for building must match exactly what Ollama vendors.

**Model too large for VRAM** — Intel integrated GPUs share system memory. Increase `shm_size` in `docker-compose.yml` or use a smaller quantization (Q4_0, Q4_K_M). See the [VRAM guide](intel-arc-a770-context-limits.md).

**Slow first inference** — SYCL JIT-compiles GPU kernels on first run. Set `SYCL_CACHE_PERSISTENT=1` so compiled kernels are cached for subsequent runs.

**`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` on kernel 6.18+** — Known Level-Zero regression. Workarounds: downgrade kernel, or switch to Vulkan backend.

## Tested Hardware

| Intel GPU | Status |
|---|---|
| Core Ultra 7 155H integrated Arc (Meteor Lake) | Verified |
| Arc A-series (A770, A750, A380) | Expected compatible |
| Data Center Flex / Max | Expected compatible |

**Requirements:** Ubuntu 24.04+, Docker with Compose, Intel GPU with Level-Zero driver support.

## References

- [llama.cpp SYCL backend docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [Intel oneAPI base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)
- [Intel GPU driver installation](https://dgpu-docs.intel.com/driver/client/overview.html)
- [Ollama Vulkan PR #11835](https://github.com/ollama/ollama/pull/11835)
- [IPEX-LLM archived repo](https://github.com/intel/ipex-llm)

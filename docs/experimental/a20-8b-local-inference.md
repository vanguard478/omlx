# Running 8B models on an Apple A20: what to take from oMLX

oMLX is a Mac server (Python, FastAPI, mlx-lm and mlx-vlm), so it cannot run on
an iPhone as it is. Most of its speed comes from a small set of parts that sit on
top of MLX. This note sorts those parts into what an on-device 8B runtime for an
Apple A20-class phone SoC should port, what needs changes, and what to leave out.
In this note, "A20" means Apple's A20 phone chip. The Allwinner A20 (dual
Cortex-A7, up to 2 GB RAM, no Metal) cannot hold an 8B model at any quantization.

## Constraints that drive the choices

- **Memory.** An 8B model at 4.75 bits per weight is about 4.8 GB. With iOS
  jetsam limits that only fits on a 12 GB device, and only with the
  `com.apple.developer.kernel.increased-memory-limit` entitlement. KV cache and
  activations have to come out of what is left.
- **Bandwidth.** Decode is bandwidth-bound: tokens/s ≈ effective GB/s ÷ weight
  bytes. Every byte saved on weights or KV turns straight into speed.
- **GPU tensor units (NAX).** oMLX's NAX gate
  (`omlx/custom_kernels/quant_matmul/fast.py`, `_NAX_ARCH_RE`) enables NAX for
  `applegpu_gNNp` (phone) GPUs from generation 18 (A19) and for Mac GPUs from
  generation 17 (M5). The A20's GPU is a later generation, so it should take
  the same NAX/TensorOps kernel paths as M5. The Python fallback also checks
  `platform.mac_ver() >= 26.2`. On iOS that call returns an empty string, so a
  port has to replace it with an OS version check (or use MLX's own
  `metal::is_nax_available()`).
- **Runtime.** Python on iOS is impractical. The target is **mlx-swift**
  (MLX, MLXNN, MLXLMCommon). JIT kernels written with `mx.fast.metal_kernel`
  carry over unchanged as `MLXFast.metalKernel`. Native C++/Metal extensions
  have to be compiled into the app's MLX build.

## Tier 1: port these (most of the gain for a dense 8B)

| Component | oMLX source | Why it matters on A20 | Port notes |
|---|---|---|---|
| **oQ / oQe mixed-precision quantization** | `omlx/oq.py`, `docs/oQ_Quantization.md` | Measures per-layer sensitivity and spends extra bits where they help (8-bit head and embeddings, more bits on sensitive attention projections). oQe adds imatrix-weighted rounding. oQ4e is about 4.7–4.8 bpw, about 4.8 GB for 8B. oQ3e (about 3.7 bpw, about 3.7 GB) leaves room for KV on a 12 GB phone. | **Run offline on a Mac.** The output is standard MLX safetensors, so mlx-swift loads it with no custom loader. Nothing to port, just ship the converted model. |
| **Affine4 / Affine8 KV cache** | `omlx/affine4.py`, `docs/affine4.md`, `docs/affine8.md` | Signed 4- or 8-bit KV with orthogonal rotation and one fp32 scale per vector. For head_dim 128, a K or V vector shrinks from 256 B to 68 B (about 3.8x). Llama-3.1-8B KV at 8K context drops from about 1 GB to about 0.27 GB. Decode attention runs on a Metal **TensorOps** kernel (`mpp::tensor_ops`), the path NAX GPUs accelerate, with a portable fallback. | All kernels are `mx.fast.metal_kernel` JIT, so they port directly to `MLXFast.metalKernel`. Keep `turboquant_skip_last` (last attention layer stays native). Compress during the prefill update so the full fp16 KV is never held in memory. |
| **Chunked prefill** | `omlx/scheduler.py` (`prefill_step_size`, `chunked_prefill`) | Splits long prompts into fixed chunks so peak activation memory stays bounded and the UI stays responsive. On a phone, an unbounded prefill gets the app killed by jetsam. | Use a smaller step on the phone (512–1024 instead of 2048). |
| **Prefix (prompt) cache, block-hashed** | `omlx/cache/paged_cache.py`, `omlx/cache/prefix_cache.py` | Chain-hashed KV blocks let a new turn reuse the system prompt and conversation history, so it skips re-prefill. On a phone, prefill compute is the scarce resource, so this is the largest TTFT win in multi-turn chat. | Drop the vLLM-style multi-request pool, ref-counting and COW. For a single user, keep `compute_block_hash` chain hashing and a small LRU of blocks. |
| **Cold KV tier on flash** | `omlx/cache/paged_ssd_cache.py` | KV blocks are serialized as safetensors and restored on a prefix match, even after restart. On iOS, the app is suspended and killed often, so a disk-backed prompt cache lets a conversation resume without re-prefilling. | Write to the app's Caches directory, cap the total size, and evict LRU. Keep the format/seed metadata check so stale or incompatible blocks become a miss. |
| **Memory guard / admission control** | `omlx/process_memory_enforcer.py`, `omlx/memory_monitor.py` | Admission and prefill-abort ceilings, an estimate of prefill transients, and `mx.set_wired_limit`/`mx.set_cache_limit` handling. This keeps requests from pushing the process over its memory limit. | Replace the macOS `sysctl iogpu.wired_limit_mb` / `vm_stat` logic with `os_proc_available_memory()`, and react to memory-warning notifications by clearing MLX's buffer cache. Keep the design (estimate, admit, abort gracefully) and drop the macOS specifics. |

## Tier 2: worth porting once Tier 1 works

| Component | oMLX source | Benefit | Notes |
|---|---|---|---|
| **DFlash block speculative decoding** | `omlx/speculative/dflash/`, `omlx/patches/mlx_lm_mtp/drafter_target.py`, `docs/experimental/dflash_mlx_integration.md` | A small drafter proposes a block of tokens from captured target hidden states, and the target verifies them in one forward. This turns bandwidth-bound decode into fewer weight passes. **Qwen3 dense** targets are supported in *single-row* mode, which is exactly the phone case (one user). | Needs a DFlash draft checkpoint for the target. The target-side changes are small: capture selected layer outputs and trim the KV on rejection. Memory cost is the drafter (quantize it, `dflash_draft_quant_*`). |
| **Multi-row verify qmv** | `omlx/patches/m1_verify_qmv.py`, `omlx/patches/row_exact_qmv.py` | Speculative verify multiplies a 2–9-row batch against the quantized weights. These kernels unpack each weight tile once and apply it to every row, instead of re-running the one-row qmv per row. They help only when speculative decoding is on. | JIT `metal_kernel`, so it can be ported. The tile and route thresholds were tuned on M1 / M3 Ultra, so re-measure on A20. `row_exact_qmv`'s narrower `one_row_qmv` tile, which spreads the weights over more threadgroups, is worth testing for plain decode on a GPU with few cores. |
| **SpecPrefill (sparse prefill)** | `omlx/patches/specprefill.py`, `omlx/specprefill/` | For long prompts, a draft model scores token importance and the target prefills only the top-K% at their original RoPE positions. Cuts TTFT on long documents. | Lossy. Use it only above a token threshold (`policy.py`). Needs a small draft model in memory. |
| **Decode SDPA kernel** | `omlx/custom_kernels/attention/csrc/sdpa_decode.*` | A port of mlx#4294 decode attention (query length ≤ 8, GQA, sinks). | Native C++ extension. Check first whether the MLX version you ship already includes the upstream change. If it does, skip this. |
| **Packed prefill** | `omlx/prefill/packed.py` | Packs several requests' chunks into one forward. | Only useful if the app runs background jobs (summarization, embeddings) alongside chat. Otherwise skip. |

## Leave out

| Component | Why it does not fit |
|---|---|
| ANE/GPU split prefill (`omlx/patches/qwen35_ane_prefill.py`, `quant_matmul/ane.py`) | Uses the private `AppleNeuralEngine.framework` (not allowed on the App Store), targets M3 Ultra's two ANEs, and only covers Qwen3.5/3.6/3.8. For ANE on iPhone, use Core ML instead. |
| Continuous batching, `batched_decode_qmm` (≥ 6 rows), engine pool, LRU model swapping | Built for many concurrent requests and many models. A phone serves one user and one model. |
| MoE expert offload, sparse/MLA/DSA attention, GDN kernels, `sdpa256_attention` | Model-specific to large MoE or hybrid models, or to head_dim 256. Dense Llama-3.1-8B and Qwen3-8B use head_dim 128 GQA and need none of them. |
| Multi-Mac cluster, admin UI, API server, tool-calling parsers, MCP | Not inference speed. (Tool-call parsing in `omlx/api/tool_calling.py` is worth copying only if the app does function calling.) |

## Rough budget for a 12 GB A20 phone

| Item | Llama-3.1-8B (32 layers, 8 KV heads, d=128) |
|---|---|
| Weights, oQ4e | ~4.8 GB (oQ3e: ~3.7 GB) |
| KV per token, fp16 | 128 KiB → 8K ctx ≈ 1.0 GB |
| KV per token, Affine4 | ~34 KiB → 8K ctx ≈ 0.27 GB |
| Prefill transients (chunk 512–1024) | a few hundred MB |
| Decode speed | about (effective bandwidth ÷ 4.8 GB). Measure on device. DFlash can multiply it by the mean accepted block length. |

## Order of work

1. Convert the model with oQ4e (or oQ3e) on a Mac and load it in mlx-swift with
   MLXLMCommon. Measure baseline decode tok/s and TTFT.
2. Add chunked prefill and the memory guard (handling for
   `os_proc_available_memory`, memory warnings and the MLX cache limit).
3. Port the Affine4 KV cache (JIT kernels plus codec) with the last layer kept
   native. Check output quality against fp16 KV.
4. Add the block-hashed prefix cache, then the on-flash cold tier.
5. Add DFlash (Qwen3 target) with the multi-row verify qmv, then SpecPrefill
   for long prompts.
6. Re-tune kernel route thresholds on A20 hardware. The values in oMLX were
   measured on M1 Max, M3 Ultra and M5.

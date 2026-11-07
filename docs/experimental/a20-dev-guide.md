# Apple A20 on-device AI: developer guide

A practical guide to developing on-device LLM apps for iPhones with Apple's A20
chip. It covers the chip architecture from an AI developer's point of view,
Apple's SDKs (compared with Qualcomm's SNPE), how to test without buying an
iPhone, and how much of the work can be done from a Linux machine.

For which oMLX components to port to an on-device 8B runtime, see
[a20-8b-local-inference.md](a20-8b-local-inference.md).

> **Status of the facts here (November 2026).** The A20 Pro was announced on
> September 9, 2026. Apple's own claims are marked as such. Numbers marked
> *unconfirmed* come from third-party spec tables or estimates. Measure on a
> real device before relying on them.

---

## 1. A20 architecture for AI developers

### The big picture: one memory pool, three kinds of compute

```
            ┌──────────────────── A20 Pro SoC (TSMC N2) ─────────────────────┐
            │  CPU: 2 performance + 4 efficiency cores                       │
            │  GPU: ~7 cores, each with a "Neural Accelerator" (matrix unit) │
            │  Neural Engine (ANE): fixed-function NPU                       │
            │  System-level cache (shared by all of the above)               │
            └───────────────┬────────────────────────────────────────────────┘
                            │  one shared bus
                  12 GB LPDDR5X on the package (unified memory)
```

All three compute blocks share the same RAM. Model weights, KV cache and
activations are never copied between CPU and GPU: MLX, Core ML and Metal all
use the same memory without copies. The flip side is that the OS, your app and
the model compete for the same 12 GB.

### Each block, and when to use it

**GPU with Neural Accelerators: the main engine for LLMs.**

- The A19 Pro added a matrix-multiply unit to every GPU core (the same design
  as the M5). The A20 Pro keeps it. Apple claims about 2x faster FP8 and about
  40% better sustained GPU performance than the A19 Pro.
- Programmable through Metal 4 tensor operations. MLX uses these units
  automatically. oMLX calls them "NAX", and its detection logic
  (`omlx/custom_kernels/quant_matmul/fast.py`) already treats A19-and-later
  phone GPUs as NAX-capable.
- **Best at prefill.** Prefill is matrix-times-matrix work, limited by compute,
  and these units speed it up a lot. Faster time to first token is the main
  gain from this hardware generation.
- **Helps much less with decode.** Generating one token at a time is a
  matrix-times-vector operation, limited by how fast weights stream from
  memory, not by math throughput.

**Neural Engine (ANE): efficient but rigid.**

- A separate NPU. Third-party tables list it as dual 16-core with about
  44 TOPS FP16 and 73 TOPS INT8 (*unconfirmed*).
- The only public way to use it is Core ML (or its successor, see section 2).
  Models need fixed shapes and work best with FP16 or INT8 weights. MLX cannot
  target the ANE through public APIs. oMLX's ANE path uses a private framework
  that cannot ship on the App Store.
- Lowest power per operation. Good for small always-on models (embeddings,
  classifiers, speech recognition) and fixed-shape prefill chunks. Poor fit for
  dynamic-length LLM decode with a growing KV cache.

**CPU: coordinator, not compute.**

- Two high-clock performance cores (reported up to about 4.9 GHz) and four
  efficiency cores, with Arm matrix extensions reachable through Accelerate.
- Use it for tokenization, sampling, grammar-constrained decoding and app
  logic. Do not put an 8B forward pass on it.

**Memory: the number that sets decode speed.**

- 12 GB LPDDR5X. Apple says bandwidth is up about 50% over the A19 Pro and
  calls it the widest memory interface in an iPhone.
- The exact GB/s figure is *unconfirmed*. One third-party estimate is about
  115 GB/s, derived from bus width and memory speed.
- The weights do not fit in the system cache, so every decoded token reads the
  whole model from RAM.

### The one formula to remember

Decode speed ≈ **memory bandwidth ÷ bytes read per token**.

| Model | Weights | Theoretical ceiling at ~115 GB/s | Realistic (~60–70% efficiency) |
|---|---|---|---|
| 8B at 4-bit (oQ4) | ~4.8 GB | ~24 tok/s | ~14–17 tok/s |
| 8B at 3-bit (oQ3) | ~3.7 GB | ~31 tok/s | ~18–22 tok/s |
| 3B at 4-bit | ~1.8 GB | ~64 tok/s | ~35–45 tok/s |

These are estimates. They explain why these techniques matter on a phone:

- Weight quantization (oQ) reduces bytes read per token.
- KV cache compression (Affine4) reduces the KV bytes read per token, which
  grow with context length.
- Speculative decoding (DFlash) gets several tokens out of each pass over the
  weights.
- A prefix cache avoids repeating prefill.

### Phone-specific constraints that Mac tooling ignores

- **Memory limits.** iOS kills apps that use too much memory ("jetsam"). An 8B
  model needs the `com.apple.developer.kernel.increased-memory-limit`
  entitlement and a 12 GB device. Budget about 5–6 GB for everything
  model-related, and use `os_proc_available_memory()` and memory warnings to
  stay safe.
- **Heat.** A phone sustains only a few watts. Peak speed drops after tens of
  seconds of continuous generation. Benchmark long runs, not short bursts.
- **No GPU in the background.** iOS restricts Metal work when the app is not in
  the foreground. Plan for generation to pause or be cancelled when the user
  leaves the app.
- **Battery.** For background-style tasks, the ANE through Core ML or a smaller
  model often beats a faster but power-hungry GPU path.

---

## 2. Apple's SDK: the equivalent of Qualcomm SNPE

The Apple equivalent of SNPE is **Core ML together with Core ML Tools
(`coremltools`)**: the vendor's official SDK for converting a trained model,
optimizing it, and running it on the chip's CPU, GPU and Neural Engine.

> **Core AI (reported, not fully confirmed).** Reports say Apple introduced
> **Core AI** at WWDC 2026 as the successor to Core ML, starting with iOS 27,
> built for LLMs, diffusion models and transformers. These reports come from
> third-party sites; check developer.apple.com before relying on them. Core ML
> reportedly keeps working, but new AI features would ship only in Core AI.

### How SNPE maps to Apple's stack

| Step | Qualcomm (SNPE / QNN) | Apple (Core ML) |
|---|---|---|
| Convert a model | `snpe-onnx-to-dlc`, `snpe-pytorch-to-dlc` | `coremltools.convert()` from PyTorch |
| Model file | `.dlc` | `.mlpackage`, compiled to `.mlmodelc` |
| Quantize | `snpe-dlc-quantize` (INT8 calibration) | `coremltools.optimize`: linear INT8/INT4, palettization (weight lookup tables), pruning |
| Choose where it runs | Runtime: CPU, GPU, DSP/HTP | `MLComputeUnits`: `.cpuOnly`, `.cpuAndGPU`, `.cpuAndNeuralEngine`, `.all` |
| Profile | `snpe-diagview`, Snapdragon Profiler | Xcode Core ML performance report, Instruments Core ML template |
| On-device runtime API | SNPE C/C++ / Java API | Core ML Swift/Objective-C API (`MLModel`) |
| Model hub | Qualcomm AI Hub | Apple's Core ML models page, Hugging Face (`apple/`, `coreml-community`) |

### The key difference

- **Qualcomm lets you reach the NPU directly.** QNN, SNPE's successor, has an
  HTP backend you can target explicitly.
- **Apple has no public low-level Neural Engine API.** You hand Core ML a model
  and a preference such as `.cpuAndNeuralEngine`. Its scheduler decides which
  layers run on the ANE and may fall back to GPU or CPU for unsupported ops or
  dynamic shapes.
- That is why on-device LLMs on Apple usually run on the GPU, not the ANE
  (MLX, llama.cpp and oMLX all do). Core ML's own LLM support has improved:
  stateful KV cache, INT4 weights and multifunction models arrived in iOS 18.

### Other Apple frameworks

| Framework | CPU | GPU and Neural Accelerators | ANE | Use for |
|---|---|---|---|---|
| MLX Swift | yes | **yes** | no | The main LLM path |
| Metal 4 (custom kernels, tensor ops) | — | yes | no | Custom kernels for the Neural Accelerators |
| Core ML (Core AI on iOS 27) | yes | yes | **yes (only public route)** | Small helper models |
| Accelerate / BNNS Graph | yes | — | — | Small low-latency CPU models (e.g. audio) |
| Foundation Models framework | Apple's own ~3B on-device model | | | Cannot load your own weights |

**Recommendation:** run the 8B LLM on the GPU through MLX Swift. Use Core ML
(or Core AI) for small helper models such as embeddings, speech recognition or
classifiers, where the ANE's power efficiency pays off.

---

## 3. Testing without buying an iPhone

You can do most development without an iPhone. Speed, memory limits and
thermal throttling can only be measured on a real A20 device.

### Option 1: an Apple Silicon Mac (best stand-in for development)

- **Same software.** MLX Swift, Core ML and Metal run the same code on macOS and
  iOS. An iPhone build can run natively on an Apple Silicon Mac ("Designed for
  iPad" mode).
- **Same GPU generation on M5 Macs.** The M5 has the same per-core Neural
  Accelerators as the A19/A20, so the NAX / Metal 4 tensor kernel paths run and
  can be checked for correctness.
- **What a Mac cannot tell you:** phone speed (more bandwidth, more GPU cores,
  better cooling), iOS memory limits (jetsam), and background GPU restrictions.
  Cap MLX's memory to mimic the phone budget; a base M5 gets closest to
  phone-class hardware.

### Option 2: the iOS Simulator (UI work only)

- **MLX does not run in the Simulator.** It lacks the GPU support MLX needs and
  crashes in confusing ways rather than giving a clear error. The MLX
  maintainers document this.
- Core ML models run there, but on the Mac's CPU, with no Neural Engine and
  nothing like phone performance.
- Use it for UI and app logic, with a mock inference backend.

### Option 3: cloud real-device farms (final validation)

These services rent real iPhones remotely:

- BrowserStack (App Live for manual tests, App Automate for automated tests)
- AWS Device Farm
- Sauce Labs, LambdaTest

Check before relying on one:

- **A20 availability.** No service was confirmed to list the iPhone 18 Pro as
  of this writing. A19 Pro phones (iPhone 17 Pro) are more likely to be
  available and share the same GPU design.
- **Signing.** You need a paid Apple Developer Program membership ($99/year) to
  produce a signed `.ipa` and to add the increased-memory-limit entitlement.
- **Upload size.** Do not bundle ~5 GB of weights into the app. Download them on
  first launch.
- **Profiling.** Most farms give logs and video, not full Xcode Instruments.
  Have the app log its own metrics: tokens/s, time to first token, memory
  headroom and thermal state.
- **Reliability.** Shared devices may be warm or busy. Repeat each run.

---

## 4. Developing from Linux

Building, signing and GPU-testing an iPhone app needs **macOS on Apple
hardware**. You do not have to own a Mac: renting one covers it.

### What needs a Mac, and why

- **Xcode and the iOS SDK only run on macOS.** That covers compiling the app,
  signing it, and compiling the Metal shaders MLX depends on.
- **MLX's GPU path needs an Apple GPU.** Even a working macOS VM on a Linux PC
  would not run it.
- **macOS VMs on PC hardware** (OSX-KVM, Docker-OSX) break Apple's license and
  still have no Apple GPU, so they do not help.

### What works on Linux

| Task | Works on Linux? | Notes |
|---|---|---|
| Convert and quantize the model | **Yes** | MLX publishes Linux wheels (`mlx 0.32.4` installs on x86_64 Linux), with CPU and CUDA backends. `mlx_lm.convert` writes the same MLX safetensors files the iPhone app loads. |
| oQ / oQe from this repo | **Probably not fully** | oQe's refit step uses a Metal-only kernel (`omlx/oq.py`, `_lsq_refit_kernel`). Plain oQ might work but is untested on Linux. |
| Check quantization quality (perplexity, sample outputs) | **Yes** | CPU, or CUDA with an NVIDIA GPU. |
| Affine4 KV cache numerics | **Partly** | The plain-MLX fallback runs, so accuracy can be checked. The Metal TensorOps kernels cannot. |
| App logic that does not touch the GPU | **Yes** | Prefix cache, memory-budget policy, sampling, tokenizer, chat templates. Keep it platform-independent and unit-test it on Linux. |
| Build, sign and install the iOS app | **No** | Experimental Linux toolchains for iOS apps exist, but none compile MLX's Metal shaders reliably. |
| Speed, memory and heat measurements | **No** | Need a real or rented iPhone. |

### Getting macOS without buying a Mac

**Bare-metal cloud Macs, reached over SSH or VNC.** Pick bare-metal, not VMs:
virtualized macOS often has no GPU access.

- **AWS EC2 Mac:** dedicated Mac minis, 24-hour minimum (Apple licensing).
- **Scaleway Apple Silicon:** dedicated Mac minis, hourly after a 24-hour
  minimum.
- **MacStadium, MacinCloud:** monthly or hourly plans.

Ask for an **M5** machine if offered: it has the same GPU Neural Accelerators
as the A20, so the fast kernel paths run.

**CI services** (GitHub Actions macOS runners, Codemagic, Bitrise) suit
automated builds and signing only. Their virtualized Macs usually cannot use
the GPU, so MLX GPU tests fail there.

**Real devices.** The device farms in section 3 accept a signed `.ipa`. If you
own an iPhone, Linux tools such as `libimobiledevice` can install an
already-signed `.ipa` on it.

---

## 5. Recommended workflow

1. **Linux:** convert and quantize the model with `mlx_lm.convert`, evaluate its
   quality, and write and unit-test the code that does not touch the GPU.
2. **Rented bare-metal Mac (in focused 24-hour blocks):** build the app in Xcode,
   run the MLX GPU path, and sign the `.ipa`.
3. **iOS Simulator:** UI work against a mock inference backend.
4. **Device farm, or later your own iPhone:** measure tokens/s, time to first
   token, memory headroom and thermal throttling. Use an A20 phone if
   available, otherwise an A19 Pro.
5. **If the project runs for months:** a used Mac mini usually costs less than
   repeated cloud rentals. An M1 is fine for builds; an M5 matches the A20's GPU
   features. For thermal and long-run memory behavior, a used or refurbished
   iPhone 17 Pro or 18 Pro is worth buying.

---

## Sources

A20 / A19 hardware:

- [Tom's Hardware: Apple A20 Pro, Apple's first 2nm smartphone chip](https://www.tomshardware.com/pc-components/cpus/apple-a20-pro-powers-iphone-18-pro-the-companys-first-2-nanometer-smartphone-chip)
- [iGeeksBlog: Apple A20 Pro, everything you need to know](https://www.igeeksblog.com/apple-a20-pro-everything-you-need-to-know/)
- [Beebom: Apple A20 Pro benchmarks and specs](https://gadgets.beebom.com/guides/apple-a20-pro-benchmark-specs)
- [Nanoreview: Apple A20 Pro specs](https://nanoreview.net/en/soc/apple-a20-pro)
- [Apple Wiki (Fandom): Apple A20 Pro](https://apple.fandom.com/wiki/Apple_A20_Pro)
- [Wikipedia: Apple A19](https://en.wikipedia.org/wiki/Apple_A19)
- [Macworld: A19, N1 and C1X](https://www.macworld.com/article/2905765/the-a19-n1-and-c1x-the-drumbeat-of-impressive-apple-silicon-continues.html)
- [Jon Peddie Research: Apple's A19 and upcoming M5 chips](https://www.jonpeddie.com/news/apples-a19-and-upcoming-m5-chips/)

Core ML / Core AI:

- [AppleInsider: WWDC 2026 to introduce Core AI as replacement for Core ML](https://appleinsider.com/articles/26/03/01/wwdc-2026-to-introduce-core-ai-as-replacement-for-core-ml)
- [9to5Mac: Apple replacing Core ML with modernized Core AI framework for iOS 27](https://9to5mac.com/2026/03/01/apple-replacing-core-ml-with-modernized-core-ai-framework-for-ios-27-at-wwdc/)
- [Let's Data Science: Apple unveils Core AI for on-device generative models](https://letsdatascience.com/news/apple-unveils-core-ai-for-on-device-generative-models-1142390f)
- [eCorpIT: iOS 27 Core AI, 5 changes for developers replacing Core ML](https://ecorpit.com/core-ai-framework-ios-27-replaces-core-ml/)
- [Stora: Core AI replacing Core ML, WWDC 2026 migration plan](https://stora.sh/blog/2026-05-10-core-ai-replacing-core-ml-wwdc-2026-migration-plan)

Simulator and device testing:

- [Compiled Thoughts: integrating local LLMs into iOS apps with MLX Swift](https://compiledthoughts.pages.dev/blog/integrating-mlx-local-llms-ios-apps)
- [mlx issue #2605: block GPU access on iOS simulator](https://upd.dev/ml-explore/mlx/issues/2605)
- [Apple Developer Forums: Metal 4 support in iOS simulator](https://developer.apple.com/forums/thread/802172)
- [BrowserStack: iOS app testing](https://browserstack.com/ios-app-testing)
- [AWS Device Farm](https://aws.amazon.com/device-farm/?p=tile)

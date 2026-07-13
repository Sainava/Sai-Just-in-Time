
### Category 1: Environment-Dependent Parameters

These parameters are entirely dictated by your physical hardware constraints and environmental boundaries. They must remain completely frozen across your Vanilla Baseline, DiCache, JiT, and Combined runs to ensure a mathematically fair comparison.

| Parameter | Standardized Value | Research / Systems Justification |
| --- | --- | --- |
| **Model Precision** | `torch.float16` + `load_in_8bit=True` | **VRAM Ceiling Enforcement:** Native FP16 FLUX parameters require $\sim$24GB VRAM. Compressing the transformer block to 8-bit drops the weight footprint to $\sim$12GB, allowing the system to run on the T4's 16GB allocation while introducing a critical thesis theme: *Robustness under severe quantization truncation noise*. |
| **Device Mapping** | Explicit Custom Dict Map:<br><br>`{ "text_encoder": 1, "text_encoder_2": 1, "vae": 0, "transformer": 0 }` | **Telemetry Isolation:** Bypasses the catastrophic PCIe bus transfer overhead caused by `enable_sequential_cpu_offload()`. By physically separating text token processing (GPU 1) from the iterative denoising transformer loop (GPU 0), we isolate our wall-clock latency metrics from external system memory noise. |
| **Resolution** | $1024 \times 1024$ | **Architectural Alignment:** The native training aspect ratio for `FLUX.1-dev`. Any variation introduces unaligned spatial patchification metrics during token reduction phases. |
| **Prompt Pack** | The 5 Stratified Golden Prompts | **Vulnerability Stress Testing:** Evaluates high-frequency spatial detail, structural boundaries, typography rendering, and low-frequency coherence without requiring a massive, compute-prohibitive dataset. |
| **Random Seeds** | `[42, 1024, 2048, 777, 999]` | **Latent Space Determinism:** Locks the initial Gaussian noise vectors. This ensures that any observed structural degradation or visual artifact is strictly caused by the acceleration algorithms, not random initialization. |



### Category 2: Coherent Parameters

These parameters are structurally and operationally identical between the underlying model implementations or share identical conventions that require zero code transformation to standardize.

| Parameter | Standardized Value | Source Grounding & Justification |
| --- | --- | --- |
| **Base Model Architecture** | `FluxTransformer2DModel` | **Unified Target:** Both repositories natively use the standard diffusers Diffusion Transformer backbone. This ensures structural adjustments map to the exact same double-stream and single-stream block architecture. |
| **Model Checkpoint** | `black-forest-labs/FLUX.1-dev` | **Checkpoint Parity:** Both repositories hardcode this exact HuggingFace model string for text-to-image pipeline loading. |
| **VAE Optimization** | `enable_vae_tiling()` | **Memory Safety Alignment:** Implemented natively in JiT's call script and explicitly integrated during DiCache's optimization runtime to prevent VRAM allocation limits from breaking during full-resolution chunked decoding. |
| **Output Format / Type** | `pil` (Saved as PNG image) | **API Agreement:** Both default pipeline generation loops yield standard PIL images mapped by prompt naming conventions. |
| **Max Token Sequence Length** | `256` | **Encoding Boundary:** Extracted from JiT's explicit invocation signature. Standardizing DiCache to this parameter ensures identical text embedding vector sizing. |

---

### Category 3: Incoherent Parameters & System Clashes

These parameters represent fundamental algorithmic conflicts between the two repositories. They are handled using different math, variables, or integration hooks. To achieve a valid standardized baseline, we must design an explicit architectural reconciliation for each clash.

| Incoherent Variable | DiCache Setup | Just-in-Time (JiT) Setup | The System Resolution for Phase 5 |
| --- | --- | --- | --- |
| **Integration Architecture** | **Monkey-Patching**<br><br>Replaces `transformer.forward = dicache_forward` inline. | **Pipeline Subclassing**<br><br>Overrides `__call__` via custom `FluxPipeline_JiT` class inheritance. | **Hierarchical Hooking:** For the combined run, the notebook must first load `FluxPipeline_JiT` to inherit the spatial token logic, then apply the `dicache_forward` monkey-patch directly onto its transformer component. |
| **Inference Step Count & Flow** | **30 Steps**<br><br>Relies on a fixed, uniform step schedule throughout the loop. | **18 Steps (`default_4x`)**<br><br>Relies on a tight, compressed schedule. | **Bimodal Scheduling:** You must run two separate test matrices. Matrix A freezes the loop at **18 steps**, forcing DiCache to find matching historical trajectories in high-speed schedules. Matrix B shifts to **30 steps**, forcing JiT to stretch its token sparsity across a finer grid. |
| **ODE Integration & Schedulers** | **Diffusers Native**<br><br>Delegates entirely to `FlowMatchEulerDiscreteScheduler`. | **Manual Integration**<br><br>Bypasses `.step()` for custom manual Euler update loops and beta timesteps. | **Track-Isolated Schedulers:** During standalone tests, each uses its native scheduler loop. For the combined run, DiCache's residual window math must be manually mapped inside JiT's custom `y_full = y_full + velocity_full * dt` manual iteration loop. |
| **CFG Guidance Logic** | **Implicit Distillation**<br><br>Inherits pipeline defaults; no explicit scalar passed. | **Explicit Embeddings**<br><br>Hardcodes `guidance_scale=3.5` via native guidance tensor injection. | **Unified Scalar Injection:** Lock all benchmark variations to an explicit `guidance_scale=3.5` parameter during the pipeline functional call. |
| **Caching/Sparsity Hyperparameters** | **Dynamic Error Evaluation**<br><br>`probe_depth=1`, `rel_l1_thresh=0.4`, `ret_ratio=0.2`. | **Multi-Stage Presets**<br><br>`stage_ratios=[0.4, 0.65, 1.0]`, `sparsity_ratios=[0.35, 0.62, 1.0]`. | **Co-existence Mapping:** The parameters do not overlap operationally. In the combined run, they operate simultaneously: JiT prunes tokens spatially based on its stage arrays, while DiCache evaluates temporal block skipping based on its relative $L_1$ threshold metrics. |


### Category 3: The Singular Phase 5 Incoherent Standard

| Parameter | Standardized Choice | Research & Systems Justification |
| --- | --- | --- |
| **Global Inference Steps** | **18 Steps** | **Strictest Bounds Testing:** JiT's entire mathematical contribution is built around hyper-accelerated schedules ($18$ steps). If an acceleration algorithm like DiCache claims to be robust, it must prove it can function within these fast schedules where the ODE trajectory is steep and compression noise is highest. |
| **Scheduler & Integration** | **JiT Manual Euler Loop** + Custom Beta Schedule | **Architecture Overriding:** Because JiT alters the base pipeline structure via a custom subclass (`FluxPipeline_JiT`), the combined code must execute through its manual integration loop (`y_full = y_full + velocity_full * dt`). DiCache will be evaluated by monkey-patching directly inside this manual step block. |
| **Guidance Scale (CFG)** | **3.5** | **Baseline Control:** Explicitly locked to JiT's hardware default, forcing DiCache to process identical guidance-embedded tensor profiles. |
| **DiCache Hyperparameters** | `probe_depth = 1`<br><br>`error_choice = 'delta_y'`<br><br>`rel_l1_thresh = 0.4`<br><br>`ret_ratio = 0.2` | **Empirical Baseline:** Extracted directly from DiCache's verified Phase 4 reproduction matrix. |
| **JiT Hyperparameters** | `default_4x` bundle (Strided checkerboard, adaptive densification, microflow steps = 3) | **Empirical Baseline:** Extracted directly from JiT's verified Phase 4 reproduction configuration. |

---

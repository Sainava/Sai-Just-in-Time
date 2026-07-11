### Phase 4 — Reproduction Configuration Matrix (JiT)

**Objective:** Execute the exact default configuration provided by the `Wenhao-Sun77/Just-in-Time` repository to verify baseline acceleration (4x and 7x) and image quality on FLUX.1-dev.

| Parameter | Configuration Value | Source Justification (Phase 2 & 3 Audits) |
| --- | --- | --- |
| **Backbone Model** | `black-forest-labs/FLUX.1-dev` | The default model ID in `flux/infer.py` (L33). |
| **Pipeline Wrapper** | `FluxPipeline_JiT` | The custom subclass overriding `__call__` for SAG-ODE execution. |
| **Precision** | `torch.float16` | Passed to `from_pretrained` (L34). Mixed precision is handled internally by JiT. |
| **Resolution** | `1024 x 1024` | Hardcoded constraint in `flux/infer.py`; must be divisible by 16. |
| **Random Seed** | `42` | The hardcoded seed used in their demo script for reproducibility. |
| **Guidance Scale (CFG)** | `3.5` | Native to FLUX.1-dev (uses guidance embeddings rather than dual-pass CFG). |
| **Sequence Length** | `256` | Set explicitly as `max_sequence_length=256` in the pipeline call. |
| **Target Prompt** | *"A grand piano made entirely of transparent..."* | The exact hardcoded prompt in `flux/infer.py` (L58) to replicate their demo output. |
| **VAE Optimization** | Enabled | `self.enable_vae_tiling()` is called to prevent memory spikes during decode. |

### The Two Execution Profiles

To fully verify JiT's claims, you must execute two distinct runs using their predefined parameter bundles.

| JiT Hyperparameter | Profile 1: `default_4x` | Profile 2: `default_7x` |
| --- | --- | --- |
| **Total Inference Steps** | 18 | 11 |
| **Stage Ratios** (Transition Timing) | `[0.4, 0.65, 1.0]` | `[0.4, 0.65, 1.0]` |
| **Sparsity Ratios** (Token Density) | `[0.35, 0.62, 1.0]` | `[0.32, 0.6, 1.0]` |
| **Token Init Strategy** | Checkerboard (`True`) | Checkerboard (`True`) |
| **Adaptive Densification** | Enabled (`True`) | Enabled (`True`) |
| **Custom Beta Schedule** | Enabled (`alpha=1.4, beta=0.425`) | Enabled (`alpha=1.4, beta=0.425`) |
| **Microflow Interpolation Steps** | 3 | 3 |

---

### How to use this for Phase 4 Execution:

1. **Environment:** Ensure you are running `diffusers==0.37.0` and `torch==2.8.0` as pinned in their requirements, otherwise their pipeline subclass will crash due to missing parent attributes.
2. **Telemetry Setup:** Before running, you must wrap `time.perf_counter()` around the `__call__` method. Do not measure model loading time; you only want the latency of the denoising loop.
3. **The Output Check:** Run both Profiles (`--preset default_4x` and `--preset default_7x`). You should expect the 4x profile to maintain high visual fidelity, while the 7x profile may exhibit slight background blurring due to the aggressive 0.32 initial sparsity ratio.

Once you have verified JiT runs locally and matches their claims, provide the Phase 2 and 3 audits for **DiCache**. I will then generate DiCache's reproduction parameters so we can prepare for the Phase 5 (Standardized Benchmark) collision.
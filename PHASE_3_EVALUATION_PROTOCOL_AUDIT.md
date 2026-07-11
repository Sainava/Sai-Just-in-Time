# PHASE 3 — Evaluation Protocol Audit: Just-in-Time (JiT)

> **Interoperability Audit Protocol (IAP) — Deliverable 2**
> Audited repository: `Wenhao-Sun77/Just-in-Time`
> All information derived exclusively from repository source code, configuration files, and README.

---

## Default Model Checkpoints

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Default checkpoint (FLUX.1-dev) | `black-forest-labs/FLUX.1-dev` (HuggingFace model ID) | `flux/infer.py` (L33) | High | `model_path = "black-forest-labs/FLUX.1-dev"` — comment says "Replace this with your local path if you are loading the model offline". |
| Default checkpoint (FLUX.2-Klein) | `black-forest-labs/FLUX.2-klein-base-9B` (HuggingFace model ID) | `flux2-klein-base-9B/infer_flux2.py` (L33) | High | `model_path = "black-forest-labs/FLUX.2-klein-base-9B"` — same offline loading comment. |
| Model loading method | `from_pretrained(model_path, torch_dtype=torch.float16)` | `flux/infer.py` (L34), `flux2-klein-base-9B/infer_flux2.py` (L34) | High | Standard diffusers `from_pretrained` with explicit float16 dtype. |

---

## Default Pipeline & Inference Scripts

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Default pipeline (FLUX.1-dev) | `FluxPipeline_JiT` | `flux/infer.py` (L1, L34) | High | `from pipeline_flux_JiT import FluxPipeline_JiT` |
| Default pipeline (FLUX.2-Klein) | `Flux2KleinPipeline_JiT` | `flux2-klein-base-9B/infer_flux2.py` (L1, L34) | High | `from pipeline_flux2_klein_JiT import Flux2KleinPipeline_JiT` |
| Default inference script (FLUX.1-dev) | `flux/infer.py` | `README.md` (L88-L89) | High | `python flux/infer.py --preset default_4x --gpu_id 0` |
| Default inference script (FLUX.2-Klein) | `flux2-klein-base-9B/infer_flux2.py` | `README.md` (L101-L102) | High | `python flux2-klein-base-9B/infer_flux2.py --preset default_4x --gpu_id 0` |

---

## Resolution

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Default resolution (FLUX.1-dev) | `1024×1024` | `flux/infer.py` (L56) | High | `resolution = 1024`, passed as both `height=resolution` and `width=resolution` (L66-L67). |
| Default resolution (FLUX.2-Klein) | `1024×1024` | `flux2-klein-base-9B/infer_flux2.py` (L51) | High | `resolution = 1024`, passed as `height=resolution, width=resolution` (L60-L61). |
| Resolution constraint | Must be divisible by 16 | `flux/pipeline_flux_JiT.py` (L605-L606) | High | `if height % 16 != 0 or width % 16 != 0: raise ValueError(...)` |
| README stated resolution | `1024×1024` | `README.md` (L95) | High | "generate a `1024x1024` image from the built-in prompt" |

---

## Number of Inference Steps

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Default steps — `default_4x` preset | **18** | `flux/pipeline_flux_JiT.py` (L132), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L77) | High | `total_steps = 18` in both pipeline set_params implementations. |
| Default steps — `default_7x` preset | **11** | `flux/pipeline_flux_JiT.py` (L122), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L70) | High | `total_steps = 11` in both pipeline set_params implementations. |
| Default steps if no preset | Pipeline `__call__` default is `num_inference_steps=28` | `flux/pipeline_flux_JiT.py` (L558), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L410) | Medium | This is the function signature default, but `set_params` overrides this via `total_steps`. |
| Default preset (both scripts) | `default_4x` (argparse default) | `flux/infer.py` (L13), `flux2-klein-base-9B/infer_flux2.py` (L13) | High | `parser.add_argument("--preset", type=str, default="default_4x", ...)` |
| README preset table | `default_4x`: 18 steps, `default_7x`: 11 steps | `README.md` (L113-L116) | High | Documented in the "JiT Presets" table. |

---

## Scheduler

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Scheduler (FLUX.1-dev) | Flow-matching Euler (inherited from `FluxPipeline`) — but JiT performs **manual Euler integration** with custom beta schedule when `use_beta_sigmas=True` | `flux/pipeline_flux_JiT.py` (L700-L728, L854) | High | When `use_beta_sigmas=True`: directly computes sigmas via `scheduler._convert_to_beta()`, sets `scheduler.sigmas`, derives timesteps, then manually does `y_full = y_full + velocity_full * dt`. When `use_beta_sigmas=False`: uses `retrieve_timesteps` with `mu` shift. |
| Scheduler (FLUX.2-Klein) | `FlowMatchEulerDiscreteScheduler` (explicitly imported) — uses standard `scheduler.step()` | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L26, L724) | High | `from diffusers.schedulers import FlowMatchEulerDiscreteScheduler`. Timesteps computed via `compute_empirical_mu` + `retrieve_timesteps`. |
| Beta schedule parameters (FLUX.1-dev) | `alpha=1.4, beta=0.425` (in code), `beta=0.42` (in README) | `flux/pipeline_flux_JiT.py` (L129, L139), `README.md` (L155) | High | Minor discrepancy: code uses 0.425, README says 0.42. Both presets use `use_beta_sigmas=True`. |
| Beta schedule (FLUX.2-Klein) | **Not used** — no `use_beta_sigmas` parameter in FLUX.2-Klein's `set_params` | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L47-L96) | High | README (L159): "For FLUX.2-klein-base-9B, we do not apply this custom beta schedule." |
| Sigma fallback | `sigmas = np.linspace(1.0, 1/num_inference_steps, num_inference_steps)` when no custom sigmas provided and scheduler doesn't use `use_flow_sigmas` | `flux/pipeline_flux_JiT.py` (L702), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L553) | High | Both pipelines share this fallback pattern. |

---

## CFG / Guidance Scale

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Default guidance scale (FLUX.1-dev) | `3.5` | `flux/infer.py` (L70), `flux/pipeline_flux_JiT.py` (L560) | High | `guidance_scale=3.5` in both inference script call and `__call__` signature default. |
| Default guidance scale (FLUX.2-Klein) | `4.0` | `flux2-klein-base-9B/infer_flux2.py` (L64), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L412) | High | `guidance_scale=4.0` in both inference script call and `__call__` signature default. |
| CFG type (FLUX.1-dev) | Guidance embeddings (not traditional CFG) — a single forward pass with `guidance` scalar | `flux/pipeline_flux_JiT.py` (L732-L736) | High | `if self.transformer.config.guidance_embeds: guidance = torch.full([1], guidance_scale, ...)`. |
| CFG type (FLUX.2-Klein) | Traditional classifier-free guidance — dual forward pass (cond + uncond) | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L688-L706) | High | `if self.do_classifier_free_guidance:` runs unconditional pass, then `noise_pred_active = neg + gs*(cond - neg)`. |

---

## Precision & Quantization

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Default precision | `torch.float16` (fp16) | `flux/infer.py` (L34), `flux2-klein-base-9B/infer_flux2.py` (L34) | High | `torch_dtype=torch.float16` passed to `from_pretrained`. |
| Internal computation precision | Mixed — intermediate computations cast to `torch.float32` for Tweedie prediction, then back to original dtype | `flux/pipeline_flux_JiT.py` (L531-L534), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L396-L400) | High | `_predict_x0_latent`: `latents.to(torch.float32) - noise_pred.to(torch.float32) * timestep.to(torch.float32) * 1e-3).to(latents_dtype)`. |
| MPS dtype fix | Explicit dtype preservation for MPS backend | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L726-L728) | Medium | `if torch.backends.mps.is_available(): y_full = y_full.to(latents_dtype)` |
| Quantization | **None** — no quantization applied anywhere in the codebase | All `.py` files | High | No bitsandbytes, no int8/int4, no quantization config. |

---

## Runtime & Memory Optimizations

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| `torch.no_grad()` | Applied to `__call__` method via decorator | `flux/pipeline_flux_JiT.py` (L551), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L402) | High | `@torch.no_grad()` decorator on `__call__`. |
| `torch.compiler.disable` | Applied to `progress_bar` method (FLUX.1-dev) | `flux/pipeline_flux_JiT.py` (L537) | High | `@torch.compiler.disable` — prevents dynamo from tracing the progress bar. |
| VAE tiling | **FLUX.1-dev only** — `self.enable_vae_tiling()` called at inference start | `flux/pipeline_flux_JiT.py` (L587) | High | Memory optimization for large image decoding. Not used in FLUX.2-Klein. |
| Precomputed coordinate grid | Cached for interpolation to avoid recomputation | `flux/pipeline_flux_JiT.py` (L319-L329, L747), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L193-L203, L582) | High | `_precompute_coords`: computes once and caches `self._coords_full`. |
| Global noise pre-generation | Single noise tensor pre-generated and reused for DMF target computation | `flux/pipeline_flux_JiT.py` (L691-L698), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L567-L571) | High | `global_noise = torch.randn(...)` created before loop, indexed by `newly_activated` during transitions. |
| Velocity caching | Last velocity field cached for importance map computation and x₀ prediction | `flux/pipeline_flux_JiT.py` (L750, L850-L851), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L584-L585, L685-L686) | High | `last_velocity_full = velocity_full` and `last_y = y_full.clone()`. |
| Model hooks cleanup | `self.maybe_free_model_hooks()` called after generation | `flux/pipeline_flux_JiT.py` (L884), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L772) | High | Standard diffusers memory cleanup. |
| torch.compile | Not applied | All `.py` files | High | No `torch.compile()` calls. Only `torch.compiler.disable` on progress_bar. |
| Additional memory optimizations | None beyond standard diffusers (no CPU offloading configured, no attention slicing, no sequential offloading) | All `.py` files | High | Though parent class methods for offloading are available via inheritance, they are not invoked in the demo scripts. |

---

## Configuration Files & Parameters

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Configuration format | **Programmatic only** — no YAML/JSON/TOML config files. All configuration via `set_params()` method or argparse CLI. | Repository-wide | High | No config files found in repository. |
| Preset configuration: `default_4x` | `total_steps=18, stage_ratios=[0.4,0.65,1.0], sparsity_ratios=[0.35,0.62,1.0], use_checkerboard_init=True, use_adaptive=True, use_beta_sigmas=True (FLUX.1 only), alpha=1.4, beta=0.425, microflow_relax_steps=3` | `flux/pipeline_flux_JiT.py` (L131-L140), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L76-L82) | High | Note: FLUX.2-Klein `default_4x` does not include `use_beta_sigmas`, `alpha`, or `beta` parameters. |
| Preset configuration: `default_7x` | `total_steps=11, stage_ratios=[0.4,0.65,1.0], sparsity_ratios=[0.32,0.6,1.0], use_checkerboard_init=True, use_adaptive=True, use_beta_sigmas=True (FLUX.1 only), alpha=1.4, beta=0.425, microflow_relax_steps=3` | `flux/pipeline_flux_JiT.py` (L121-L130), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L69-L75) | High | Same backbone-specific differences as above. |
| Configuration hierarchy | 1. Preset overrides all, 2. Individual params via `set_params()`, 3. Pipeline `__call__` defaults | `flux/pipeline_flux_JiT.py` (L94-L159) | High | If `preset` is set, it overrides all other arguments to `set_params`. |
| CLI arguments (FLUX.1-dev) | `--preset` (default: `default_4x`), `--gpu_id` (default: `0`) | `flux/infer.py` (L8-L22) | High | Two command-line arguments only. |
| CLI arguments (FLUX.2-Klein) | `--preset` (default: `default_4x`), `--gpu_id` (default: `0`) | `flux2-klein-base-9B/infer_flux2.py` (L8-L22) | High | Identical CLI structure. |

---

## Launch Commands

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| FLUX.1-dev launch (4x) | `python flux/infer.py --preset default_4x --gpu_id 0` | `README.md` (L88) | High | Documented in README "Quick Start". |
| FLUX.1-dev launch (7x) | `python flux/infer.py --preset default_7x --gpu_id 0` | `README.md` (L89) | High | Documented in README "Quick Start". |
| FLUX.2-Klein launch (4x) | `python flux2-klein-base-9B/infer_flux2.py --preset default_4x --gpu_id 0` | `README.md` (L101) | High | Documented in README "Quick Start". |
| FLUX.2-Klein launch (7x) | `python flux2-klein-base-9B/infer_flux2.py --preset default_7x --gpu_id 0` | `README.md` (L102) | High | Documented in README "Quick Start". |
| Environment setup | `conda create -n jit python=3.10 -y && conda activate jit && pip install -r requirement.txt` | `README.md` (L59-L65) | High | Documented in README "Getting Started". |

---

## Default Inference Parameters (Combined)

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Default seed | `42` | `flux/infer.py` (L29), `flux2-klein-base-9B/infer_flux2.py` (L29) | High | `seed = 42` in both inference scripts. |
| Default prompt (FLUX.1-dev) | "A grand piano made entirely of transparent, crystal-clear ice, with delicate frost patterns on its surface. It sits in a warm, sunlit concert hall, slowly melting, with water dripping onto the polished wooden floor. Photorealistic, poignant." | `flux/infer.py` (L58) | High | Hardcoded in inference script. |
| Default prompt (FLUX.2-Klein) | "A futuristic CPU chip with the text 'JiT' laser-etched on the center, intricate circuits, macro shot." | `flux2-klein-base-9B/infer_flux2.py` (L52) | High | Hardcoded in inference script. |
| max_sequence_length (FLUX.1-dev) | `256` | `flux/infer.py` (L68) | High | `max_sequence_length=256` passed to pipeline call. |
| max_sequence_length (FLUX.2-Klein default) | `512` | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L423) | Medium | Default in `__call__` signature. Not explicitly set in inference script, so defaults apply. |
| num_images_per_prompt | `1` (default in both `__call__` signatures) | `flux/pipeline_flux_JiT.py` (L561), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L413) | High | Not modified in inference scripts. |
| output_type | `"pil"` (default) | `flux/pipeline_flux_JiT.py` (L566), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L418) | High | Default in `__call__` signature. |
| Output directory (FLUX.1-dev) | `./outputs/` | `flux/infer.py` (L76) | High | `output_dir = "./outputs/"` |
| Output directory (FLUX.2-Klein) | `./outputs_flux2/` | `flux2-klein-base-9B/infer_flux2.py` (L71) | High | `out_dir = f"./outputs_flux2/"` |
| Output filename convention | First 50 characters of prompt + `.png` | `flux/infer.py` (L78), `flux2-klein-base-9B/infer_flux2.py` (L73) | High | `f"{prompt[:50]}.png"` |
| num_inference_steps passed to call (FLUX.2-Klein) | `pipeline.params['total_steps']` (from preset) | `flux2-klein-base-9B/infer_flux2.py` (L62) | High | Explicitly reads from `params` dict. FLUX.1-dev does not pass `num_inference_steps` explicitly (handled internally by `set_params`). |

---

## JiT-Specific Parameters Summary

| Parameter | `default_4x` | `default_7x` | Source File |
|-----------|-------------|-------------|-------------|
| `total_steps` | 18 | 11 | `flux/pipeline_flux_JiT.py` (L122, L132) |
| `stage_ratios` | [0.4, 0.65, 1.0] | [0.4, 0.65, 1.0] | `flux/pipeline_flux_JiT.py` (L123, L133) |
| `sparsity_ratios` | [0.35, 0.62, 1.0] | [0.32, 0.6, 1.0] | `flux/pipeline_flux_JiT.py` (L124, L134) |
| `use_checkerboard_init` | True | True | `flux/pipeline_flux_JiT.py` (L125, L135) |
| `use_adaptive` | True | True | `flux/pipeline_flux_JiT.py` (L126, L136) |
| `use_beta_sigmas` (FLUX.1 only) | True | True | `flux/pipeline_flux_JiT.py` (L127, L137) |
| `alpha` (FLUX.1 only) | 1.4 | 1.4 | `flux/pipeline_flux_JiT.py` (L128, L138) |
| `beta` (FLUX.1 only) | 0.425 | 0.425 | `flux/pipeline_flux_JiT.py` (L129, L139) |
| `microflow_relax_steps` | 3 | 3 | `flux/pipeline_flux_JiT.py` (L130, L140) |

---

## Stage Step Computation

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Stage step formula | `stage_steps = [int(total_steps * r) for r in stage_ratios]` | `flux/pipeline_flux_JiT.py` (L152), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L92) | High | For `default_4x` (18 steps): `[7, 11, 18]`. For `default_7x` (11 steps): `[4, 7, 11]`. |
| Stage transition logic | Highest stage index first (most sparse), transitions to lower stages (denser) as steps progress | `flux/pipeline_flux_JiT.py` (L764-L768) | High | `target_stage = num_stages - 1 - s_idx` where `s_idx` is found by `if i < s_step`. |
| Density per stage (ascending ratio order) | Stage 2 (first) gets `sparsity_ratios[0]` (sparsest), Stage 0 (last) gets `sparsity_ratios[-1]` (densest=1.0) | `flux/pipeline_flux_JiT.py` (L217-L224) | High | `_ratio_of_stage`: ascending order returns `ratios[num_stages-1-stage_k]`. |

---

# Cannot Be Determined From Code

The following evaluation protocol questions **require reading the paper** (arXiv 2603.10744) or external documentation and cannot be answered from the repository alone:

1. **Evaluation hardware** — No GPU model, VRAM, or hardware specification is mentioned anywhere in the code or README. Cannot determine what hardware was used for benchmarking.
2. **Benchmark datasets** — No dataset paths, dataset names, or dataset loading code. The inference scripts use single hardcoded prompts only.
3. **Evaluation prompts** — Only two demo prompts are hardcoded in the inference scripts. The prompts used for formal evaluation (e.g., PartiPrompts, DrawBench, COCO, or custom prompt sets) are not documented.
4. **Evaluation seeds** — Only `seed=42` is used in the demo scripts. Whether the paper uses multiple seeds or a specific seed set for evaluation cannot be determined.
5. **Evaluation metrics** — No metric computation code (FID, CLIP score, LPIPS, aesthetic score, etc.) exists in the repository.
6. **Claimed speedups** — README mentions "up to **7x** wall-clock acceleration", but no benchmark results, timing logs, or performance measurement scripts are provided.
7. **Official golden evaluation regime** — No evaluation pipeline, batch generation script, or automated comparison framework.
8. **Baseline comparison methodology** — No vanilla/baseline inference script for controlled A/B comparison.
9. **Number of generated samples for evaluation** — Cannot determine how many images were generated for quantitative evaluation.
10. **Image quality assessment protocol** — No human evaluation setup, no automated quality scripts.
11. **Multi-resolution evaluation** — Only 1024×1024 is demonstrated; whether other resolutions were evaluated cannot be determined.
12. **Latency measurement methodology** — Timing is done with `time.perf_counter()` in inference scripts (warm-up strategy, number of runs, averaging not specified).
13. **Memory profiling** — No memory measurement code or peak VRAM reporting.
14. **Batch size for evaluation** — Demo scripts use `batch_size=1` only. Paper batch size unknown.
15. **Comparison with other acceleration methods** — No code for running competing methods (e.g., token merging, step distillation, caching methods).

# PHASE 2 — Architectural Audit: Just-in-Time (JiT)

> **Interoperability Audit Protocol (IAP) — Deliverable 1**
> Audited repository: `Wenhao-Sun77/Just-in-Time`
> All information derived exclusively from repository source code, configuration files, and README.

---

## Supported Backbones

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Supported backbone(s) | **FLUX.1-dev** and **FLUX.2-klein-base-9B** | `README.md` (L84-L108), directory structure (`flux/`, `flux2-klein-base-9B/`) | High | Two separate pipeline files, one per backbone. README also lists **Qwen-image** and **HunyuanVideo 1.5** as "Coming Soon" but no code exists for them. |
| Backbone architecture family | Diffusion Transformer (DiT) | `README.md` (L40), `flux/pipeline_flux_JiT.py` (L618) | High | README: "acceleration framework for Diffusion Transformers (DiTs)". Code comment: "DiT forward pass on anchor tokens" (L821). |
| Model-agnostic claim | Yes, claimed training-free and model-agnostic | `README.md` (L40) | Medium | README states "model-agnostic acceleration framework", but code only implements two FLUX backbones. |

---

## Model Architecture

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| FLUX.1-dev pipeline class | `FluxPipeline_JiT` (subclass of `FluxPipeline` from diffusers) | `flux/pipeline_flux_JiT.py` (L92) | High | `class FluxPipeline_JiT(FluxPipeline)` — single inheritance from `diffusers.FluxPipeline`. |
| FLUX.2-Klein pipeline class | `Flux2KleinPipeline_JiT` (subclass of `Flux2KleinPipeline` from diffusers) | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L45) | High | `class Flux2KleinPipeline_JiT(Flux2KleinPipeline)` — single inheritance from `diffusers.pipelines.flux2.pipeline_flux2_klein.Flux2KleinPipeline`. |
| FLUX.1-dev model class (transformer) | Inherited from `FluxPipeline` — uses `self.transformer` (diffusers `FluxTransformer2DModel`) | `flux/pipeline_flux_JiT.py` (L827-L837) | High | No custom transformer defined; accesses `self.transformer` inherited from `FluxPipeline`. References `self.transformer.config.in_channels` (L669) and `self.transformer.config.guidance_embeds` (L732). |
| FLUX.2-Klein model class (transformer) | `Flux2Transformer2DModel` (imported from diffusers) | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L25) | High | `from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel`. Used via `self.transformer` inherited from `Flux2KleinPipeline`. |
| VAE (FLUX.1-dev) | Inherited from `FluxPipeline` — standard diffusers FLUX VAE | `flux/pipeline_flux_JiT.py` (L875-L877) | High | `self.vae.decode(...)`, references `self.vae.config.scaling_factor` and `self.vae.config.shift_factor`. |
| VAE (FLUX.2-Klein) | `AutoencoderKLFlux2` (imported from diffusers) | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L25) | High | Also uses `self.vae.bn.running_mean` / `self.vae.bn.running_var` for batch-norm denormalization (L760-L764). |
| Text encoder (FLUX.1-dev) | Inherited from `FluxPipeline` (CLIP + T5 dual encoder) | `flux/pipeline_flux_JiT.py` (L654-L666) | Medium | Calls `self.encode_prompt(...)` with `prompt`, `prompt_2`, `pooled_prompt_embeds` — standard FLUX dual-encoder signature. |
| Text encoder (FLUX.2-Klein) | `Qwen2TokenizerFast` + `Qwen3ForCausalLM` (imported from transformers) | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L22) | High | Imports these tokenizer/model classes. `self.encode_prompt(...)` called at L460-L467. |

---

## Attention Mechanism & Positional Encoding

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Attention mechanism | Not modified by JiT — delegates entirely to the parent transformer's forward pass | `flux/pipeline_flux_JiT.py` (L827-L837), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L660-L670) | High | JiT does not define, override, or monkey-patch any attention module. The transformer is called as-is with `hidden_states`, `encoder_hidden_states`, `img_ids`, `txt_ids`. |
| Positional encoding (FLUX.1-dev) | 3D positional IDs `[batch_idx, height_pos, width_pos]` — recomputed per-step for active tokens only | `flux/pipeline_flux_JiT.py` (L497-L514) | High | `_prepare_latent_image_ids` returns `[M, 3]` tensor indexed from full `[H_packed*W_packed, 3]` grid by `indices`. |
| Positional encoding (FLUX.2-Klein) | 4D positional IDs `[t, h, w, l]` — recomputed per-step for active tokens only | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L362-L381) | High | `_prepare_latent_image_ids` returns `[M, 4]` tensor with `t=0, h=arange, w=arange, l=0`. |
| Joint attention kwargs | Passed through to transformer: `joint_attention_kwargs` (FLUX.1-dev) / `attention_kwargs` (FLUX.2-Klein) | `flux/pipeline_flux_JiT.py` (L835), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L668) | High | No modification to kwargs — transparent passthrough. |

---

## Token Representation & Tensor Layout

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Token representation | 2×2 spatial patchification: `[B, C, H, W]` → `[B, H//2 * W//2, C*4]` | `flux/pipeline_flux_JiT.py` (L176-L195) | High | `_pack_latents`: reshapes `[B,C,H//2,2,W//2,2]` → `[B, H//2*W//2, C*4]`. Reverse in `_unpack_latents` (L197-L215). |
| Token representation (FLUX.2-Klein) | Uses parent `prepare_latents` which returns `[B, N_packed, C]` packed tokens | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L514-L527) | High | Comments: "latents is [B, N_packed, C], latent_ids is [B, N_packed, 4]" (L525). |
| Tensor layout during denoising | `[B, N_packed, D]` — full packed token sequence (`y_full`) | `flux/pipeline_flux_JiT.py` (L689), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L527) | High | `y_full` is the maintained full latent state throughout the loop. Active subset `y_active = y_full[:, active_indices, :]`. |
| Hidden state representation during sparse stages | Only active tokens `y_active = y_full[:, active_indices, :]` of shape `[B, M, D]` are passed to transformer | `flux/pipeline_flux_JiT.py` (L818-L837), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L647-L670) | High | `_extract_active_tokens` (L462/L325) selects subset. Only M < N tokens enter the transformer forward pass. |
| Latent channel count | `self.transformer.config.in_channels // 4` (packed from 4× spatial blocks) | `flux/pipeline_flux_JiT.py` (L669), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L510) | High | Standard FLUX packing convention. |

---

## Execution Location Within Pipeline

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Where JiT operates | JiT operates at the **denoising loop level** within the pipeline's `__call__` method — it wraps the standard denoising loop with spatial token selection, sparse transformer forward, interpolation, and stage transitions | `flux/pipeline_flux_JiT.py` (L552-L888), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L402-L777) | High | The entire `__call__` is overridden. The prompt encoding, latent prep, and VAE decode remain standard; only the denoising loop logic is replaced. |
| Does JiT modify the transformer? | **No** — the transformer is called as a black box with the standard forward signature | `flux/pipeline_flux_JiT.py` (L827-L837) | High | No subclassing, no monkey-patching, no custom forward function on the transformer. |
| Does JiT modify the VAE? | **No** — standard `self.vae.decode(...)` | `flux/pipeline_flux_JiT.py` (L876) | High | Decode path is identical to parent. |
| Does JiT modify the scheduler? | **Partially** — for FLUX.1-dev with `use_beta_sigmas=True`, it directly sets `self.scheduler.sigmas` using `_convert_to_beta()`. For FLUX.2-Klein, it uses the scheduler's `step()` method normally. | `flux/pipeline_flux_JiT.py` (L706-L726), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L552-L564) | High | FLUX.1-dev: manual Euler step `y_full = y_full + velocity_full * dt` (L854). FLUX.2-Klein: `self.scheduler.step(velocity_full_step, t, y_full)` (L724). |

---

## Diffusers Compatibility & API

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Diffusers version | `diffusers==0.37.0` (pinned) | `requirement.txt` (L3) | High | Exact version pinned in requirements. |
| Diffusers compatibility mechanism | Inherits from diffusers pipeline classes via standard Python subclassing | `flux/pipeline_flux_JiT.py` (L11-L13), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L24-L32) | High | Uses `from_pretrained`, `check_inputs`, `encode_prompt`, `image_processor.postprocess` etc. from parent. |
| `from_pretrained` compatibility | **Full** — inherits `from_pretrained` from parent diffusers pipeline | `flux/infer.py` (L34), `flux2-klein-base-9B/infer_flux2.py` (L34) | High | `FluxPipeline_JiT.from_pretrained(model_path, torch_dtype=torch.float16)` — standard diffusers loading. |
| Pipeline output type (FLUX.1-dev) | `FluxPipelineOutput` (from diffusers) | `flux/pipeline_flux_JiT.py` (L888) | High | `return FluxPipelineOutput(images=image)` |
| Pipeline output type (FLUX.2-Klein) | `Flux2PipelineOutput` (from diffusers) | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L777) | High | `return Flux2PipelineOutput(images=image)` |
| API compatibility — `__call__` signature (FLUX.1-dev) | Matches diffusers `FluxPipeline.__call__` with extra `**kwargs` | `flux/pipeline_flux_JiT.py` (L552-L573) | High | Parameters: `prompt`, `prompt_2`, `height`, `width`, `num_inference_steps`, `sigmas`, `guidance_scale`, `num_images_per_prompt`, `generator`, `latents`, `prompt_embeds`, `pooled_prompt_embeds`, `output_type`, `return_dict`, `joint_attention_kwargs`, `callback_on_step_end`, `max_sequence_length`. |
| API compatibility — `__call__` signature (FLUX.2-Klein) | Matches diffusers `Flux2KleinPipeline.__call__` signature | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L404-L425) | High | Parameters: `image`, `prompt`, `height`, `width`, `num_inference_steps`, `sigmas`, `guidance_scale`, `num_images_per_prompt`, `generator`, `latents`, `prompt_embeds`, `negative_prompt_embeds`, `output_type`, `return_dict`, `attention_kwargs`, `callback_on_step_end`, `max_sequence_length`, `text_encoder_out_layers`. |

---

## Custom Wrappers, Monkey Patches & Custom Forward Functions

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Custom pipeline wrappers | **Yes** — both `FluxPipeline_JiT` and `Flux2KleinPipeline_JiT` are subclass wrappers that override `__call__` | `flux/pipeline_flux_JiT.py` (L92), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L45) | High | Clean subclass pattern. The `__call__` method is fully overridden with the JiT denoising loop. |
| Monkey patches | **None** | All `.py` files | High | No dynamic attribute replacement, no `setattr` on external classes, no module-level patching detected across all source files. |
| Custom forward functions on transformer/VAE | **None** | All `.py` files | High | The transformer's forward is called via the standard `self.transformer(...)` interface. No custom forward hook, wrapper, or replacement. |
| Custom forward functions within pipeline | **Yes** — many JiT-specific methods added to the pipeline subclass | `flux/pipeline_flux_JiT.py`, `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` | High | Methods: `set_params`, `_pack_latents`, `_unpack_latents`, `_ratio_of_stage`, `_create_sparse_grid`, `_calculate_blur_params`, `_precompute_coords`, `_irregular_interpolation`, `_compute_importance_map`, `_adaptive_densify`, `_extract_active_tokens`, `_microflow_bridge`, `_prepare_latent_image_ids`, `_predict_x0_latent`, `progress_bar`. |
| External standalone helper functions | `calculate_shift()` and `retrieve_timesteps()` (FLUX.1-dev only) | `flux/pipeline_flux_JiT.py` (L20-L90) | High | Both are copied from diffusers (`retrieve_timesteps` explicitly marked as "Copied from diffusers"). FLUX.2-Klein imports these from `diffusers.pipelines.flux2.pipeline_flux2_klein`. |

---

## Tensor & API Compatibility for Interoperability

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Expected tensor layout at transformer input | `[B, M, D]` where M ≤ N_packed (variable per step/stage) | `flux/pipeline_flux_JiT.py` (L818-L828) | High | Active token subset with matching `img_ids` of shape `[M, 3]` (FLUX.1) or `[B, M, 4]` (FLUX.2-Klein). |
| Expected tensor layout at transformer output | `[B, M, D]` — velocity/noise prediction for active tokens only | `flux/pipeline_flux_JiT.py` (L837) | High | `noise_pred_active = self.transformer(...)[0]` — output indexed at [0] of return tuple. |
| Full tensor maintained across loop | `y_full: [B, N_packed, D]` — the complete latent state, updated via Euler step or scheduler step | `flux/pipeline_flux_JiT.py` (L854), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L724) | High | FLUX.1-dev: manual Euler `y_full = y_full + velocity_full * dt`. FLUX.2-Klein: `scheduler.step(velocity_full_step, t, y_full)`. |
| Velocity interpolation method | Nearest-neighbor fill + masked Gaussian blur (zero-order hold with smoothing) | `flux/pipeline_flux_JiT.py` (L331-L386) | High | `_irregular_interpolation`: (1) nearest neighbor from active coords, (2) Gaussian blur on 2D grid, (3) active mask preserves exact anchor values. |
| Wrappers likely required for interoperability? | **No wrapper required for the transformer** — JiT treats the transformer as a black-box callable. Any diffusion acceleration method that operates at the pipeline/loop level should be composable. **However**, the `__call__` method is fully overridden, meaning any other method that also overrides `__call__` would conflict. | `flux/pipeline_flux_JiT.py` (L552-L888) | Medium | The JiT spatial token selection, interpolation, and stage management logic is deeply embedded in the `__call__` loop body. Composition with other pipeline-level methods requires modification of the `__call__` method. |

---

## Core JiT Algorithm Components (from code)

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| SAG-ODE (Spatially Approximated Generative ODE) | Implemented as the main denoising loop: only anchor tokens (Ω_K) are passed through the transformer, then velocity is spatially interpolated to all tokens | `flux/pipeline_flux_JiT.py` (L752-L868) | High | Docstrings explicitly name "SAG-ODE" and "anchor tokens". |
| DMF (Deterministic Micro-Flow) | Linear interpolation bridge for newly activated tokens during stage transitions | `flux/pipeline_flux_JiT.py` (L466-L493), `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L333-L358) | High | `_microflow_bridge`: `y = (1-w)*current + w*target`, w = 1/microflow_relax_steps. |
| Multi-stage design | 3 stages by default (defined by `stage_ratios`), with progressively increasing token density | `flux/pipeline_flux_JiT.py` (L98-L152) | High | `self.num_stages = len(stage_ratios)`. Default: 3 stages at ratios [0.4, 0.65, 1.0]. |
| Sparse anchor initialization strategies | Two modes: (1) **Checkerboard** — stride-2 grid + boundary, (2) **Adaptive stride** — dynamic stride with random supplement | `flux/pipeline_flux_JiT.py` (L228-L295) | High | `_create_sparse_grid` with `use_checkerboard` parameter. |
| Adaptive densification | Variance-based importance map from cached velocity field → top-K token selection | `flux/pipeline_flux_JiT.py` (L389-L460) | High | `_compute_importance_map` uses local variance of velocity via `F.avg_pool2d`. `_adaptive_densify` selects top-K by normalized importance. |
| Tweedie x₀ prediction | `x₀ = x_t - t * v_t` (flow matching convention) | `flux/pipeline_flux_JiT.py` (L516-L535) | High | `_predict_x0_latent`: used during stage transitions to estimate clean data for DMF target computation. |
| Beta timestep schedule (FLUX.1-dev only) | Uses `scipy.stats.beta` distribution with `alpha=1.4, beta=0.425` via `scheduler._convert_to_beta()` | `flux/pipeline_flux_JiT.py` (L706-L714), `README.md` (L153-L159) | High | README confirms FLUX.2-Klein does **not** use this beta schedule. |
| ODE integration method (FLUX.1-dev) | **Manual Euler** — `y_full = y_full + velocity_full * dt`, where `dt = (t_next - t_curr) * 1e-3` | `flux/pipeline_flux_JiT.py` (L854) | High | Does not use `scheduler.step()` for FLUX.1-dev. |
| ODE integration method (FLUX.2-Klein) | **Scheduler step** — `self.scheduler.step(velocity_full_step, t, y_full)` | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L724) | High | Uses the standard diffusers scheduler step. |
| CFG handling (FLUX.1-dev) | Guidance embedding via `self.transformer.config.guidance_embeds` — passed as `guidance` tensor, no separate unconditional pass | `flux/pipeline_flux_JiT.py` (L732-L736) | High | FLUX.1-dev uses guidance embeddings natively, not traditional CFG. |
| CFG handling (FLUX.2-Klein) | Traditional CFG with separate conditional + unconditional passes, both on sparse active tokens | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L688-L717) | High | Uses `self.transformer.cache_context("cond")` / `cache_context("uncond")` with re-interpolation after CFG combination. |
| VAE tiling | Enabled for FLUX.1-dev via `self.enable_vae_tiling()` | `flux/pipeline_flux_JiT.py` (L587) | High | Called at the start of `__call__`. Not present in FLUX.2-Klein pipeline. |

---

## Dependencies

| Question | Answer | Source File | Confidence | Notes |
|----------|--------|-------------|------------|-------|
| Python version | 3.10 (conda environment) | `README.md` (L62) | High | `conda create -n jit python=3.10 -y` |
| PyTorch version | `torch==2.8.0` | `requirement.txt` (L1) | High | Pinned. |
| Diffusers version | `diffusers==0.37.0` | `requirement.txt` (L3) | High | Pinned. |
| Transformers version | `transformers==4.55.0` | `requirement.txt` (L4) | High | Pinned. |
| Scipy dependency | `scipy==1.16.0` (used for beta schedule) | `requirement.txt` (L12) | High | Required by `scheduler._convert_to_beta()`. |
| XLA support | Conditional import for `torch_xla` in FLUX.2-Klein | `flux2-klein-base-9B/pipeline_flux2_klein_JiT.py` (L35-L39) | Medium | `if is_torch_xla_available()` — optional, not in requirements. |

---

# Cannot Be Determined From Code

The following architectural questions **require reading the paper** (arXiv 2603.10744) and cannot be answered from the repository alone:

1. **Mathematical formulation of SAG-ODE** — The code implements the algorithm but the formal ODE formulation (equations, convergence guarantees, error bounds) is not in the codebase.
2. **Theoretical justification for the interpolation operator** — Why nearest-neighbor + Gaussian blur? The code implements it but does not explain the theoretical basis.
3. **DMF convergence properties** — The micro-flow uses linear interpolation with fixed weight `1/steps`, but the theoretical motivation is not in the code.
4. **Optimal stage ratio and sparsity ratio derivation** — The default values `[0.4, 0.65, 1.0]` and `[0.32, 0.6, 1.0]` are hardcoded presets; how they were determined is not documented.
5. **Relationship between JiT and token merging / token pruning** — The code does spatial token *selection* but the paper may discuss relationships to other methods.
6. **Error analysis of spatial approximation** — The code computes interpolated velocities but does not document approximation error bounds.
7. **Formal definition of "information density"** — The code uses velocity variance as a proxy, but the paper may define this more formally.
8. **Cross-modality (video) extension details** — README mentions 3D video generalizability, but no video code exists in the repository.
9. **Comparison architecture or baseline implementation** — No baseline/vanilla inference scripts are provided.
10. **Theoretical complexity analysis (FLOPs/MACs)** — Not computed or documented in code.

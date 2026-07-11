# Diffusers 0.34.0 Multi-GPU 8-bit Survival Guide

This document outlines the current state of calling `diffusers` (and `transformers`) APIs when operating in highly constrained environments—specifically when attempting to load massive models (like FLUX.1-dev) across multiple GPUs using `bitsandbytes` 8-bit quantization. 

Due to recent architectural shifts in Hugging Face libraries, relying on default parameters like `device_map="balanced"` can lead to devastating crashes. This report details the exact errors encountered and the battle-tested workarounds implemented to achieve absolute stability.

> [!WARNING]
> The behaviors described here specifically target `diffusers==0.34.0` and `transformers==4.53.0`. Future versions of these libraries may eventually patch these native bugs.

---

## 1. The Disk Memory Miscalculation (`ValueError: Some modules are dispatched on the CPU or the disk`)

### The Problem
When passing `device_map="balanced"` directly to `DiffusionPipeline.from_pretrained` alongside a `PipelineQuantizationConfig`, `accelerate` calculates the required VRAM based on the **unquantized** size of the models on disk (~35GB for FLUX). Because Kaggle's 2x T4 GPUs only offer 32GB total, `accelerate` preemptively assigns components to the CPU. `bitsandbytes` then violently crashes because 8-bit quantized models physically cannot be loaded onto the CPU.

### The Fix: Manual Component Isolation
To bypass the memory check, you must instantiate the heaviest components manually and assign them explicitly to individual GPUs before constructing the pipeline.

```python
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import FluxTransformer2DModel, BitsAndBytesConfig as DiffusersBitsAndBytesConfig

# Isolate T5 to GPU 1
text_encoder_2 = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder_2",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map={"": 1},
    torch_dtype=torch.float16
)

# Isolate DiT to GPU 0
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
    device_map={"": 0},
    torch_dtype=torch.float16
)
```

---

## 2. The Illegal `.to()` Call (`ValueError: .to is not supported for 8-bit bitsandbytes models`)

### The Problem
When Diffusers or Transformers finish dispatching a model, or when you explicitly call `pipeline.to("cuda:0")`, internal loops trigger catch-all `.to(device)` calls on all sub-components. Because `bitsandbytes` tensors are already loaded natively onto the GPU in C++, they explicitly forbid `.to()` calls and throw an error. Both libraries currently fail to properly silence this for their base classes.

### The Fix: The 8-bit `.to()` Monkey-Patch
You must intercept the `.to()` method at the highest class level for both libraries and silently return `self` if the model is quantized.

```python
from diffusers.models.modeling_utils import ModelMixin
from transformers.modeling_utils import PreTrainedModel

# Diffusers Patch
original_to = ModelMixin.to
def safe_to(self, *args, **kwargs):
    if getattr(self, "is_loaded_in_8bit", False) or getattr(self, "is_loaded_in_4bit", False):
        return self
    return original_to(self, *args, **kwargs)
ModelMixin.to = safe_to

# Transformers Patch
original_tf_to = PreTrainedModel.to
def safe_tf_to(self, *args, **kwargs):
    if getattr(self, "is_loaded_in_8bit", False) or getattr(self, "is_loaded_in_4bit", False):
        return self
    return original_tf_to(self, *args, **kwargs)
PreTrainedModel.to = safe_tf_to
```

---

## 3. The Tensor Routing Mismatch (`RuntimeError: Expected all tensors to be on the same device`)

### The Problem
Because we bypassed `device_map="balanced"` to avoid the CPU crash (see Section 1), `accelerate` did not automatically attach its tensor-routing hooks. When the pipeline (running on GPU 0) passed input tensors into `text_encoder_2` (running on GPU 1), PyTorch crashed because you cannot perform operations across GPUs without explicit transfers.

### The Fix: Manual `AlignDevicesHook` Injection
You must manually inject `accelerate`'s hardware isolation hooks directly onto the components before passing them to the pipeline. These hooks intercept inputs and dynamically route them to the correct GPU microseconds before the forward pass runs.

```python
from accelerate.hooks import add_hook_to_module, AlignDevicesHook

# Attach hardware isolation hooks
add_hook_to_module(text_encoder_2, AlignDevicesHook(execution_device=1))
add_hook_to_module(transformer, AlignDevicesHook(execution_device=0))

# Now construct the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    text_encoder_2=text_encoder_2,
    transformer=transformer,
    torch_dtype=torch.float16
)

# Crucial: Sync the pipeline's execution device so the tiny VAE/CLIP models land safely next to the DiT
pipeline.to("cuda:0")
```

---

## 4. Monkey-Patching Internal Logic (`NameError` within custom forward passes)

### The Problem
When overriding core methods like `FluxTransformer2DModel.forward` with custom implementations (e.g., DiCache's `dicache_forward`), it is extremely easy to miss internal utility imports that the original source code relied on. Because Python resolves function dependencies dynamically, missing imports will not throw errors until the generation loop actually reaches that exact line of code.

### The Fix
Always ensure the enclosing global scope of your custom function perfectly mirrors the source file's imports. For `diffusers` DiT replacements, ensure these are present:

```python
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
```

## 5. The VAE Spatial Allocation Spike (`OutOfMemoryError` during VAE Decode)

### The Problem
FLUX models generate sequence-packed latents that are unpacked and decoded at a 16x scale. To decode a 1024x1024 image, the pipeline inherently attempts to allocate memory for the entire high-resolution spatial map in one massive pass (~4.5GB). If the 12GB DiT is already occupying `cuda:0`, this final 4.5GB spike causes an instant Out Of Memory error on 16GB cards.

### The Fix: VAE Spatial Tiling
You must force the pipeline to decode the latents in smaller mathematical tiles rather than processing the entire 1024x1024 spatial plane at once.

```python
pipeline.enable_vae_tiling()
pipeline.enable_vae_slicing()
```

*(Note: Diffusers relies on `tile_sample_min_size` to determine when to tile. For maximum safety on sequence-packed latents like FLUX, you should also manually offload the VAE entirely. See Section 6.)*

---

## 6. The Inference Cache Memory Leak (Accumulated tensors in lists)

### The Problem
Caching algorithms (like DiCache or TeaCache) intrinsically track inference histories across timesteps by appending PyTorch tensors to Python lists (e.g., `residual_window.append(hidden_states)`). Even though `torch.no_grad()` is active, these massive intermediate tensors (~100MB each) are kept alive in VRAM by the list references. By step 30, this list bloats to ~3GB of wasted VRAM. If this list is not forcefully cleared *before* VAE decoding, the VAE will fail from extreme memory pressure.

### The Fix: `output_type="latent"` and Manual Multi-GPU VAE Offload
To absolutely guarantee memory safety on 2x 16GB GPUs, we completely bypass the pipeline's internal VAE decoder. We halt generation early, manually destroy the cache references to free 3GB of VRAM, and mathematically shift the VAE workload to the secondary GPU (which has 10GB of free VRAM since it only holds the Text Encoder).

```python
# 1. Manually isolate the VAE to the empty GPU 1
pipeline.vae.to("cuda:1")

# 2. Instruct the pipeline to return UNDECODED latents
output = pipeline(
    prompt,
    num_inference_steps=30,
    output_type="latent" # Halts generation right before VAE Decoding
)

# 3. Forcefully annihilate the DiCache lists and purge CUDA memory on GPU 0
pipeline.transformer.__class__.residual_window = []
pipeline.transformer.__class__.probe_residual_window = []
import gc
torch.cuda.empty_cache()
gc.collect()

# 4. Safely shuttle latents to GPU 1 and decode with zero memory pressure
latents = output.images[0] if isinstance(output.images, list) else output.images

# Unpack FLUX sequence latents
latents = pipeline._unpack_latents(latents, 1024, 1024, pipeline.vae_scale_factor)
latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor

with torch.no_grad():
    image = pipeline.vae.decode(latents.to("cuda:1"), return_dict=False)[0]

# 5. Native post-processing shifts the final image to CPU automatically
image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
```
## 7. The Sequence-Packed Latent Trap (`ValueError: too many values to unpack`)

### The Problem
Legacy diffusion models (like Stable Diffusion 1.5 or XL) output their latents as 4-dimensional spatial grids `(batch_size, channels, height, width)`. FLUX, however, is structurally a transformer that treats the image as a sequence of tokens. As a result, its raw latent output is strictly 3-dimensional `(batch_size, num_patches, channels)`. If you attempt to apply standard UNet memory/shape safeguards (like `latents.unsqueeze(0)` to force a batch dimension or spatial dimensions), you will fatally break `pipeline._unpack_latents()`.

### The Fix
Do not manually reshape FLUX latents before passing them into `pipeline._unpack_latents()`. Diffusers internally expects exactly 3 dimensions in order to execute its 8x compression decompression math:

```python
# Correct FLUX Latent Handling:
latents = output.images[0] if isinstance(output.images, list) else output.images

# NEVER do this for FLUX: latents = latents.unsqueeze(0)

# Pass the raw 3D tensor directly to unpack:
latents = pipeline._unpack_latents(latents, 1024, 1024, pipeline.vae_scale_factor)
```
# JustDubit Pipeline

A two-stage audio-video generation pipeline for **video dubbing** tasks, built on Lightricks' LTX-2 model.

JustDubit generates synchronized audio and video from source content, enabling high-quality dubbing with natural lip movements and speech alignment.

> For other LTX-2 pipelines (text-to-video, image-to-video, keyframe interpolation), see the [main LTX-2 repository](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/).

---

## 📋 Overview

**Key Features:**

- 🎙️ **Synchronized Audio-Video Generation**: Generates aligned audio and video for dubbing tasks
- 🎬 **Two-Stage Architecture**: Stage 1 generates at target resolution, Stage 2 upsamples 2x with refinement
- 📹 **Video Conditioning**: Condition on source video for lip-sync and motion preservation
- 🖼️ **Automatic First-Frame Extraction**: Seamlessly extracts conditioning frames from source video
- 🔧 **LoRA Support**: Easy integration with custom LoRA adapters

---

## 🚀 Quick Start

### Installation

```bash
# From the repository root
uv sync --frozen
```

### Usage


```bash
# Model paths
CHECKPOINT=/path/to/ltx-av-checkpoint.safetensors
GEMMA_ROOT=/path/to/gemma-text-encoder
SPATIAL_UPSAMPLER=/path/to/spatial-upscaler.safetensors
DISTILLED_LORA=/path/to/distilled-lora.safetensors

uv run python src/ltx_pipelines/pipeline_justdubit.py \
    --checkpoint_path ${CHECKPOINT} \
    --gemma_root ${GEMMA_ROOT} \
    --distilled_lora_path ${DISTILLED_LORA} \
    --distilled_lora_strength 1.0 \
    --spatial_upsampler_path ${SPATIAL_UPSAMPLER} \
    --lora /path/to/justdubit-lora.safetensors \
    --lora_strength 1.0 \
    --video_conditioning /path/to/source-video.mp4 1.0 \
    --prompt "The man is speaking English, saying: 'Hello, world!' " \
    --height 512 --width 768 \
    --num_inference_steps 30 \
    --cfg_guidance_scale 3.0 \
    --frame_rate 25 \
    --seed 42 \
    --output_path ./output.mp4
```

> **Resolution Note:** The `--height` and `--width` specify Stage 1 resolution. The final output is **2x upsampled** in Stage 2. For example, `512x768` input produces a `1024x1536` output video.

---

## 🎛️ CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--checkpoint_path` | ✅ | Path to LTX-2 AV model checkpoint - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors) |
| `--gemma_root` | ✅ | Path to Gemma text encoder directory - [Download](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main) |
| `--distilled_lora_path` | ✅ | Path to distilled LoRA for Stage 2 - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors) |
| `--spatial_upsampler_path` | ✅ | Path to spatial upsampler model - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors) |
| `--output_path` | ✅ | Path for output MP4 file |
| `--prompt` | ✅ | Text prompt describing desired output |
| `--video_conditioning` | ✅ | Source video path and strength (e.g., `video.mp4 1.0`) |
| `--distilled_lora_strength` | | Strength of distilled LoRA (default: 1.0) |
| `--lora` | ✅ | JustDubit LoRA path (use with `--lora_strength`) - [Download](https://huggingface.co/justdubit/justdubit/resolve/main/ltx-2-19b-ic-lora-lipdubbing.safetensors) |
| `--lora_strength` | | Strength of custom LoRA (default: 1.0) |
| `--negative_prompt` | | Negative prompt for CFG guidance |
| `--height` | | Stage 1 video height in pixels (default: 512, final output: 1024) |
| `--width` | | Stage 1 video width in pixels (default: 768, final output: 1536) |
| `--num_inference_steps` | | Number of denoising steps (default: 30) |

> **Note:** The final output resolution is **2x** the specified height/width due to Stage 2 upsampling. For example, `--height 512 --width 768` produces a **1024x1536** output video.
| `--cfg_guidance_scale` | | CFG guidance scale (default: 3.0) |
| `--frame_rate` | | Output frame rate in fps (default: 25) |
| `--seed` | | Random seed for reproducibility |

---

## 📝 Prompt Format

Prompts for JustDubit should follow a specific structure to achieve best results:

```
[Speaker] is speaking [Language/Accent], saying: "[Dialogue]"
```

### Components

| Component | Description | Examples |
|-----------|-------------|----------|
| **Speaker** | Description of who is speaking | "The girl", "The man", "The young woman", "The elderly man" |
| **Language/Accent** | The language and accent style | "standard English", "British English", "American English", "French" |
| **Dialogue** | The actual words to be spoken (in quotes) | "Bonjour!", "I'm allowed to do whatever I want." |

### Examples

```text
A young woman is speaking American English, saying: "This is so exciting! I can't wait to get started."

The elderly gentleman is speaking French, saying: "Bonjour, comment allez-vous aujourd'hui?"
```

### Tips

- **Be specific about the speaker**: Match the speaker description to the person visible in the source video
- **Specify the accent**: Adding accent details (e.g., "British English", "American English") helps generate more natural speech
- **Use natural punctuation**: Include commas, periods, and ellipses in the dialogue for natural pacing

### Translating Dialogues with LLMs

For dubbing videos into different languages, we recommend using an LLM to translate the original dialogue. Use the following system prompt for best results:

```
You are a professional translator.
Your task is to translate the given dialogue into the target language while preserving the original meaning, tone, and linguistic style as closely as possible.

Guidelines:
- Keep the intent and nuance of each sentence intact.
- Preserve formality or informality, emphasis, repetition, and emotional intensity using linguistic means only.
- Translate idioms and expressions naturally into the target language rather than literally, when appropriate.
- Maintain similar sentence structure and flow where possible.
- Do not add, omit, or reinterpret content.
- Output only the translated text, keeping any original formatting or speaker labels.
```

Then construct your JustDubit prompt with the translated dialogue:

```
The woman is speaking French, saying: "[translated dialogue here]"
```

---

## 🐍 Python API

```python
from ltx_pipelines.pipeline_justdubit import JustDubitPipeline
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE

# Initialize pipeline
pipeline = JustDubitPipeline(
    checkpoint_path="/path/to/checkpoint.safetensors",
    distilled_lora_path="/path/to/distilled-lora.safetensors",
    distilled_lora_strength=1.0,
    spatial_upsampler_path="/path/to/upsampler.safetensors",
    gemma_root="/path/to/gemma",
    loras=[
        LoraPathStrengthAndSDOps(
            "/path/to/justdubit-lora.safetensors", 
            1.0, 
            LTXV_LORA_COMFY_RENAMING_MAP
        )
    ],
)

# Run inference
# Note: height/width are Stage 1 resolution; final output is 2x (1024x1536)
video, audio = pipeline(
    prompt="The man is speaking English, saying: 'hello world!' ",
    negative_prompt="blurry, low quality, distorted",
    seed=42,
    height=512,   # Stage 1 height (final output: 1024)
    width=768,    # Stage 1 width (final output: 1536)
    num_frames=121,
    frame_rate=25,
    num_inference_steps=30,
    cfg_guidance_scale=3.0,
    images=[],  # Auto-extracted from video conditioning
    video_conditioning=[("/path/to/source-video.mp4", 1.0)],
    tiling_config=TilingConfig.default(),
)

# Save output
encode_video(
    video=video,
    fps=25,
    audio=audio,
    audio_sample_rate=AUDIO_SAMPLE_RATE,
    output_path="./output.mp4",
)
```

---

## 🏗️ Pipeline Architecture

JustDubit follows a two-stage architecture optimized for video dubbing:

### Stage 1: Generation

1. **Text Encoding**: Prompt encoded via Gemma into video and audio context embeddings
2. **Conditioning Setup**:
   - First frame auto-extracted from source video for image conditioning
   - Audio extracted from source video for audio conditioning
   - Video frames encoded as latent conditionings
3. **Diffusion Process**:
   - CFG-guided denoising for specified number of steps
   - Simultaneous video and audio latent generation

### Stage 2: Refinement

1. **Upsampling**: Video latent upsampled 2x using spatial upsampler
2. **Distilled Refinement**: Fast high-resolution refinement with distilled LoRA
3. **Audio Passthrough**: Audio latent used as conditioning (no re-denoising)

### Decoding

1. **Video Decoding**: VAE decodes video latent to pixel space
2. **Audio Decoding**: Audio VAE + vocoder decodes audio latent to waveform
3. **Output**: Synchronized MP4 with video and audio tracks

---

### 🔧 Required Model Checkpoints

Download the required model files from Hugging Face:

| Model | Description | Download |
|-------|-------------|----------|
| **LTX-2 AV Checkpoint** | Main model checkpoint | [`ltx-2-19b-dev.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors) |
| **JustDubit LoRA** 💋 | LoRA for lip-sync dubbing | [`ltx-2-19b-ic-lora-lipdubbing.safetensors`](https://huggingface.co/justdubit/justdubit/resolve/main/ltx-2-19b-ic-lora-lipdubbing.safetensors) |
| **Distilled LoRA** | Required for Stage 2 refinement | [`ltx-2-19b-distilled-lora-384.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors) |
| **Spatial Upscaler** | Required for 2x upsampling | [`ltx-2-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors) |
| **Gemma Text Encoder** | Text encoder (download all assets) | [`gemma-3-12b-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main) |

---

## 🔗 Related

- **[Offical LTX-2 Pipelines](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/)** - Other pipelines (text-to-video, image-to-video, keyframe interpolation)
- **[Train your own JustDubIt](../ltx-trainer/README.md)** - Training tools

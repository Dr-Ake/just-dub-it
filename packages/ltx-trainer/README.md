# JustDubit Training

Training tools for the **JustDubit**, built on Lightricks' LTX-2.

JustDubit training enables fine-tuning the LTX-2 model for synchronized audio-video generation, where reference video/audio conditions the generation of dubbed output with natural lip movements.

> For other training strategies (text-to-video, image-to-video, IC-LoRA), see the [main LTX-2 Trainer](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer).

---

## 📋 Overview

**Training Features:**

- 🎙️ **Joint Audio-Video Training**: Train synchronized audio and video generation
- 📹 **Reference Conditioning**: Use reference video/audio to guide generation
- 🎭 **Masked Loss**: Focus training on lip/face regions with segmentation masks
- 🔧 **LoRA Training**: Efficient fine-tuning with Low-Rank Adaptation
- ⚡ **Cross-Attention Masking**: Improved audio-video synchronization

---

## 🚀 Quick Start

### 1. Installation

```bash
# From the repository root
uv sync --frozen
```

### 2. Download Dataset

Download the JustDubit training dataset from Hugging Face:

```bash
# Using huggingface-cli
huggingface-cli download justdubit/audiovisual_translation_dub --local-dir ./data/audiovisual_translation_dub --repo-type dataset

# Or using git lfs
git lfs install
git clone https://huggingface.co/datasets/justdubit/audiovisual_translation_dub ./data/audiovisual_translation_dub
```

The dataset contains paired video samples with:
- Target videos (dubbed output)
- Reference videos (original source)
- Face segmentation masks (optional, for masked loss training)
- Text captions with speaker and dialogue information

### 3. Preprocess Dataset

Preprocess the dataset to compute latent representations:

```bash
uv run python scripts/process_dataset.py ./data/audiovisual_translation_dub/dataset.json \
    --resolution-buckets "640x352x121" \
    --model-path /path/to/ltx-2-checkpoint.safetensors \
    --text-encoder-path /path/to/gemma-text-encoder \
    --with-audio \
    --reference-column reference_path \
    --mask-column mask_path  # Optional: only needed for masked loss training
```

**Preprocessing Arguments:**

| Argument | Description |
|----------|-------------|
| `--resolution-buckets` | Video dimensions as `WxHxF` (width x height x frames) |
| `--model-path` | Path to LTX-2 checkpoint |
| `--text-encoder-path` | Path to Gemma text encoder directory |
| `--with-audio` | Enable audio latent extraction |
| `--reference-column` | Dataset column containing reference video paths |
| `--mask-column` | (Optional) Dataset column containing face mask paths for masked loss |
| `--decode` | (Optional) Decode latents for verification |

This creates a `.precomputed` directory with:
```
data/audiovisual_translation_dub/
└── .precomputed/
    ├── latents/                  # Target video latents
    ├── conditions/               # Text embeddings
    ├── audio_latents/            # Target audio latents
    ├── reference_latents/        # Reference video latents
    ├── reference_audio_latents/  # Reference audio latents
    └── masks/                    # Face segmentation masks (if --mask-column provided)
```

### 4. Configure Training

Create a training configuration file `configs/justdubit.yaml`:

```yaml
# Model Configuration
model:
  model_path: "/path/to/ltx-2-checkpoint.safetensors"
  text_encoder_path: "/path/to/gemma-text-encoder"
  training_mode: "lora"

# LoRA Configuration
lora:
  rank: 128
  alpha: 128
  dropout: 0.0
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"
    - "net.0.proj"
    - "net.2"

# Training Strategy
training_strategy:
  name: "justdubit"
  first_frame_conditioning_p: 0.1
  with_audio: true
  enable_cross_attention_masking: true
  audio_latents_dir: "audio_latents"
  reference_latents_dir: "reference_latents"
  reference_audio_latents_dir: "reference_audio_latents"
  # Masked loss configuration (optional - for focusing training on face regions)
  # Set use_masked_loss to false if not using face masks
  mask_config:
    mask_dir: "masks"
    use_masked_loss: true  # Set to false if not using masks
    mask_loss_weight: 1.0
    background_loss_weight: 0.1
    mask_threshold: 0.1

# Optimization
optimization:
  learning_rate: 2e-4
  learning_rate_groups:
    "audio_attn": 1e-6
    "audio_ff": 1e-6
  steps: 2000
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw"
  scheduler_type: "linear"
  enable_gradient_checkpointing: true

# Acceleration
acceleration:
  mixed_precision_mode: "bf16"
  quantization: null

# Data
data:
  preprocessed_data_root: "./data/audiovisual_translation_dub/.precomputed"
  num_dataloader_workers: 2

# Validation
validation:
  prompts:
    - "A woman is speaking English, saying: 'Hello, how are you today?'"
  reference_videos:
    - "./data/audiovisual_translation_dub/validation/sample.mp4"
  video_dims: [640, 352, 121]
  frame_rate: 25.0
  seed: 42
  inference_steps: 30
  interval: 100
  guidance_scale: 3.0
  generate_audio: true

# Checkpoints
checkpoints:
  interval: 250
  keep_last_n: -1

# Flow Matching
flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"

# Output
seed: 42
output_dir: "outputs/justdubit"
```

### 5. Train

Run training with single GPU:

```bash
uv run python scripts/train.py configs/justdubit.yaml
```

For multi-GPU training with Accelerate:

```bash
accelerate config  # Configure distributed training
accelerate launch scripts/train.py configs/justdubit.yaml
```

---

## 🎛️ Key Configuration Options

### Training Strategy

| Option | Description |
|--------|-------------|
| `name` | Set to `"justdubit"` for video dubbing training |
| `first_frame_conditioning_p` | Probability of first-frame conditioning (default: 0.1) |
| `with_audio` | Enable joint audio-video training (default: true) |
| `enable_cross_attention_masking` | Mask cross-attention between reference tokens and noisy tokens |

### Masked Loss

| Option | Description |
|--------|-------------|
| `use_masked_loss` | Enable masked loss computation |
| `mask_loss_weight` | Weight for foreground (face) regions |
| `background_loss_weight` | Weight for background regions |
| `mask_threshold` | Binary mask threshold |

### Learning Rate Groups

For JustDubit, we recommend using lower learning rates for audio modules:

```yaml
learning_rate_groups:
  "audio_attn": 1e-6   # Audio attention layers
  "audio_ff": 1e-6     # Audio feed-forward layers
```

---

## 📁 Dataset Format

The dataset must be a JSON/JSONL file with the following structure:

```json
[
  {
    "caption": "The woman is speaking English, saying: 'Hello, world!'",
    "media_path": "target_001.mp4",
    "reference_path": "reference_001.mp4",
    "mask_path": "facemask_001.mp4"
  }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `caption` | ✅ | Text prompt with speaker, language, and dialogue |
| `media_path` | ✅ | Path to target (dubbed) video |
| `reference_path` | ✅ | Path to reference (source) video |
| `mask_path` | | (Optional) Path to face segmentation mask video for masked loss |

> **Note:** The `mask_path` field is only required if you want to train with masked loss (`use_masked_loss: true`), which focuses the loss computation on facial regions. If not using masked loss, you can omit this field and set `use_masked_loss: false` in your training config.

---

## 🔧 Requirements

- **Python 3.12+**
- **Linux with CUDA 12.1+**
- **Nvidia GPU with 80GB+ VRAM** (recommended)
- **Model Checkpoints**:
  - LTX-2 AV checkpoint - [Download](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors)
  - Gemma text encoder - [Download](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main)

---

## 🔗 Related

- **[JustDubit Pipeline](../ltx-pipelines/README.md)** - Inference pipeline for video dubbing
- **[LTX-2 Trainer](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer)** - Other training strategies

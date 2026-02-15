from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_gradio():
    try:
        import gradio as gr  # noqa: PLC0415
        return gr
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        import gradio as gr  # noqa: PLC0415
        return gr


def _resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    return candidate


def _require_existing_path(label: str, raw_path: str, expect_dir: bool = False) -> Path:
    if not raw_path or not raw_path.strip():
        raise ValueError(f"{label} is required.")
    resolved = _resolve_path(raw_path.strip())
    if not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    if expect_dir and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    if not expect_dir and not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    return resolved


def _resolve_source_video(source_video_path: str, uploaded_video: str | None) -> Path:
    if uploaded_video and uploaded_video.strip():
        return _require_existing_path("Uploaded source video", uploaded_video, expect_dir=False)
    return _require_existing_path("Source video path", source_video_path, expect_dir=False)


def run_pipeline(
    checkpoint_path: str,
    gemma_root: str,
    distilled_lora_path: str,
    spatial_upsampler_path: str,
    justdubit_lora_path: str,
    source_video_path: str,
    uploaded_video: str | None,
    prompt: str,
    negative_prompt: str,
    lora_strength: float,
    distilled_lora_strength: float,
    width: int,
    height: int,
    num_inference_steps: int,
    cfg_guidance_scale: float,
    frame_rate: float,
    seed: int,
) -> tuple[str | None, str]:
    try:
        checkpoint = _require_existing_path("Checkpoint path", checkpoint_path, expect_dir=False)
        gemma = _require_existing_path("Gemma root", gemma_root, expect_dir=True)
        distilled_lora = _require_existing_path("Distilled LoRA path", distilled_lora_path, expect_dir=False)
        upsampler = _require_existing_path("Spatial upsampler path", spatial_upsampler_path, expect_dir=False)
        justdubit_lora = _require_existing_path("JustDubit LoRA path", justdubit_lora_path, expect_dir=False)
        source_video = _resolve_source_video(source_video_path, uploaded_video)
    except ValueError as exc:
        return None, str(exc)

    if not prompt or not prompt.strip():
        return None, "Prompt is required."

    try:
        width_i = int(width)
        height_i = int(height)
        steps_i = int(num_inference_steps)
        seed_i = int(seed)
    except (TypeError, ValueError):
        return None, "Width, height, inference steps, and seed must be valid numbers."

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_path = OUTPUT_DIR / f"justdubit_{run_id}.mp4"

    command = [
        sys.executable,
        "-m",
        "ltx_pipelines.pipeline_justdubit",
        "--checkpoint_path",
        str(checkpoint),
        "--gemma_root",
        str(gemma),
        "--distilled_lora_path",
        str(distilled_lora),
        "--distilled_lora_strength",
        str(distilled_lora_strength),
        "--spatial_upsampler_path",
        str(upsampler),
        "--lora",
        str(justdubit_lora),
        "--lora_strength",
        str(lora_strength),
        "--video_conditioning",
        str(source_video),
        "1.0",
        "--prompt",
        prompt.strip(),
        "--height",
        str(height_i),
        "--width",
        str(width_i),
        "--num_inference_steps",
        str(steps_i),
        "--cfg_guidance_scale",
        str(cfg_guidance_scale),
        "--frame_rate",
        str(frame_rate),
        "--seed",
        str(seed_i),
        "--output_path",
        str(output_path),
    ]

    if negative_prompt and negative_prompt.strip():
        command.extend(["--negative_prompt", negative_prompt.strip()])

    quoted_cmd = " ".join(shlex.quote(part) for part in command)
    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    logs = f"$ {quoted_cmd}\n\n{process.stdout}\n{process.stderr}".strip()

    if process.returncode != 0:
        return None, logs
    if not output_path.exists():
        return None, logs + f"\n\nExpected output file was not created: {output_path}"

    return str(output_path), logs


def build_ui(gr):
    with gr.Blocks(title="JustDubit (Pinokio)") as demo:
        gr.Markdown(
            """
            # JustDubit

            Set the model paths, provide a source video, and run dubbing.
            Recommended default folders:
            - `models/` for checkpoints and LoRAs
            - `inputs/` for source videos
            - `outputs/` for generated videos
            """
        )

        with gr.Row():
            checkpoint_path = gr.Textbox(
                label="LTX-2 AV checkpoint",
                value="models/ltx-2-19b-dev.safetensors",
            )
            gemma_root = gr.Textbox(
                label="Gemma root directory",
                value="models/gemma-3-12b-it-qat-q4_0-unquantized",
            )

        with gr.Row():
            distilled_lora_path = gr.Textbox(
                label="Distilled LoRA",
                value="models/ltx-2-19b-distilled-lora-384.safetensors",
            )
            spatial_upsampler_path = gr.Textbox(
                label="Spatial upsampler",
                value="models/ltx-2-spatial-upscaler-x2-1.0.safetensors",
            )

        justdubit_lora_path = gr.Textbox(
            label="JustDubit LoRA",
            value="models/ltx-2-19b-ic-lora-lipdubbing.safetensors",
        )

        with gr.Row():
            source_video_path = gr.Textbox(
                label="Source video path",
                value="inputs/source.mp4",
            )
            uploaded_video = gr.File(
                label="Or upload source video",
                type="filepath",
            )

        prompt = gr.Textbox(
            label="Prompt",
            value='The person is speaking English, saying: "Hello, world!"',
            lines=2,
        )
        negative_prompt = gr.Textbox(
            label="Negative prompt (optional)",
            value="",
            lines=2,
        )

        with gr.Row():
            lora_strength = gr.Slider(
                label="JustDubit LoRA strength",
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.05,
            )
            distilled_lora_strength = gr.Slider(
                label="Distilled LoRA strength",
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.05,
            )

        with gr.Row():
            width = gr.Slider(label="Width (stage 1)", minimum=256, maximum=1024, value=768, step=32)
            height = gr.Slider(label="Height (stage 1)", minimum=256, maximum=1024, value=512, step=32)

        with gr.Row():
            num_inference_steps = gr.Slider(label="Inference steps", minimum=5, maximum=80, value=30, step=1)
            cfg_guidance_scale = gr.Slider(label="CFG guidance scale", minimum=1.0, maximum=10.0, value=3.0, step=0.1)
            frame_rate = gr.Slider(label="Frame rate", minimum=8, maximum=60, value=25, step=1)
            seed = gr.Number(label="Seed", value=42, precision=0)

        run_button = gr.Button("Run Dubbing", variant="primary")
        output_video = gr.Video(label="Output video")
        logs = gr.Textbox(label="Logs", lines=20, max_lines=30)

        run_button.click(
            fn=run_pipeline,
            inputs=[
                checkpoint_path,
                gemma_root,
                distilled_lora_path,
                spatial_upsampler_path,
                justdubit_lora_path,
                source_video_path,
                uploaded_video,
                prompt,
                negative_prompt,
                lora_strength,
                distilled_lora_strength,
                width,
                height,
                num_inference_steps,
                cfg_guidance_scale,
                frame_rate,
                seed,
            ],
            outputs=[output_video, logs],
        )

    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pinokio launcher for JustDubit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gr = _load_gradio()
    ui = build_ui(gr)
    ui.queue(max_size=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

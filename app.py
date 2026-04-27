#!/usr/bin/env python3
"""
MuseTalk Gradio web interface.

Launch:
    python app.py [--share] [--port 7860]
"""

import argparse
import os
import sys
import tempfile

# MuseTalk source lives in ./MuseTalk
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MuseTalk"))

# Load FFMPEG path from .env if present
_env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_file):
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

import gradio as gr


def _run_musetalk(video_path: str, audio_path: str, bbox_shift: int) -> str:
    """Run MuseTalk inference and return the path to the output video."""
    import yaml
    from scripts.inference import main as musetalk_main

    output_dir = os.path.join(os.path.dirname(__file__), "results", "output")
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "task_0": {
            "video_path": video_path,
            "audio_path": audio_path,
            **({"bbox_shift": bbox_shift} if bbox_shift != 0 else {}),
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "inference.py",
            "--inference_config", tmp_path,
            "--result_dir", output_dir,
        ]
        musetalk_main()
    finally:
        sys.argv = original_argv
        os.unlink(tmp_path)

    # MuseTalk names output after the video stem
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    audio_stem = os.path.splitext(os.path.basename(audio_path))[0]
    candidate = os.path.join(output_dir, f"{video_stem}_{audio_stem}.mp4")
    if os.path.exists(candidate):
        return candidate

    # Fallback: return the most recent mp4 in the output dir
    mp4s = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp4")],
        key=os.path.getmtime,
    )
    return mp4s[-1] if mp4s else ""


def process(video_file, audio_file, bbox_shift: int):
    if video_file is None:
        return None, "Please upload a video."
    if audio_file is None:
        return None, "Please upload an audio file."

    try:
        result_path = _run_musetalk(video_file, audio_file, bbox_shift)
        if result_path and os.path.exists(result_path):
            return result_path, f"Done! Saved to: {result_path}"
        return None, "Inference finished but output file not found."
    except Exception as e:
        return None, f"Error: {e}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MuseTalk — AI Lip Sync") as demo:
        gr.Markdown(
            """
            # MuseTalk — AI Lip Sync
            Upload a **portrait video** and an **audio clip**. MuseTalk will generate a version
            of the video with the mouth movements synced to the audio.
            """
        )

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input Video", sources=["upload"])
                audio_input = gr.Audio(label="Input Audio", type="filepath", sources=["upload"])
                bbox_shift = gr.Slider(
                    minimum=-25, maximum=25, step=1, value=0,
                    label="Bounding-box shift",
                    info="Adjust if the lip region is cut off. Negative = shift up, positive = shift down.",
                )
                run_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output_video = gr.Video(label="Output Video")
                status_box = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            fn=process,
            inputs=[video_input, audio_input, bbox_shift],
            outputs=[output_video, status_box],
        )

        gr.Examples(
            examples=[
                ["MuseTalk/data/video/yongen.mp4", "MuseTalk/data/audio/yongen.wav", 0],
                ["MuseTalk/data/video/sun.mp4",    "MuseTalk/data/audio/sun.wav",    -7],
            ],
            inputs=[video_input, audio_input, bbox_shift],
            label="Example inputs (requires MuseTalk repo to be present)",
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="MuseTalk Gradio web UI")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

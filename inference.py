#!/usr/bin/env python3
"""
MuseTalk CLI inference.

Usage:
    # Single task via flags:
    python inference.py --video data/video/person.mp4 --audio data/audio/speech.wav

    # Multiple tasks via YAML config:
    python inference.py --config configs/inference/test.yaml

    # With optional bbox shift (positive = expand down, negative = contract):
    python inference.py --video data/video/person.mp4 --audio data/audio/speech.wav --bbox_shift -7
"""

import argparse
import os
import sys

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


def build_config_from_flags(video: str, audio: str, bbox_shift: int) -> dict:
    return {
        "task_0": {
            "video_path": video,
            "audio_path": audio,
            **({"bbox_shift": bbox_shift} if bbox_shift != 0 else {}),
        }
    }


def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def run_inference(config: dict, output_dir: str = "./results") -> None:
    import tempfile, yaml

    # MuseTalk uses relative paths anchored at the repo root (./musetalk/utils/dwpose/...).
    # We must cd into the MuseTalk subdirectory before importing or running anything.
    musetalk_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MuseTalk")
    original_dir = os.getcwd()
    output_dir_abs = os.path.abspath(output_dir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    try:
        os.chdir(musetalk_dir)
        sys.argv = [
            "inference.py",
            "--inference_config", tmp_path,
            "--result_dir", output_dir_abs,
        ]
        import runpy
        runpy.run_path(
            os.path.join(musetalk_dir, "scripts", "inference.py"),
            run_name="__main__",
        )
    finally:
        os.chdir(original_dir)
        os.unlink(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuseTalk lip-sync inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", metavar="PATH",
                        help="YAML config file with one or more tasks")
    source.add_argument("--video", metavar="PATH",
                        help="Input video file")

    parser.add_argument("--audio", metavar="PATH",
                        help="Input audio file (required when --video is used)")
    parser.add_argument("--bbox_shift", type=int, default=0,
                        help="Bounding-box vertical shift in pixels")
    parser.add_argument("--output_dir", default="./results",
                        help="Directory to save output videos")

    args = parser.parse_args()

    if args.video and not args.audio:
        parser.error("--audio is required when --video is specified")

    if args.config:
        config = load_yaml_config(args.config)
    else:
        config = build_config_from_flags(args.video, args.audio, args.bbox_shift)

    os.makedirs(args.output_dir, exist_ok=True)
    run_inference(config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Chunked wrapper around inverse and forward renderer inference scripts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image


INVERSE_PASSES: Sequence[str] = ("basecolor", "normal", "depth", "roughness", "metallic")


@dataclass(frozen=True)
class ChunkSpec:
    start: int
    valid_length: int


def log(msg: str) -> None:
    print(f"[render-pipeline] {msg}")


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    pretty = " ".join(cmd)
    log(pretty)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ensure_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def ffprobe_stream_info(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,nb_read_packets",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    return streams[0]


def count_video_frames(video_path: Path) -> int:
    info = ffprobe_stream_info(video_path)
    for key in ("nb_read_packets", "nb_frames"):
        raw = info.get(key)
        if raw and str(raw).isdigit():
            value = int(raw)
            if value > 0:
                return value
    raise RuntimeError(f"Unable to determine frame count for {video_path}")


def read_video_fps(video_path: Path) -> float:
    info = ffprobe_stream_info(video_path)
    raw = str(info.get("avg_frame_rate", "0/1"))
    if "/" in raw:
        num, den = raw.split("/", 1)
        if den != "0":
            fps = float(num) / float(den)
            if fps > 0:
                return fps
    return 24.0


def validate_same_frame_count(paths: Iterable[Path]) -> int:
    counts = {str(path): count_video_frames(path) for path in paths}
    unique = sorted(set(counts.values()))
    if len(unique) != 1:
        details = ", ".join(f"{path}={count}" for path, count in counts.items())
        raise ValueError(f"Input videos must have identical frame counts: {details}")
    return unique[0]


def sliding_window_specs(total_frames: int, chunk_size: int, overlap: int) -> List[ChunkSpec]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    if total_frames <= chunk_size:
        return [ChunkSpec(start=0, valid_length=total_frames)]

    step = chunk_size - overlap
    starts: List[int] = []
    cutoff = total_frames - overlap
    start = 0
    while start < cutoff:
        starts.append(start)
        start += step

    return [ChunkSpec(start=s, valid_length=min(chunk_size, total_frames - s)) for s in starts]


def extract_video_frames(input_video: Path, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-start_number",
        "0",
        str(output_dir / "%06d.png"),
    ]
    run_cmd(cmd)
    frames = sorted(output_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No frames extracted from {input_video}")
    return frames


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def encode_video(frame_pattern: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_path),
    ]
    run_cmd(cmd)


def trim_video(input_video: Path, output_video: Path, frame_count: int, fps: float) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-frames:v",
        str(frame_count),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_video),
    ]
    run_cmd(cmd)


def average_images(image_paths: Sequence[Path]) -> Image.Image:
    if not image_paths:
        raise ValueError("Cannot average an empty image list")
    total = None
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image, dtype=np.float64)
        if total is None:
            total = array
        else:
            total += array
    assert total is not None
    averaged = np.clip(total / len(image_paths), 0, 255).astype(np.uint8)
    return Image.fromarray(averaged)


def merge_chunk_videos(
    chunk_video_paths: Sequence[Path],
    chunk_specs: Sequence[ChunkSpec],
    total_frames: int,
    fps: float,
    output_video: Path,
) -> None:
    if len(chunk_video_paths) != len(chunk_specs):
        raise ValueError("chunk_video_paths and chunk_specs length mismatch")
    if len(chunk_video_paths) == 1 and chunk_specs[0].valid_length == total_frames:
        trim_video(chunk_video_paths[0], output_video, total_frames, fps)
        return

    with tempfile.TemporaryDirectory(prefix="render-pipeline-merge-") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        frame_sources: List[List[Path]] = [[] for _ in range(total_frames)]

        for chunk_idx, (video_path, spec) in enumerate(zip(chunk_video_paths, chunk_specs)):
            chunk_dir = tmpdir / f"chunk_{chunk_idx:04d}"
            frames = extract_video_frames(video_path, chunk_dir)
            if len(frames) < spec.valid_length:
                raise RuntimeError(
                    f"Chunk video {video_path} has only {len(frames)} frames, expected at least {spec.valid_length}"
                )
            for local_idx in range(spec.valid_length):
                global_idx = spec.start + local_idx
                if global_idx >= total_frames:
                    break
                frame_sources[global_idx].append(frames[local_idx])

        merged_dir = tmpdir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, sources in enumerate(frame_sources):
            if not sources:
                raise RuntimeError(f"No sources found for global frame {frame_idx}")
            output_frame = merged_dir / f"frame_{frame_idx:06d}.png"
            average_images(sources).save(output_frame)

        encode_video(merged_dir / "frame_%06d.png", output_video, fps=fps)


def find_single_match(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} under {folder}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one file matching {pattern} under {folder}, found {len(matches)}")
    return matches[0]


def stage_inverse_chunk(chunk_root: Path, all_rgb_frames: Sequence[Path], spec: ChunkSpec, chunk_size: int) -> Path:
    clip_dir = chunk_root / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    last_idx = len(all_rgb_frames) - 1
    for local_idx in range(chunk_size):
        src_idx = min(spec.start + local_idx, last_idx)
        dst = clip_dir / f"{local_idx:06d}.png"
        link_or_copy(all_rgb_frames[src_idx], dst)
    return chunk_root


def stage_forward_chunk(
    chunk_root: Path,
    frame_map: dict[str, Sequence[Path]],
    spec: ChunkSpec,
    chunk_size: int,
) -> Path:
    clip_dir = chunk_root / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    last_idx = min(len(next(iter(frame_map.values()))) - 1, max(len(v) - 1 for v in frame_map.values()))
    for local_idx in range(chunk_size):
        src_idx = min(spec.start + local_idx, last_idx)
        for pass_name, frames in frame_map.items():
            dst = clip_dir / f"{local_idx:06d}.{pass_name}.png"
            link_or_copy(frames[src_idx], dst)
    return chunk_root


def build_common_inference_args(args: argparse.Namespace, fps: int) -> List[str]:
    cmd = [
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--num_steps",
        str(args.num_steps),
        "--guidance",
        str(args.guidance),
        "--num_video_frames",
        str(args.chunk_size),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--fps",
        str(fps),
        "--seed",
        str(args.seed),
    ]
    if args.resize_resolution is not None:
        cmd.extend(["--resize_resolution", str(args.resize_resolution[0]), str(args.resize_resolution[1])])
    if args.offload_diffusion_transformer:
        cmd.append("--offload_diffusion_transformer")
    if args.offload_tokenizer:
        cmd.append("--offload_tokenizer")
    if args.offload_text_encoder_model:
        cmd.append("--offload_text_encoder_model")
    if args.offload_guardrail_models:
        cmd.append("--offload_guardrail_models")
    return cmd


def run_inverse(args: argparse.Namespace) -> None:
    input_rgb = Path(args.input_rgb)
    ensure_file(input_rgb, "input RGB video")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = count_video_frames(input_rgb)
    fps = int(round(args.fps if args.fps is not None else read_video_fps(input_rgb)))
    chunk_specs = sliding_window_specs(total_frames, args.chunk_size, args.overlap)

    tmp_root = Path(args.tmp_root) / f"inverse_{input_rgb.stem}"
    if tmp_root.exists() and not args.keep_tmp:
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    log(f"Extracting RGB frames from {input_rgb}")
    rgb_frames = extract_video_frames(input_rgb, tmp_root / "rgb_frames")

    pass_to_chunk_videos = {name: [] for name in args.inference_passes}
    try:
        for spec in chunk_specs:
            chunk_name = f"chunk_{spec.start:06d}"
            chunk_input_root = stage_inverse_chunk(tmp_root / "chunks" / chunk_name, rgb_frames, spec, args.chunk_size)
            chunk_output_root = tmp_root / "outputs" / chunk_name
            chunk_output_root.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "cosmos_predict1/diffusion/inference/inference_inverse_renderer.py",
                "--dataset_path",
                str(chunk_input_root),
                "--group_mode",
                "folder",
                "--chunk_mode",
                "first",
                "--save_image",
                "False",
                "--save_video",
                "True",
                "--video_save_folder",
                str(chunk_output_root),
                "--inference_passes",
                *args.inference_passes,
            ]
            cmd.extend(build_common_inference_args(args, fps=fps))
            if args.normalize_normal:
                cmd.extend(["--normalize_normal", "True"])
            run_cmd(cmd, dry_run=args.dry_run)
            if args.dry_run:
                continue

            for pass_name in args.inference_passes:
                pass_to_chunk_videos[pass_name].append(find_single_match(chunk_output_root, f"*.{pass_name}.mp4"))

        if args.dry_run:
            log("Dry run only; skipping merge.")
            return

        for pass_name in args.inference_passes:
            output_video = output_dir / f"{pass_name}.mp4"
            merge_chunk_videos(pass_to_chunk_videos[pass_name], chunk_specs, total_frames, fps, output_video)
            log(f"Saved {pass_name} video to {output_video}")
    finally:
        if not args.keep_tmp and not args.dry_run:
            shutil.rmtree(tmp_root, ignore_errors=True)


def run_forward(args: argparse.Namespace) -> None:
    input_videos = {
        "basecolor": Path(args.basecolor),
        "normal": Path(args.normal),
        "depth": Path(args.depth),
        "roughness": Path(args.roughness),
        "metallic": Path(args.metallic),
    }
    for pass_name, path in input_videos.items():
        ensure_file(path, f"{pass_name} video")
    env_map = Path(args.env_map)
    ensure_file(env_map, "environment map")

    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    total_frames = validate_same_frame_count(input_videos.values())
    fps = int(round(args.fps if args.fps is not None else read_video_fps(input_videos["basecolor"])))
    chunk_specs = sliding_window_specs(total_frames, args.chunk_size, args.overlap)

    tmp_root = Path(args.tmp_root) / f"forward_{output_video.stem}"
    if tmp_root.exists() and not args.keep_tmp:
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    frame_map: dict[str, Sequence[Path]] = {}
    log("Extracting intrinsic frames")
    for pass_name, path in input_videos.items():
        frame_map[pass_name] = extract_video_frames(path, tmp_root / f"{pass_name}_frames")

    chunk_videos: List[Path] = []
    try:
        for spec in chunk_specs:
            chunk_name = f"chunk_{spec.start:06d}"
            chunk_input_root = stage_forward_chunk(tmp_root / "chunks" / chunk_name, frame_map, spec, args.chunk_size)
            chunk_output_root = tmp_root / "outputs" / chunk_name
            chunk_output_root.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "cosmos_predict1/diffusion/inference/inference_forward_renderer.py",
                "--dataset_path",
                str(chunk_input_root),
                "--save_image",
                "False",
                "--video_save_folder",
                str(chunk_output_root),
                "--use_custom_envmap",
                "True",
                "--env_map",
                str(env_map),
            ]
            cmd.extend(build_common_inference_args(args, fps=fps))
            if args.rotate_light:
                cmd.extend(["--rotate_light", "True"])
            if args.use_fixed_frame_ind:
                cmd.extend(["--use_fixed_frame_ind", "True", "--fixed_frame_ind", str(args.fixed_frame_ind)])
            run_cmd(cmd, dry_run=args.dry_run)
            if args.dry_run:
                continue
            chunk_videos.append(find_single_match(chunk_output_root, "*.relit_*.mp4"))

        if args.dry_run:
            log("Dry run only; skipping merge.")
            return

        merge_chunk_videos(chunk_videos, chunk_specs, total_frames, fps, output_video)
        log(f"Saved relit video to {output_video}")
    finally:
        if not args.keep_tmp and not args.dry_run:
            shutil.rmtree(tmp_root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sliding-window wrapper around forward and inverse renderers")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
        subparser.add_argument("--num_steps", type=int, default=15)
        subparser.add_argument("--guidance", type=float, default=0.0)
        subparser.add_argument("--height", type=int, default=704)
        subparser.add_argument("--width", type=int, default=1280)
        subparser.add_argument("--fps", type=float, default=None)
        subparser.add_argument("--seed", type=int, default=1000)
        subparser.add_argument("--chunk_size", type=int, default=57)
        subparser.add_argument("--overlap", type=int, default=50)
        subparser.add_argument("--resize_resolution", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"))
        subparser.add_argument("--offload_diffusion_transformer", action="store_true")
        subparser.add_argument("--offload_tokenizer", action="store_true")
        subparser.add_argument("--offload_text_encoder_model", action="store_true")
        subparser.add_argument("--offload_guardrail_models", action="store_true")
        subparser.add_argument("--tmp_root", type=str, default=".render_tmp")
        subparser.add_argument("--keep_tmp", action="store_true")
        subparser.add_argument("--dry_run", action="store_true")

    inverse = subparsers.add_parser("inverse", help="Run inverse renderer on an RGB video")
    add_common(inverse)
    inverse.add_argument("--input_rgb", type=str, required=True)
    inverse.add_argument("--output_dir", type=str, required=True)
    inverse.add_argument("--inference_passes", type=str, nargs="+", default=list(INVERSE_PASSES), choices=list(INVERSE_PASSES))
    inverse.add_argument("--normalize_normal", action="store_true")
    inverse.set_defaults(func=run_inverse)

    forward = subparsers.add_parser("forward", help="Run forward renderer on intrinsic videos")
    add_common(forward)
    forward.add_argument("--basecolor", type=str, required=True)
    forward.add_argument("--normal", type=str, required=True)
    forward.add_argument("--depth", type=str, required=True)
    forward.add_argument("--roughness", type=str, required=True)
    forward.add_argument("--metallic", type=str, required=True)
    forward.add_argument("--env_map", type=str, required=True)
    forward.add_argument("--output_video", type=str, required=True)
    forward.add_argument("--rotate_light", action="store_true")
    forward.add_argument("--use_fixed_frame_ind", action="store_true")
    forward.add_argument("--fixed_frame_ind", type=int, default=0)
    forward.set_defaults(func=run_forward)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

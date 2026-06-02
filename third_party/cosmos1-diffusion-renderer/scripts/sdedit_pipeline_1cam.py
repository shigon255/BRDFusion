#!/usr/bin/env python3
"""Orchestrate SDEdit inverse/forward workflows for a single camera (cam 0)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

CAM_ID = 0
INVERSE_PASS_ORDER: Sequence[str] = ("basecolor", "normal", "depth", "roughness", "metallic")
INVERSE_PASS_NAME_MAP: Dict[str, str] = {
    "basecolor": "albedo",
    "normal": "normal",
    "depth": "normalized_depth",
    "roughness": "roughness",
    "metallic": "metallic",
}
INVERSE_CLI_TO_INTERNAL: Dict[str, str] = {
    "albedo": "basecolor",
    "basecolor": "basecolor",
    "normal": "normal",
    "normalized_depth": "depth",
    "depth": "depth",
    "roughness": "roughness",
    "metallic": "metallic",
}
FORWARD_COPY_PASS_ORDER: Sequence[str] = ("basecolor", "normal", "depth", "roughness", "metallic")
FORWARD_PASS_NAME_MAP: Dict[str, str] = {
    "basecolor": "albedo",
    "normal": "normal",
    "depth": "normalized_depth",
    "roughness": "roughness",
    "metallic": "metallic",
}


@dataclass(frozen=True)
class VideoInputs:
    rgb: Path
    basecolor: Path
    normal: Path
    depth: Path
    roughness: Path
    metallic: Path


def log(msg: str) -> None:
    print(f"[sdedit-pipeline-1cam] {msg}")


def run_cmd(cmd: List[str], dry_run: bool = False, cwd: Path | None = None) -> None:
    pretty = " ".join(cmd)
    if cwd is not None:
        pretty = f"(cd {cwd} && {pretty})"
    log(pretty)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def strength_tag(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def resolve_sdedit_values(args: argparse.Namespace) -> tuple[list[float], str]:
    """Resolve SDEdit sweep values and control mode."""
    has_strengths = args.strengths is not None and len(args.strengths) > 0
    has_sigmas = args.sigmas is not None and len(args.sigmas) > 0
    if has_strengths == has_sigmas:
        raise ValueError("Provide exactly one of --strengths or --sigmas.")
    if has_sigmas:
        for sigma in args.sigmas:
            if sigma < 0:
                raise ValueError("--sigmas values must be >= 0.")
        return list(args.sigmas), "sigma"
    return list(args.strengths), "strength"


def ffprobe_stream_info(video_path: Path) -> Dict[str, object]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,nb_read_packets",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {video_path}")
    return streams[0]


def count_video_frames(video_path: Path) -> int:
    info = ffprobe_stream_info(video_path)
    for key in ("nb_read_packets", "nb_frames"):
        raw = info.get(key)
        if raw and str(raw).isdigit():
            count = int(raw)
            if count > 0:
                return count
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
    frame_counts = {str(p): count_video_frames(p) for p in paths}
    uniq = sorted(set(frame_counts.values()))
    if len(uniq) != 1:
        details = ", ".join(f"{k}={v}" for k, v in frame_counts.items())
        raise ValueError(f"Input videos must have same frame count. Found: {details}")
    return uniq[0]


def resolve_envmap_path(input_root: Path, prefix: str, cam_id: int) -> Path:
    candidates = [
        input_root / f"{prefix}{cam_id}.hdr",
        input_root / f"{prefix}{cam_id}.exr",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Missing envmap for cam {cam_id}. Tried: {tried}")


def ensure_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def sliding_window_starts(total_frames: int, chunk_size: int, overlap: int) -> List[int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    if total_frames <= chunk_size:
        return [0]

    cutoff = total_frames - overlap
    starts: List[int] = []
    start = 0
    while start < cutoff:
        starts.append(start)
        start += step
    if not starts:
        starts = [0]
    return starts


def trim_video(input_video: Path, output_video: Path, frame_count: int, fps: float, dry_run: bool = False) -> None:
    if not dry_run:
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
    run_cmd(cmd, dry_run=dry_run)


def resize_video(
    input_video: Path,
    output_video: Path,
    width: int | None,
    height: int | None,
    dry_run: bool = False,
) -> None:
    if not width or not height:
        if input_video != output_video and not dry_run:
            output_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_video, output_video)
        return
    if not dry_run:
        output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-vf",
        f"scale={width}:{height}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_video),
    ]
    run_cmd(cmd, dry_run=dry_run)


def extract_images(
    video_path: Path,
    output_dir: Path,
    cam_id: int,
    dry_run: bool = False,
    resize_width: int | None = None,
    resize_height: int | None = None,
) -> None:
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = output_dir / f"%03d_{cam_id}.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-start_number",
        "0",
    ]
    if resize_width and resize_height:
        cmd.extend(["-vf", f"scale={resize_width}:{resize_height}"])
    cmd.append(str(out_pattern))
    run_cmd(cmd, dry_run=dry_run)


def find_buffer_video(folder: Path, buffer_type: str) -> Path:
    matches = sorted(folder.glob(f"*{buffer_type}.mp4"))
    if not matches:
        raise FileNotFoundError(f"No *{buffer_type}.mp4 found in {folder}")
    return matches[0]


def run_merge_overlap(
    input_dir: Path,
    output_dir: Path,
    strength: str,
    cam_id: int,
    seed: int,
    buffer_type: str,
    chunk_size: int,
    step: int,
    fps: int,
    dry_run: bool,
) -> Path:
    cmd = [
        sys.executable,
        "merge_overlap_videos.py",
        str(input_dir),
        str(output_dir),
        strength,
        str(cam_id),
        str(seed),
        buffer_type,
        "--chunk_size",
        str(chunk_size),
        "--step",
        str(step),
        "--fps",
        str(fps),
    ]
    run_cmd(cmd, dry_run=dry_run)
    return output_dir / f"sdedit_w{strength}_cam{cam_id}_seed{seed}"


def resolve_inverse_passes(intrinsics: Sequence[str]) -> List[str]:
    selected: List[str] = []
    seen = set()
    for item in intrinsics:
        key = item.strip().lower()
        if key not in INVERSE_CLI_TO_INTERNAL:
            allowed = ", ".join(sorted(INVERSE_CLI_TO_INTERNAL.keys()))
            raise ValueError(f"Unknown intrinsic '{item}'. Allowed: {allowed}")
        internal = INVERSE_CLI_TO_INTERNAL[key]
        if internal not in seen:
            seen.add(internal)
            selected.append(internal)
    if not selected:
        raise ValueError("At least one intrinsic must be selected")
    return selected


def build_inverse_inputs(args: argparse.Namespace, selected_passes: Sequence[str]) -> Dict[str, Path]:
    root = Path(args.input_root)
    inputs: Dict[str, Path] = {
        "rgb": root / args.gt_rgb,
        "basecolor": root / args.albedo,
        "normal": root / args.normal,
        "depth": root / args.normalized_depth,
        "roughness": root / args.roughness,
        "metallic": root / args.metallic,
    }
    ensure_file(inputs["rgb"], "gt_rgb video")
    for pass_name in selected_passes:
        ensure_file(inputs[pass_name], f"{INVERSE_PASS_NAME_MAP[pass_name]} video")
    return inputs


def build_forward_inputs(args: argparse.Namespace) -> VideoInputs:
    root = Path(args.input_root)
    inputs = VideoInputs(
        rgb=root / args.pbr_rgb,
        basecolor=root / args.albedo,
        normal=root / args.normal,
        depth=root / args.normalized_depth,
        roughness=root / args.roughness,
        metallic=root / args.metallic,
    )
    ensure_file(inputs.rgb, "pbr_rgb video")
    ensure_file(inputs.basecolor, "albedo video")
    ensure_file(inputs.normal, "normal video")
    ensure_file(inputs.depth, "normalized_depth video")
    ensure_file(inputs.roughness, "roughness video")
    ensure_file(inputs.metallic, "metallic video")
    return inputs


def run_inverse(args: argparse.Namespace) -> None:
    values, mode = resolve_sdedit_values(args)
    selected_passes = resolve_inverse_passes(args.intrinsics)
    inputs = build_inverse_inputs(args, selected_passes)
    frame_count = validate_same_frame_count([inputs["rgb"]] + [inputs[p] for p in selected_passes])
    fps = int(read_video_fps(inputs["rgb"]))
    starts = sliding_window_starts(frame_count, args.chunk_size, args.overlap)
    step = args.chunk_size - args.overlap

    log(f"Discovered inverse inputs under {args.input_root}")
    log(f"Input frame count={frame_count}, fps={fps}, chunk_starts={starts}")

    tmp_root = Path(args.tmp_root)
    chunks_root = tmp_root / "chunks_inverse"
    merged_root = tmp_root / "merged_inverse"

    if not args.keep_tmp and not args.dry_run:
        shutil.rmtree(chunks_root, ignore_errors=True)
        shutil.rmtree(merged_root, ignore_errors=True)

    for value in values:
        s_tag = strength_tag(value)
        log(f"Running inverse pipeline for {mode}={s_tag}")

        for start_frame in starts:
            chunk_dir = chunks_root / f"sdedit_w{s_tag}_cam{CAM_ID}_seed{args.seed}_start{start_frame}"
            for pass_name in selected_passes:
                cmd = [
                    sys.executable,
                    "sdedit_inverse_renderer.py",
                    "--input_rgb_video",
                    str(inputs["rgb"]),
                    "--basecolor_video",
                    str(inputs["basecolor"]),
                    "--normal_video",
                    str(inputs["normal"]),
                    "--depth_video",
                    str(inputs["depth"]),
                    "--roughness_video",
                    str(inputs["roughness"]),
                    "--metallic_video",
                    str(inputs["metallic"]),
                    "--output_dir",
                    str(chunk_dir),
                    "--save_image",
                    str(args.save_chunk_images),
                    "--start_frame",
                    str(start_frame),
                    "--guidance",
                    str(args.guidance),
                    "--num_steps",
                    str(args.num_steps),
                    "--s_churn",
                    str(args.s_churn),
                    "--s_noise",
                    str(args.s_noise),
                    "--s_tmin",
                    str(args.s_tmin),
                    "--s_tmax",
                    str(args.s_tmax),
                    "--height",
                    str(args.height),
                    "--width",
                    str(args.width),
                    "--scheduler_sigma_min",
                    str(args.scheduler_sigma_min),
                    "--scheduler_sigma_max",
                    str(args.scheduler_sigma_max),
                    "--scheduler_rho",
                    str(args.scheduler_rho),
                    "--max_frames",
                    str(args.chunk_size),
                    "--seed",
                    str(args.seed),
                    "--normalize_normal",
                    str(args.normalize_normal),
                    "--inference_passes",
                    pass_name,
                    "--checkpoint_dir",
                    str(args.checkpoint_dir),
                ]
                if mode == "sigma":
                    cmd.extend(["--sdedit_sigma", str(value)])
                else:
                    cmd.extend(["--sdedit_strength", str(value)])
                if args.guidance_start is not None:
                    cmd.extend(["--guidance_start", str(args.guidance_start), "--guidance_end", str(args.guidance_end)])
                if args.offload_diffusion_transformer:
                    cmd.append("--offload_diffusion_transformer")
                if args.offload_tokenizer:
                    cmd.append("--offload_tokenizer")
                run_cmd(cmd, dry_run=args.dry_run, cwd=Path.cwd())

        final_root = Path(args.output_root) / f"refined_intrinsic_w{s_tag}"
        final_videos = final_root / "videos"
        final_images = final_root / "images"
        if not args.dry_run:
            final_videos.mkdir(parents=True, exist_ok=True)
            final_images.mkdir(parents=True, exist_ok=True)

        for pass_name in selected_passes:
            merged_folder = run_merge_overlap(
                input_dir=chunks_root,
                output_dir=merged_root,
                strength=s_tag,
                cam_id=CAM_ID,
                seed=args.seed,
                buffer_type=pass_name,
                chunk_size=args.chunk_size,
                step=step,
                fps=fps,
                dry_run=args.dry_run,
            )
            merged_video = merged_folder / f"dryrun.{pass_name}.mp4" if args.dry_run else find_buffer_video(merged_folder, pass_name)

            normalized = INVERSE_PASS_NAME_MAP[pass_name]
            out_video = final_videos / f"{normalized}.mp4"
            out_raw = final_videos / f"{normalized}.pre_resize.mp4" if args.post_resize_width and args.post_resize_height else out_video
            trim_video(merged_video, out_raw, frame_count, fps, dry_run=args.dry_run)
            resize_video(
                input_video=out_raw,
                output_video=out_video,
                width=args.post_resize_width,
                height=args.post_resize_height,
                dry_run=args.dry_run,
            )
            if out_raw != out_video and not args.dry_run and out_raw.exists():
                out_raw.unlink()

            image_dir = final_images / normalized
            extract_images(
                out_video,
                image_dir,
                CAM_ID,
                dry_run=args.dry_run,
                resize_width=args.post_resize_width,
                resize_height=args.post_resize_height,
            )

        log(f"Completed inverse outputs: {final_root}")


def run_forward(args: argparse.Namespace) -> None:
    values, mode = resolve_sdedit_values(args)
    inputs = build_forward_inputs(args)
    frame_count = validate_same_frame_count([inputs.rgb, inputs.basecolor, inputs.normal, inputs.depth, inputs.roughness, inputs.metallic])
    fps = int(read_video_fps(inputs.rgb))
    starts = sliding_window_starts(frame_count, args.chunk_size, args.overlap)
    step = args.chunk_size - args.overlap

    envmap = resolve_envmap_path(Path(args.input_root), args.envmap_prefix, CAM_ID)

    log(f"Discovered forward inputs under {args.input_root}")
    log(f"Input frame count={frame_count}, fps={fps}, chunk_starts={starts}")

    tmp_root = Path(args.tmp_root)
    chunks_root = tmp_root / "chunks_forward"
    merged_root = tmp_root / "merged_forward"

    if not args.keep_tmp and not args.dry_run:
        shutil.rmtree(chunks_root, ignore_errors=True)
        shutil.rmtree(merged_root, ignore_errors=True)

    for value in values:
        s_tag = strength_tag(value)
        log(f"Running forward pipeline for {mode}={s_tag}")

        for start_frame in starts:
            chunk_dir = chunks_root / f"sdedit_w{s_tag}_cam{CAM_ID}_seed{args.seed}_start{start_frame}"
            output_video = chunk_dir / "video.rgb.mp4"
            cmd = [
                sys.executable,
                "sdedit_forward_renderer.py",
                "--input_rgb_video",
                str(inputs.rgb),
                "--basecolor_video",
                str(inputs.basecolor),
                "--normal_video",
                str(inputs.normal),
                "--depth_video",
                str(inputs.depth),
                "--roughness_video",
                str(inputs.roughness),
                "--metallic_video",
                str(inputs.metallic),
                "--env_map",
                str(envmap),
                "--output_video",
                str(output_video),
                "--start_frame",
                str(start_frame),
                "--max_frames",
                str(args.chunk_size),
                "--guidance",
                str(args.guidance),
                "--num_steps",
                str(args.num_steps),
                "--s_churn",
                str(args.s_churn),
                "--s_noise",
                str(args.s_noise),
                "--s_tmin",
                str(args.s_tmin),
                "--s_tmax",
                str(args.s_tmax),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--scheduler_sigma_min",
                str(args.scheduler_sigma_min),
                "--scheduler_sigma_max",
                str(args.scheduler_sigma_max),
                "--scheduler_rho",
                str(args.scheduler_rho),
                "--seed",
                str(args.seed),
                "--checkpoint_dir",
                str(args.checkpoint_dir),
            ]
            if mode == "sigma":
                cmd.extend(["--sdedit_sigma", str(value)])
            else:
                cmd.extend(["--sdedit_strength", str(value)])
            if args.guidance_start is not None:
                cmd.extend(["--guidance_start", str(args.guidance_start), "--guidance_end", str(args.guidance_end)])
            if args.rotate_light:
                cmd.append("--rotate_light")
            if args.offload_diffusion_transformer:
                cmd.append("--offload_diffusion_transformer")
            if args.offload_tokenizer:
                cmd.append("--offload_tokenizer")
            run_cmd(cmd, dry_run=args.dry_run, cwd=Path.cwd())

        final_root = Path(args.output_root) / f"refined_render_w{s_tag}"
        videos_root = final_root / "videos"
        vis_root = final_root / "visualization"
        if not args.dry_run:
            (videos_root / str(CAM_ID)).mkdir(parents=True, exist_ok=True)
            vis_root.mkdir(parents=True, exist_ok=True)

        merged_folder = run_merge_overlap(
            input_dir=chunks_root,
            output_dir=merged_root,
            strength=s_tag,
            cam_id=CAM_ID,
            seed=args.seed,
            buffer_type="rgb",
            chunk_size=args.chunk_size,
            step=step,
            fps=fps,
            dry_run=args.dry_run,
        )
        merged_video = merged_folder / "dryrun.rgb.mp4" if args.dry_run else find_buffer_video(merged_folder, "rgb")

        cam_out = videos_root / str(CAM_ID) / "pbr_rgb.mp4"
        cam_raw = videos_root / str(CAM_ID) / "pbr_rgb.pre_resize.mp4" if args.post_resize_width and args.post_resize_height else cam_out
        trim_video(merged_video, cam_raw, frame_count, fps, dry_run=args.dry_run)
        resize_video(
            input_video=cam_raw,
            output_video=cam_out,
            width=args.post_resize_width,
            height=args.post_resize_height,
            dry_run=args.dry_run,
        )
        if cam_raw != cam_out and not args.dry_run and cam_raw.exists():
            cam_raw.unlink()

        concat_out = vis_root / "pbr_rgb_concat.mp4"
        if not args.dry_run:
            shutil.copy2(cam_out, concat_out)

        for pass_name in FORWARD_COPY_PASS_ORDER:
            output_name = FORWARD_PASS_NAME_MAP[pass_name]
            src_video = getattr(inputs, pass_name)
            out_video = videos_root / str(CAM_ID) / f"{output_name}.mp4"
            out_raw = videos_root / str(CAM_ID) / f"{output_name}.pre_resize.mp4" if args.post_resize_width and args.post_resize_height else out_video
            trim_video(src_video, out_raw, frame_count, fps, dry_run=args.dry_run)
            resize_video(
                input_video=out_raw,
                output_video=out_video,
                width=args.post_resize_width,
                height=args.post_resize_height,
                dry_run=args.dry_run,
            )
            if out_raw != out_video and not args.dry_run and out_raw.exists():
                out_raw.unlink()

            concat_pass_out = vis_root / f"{output_name}_concat.mp4"
            if not args.dry_run:
                shutil.copy2(out_video, concat_pass_out)

        log(f"Completed forward outputs: {final_root}")


def add_common_args(parser: argparse.ArgumentParser, *, forward: bool) -> None:
    parser.add_argument("--strengths", type=float, nargs="+", default=None, help="(Deprecated) SDEdit strengths")
    parser.add_argument("--sigmas", type=float, nargs="+", default=None, help="SDEdit start sigmas")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--chunk_size", type=int, default=57)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--guidance_start", type=float, default=None)
    parser.add_argument("--guidance_end", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=15)
    parser.add_argument("--s_churn", type=float, default=0.0)
    parser.add_argument("--s_noise", type=float, default=1.0)
    parser.add_argument("--s_tmin", type=float, default=0.0)
    parser.add_argument("--s_tmax", type=float, default=float("inf"))
    parser.add_argument("--scheduler_sigma_min", type=float, default=0.02)
    parser.add_argument("--scheduler_sigma_max", type=float, default=80.0)
    parser.add_argument("--scheduler_rho", type=float, default=7.0)
    parser.add_argument("--offload_diffusion_transformer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload_tokenizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tmp_root", type=str, default=".sdedit_tmp")
    parser.add_argument("--post_resize_width", type=int, default=None)
    parser.add_argument("--post_resize_height", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    if forward:
        parser.add_argument("--rotate_light", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SDEdit 1-cam workflow orchestrator")
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    inv = subparsers.add_parser("inverse", help="Run inverse SDEdit workflow")
    inv.add_argument("--input_root", type=str, default="intrinsic_refinement/raw_intrinsic")
    inv.add_argument("--output_root", type=str, default="intrinsic_refinement")
    inv.add_argument("--gt_rgb", type=str, default="gt_rgb.mp4")
    inv.add_argument("--albedo", type=str, default="albedo.mp4")
    inv.add_argument("--normal", type=str, default="normal.mp4")
    inv.add_argument("--normalized_depth", type=str, default="normalized_depth.mp4")
    inv.add_argument("--roughness", type=str, default="roughness.mp4")
    inv.add_argument("--metallic", type=str, default="metallic.mp4")
    inv.add_argument("--normalize_normal", action=argparse.BooleanOptionalAction, default=True)
    inv.add_argument("--save_chunk_images", action=argparse.BooleanOptionalAction, default=False)
    inv.add_argument(
        "--intrinsics",
        type=str,
        nargs="+",
        default=["albedo", "normal", "normalized_depth", "roughness", "metallic"],
        help="Subset of intrinsics to process (aliases allowed: albedo/basecolor, normalized_depth/depth)",
    )
    add_common_args(inv, forward=False)

    fwd = subparsers.add_parser("forward", help="Run forward SDEdit workflow")
    fwd.add_argument("--input_root", type=str, default="pbr_refinement/raw_render")
    fwd.add_argument("--output_root", type=str, default="pbr_refinement")
    fwd.add_argument("--pbr_rgb", type=str, default="pbr_rgb.mp4")
    fwd.add_argument("--albedo", type=str, default="albedo.mp4")
    fwd.add_argument("--normal", type=str, default="normal.mp4")
    fwd.add_argument("--normalized_depth", type=str, default="normalized_depth.mp4")
    fwd.add_argument("--roughness", type=str, default="roughness.mp4")
    fwd.add_argument("--metallic", type=str, default="metallic.mp4")
    fwd.add_argument("--envmap_prefix", type=str, default="envmap_")
    add_common_args(fwd, forward=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (args.guidance_start is None) != (args.guidance_end is None):
        raise ValueError("Both --guidance_start and --guidance_end must be provided together, or neither.")
    if args.workflow == "inverse":
        run_inverse(args)
    elif args.workflow == "forward":
        run_forward(args)
    else:
        raise ValueError(f"Unknown workflow: {args.workflow}")


if __name__ == "__main__":
    main()

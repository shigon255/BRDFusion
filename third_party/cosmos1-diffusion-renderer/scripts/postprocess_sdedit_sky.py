#!/usr/bin/env python3
"""Post-process SDEdit rendered PBR videos with sky replacement using opacity mask."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image

CAM_IDS: Sequence[int] = (0, 1, 2)
CONCAT_ORDER: Sequence[int] = (1, 0, 2)
INTRINSIC_NAMES: Sequence[str] = ("albedo", "normal", "normalized_depth", "roughness", "metallic")


def log(msg: str) -> None:
    print(f"[sdedit-sky] {msg}")


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    log(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def strength_tag(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def ffprobe_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        return 24.0
    raw = str(streams[0].get("avg_frame_rate", "0/1"))
    if "/" in raw:
        num, den = raw.split("/", 1)
        if den != "0":
            fps = float(num) / float(den)
            if fps > 0:
                return fps
    return 24.0


def split_concat_video(input_video: Path, output_dir: Path, dry_run: bool = False) -> Dict[int, Path]:
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    base = input_video.stem
    out_paths = {
        1: output_dir / f"{base}_1.mp4",
        0: output_dir / f"{base}_0.mp4",
        2: output_dir / f"{base}_2.mp4",
    }
    split_cmds = {
        1: ["crop=iw/3:ih:0:0", out_paths[1]],
        0: ["crop=iw/3:ih:iw/3:0", out_paths[0]],
        2: ["crop=iw/3:ih:2*iw/3:0", out_paths[2]],
    }
    for cam_id in (1, 0, 2):
        vf, out_path = split_cmds[cam_id]
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_video),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(out_path),
        ]
        run_cmd(cmd, dry_run=dry_run)
    return out_paths


def extract_frames(video_path: Path, output_dir: Path, dry_run: bool = False) -> List[Path]:
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%05d.png"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-start_number",
        "0",
        str(pattern),
    ]
    run_cmd(cmd, dry_run=dry_run)
    if dry_run:
        return [output_dir / "frame_00000.png"]
    return sorted(output_dir.glob("frame_*.png"))


def encode_video(frame_pattern: Path, output_video: Path, fps: float, dry_run: bool = False) -> None:
    if not dry_run:
        output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_video),
    ]
    run_cmd(cmd, dry_run=dry_run)


def merge_concat(cam_videos: Dict[int, Path], output_video: Path, dry_run: bool = False) -> None:
    cmd = [
        sys.executable,
        "merge.py",
        str(cam_videos[1]),
        str(cam_videos[0]),
        str(cam_videos[2]),
        "-o",
        str(output_video),
    ]
    run_cmd(cmd, dry_run=dry_run)


def copy_video(src: Path, dst: Path, dry_run: bool = False) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source video to copy: {src}")
    log(f"copy {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def blend_frame(pbr_path: Path, sky_path: Path, opacity_path: Path, out_path: Path) -> None:
    pbr = Image.open(pbr_path).convert("RGB")
    sky = Image.open(sky_path).convert("RGB")
    # Convert opacity video frame to a single-channel soft mask in [0, 255].
    alpha = Image.open(opacity_path).convert("L")

    # out = pbr*alpha + sky*(1-alpha), where alpha is normalized by 255.
    out = Image.composite(pbr, sky, alpha)
    out.save(out_path)


def process_strength(
    output_root: Path,
    strength: float,
    sky_cam_videos: Dict[int, Path],
    opacity_cam_videos: Dict[int, Path],
    tmp_root: Path,
    keep_tmp: bool,
    dry_run: bool,
) -> None:
    s_tag = strength_tag(strength)
    src_root = output_root / f"refined_render_w{s_tag}"
    dst_root = output_root / f"refined_render_w{s_tag}_sky"

    if not src_root.exists():
        raise FileNotFoundError(f"Missing source folder: {src_root}")

    log(f"Processing strength={s_tag}")
    log("Sky postprocess behavior: blend sky for pbr_rgb only; copy intrinsic videos unchanged.")

    cam_outputs: Dict[int, Path] = {}
    for cam_id in CAM_IDS:
        src_video = src_root / "videos" / str(cam_id) / "pbr_rgb.mp4"
        if not src_video.exists():
            raise FileNotFoundError(f"Missing source video: {src_video}")

        fps = ffprobe_fps(src_video)
        cam_tmp = tmp_root / f"w{s_tag}_cam{cam_id}"
        pbr_frames_dir = cam_tmp / "pbr"
        sky_frames_dir = cam_tmp / "sky"
        opa_frames_dir = cam_tmp / "opacity"
        out_frames_dir = cam_tmp / "out"

        if not dry_run:
            if cam_tmp.exists() and not keep_tmp:
                shutil.rmtree(cam_tmp)
            out_frames_dir.mkdir(parents=True, exist_ok=True)

        pbr_frames = extract_frames(src_video, pbr_frames_dir, dry_run=dry_run)
        sky_frames = extract_frames(sky_cam_videos[cam_id], sky_frames_dir, dry_run=dry_run)
        opa_frames = extract_frames(opacity_cam_videos[cam_id], opa_frames_dir, dry_run=dry_run)

        n = min(len(pbr_frames), len(sky_frames), len(opa_frames))
        if n == 0:
            raise RuntimeError(f"No frames to blend for cam {cam_id}, strength {s_tag}")

        if not dry_run:
            for i in range(n):
                out_path = out_frames_dir / f"frame_{i:05d}.png"
                blend_frame(pbr_frames[i], sky_frames[i], opa_frames[i], out_path)

        out_video = dst_root / "videos" / str(cam_id) / "pbr_rgb.mp4"
        encode_video(out_frames_dir / "frame_%05d.png", out_video, fps=fps, dry_run=dry_run)
        cam_outputs[cam_id] = out_video

    vis_out = dst_root / "visualization" / "pbr_rgb_concat.mp4"
    merge_concat(cam_outputs, vis_out, dry_run=dry_run)

    # Copy non-SDEdit intrinsic videos and visualizations from source to _sky target.
    for intrinsic in INTRINSIC_NAMES:
        for cam_id in CAM_IDS:
            src_video = src_root / "videos" / str(cam_id) / f"{intrinsic}.mp4"
            dst_video = dst_root / "videos" / str(cam_id) / f"{intrinsic}.mp4"
            copy_video(src_video, dst_video, dry_run=dry_run)

        src_concat = src_root / "visualization" / f"{intrinsic}_concat.mp4"
        dst_concat = dst_root / "visualization" / f"{intrinsic}_concat.mp4"
        copy_video(src_concat, dst_concat, dry_run=dry_run)

    log(f"Saved: {dst_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sky replacement postprocess for refined_render_w{strength}")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--strengths", type=float, nargs="+", required=True)
    parser.add_argument("--sky_video", type=str, required=True, help="Concatenated 3-camera sky RGB video")
    parser.add_argument("--opacity_video", type=str, required=True, help="Concatenated 3-camera opacity video")
    parser.add_argument("--tmp_root", type=str, default=".sdedit_tmp/sky")
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    sky_video = Path(args.sky_video)
    opacity_video = Path(args.opacity_video)
    tmp_root = Path(args.tmp_root)

    if not sky_video.exists():
        raise FileNotFoundError(f"Missing sky video: {sky_video}")
    if not opacity_video.exists():
        raise FileNotFoundError(f"Missing opacity video: {opacity_video}")

    split_root = tmp_root / "split"
    if not args.dry_run:
        split_root.mkdir(parents=True, exist_ok=True)

    log(f"Using output root: {output_root}")
    log(f"Using sky video: {sky_video}")
    log(f"Using opacity video: {opacity_video}")

    sky_cam = split_concat_video(sky_video, split_root, dry_run=args.dry_run)
    opa_cam = split_concat_video(opacity_video, split_root, dry_run=args.dry_run)

    for strength in args.strengths:
        process_strength(
            output_root=output_root,
            strength=strength,
            sky_cam_videos=sky_cam,
            opacity_cam_videos=opa_cam,
            tmp_root=tmp_root,
            keep_tmp=args.keep_tmp,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

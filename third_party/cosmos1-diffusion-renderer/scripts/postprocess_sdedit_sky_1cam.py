#!/usr/bin/env python3
"""Post-process SDEdit rendered PBR videos with sky replacement for one camera."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Sequence

from PIL import Image

CAM_ID = 0
INTRINSIC_NAMES: Sequence[str] = ("albedo", "normal", "normalized_depth", "roughness", "metallic")


def log(msg: str) -> None:
    print(f"[sdedit-sky-1cam] {msg}")


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
    alpha = Image.open(opacity_path).convert("L")
    out = Image.composite(pbr, sky, alpha)
    out.save(out_path)


def process_strength(
    output_root: Path,
    strength: float,
    sky_video: Path,
    opacity_video: Path,
    tmp_root: Path,
    keep_tmp: bool,
    dry_run: bool,
) -> None:
    s_tag = strength_tag(strength)
    src_root = output_root / f"refined_render_w{s_tag}"
    dst_root = output_root / f"refined_render_w{s_tag}_sky"

    if not src_root.exists():
        raise FileNotFoundError(f"Missing source folder: {src_root}")

    src_video = src_root / "videos" / str(CAM_ID) / "pbr_rgb.mp4"
    if not src_video.exists():
        raise FileNotFoundError(f"Missing source video: {src_video}")

    fps = ffprobe_fps(src_video)
    cam_tmp = tmp_root / f"w{s_tag}_cam{CAM_ID}"
    pbr_frames_dir = cam_tmp / "pbr"
    sky_frames_dir = cam_tmp / "sky"
    opa_frames_dir = cam_tmp / "opacity"
    out_frames_dir = cam_tmp / "out"

    if not dry_run:
        if cam_tmp.exists() and not keep_tmp:
            shutil.rmtree(cam_tmp)
        out_frames_dir.mkdir(parents=True, exist_ok=True)

    pbr_frames = extract_frames(src_video, pbr_frames_dir, dry_run=dry_run)
    sky_frames = extract_frames(sky_video, sky_frames_dir, dry_run=dry_run)
    opa_frames = extract_frames(opacity_video, opa_frames_dir, dry_run=dry_run)

    n = min(len(pbr_frames), len(sky_frames), len(opa_frames))
    if n == 0:
        raise RuntimeError(f"No frames to blend for strength {s_tag}")

    if not dry_run:
        for i in range(n):
            out_path = out_frames_dir / f"frame_{i:05d}.png"
            blend_frame(pbr_frames[i], sky_frames[i], opa_frames[i], out_path)

    out_video = dst_root / "videos" / str(CAM_ID) / "pbr_rgb.mp4"
    encode_video(out_frames_dir / "frame_%05d.png", out_video, fps=fps, dry_run=dry_run)

    vis_out = dst_root / "visualization" / "pbr_rgb_concat.mp4"
    if not dry_run:
        vis_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_video, vis_out)

    for intrinsic in INTRINSIC_NAMES:
        src_cam = src_root / "videos" / str(CAM_ID) / f"{intrinsic}.mp4"
        dst_cam = dst_root / "videos" / str(CAM_ID) / f"{intrinsic}.mp4"
        copy_video(src_cam, dst_cam, dry_run=dry_run)

        src_concat = src_root / "visualization" / f"{intrinsic}_concat.mp4"
        dst_concat = dst_root / "visualization" / f"{intrinsic}_concat.mp4"
        if src_concat.exists():
            copy_video(src_concat, dst_concat, dry_run=dry_run)
        else:
            copy_video(src_cam, dst_concat, dry_run=dry_run)

    log(f"Saved: {dst_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sky replacement postprocess for 1-camera refined_render_w{strength}")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--strengths", type=float, nargs="+", required=True)
    parser.add_argument("--sky_video", type=str, required=True, help="Single-camera sky RGB video")
    parser.add_argument("--opacity_video", type=str, required=True, help="Single-camera opacity video")
    parser.add_argument("--tmp_root", type=str, default=".sdedit_tmp/sky_1cam")
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

    log(f"Using output root: {output_root}")
    log(f"Using sky video: {sky_video}")
    log(f"Using opacity video: {opacity_video}")

    for strength in args.strengths:
        process_strength(
            output_root=output_root,
            strength=strength,
            sky_video=sky_video,
            opacity_video=opacity_video,
            tmp_root=tmp_root,
            keep_tmp=args.keep_tmp,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage DriveStudio/BRDFusion eval videos for DiffusionRenderer SDEdit.

DriveStudio writes videos with verbose render keys, for example
``full_set_30000_raw_render_pbr_colors.mp4``. The DiffusionRenderer SDEdit
wrappers expect a compact folder layout such as ``pbr_rgb.mp4`` and
``albedo.mp4``. This tool creates that layout with symlinks by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable


VIDEO_SUFFIXES: Dict[str, Dict[str, str]] = {
    "inverse": {
        "gt_rgb.mp4": "gt_rgbs",
        "albedo.mp4": "albedos",
        "normal.mp4": "normals",
        "normalized_depth.mp4": "normalized_depths",
        "roughness.mp4": "roughnesses",
        "metallic.mp4": "metallics",
    },
    "forward": {
        "pbr_rgb.mp4": "pbr_colors",
        "albedo.mp4": "albedos",
        "normal.mp4": "normals",
        "normalized_depth.mp4": "normalized_depths",
        "roughness.mp4": "roughnesses",
        "metallic.mp4": "metallics",
        "rgb_sky.mp4": "rgb_sky",
        "opacity.mp4": "opacities",
    },
}


def _find_video(video_root: Path, key: str, prefer: str | None) -> Path:
    candidates = sorted(video_root.rglob(f"*_{key}.mp4"))
    if not candidates:
        candidates = sorted(video_root.rglob(f"{key}.mp4"))
    if not candidates:
        raise FileNotFoundError(f"Could not find video for key '{key}' under {video_root}")
    if prefer:
        preferred = [p for p in candidates if prefer in p.name]
        if preferred:
            candidates = preferred
    # Render keys can be suffixes of other render keys, e.g. "normals" also
    # matches "depth_normals" and "roughnesses" also matches
    # "full_roughnesses". The exact key output is the shorter filename.
    return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]


def _replace_or_link(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _cam_values(raw: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in raw:
        for item in str(value).split():
            if item:
                out.append(item)
    return out or ["0"]



def _run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _ffprobe_fps(video_path: Path) -> float:
    result = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
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


def _extract_frames(video_path: Path, output_dir: Path) -> list[Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_cmd([
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-start_number",
        "0",
        str(output_dir / "frame_%05d.png"),
    ])
    frames = sorted(output_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frames


def _encode_frames(frame_dir: Path, output_video: Path, fps: float) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    _run_cmd([
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(frame_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_video),
    ])


def _blend_frame(
    foreground_path: Path,
    opacity_path: Path,
    out_path: Path,
    background_path: Path | None = None,
    background_color: tuple[int, int, int] | None = None,
) -> None:
    from PIL import Image

    foreground = Image.open(foreground_path).convert("RGB")
    alpha = Image.open(opacity_path).convert("L")
    if background_path is not None:
        background = Image.open(background_path).convert("RGB")
    elif background_color is not None:
        background = Image.new("RGB", foreground.size, background_color)
    else:
        raise ValueError("Either background_path or background_color must be provided.")
    if background.size != foreground.size:
        background = background.resize(foreground.size, Image.BILINEAR)
    if alpha.size != foreground.size:
        alpha = alpha.resize(foreground.size, Image.BILINEAR)
    Image.composite(foreground, background, alpha).save(out_path)


def _paste_condition_video(
    foreground_video: Path,
    opacity_video: Path,
    output_video: Path,
    tmp_root: Path,
    background_video: Path | None = None,
    background_color: tuple[int, int, int] | None = None,
) -> None:
    fps = _ffprobe_fps(foreground_video)
    fg_dir = tmp_root / "foreground"
    opacity_dir = tmp_root / "opacity"
    bg_dir = tmp_root / "background"
    out_dir = tmp_root / "out"
    fg_frames = _extract_frames(foreground_video, fg_dir)
    opacity_frames = _extract_frames(opacity_video, opacity_dir)
    bg_frames = _extract_frames(background_video, bg_dir) if background_video is not None else None
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_count = len(bg_frames) if bg_frames is not None else len(fg_frames)
    frame_count = min(len(fg_frames), len(opacity_frames), bg_count)
    if frame_count == 0:
        raise RuntimeError(f"No frames to blend for {foreground_video}")
    for idx in range(frame_count):
        _blend_frame(
            fg_frames[idx],
            opacity_frames[idx],
            out_dir / f"frame_{idx:05d}.png",
            background_path=bg_frames[idx] if bg_frames is not None else None,
            background_color=background_color,
        )

    tmp_video = output_video.with_name(f".{output_video.stem}.sky_paste_tmp{output_video.suffix}")
    _encode_frames(out_dir, tmp_video, fps=fps)
    if output_video.exists() or output_video.is_symlink():
        output_video.unlink()
    shutil.move(str(tmp_video), output_video)


def _paste_forward_conditions(output_root: Path) -> None:
    tmp_root = output_root / ".sky_paste_tmp"
    try:
        _paste_condition_video(
            foreground_video=output_root / "albedo.mp4",
            background_video=output_root / "rgb_sky.mp4",
            opacity_video=output_root / "opacity.mp4",
            output_video=output_root / "albedo.mp4",
            tmp_root=tmp_root / "albedo",
        )
        _paste_condition_video(
            foreground_video=output_root / "roughness.mp4",
            background_color=(128, 128, 128),
            opacity_video=output_root / "opacity.mp4",
            output_video=output_root / "roughness.mp4",
            tmp_root=tmp_root / "roughness",
        )
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(VIDEO_SUFFIXES), required=True)
    parser.add_argument("--video_root", required=True, help="Directory containing eval videos from tools/eval.py")
    parser.add_argument("--output_root", required=True, help="SDEdit input directory to create")
    parser.add_argument("--prefer", default="full_set", help="Prefer matching source filenames containing this token")
    parser.add_argument("--envmap_path", default=None, help="Optional envmap copied/linked as envmap_<cam>.<ext>")
    parser.add_argument("--cam_ids", nargs="+", default=["0"], help="Camera ids for envmap links")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not video_root.is_dir():
        raise NotADirectoryError(f"video_root does not exist: {video_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for output_name, source_key in VIDEO_SUFFIXES[args.mode].items():
        src = _find_video(video_root, source_key, args.prefer)
        dst = output_root / output_name
        _replace_or_link(src, dst, copy=args.copy)
        manifest.append(f"{output_name}\t{src}")

    if args.mode == "forward":
        _paste_forward_conditions(output_root)
        manifest.append("albedo.mp4\t<sky-pasted with rgb_sky and opacity>")
        manifest.append("roughness.mp4\t<sky-pasted with 0.5 background and opacity>")

    if args.envmap_path:
        envmap = Path(args.envmap_path).resolve()
        if not envmap.is_file():
            raise FileNotFoundError(f"envmap_path does not exist: {envmap}")
        ext = envmap.suffix or ".hdr"
        for cam_id in _cam_values(args.cam_ids):
            dst = output_root / f"envmap_{cam_id}{ext}"
            _replace_or_link(envmap, dst, copy=args.copy)
            manifest.append(f"{dst.name}\t{envmap}")

    manifest_path = output_root / "sdedit_inputs.tsv"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Prepared {args.mode} SDEdit inputs: {output_root}")


if __name__ == "__main__":
    main()

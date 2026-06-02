#!/usr/bin/env python3
"""Prepare per-camera videos for BRDFusion metric computation.

Rendering/SDEdit staging keeps compact videos in a single directory. The metric
script expects one directory per camera, e.g. videos/0/pbr_rgb.mp4. For one
camera this tool links the videos. For multi-camera tiled videos, it splits each
frame horizontally into camera-specific videos, with an explicit left-to-right
tile camera order when needed.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio


VIDEO_NAMES = [
    "pbr_rgb.mp4",
    "albedo.mp4",
    "normal.mp4",
    "normalized_depth.mp4",
    "roughness.mp4",
    "metallic.mp4",
]


def parse_cam_ids(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(item for item in str(value).split() if item)
    return out or ["0"]


def replace_or_link(src: Path, dst: Path, copy: bool) -> None:
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


def validate_tile_cam_order(cam_ids: list[str], tile_cam_order: list[str] | None) -> list[str]:
    if tile_cam_order is None:
        return cam_ids
    if len(tile_cam_order) != len(cam_ids):
        raise ValueError(
            f"tile_cam_order must contain {len(cam_ids)} ids for cam_ids={cam_ids}, got {tile_cam_order}"
        )
    if set(tile_cam_order) != set(cam_ids):
        raise ValueError(
            "tile_cam_order must contain the same ids as cam_ids: "
            f"cam_ids={cam_ids}, tile_cam_order={tile_cam_order}"
        )
    return tile_cam_order


def split_video(
    src: Path,
    output_root: Path,
    cam_ids: list[str],
    tile_cam_order: list[str],
    video_name: str,
    fps: float | None,
) -> None:
    reader = imageio.get_reader(src)
    writers = {}
    try:
        meta = reader.get_meta_data()
        source_fps = fps or meta.get("fps", 24)
        first = reader.get_data(0)
        width = first.shape[1]
        num_cams = len(cam_ids)
        if width % num_cams != 0:
            raise ValueError(f"Cannot split {src}: width {width} is not divisible by {num_cams} cameras")
        cam_w = width // num_cams
        for cam_id in cam_ids:
            dst = output_root / cam_id / video_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            writers[cam_id] = imageio.get_writer(dst, fps=source_fps)
        for frame_idx, frame in enumerate(reader):
            if frame_idx == 0:
                frame_to_split = first
            else:
                frame_to_split = frame
            for idx, cam_id in enumerate(tile_cam_order):
                writer = writers[cam_id]
                writer.append_data(frame_to_split[:, idx * cam_w : (idx + 1) * cam_w])
    finally:
        reader.close()
        for writer in writers.values():
            writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True, help="Directory with compact videos such as pbr_rgb.mp4")
    parser.add_argument("--output_root", required=True, help="Directory that receives <cam_id>/<video>.mp4")
    parser.add_argument("--cam_ids", nargs="+", default=["0"])
    parser.add_argument(
        "--tile_cam_order",
        nargs="+",
        default=None,
        help="Camera id assigned to each horizontal tile, left-to-right. Defaults to cam_ids order.",
    )
    parser.add_argument("--copy", action="store_true", help="Copy one-camera videos instead of symlinking")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS when splitting multi-camera videos")
    parser.add_argument("--required", nargs="+", default=["pbr_rgb.mp4"], help="Video names that must exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    cam_ids = parse_cam_ids(args.cam_ids)
    tile_cam_order = validate_tile_cam_order(
        cam_ids,
        parse_cam_ids(args.tile_cam_order) if args.tile_cam_order else None,
    )
    if not input_root.is_dir():
        raise NotADirectoryError(f"input_root does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    missing_required = [name for name in args.required if not (input_root / name).is_file()]
    if missing_required:
        raise FileNotFoundError(f"Missing required video(s) under {input_root}: {missing_required}")

    for video_name in VIDEO_NAMES:
        src = input_root / video_name
        if not src.is_file():
            continue
        if len(cam_ids) == 1:
            replace_or_link(src, output_root / cam_ids[0] / video_name, copy=args.copy)
        else:
            split_video(src, output_root, cam_ids, tile_cam_order, video_name, args.fps)
    print(f"Prepared metric videos: {output_root}")


if __name__ == "__main__":
    main()

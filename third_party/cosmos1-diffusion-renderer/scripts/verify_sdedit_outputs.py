#!/usr/bin/env python3
"""Verify SDEdit pipeline outputs against expected structure and frame counts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Sequence


INVERSE_OUTPUTS: Sequence[str] = ("albedo", "normal", "normalized_depth", "roughness", "metallic")
FORWARD_OUTPUTS: Sequence[str] = ("pbr_rgb", "albedo", "normal", "normalized_depth", "roughness", "metallic")
CAM_IDS: Sequence[int] = (0, 1, 2)


def count_video_frames(path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {path}")
    info = streams[0]
    for key in ("nb_read_packets", "nb_frames"):
        raw = info.get(key)
        if raw and str(raw).isdigit():
            frames = int(raw)
            if frames > 0:
                return frames
    raise RuntimeError(f"Unable to determine frame count for {path}")


def strength_tag(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def verify_inverse(args: argparse.Namespace) -> int:
    root = Path(args.output_root)
    failures: List[str] = []

    for s in args.strengths:
        tag = strength_tag(s)
        strength_root = root / f"refined_intrinsic_w{tag}"
        for name in INVERSE_OUTPUTS:
            path = strength_root / "videos" / f"{name}.mp4"
            if not path.exists():
                failures.append(f"missing file: {path}")
                continue
            frames = count_video_frames(path)
            if frames != args.expected_frames:
                failures.append(f"frame mismatch: {path} expected={args.expected_frames} got={frames}")

    if failures:
        print("[verify-sdedit] FAILED")
        for item in failures:
            print(f" - {item}")
        return 1

    print("[verify-sdedit] OK (inverse)")
    return 0


def verify_forward(args: argparse.Namespace) -> int:
    root = Path(args.output_root)
    failures: List[str] = []

    for s in args.strengths:
        tag = strength_tag(s)
        strength_root = root / f"refined_render_w{tag}"

        for output_name in FORWARD_OUTPUTS:
            for cam in CAM_IDS:
                path = strength_root / "videos" / str(cam) / f"{output_name}.mp4"
                if not path.exists():
                    failures.append(f"missing file: {path}")
                    continue
                frames = count_video_frames(path)
                if frames != args.expected_frames:
                    failures.append(f"frame mismatch: {path} expected={args.expected_frames} got={frames}")

            concat = strength_root / "visualization" / f"{output_name}_concat.mp4"
            if not concat.exists():
                failures.append(f"missing file: {concat}")
            else:
                frames = count_video_frames(concat)
                if frames != args.expected_frames:
                    failures.append(f"frame mismatch: {concat} expected={args.expected_frames} got={frames}")

    if failures:
        print("[verify-sdedit] FAILED")
        for item in failures:
            print(f" - {item}")
        return 1

    print("[verify-sdedit] OK (forward)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SDEdit output contracts")
    sub = parser.add_subparsers(dest="workflow", required=True)

    inv = sub.add_parser("inverse")
    inv.add_argument("--output_root", type=str, default="intrinsic_refinement")
    inv.add_argument("--strengths", type=float, nargs="+", required=True)
    inv.add_argument("--expected_frames", type=int, required=True)

    fwd = sub.add_parser("forward")
    fwd.add_argument("--output_root", type=str, default="pbr_refinement")
    fwd.add_argument("--strengths", type=float, nargs="+", required=True)
    fwd.add_argument("--expected_frames", type=int, required=True)

    args = parser.parse_args()
    if args.workflow == "inverse":
        return verify_inverse(args)
    if args.workflow == "forward":
        return verify_forward(args)
    raise ValueError(f"unknown workflow {args.workflow}")


if __name__ == "__main__":
    raise SystemExit(main())

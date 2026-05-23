"""Compatibility wrapper for 1-camera metrics.

Note:
- `tools/compute_video_metrics.py` already supports arbitrary camera counts from dataset config.
- This wrapper is optional; it only normalizes single-video inputs into `video_root/0/<video_name>`
  before delegating to `compute_video_metrics.py`.
"""

import argparse
import os
import subprocess
import sys
import tempfile


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _prepare_1cam_video_root(video_root: str, video_name: str) -> str:
    direct_cam_path = os.path.join(video_root, "0", video_name)
    if os.path.exists(direct_cam_path):
        return video_root

    direct_name_path = os.path.join(video_root, video_name)
    if os.path.exists(direct_name_path):
        tmp = tempfile.mkdtemp(prefix="metrics_1cam_")
        os.makedirs(os.path.join(tmp, "0"), exist_ok=True)
        _link_or_copy(direct_name_path, os.path.join(tmp, "0", video_name))
        return tmp

    cam_mp4_path = os.path.join(video_root, "0.mp4")
    if os.path.exists(cam_mp4_path):
        tmp = tempfile.mkdtemp(prefix="metrics_1cam_")
        os.makedirs(os.path.join(tmp, "0"), exist_ok=True)
        _link_or_copy(cam_mp4_path, os.path.join(tmp, "0", video_name))
        return tmp

    if os.path.isfile(video_root):
        tmp = tempfile.mkdtemp(prefix="metrics_1cam_")
        os.makedirs(os.path.join(tmp, "0"), exist_ok=True)
        _link_or_copy(video_root, os.path.join(tmp, "0", video_name))
        return tmp

    raise FileNotFoundError(
        f"Cannot resolve 1-cam video input from '{video_root}'. "
        f"Tried '{direct_cam_path}', '{direct_name_path}', '{cam_mp4_path}', or file path."
    )


def main() -> None:
    parser = argparse.ArgumentParser("Compute image metrics for Waymo 1-camera rendering.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--config_overlay", action="append", default=[])
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--video_name", type=str, default="pbr_rgb.mp4")
    parser.add_argument("--start_timestep", type=int, required=True)
    parser.add_argument("--end_timestep", type=int, required=True)
    parser.add_argument("--test_image_stride", type=int, required=True)
    parser.add_argument("--image_output_json", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="waymo/1cams")
    parser.add_argument("--dataset_source", type=str, default=None, choices=["primary", "external"], help="For self dataset, choose GT source branch")
    parser.add_argument("--cam_ids", type=str, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    prepared_video_root = (
        _prepare_1cam_video_root(args.video_root, args.video_name)
        if args.video_root is not None
        else None
    )
    compute_script = os.path.join(os.path.dirname(__file__), "compute_video_metrics.py")

    cmd = [
        sys.executable,
        "-u",
        compute_script,
        "--config_file",
        args.config_file,
        "--dataset",
        args.dataset,
    ]
    for overlay in args.config_overlay:
        cmd.extend(["--config_overlay", overlay])
    if args.dataset_source is not None:
        cmd.extend(["--dataset_source", args.dataset_source])
    cmd.extend([
        "--cam_ids",
        (args.cam_ids if args.cam_ids is not None else "0"),
    ])
    if prepared_video_root is not None:
        cmd.extend([
            "--video_root",
            prepared_video_root,
        ])
    cmd.extend([
        "--video_name",
        args.video_name,
        "--start_timestep",
        str(args.start_timestep),
        "--end_timestep",
        str(args.end_timestep),
        "--test_image_stride",
        str(args.test_image_stride),
        "--image_output_json",
        args.image_output_json,
    ])

    if args.opts:
        cmd.extend(args.opts)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

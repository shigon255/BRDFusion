#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


CONCAT_ORDER = (1, 0, 2)  # left -> 1, middle -> 0, right -> 2
TARGET_FRAMES = 57

INVERSE_PASS_TO_OUT = {
    "basecolor": "albedo",
    "normal": "normal",
    "depth": "normalized_depth",
    "roughness": "roughness",
    "metallic": "metallic",
}


def run_cmd(cmd, cwd: Path = None) -> None:
    pretty = " ".join(str(x) for x in cmd)
    if cwd is not None:
        pretty = f"(cd {cwd} && {pretty})"
    print(f"[prepare-drivestudio-eval] {pretty}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ffprobe_count_fps_size(video_path: Path) -> Tuple[int, float, int, int]:
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
    stream = payload["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    raw_rate = str(stream.get("avg_frame_rate", "0/1"))
    num, den = raw_rate.split("/", 1)
    fps = float(num) / float(den) if den != "0" else 24.0

    frame_count = None
    for key in ("nb_read_packets", "nb_frames"):
        raw = stream.get(key)
        if raw is not None and str(raw).isdigit():
            val = int(raw)
            if val > 0:
                frame_count = val
                break
    if frame_count is None:
        raise RuntimeError(f"Could not infer frame count for {video_path}")
    return frame_count, fps, width, height


def split_concat_to_cam_videos(concat_video: Path, split_dir: Path, fps: float) -> Dict[int, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    out = {
        1: split_dir / "cam1.mp4",
        0: split_dir / "cam0.mp4",
        2: split_dir / "cam2.mp4",
    }
    filters = {
        1: "crop=iw/3:ih:0:0",
        0: "crop=iw/3:ih:iw/3:0",
        2: "crop=iw/3:ih:2*iw/3:0",
    }
    for cam in CONCAT_ORDER:
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(concat_video),
                "-vf",
                filters[cam],
                "-r",
                f"{fps}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(out[cam]),
            ]
        )
    return out


def pad_frames_to_57(input_video: Path, output_video: Path, fps: float) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_video),
            "-vf",
            "tpad=stop_mode=clone:stop_duration=1000,trim=end_frame=57",
            "-frames:v",
            str(TARGET_FRAMES),
            "-r",
            f"{fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_video),
        ]
    )


def trim_video_to_nframes(input_video: Path, output_video: Path, frame_count: int, fps: float) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_video),
            "-frames:v",
            str(frame_count),
            "-r",
            f"{fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_video),
        ]
    )


def extract_video_frames(input_video: Path, output_dir: Path, pattern: str = "%03d.png") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_video),
            "-start_number",
            "0",
            str(output_dir / pattern),
        ]
    )


def encode_frames_to_video(frame_paths, output_video: Path, fps: float) -> None:
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found to encode for {output_video}")
    src_ext = frame_paths[0].suffix.lower() or ".jpg"
    seq_dir = output_video.parent / f".seq_{output_video.stem}"
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(frame_paths):
        dst = seq_dir / f"{idx:04d}{src_ext}"
        shutil.copy2(src, dst)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps}",
            "-start_number",
            "0",
            "-i",
            str(seq_dir / f"%04d{src_ext}"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_video),
        ]
    )
    shutil.rmtree(seq_dir)


def ensure_metric_layout_and_write(output_root: Path, task: str, cam_ids) -> Dict[int, Path]:
    base = output_root / task / "videos"
    out = {}
    for cam in cam_ids:
        cam_dir = base / str(cam)
        cam_dir.mkdir(parents=True, exist_ok=True)
        out[cam] = cam_dir
    return out


def resolve_cam_ids(args: argparse.Namespace) -> Tuple[int, ...]:
    if args.num_cams == 3:
        return (0, 1, 2)
    if args.num_cams == 1:
        if args.cam_id not in (0, 1, 2):
            raise ValueError(f"--cam_id must be one of 0,1,2. Got {args.cam_id}")
        return (args.cam_id,)
    raise ValueError(f"Unsupported --num_cams: {args.num_cams}")


def run_inverse_for_cam(
    repo_root: Path,
    input_frames_dir: Path,
    out_dir: Path,
    checkpoint_dir: str,
    height: int,
    width: int,
    fps: int,
    seed: int,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "cosmos_predict1.diffusion.inference.inference_inverse_renderer",
        "--checkpoint_dir",
        checkpoint_dir,
        "--diffusion_transformer_dir",
        "Diffusion_Renderer_Inverse_Cosmos_7B",
        "--dataset_path",
        str(input_frames_dir),
        "--num_video_frames",
        str(TARGET_FRAMES),
        "--group_mode",
        "folder",
        "--video_save_folder",
        str(out_dir),
        "--save_image",
        "True",
        "--save_video",
        "False",
        "--normalize_normal",
        "True",
        "--height",
        str(height),
        "--width",
        str(width),
        "--fps",
        str(fps),
        "--seed",
        str(seed),
        "--inference_passes",
        "basecolor",
        "normal",
        "depth",
        "roughness",
        "metallic",
    ]
    run_cmd(cmd, cwd=repo_root)


def run_forward_for_cam(
    repo_root: Path,
    gbuffer_root: Path,
    out_dir: Path,
    env_map: Path,
    checkpoint_dir: str,
    height: int,
    width: int,
    fps: int,
    seed: int,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "cosmos_predict1.diffusion.inference.inference_forward_renderer",
        "--checkpoint_dir",
        checkpoint_dir,
        "--diffusion_transformer_dir",
        "Diffusion_Renderer_Forward_Cosmos_7B",
        "--dataset_path",
        str(gbuffer_root),
        "--num_video_frames",
        str(TARGET_FRAMES),
        "--video_save_folder",
        str(out_dir),
        "--save_image",
        "False",
        "--use_custom_envmap",
        "True",
        "--env_map",
        str(env_map),
        "--envlight_ind",
        "0",
        "--height",
        str(height),
        "--width",
        str(width),
        "--fps",
        str(fps),
        "--seed",
        str(seed),
    ]
    run_cmd(cmd, cwd=repo_root)


def resolve_intrinsic_rgb_video(args: argparse.Namespace) -> Path:
    if args.ori_rgb_video is not None:
        return Path(args.ori_rgb_video)
    if args.concat_rgb_video is not None:
        return Path(args.concat_rgb_video)
    raise ValueError("Missing intrinsic RGB input. Provide --ori_rgb_video (preferred) or --concat_rgb_video.")


def prepare_intrinsic(args: argparse.Namespace, repo_root: Path) -> None:
    concat_rgb = resolve_intrinsic_rgb_video(args)
    if not concat_rgb.exists():
        raise FileNotFoundError(f"Missing input video: {concat_rgb}")

    nframes, fps_probe, width, _ = ffprobe_count_fps_size(concat_rgb)
    if nframes > TARGET_FRAMES:
        raise ValueError(f"Expected <= {TARGET_FRAMES} frames, got {nframes}")
    if args.num_cams == 3 and width % 3 != 0:
        raise ValueError(f"Concatenated video width must be divisible by 3, got {width}")
    fps = float(args.fps) if args.fps is not None else fps_probe

    cam_ids = resolve_cam_ids(args)
    output_root = Path(args.output_root)
    final_dirs = ensure_metric_layout_and_write(output_root, "intrinsic", cam_ids)
    tmp_root = Path(args.tmp_root) if args.tmp_root else (output_root / ".tmp_prepare_intrinsic")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    if args.num_cams == 3:
        split_videos = split_concat_to_cam_videos(concat_rgb, tmp_root / "split", fps)
    else:
        split_videos = {args.cam_id: concat_rgb}

    for cam in cam_ids:
        cam_video = split_videos[cam]
        padded_video = cam_video
        if nframes < TARGET_FRAMES:
            padded_video = tmp_root / "padded" / f"cam{cam}.mp4"
            pad_frames_to_57(cam_video, padded_video, fps)

        cam_frames_dir = tmp_root / "inverse_inputs" / f"cam{cam}"
        extract_video_frames(padded_video, cam_frames_dir, pattern="%03d.png")

        cam_out_dir = tmp_root / "inverse_out" / f"cam{cam}"
        run_inverse_for_cam(
            repo_root=repo_root,
            input_frames_dir=cam_frames_dir,
            out_dir=cam_out_dir,
            checkpoint_dir=args.checkpoint_dir,
            height=args.height,
            width=args.width,
            fps=int(round(fps)),
            seed=args.seed,
        )

        gbuf_dir = cam_out_dir / "gbuffer_frames"
        for pass_name, out_name in INVERSE_PASS_TO_OUT.items():
            pass_frames = sorted(gbuf_dir.glob(f"*.{pass_name}.jpg"))
            if len(pass_frames) < TARGET_FRAMES:
                raise RuntimeError(
                    f"Expected at least {TARGET_FRAMES} frames for cam {cam}, pass {pass_name}, got {len(pass_frames)}"
                )
            selected = pass_frames[:nframes]
            dst_video = final_dirs[cam] / f"{out_name}.mp4"
            encode_frames_to_video(selected, dst_video, fps)

    print(f"[prepare-drivestudio-eval] intrinsic outputs ready at: {output_root / 'intrinsic' / 'videos'}")


def _split_all_relight_inputs(
    args: argparse.Namespace, tmp_root: Path, fps: float, cam_ids: Tuple[int, ...]
) -> Dict[str, Dict[int, Path]]:
    inputs = {
        "albedo": Path(args.albedo_video),
        "normal": Path(args.normal_video),
        "normalized_depth": Path(args.normalized_depth_video),
        "roughness": Path(args.roughness_video),
        "metallic": Path(args.metallic_video),
    }
    for name, p in inputs.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} video: {p}")
    split = {}
    for name, p in inputs.items():
        if args.num_cams == 3:
            split[name] = split_concat_to_cam_videos(p, tmp_root / "split" / name, fps)
        else:
            split[name] = {cam_ids[0]: p}
    return split


def prepare_relight(args: argparse.Namespace, repo_root: Path) -> None:
    relight_inputs = {
        "albedo": Path(args.albedo_video),
        "normal": Path(args.normal_video),
        "normalized_depth": Path(args.normalized_depth_video),
        "roughness": Path(args.roughness_video),
        "metallic": Path(args.metallic_video),
    }
    input_meta = {}
    for name, p in relight_inputs.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} video: {p}")
        input_meta[name] = ffprobe_count_fps_size(p)

    nframes, fps_probe, width, _ = input_meta["albedo"]
    if nframes > TARGET_FRAMES:
        raise ValueError(f"Expected <= {TARGET_FRAMES} frames, got {nframes}")
    for name, (nf, _, _, _) in input_meta.items():
        if nf != nframes:
            raise ValueError(f"All relight input videos must have same frame count; albedo={nframes}, {name}={nf}")
    if args.num_cams == 3:
        for name, (_, _, w, _) in input_meta.items():
            if w % 3 != 0:
                raise ValueError(f"Concatenated {name} video width must be divisible by 3, got {w}")
    fps = float(args.fps) if args.fps is not None else fps_probe

    cam_ids = resolve_cam_ids(args)
    if args.num_cams == 3:
        envmaps = {
            0: Path(args.envmap_0),
            1: Path(args.envmap_1),
            2: Path(args.envmap_2),
        }
    else:
        envmaps = {cam_ids[0]: Path(args.envmap)}
    for cam, p in envmaps.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing envmap for cam {cam}: {p}")

    output_root = Path(args.output_root)
    final_dirs = ensure_metric_layout_and_write(output_root, "relight", cam_ids)
    tmp_root = Path(args.tmp_root) if args.tmp_root else (output_root / ".tmp_prepare_relight")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    split = _split_all_relight_inputs(args, tmp_root, fps, cam_ids)

    for cam in cam_ids:
        padded = {}
        for name in split.keys():
            src = split[name][cam]
            if nframes < TARGET_FRAMES:
                dst = tmp_root / "padded" / name / f"cam{cam}.mp4"
                pad_frames_to_57(src, dst, fps)
                padded[name] = dst
            else:
                padded[name] = src

        gbuffer_video1 = tmp_root / "gbuffer_root" / f"cam{cam}" / "video1"
        gbuffer_video1.mkdir(parents=True, exist_ok=True)
        pass_map = {
            "albedo": "basecolor",
            "normal": "normal",
            "normalized_depth": "depth",
            "roughness": "roughness",
            "metallic": "metallic",
        }
        for in_name, out_pass in pass_map.items():
            run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(padded[in_name]),
                    "-start_number",
                    "0",
                    str(gbuffer_video1 / f"0000.%04d.{out_pass}.png"),
                ]
            )

        cam_forward_out = tmp_root / "forward_out" / f"cam{cam}"
        run_forward_for_cam(
            repo_root=repo_root,
            gbuffer_root=gbuffer_video1.parent,
            out_dir=cam_forward_out,
            env_map=envmaps[cam],
            checkpoint_dir=args.checkpoint_dir,
            height=args.height,
            width=args.width,
            fps=int(round(fps)),
            seed=args.seed,
        )

        relit_candidates = sorted(cam_forward_out.glob("*.relit_*.mp4"))
        if len(relit_candidates) == 0:
            raise RuntimeError(f"No relit output found for cam {cam} under {cam_forward_out}")
        relit_video = relit_candidates[0]
        trim_video_to_nframes(relit_video, final_dirs[cam] / "relight_rgb.mp4", nframes, fps)

    print(f"[prepare-drivestudio-eval] relight outputs ready at: {output_root / 'relight' / 'videos'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Prepare diffusion-renderer outputs for drivestudio metric computation.")
    sub = parser.add_subparsers(dest="task", required=True)

    inv = sub.add_parser("intrinsic")
    inv.add_argument("--ori_rgb_video", type=str, default=None)
    inv.add_argument("--concat_rgb_video", type=str, default=None)  # backward-compatible alias
    inv.add_argument("--output_root", type=str, required=True)
    inv.add_argument("--num_cams", type=int, choices=[1, 3], default=3)
    inv.add_argument("--cam_id", type=int, default=0)
    inv.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    inv.add_argument("--seed", type=int, default=1000)
    inv.add_argument("--height", type=int, default=704)
    inv.add_argument("--width", type=int, default=1280)
    inv.add_argument("--fps", type=float, default=None)
    inv.add_argument("--tmp_root", type=str, default=None)

    rel = sub.add_parser("relight")
    rel.add_argument("--albedo_video", type=str, required=True)
    rel.add_argument("--normal_video", type=str, required=True)
    rel.add_argument("--normalized_depth_video", type=str, required=True)
    rel.add_argument("--roughness_video", type=str, required=True)
    rel.add_argument("--metallic_video", type=str, required=True)
    rel.add_argument("--num_cams", type=int, choices=[1, 3], default=3)
    rel.add_argument("--cam_id", type=int, default=0)
    rel.add_argument("--envmap_0", type=str, default=None)
    rel.add_argument("--envmap_1", type=str, default=None)
    rel.add_argument("--envmap_2", type=str, default=None)
    rel.add_argument("--envmap", type=str, default=None)
    rel.add_argument("--output_root", type=str, required=True)
    rel.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    rel.add_argument("--seed", type=int, default=1000)
    rel.add_argument("--height", type=int, default=704)
    rel.add_argument("--width", type=int, default=1280)
    rel.add_argument("--fps", type=float, default=None)
    rel.add_argument("--tmp_root", type=str, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.task == "intrinsic":
        if args.ori_rgb_video is None and args.concat_rgb_video is None:
            raise ValueError("For intrinsic, provide --ori_rgb_video (preferred) or --concat_rgb_video")
    if args.task == "relight":
        if args.num_cams == 3:
            missing = [k for k in ("envmap_0", "envmap_1", "envmap_2") if getattr(args, k) is None]
            if missing:
                raise ValueError(f"For --num_cams 3, required args missing: {', '.join('--' + m for m in missing)}")
        else:
            if args.envmap is None:
                raise ValueError("For --num_cams 1, --envmap is required")
    repo_root = Path(__file__).resolve().parents[1]
    if args.task == "intrinsic":
        prepare_intrinsic(args, repo_root)
    elif args.task == "relight":
        prepare_relight(args, repo_root)
    else:
        raise ValueError(args.task)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage an external BRDFusion checkpoint into a render-ready run folder."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

from omegaconf import OmegaConf

try:
    from utils.config import resolve_config
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.config import resolve_config  # type: ignore


DATASET_CONFIGS = {
    ("waymo", 1): "waymo/brdfusion_1cam",
    ("waymo", 3): "waymo/brdfusion_3cams",
    ("self", 1): "self/brdfusion_1cam",
    ("self", 3): "self/brdfusion_3cams",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _abs_path(repo_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _default_data_root(dataset: str, path_id: str | None) -> str:
    if dataset == "waymo":
        return "data/waymo/processed/training"
    if path_id is None:
        raise ValueError("Self staging requires --path_id unless --data_root is provided.")
    return f"data/self/path{path_id}_fixed_tree_gamma_full"


def _default_project(dataset: str, cams: int, scene_idx: str, path_id: str | None) -> str:
    if dataset == "waymo":
        return f"waymo_{cams}cam_{scene_idx}"
    if path_id is None:
        return f"self_{cams}cam_{scene_idx}"
    return f"self_{cams}cam_path{path_id}_{scene_idx}"


def _scene_path(data_root: Path, dataset: str, scene_idx: str) -> Path:
    if dataset == "waymo":
        return data_root / f"{int(scene_idx):03d}"
    return data_root / str(scene_idx)


def _extra_opts(args: argparse.Namespace) -> List[str]:
    opts = list(args.opts or [])
    if opts and opts[0] == "--":
        opts = opts[1:]
    return opts


def _config_opts(args: argparse.Namespace, data_root: Path, end_timestep: int) -> List[str]:
    return [
        f"data.data_root={data_root}",
        f"data.scene_idx={args.scene_idx}",
        f"data.start_timestep={args.start_timestep}",
        f"data.end_timestep={end_timestep}",
        f"data.pixel_source.test_image_stride={args.test_image_stride}",
    ]


def _write_config(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    stage_dir: Path,
    data_root: Path,
    end_timestep: int,
) -> Path:
    dataset_override = DATASET_CONFIGS[(args.dataset, args.cams)]
    overlays = list(args.config_overlay or [])
    if not overlays:
        overlays = ["configs/stage/stage3_finetune.yaml"]
    cfg = resolve_config(
        config_file=str(_abs_path(repo_root, args.config_file)),
        dataset_override=dataset_override,
        overlays=[str(_abs_path(repo_root, item)) for item in overlays],
        opts=_config_opts(args, data_root, end_timestep) + _extra_opts(args),
    )
    cfg.log_dir = str(stage_dir)

    config_path = stage_dir / "config.yaml"
    if config_path.exists() and not args.overwrite_config:
        raise FileExistsError(f"config already exists: {config_path}. Use --overwrite_config to replace it.")
    with config_path.open("w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f)

    return config_path


def _place_checkpoint(source: Path, target: Path, *, symlink: bool, overwrite: bool) -> None:
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"checkpoint already exists: {target}. Use --overwrite_checkpoint to replace it.")
        target.unlink()
    if symlink:
        os.symlink(source, target)
    else:
        shutil.copy2(source, target)


def stage_checkpoint(args: argparse.Namespace) -> dict:
    if args.num_frames <= 0:
        raise ValueError("--num_frames must be > 0.")
    if args.start_timestep < 0:
        raise ValueError("--start_timestep must be >= 0.")

    repo_root = _repo_root()
    os.chdir(repo_root)
    source_ckpt = _abs_path(repo_root, args.checkpoint)
    if not source_ckpt.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source_ckpt}")

    data_root_arg = args.data_root or _default_data_root(args.dataset, args.path_id)
    data_root = _abs_path(repo_root, data_root_arg)
    scene_path = _scene_path(data_root, args.dataset, args.scene_idx)
    if not scene_path.exists():
        raise FileNotFoundError(f"dataset scene path does not exist: {scene_path}")

    project = args.project or _default_project(args.dataset, args.cams, args.scene_idx, args.path_id)
    output_root = _abs_path(repo_root, args.output_root)
    stage_dir = output_root / project / "stage3_finetune"
    stage_dir.mkdir(parents=True, exist_ok=True)

    target_ckpt = stage_dir / "checkpoint_final.pth"
    target_config = stage_dir / "config.yaml"
    if target_config.exists() and not args.overwrite_config:
        raise FileExistsError(f"config already exists: {target_config}. Use --overwrite_config to replace it.")
    _place_checkpoint(source_ckpt, target_ckpt, symlink=args.symlink, overwrite=args.overwrite_checkpoint)

    end_timestep = args.start_timestep + args.num_frames - 1
    config_path = _write_config(
        args=args,
        repo_root=repo_root,
        stage_dir=stage_dir,
        data_root=data_root,
        end_timestep=end_timestep,
    )
    return {
        "stage_dir": stage_dir,
        "checkpoint": target_ckpt,
        "config": config_path,
        "scene_path": scene_path,
        "project": project,
        "end_timestep": end_timestep,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Stage an external checkpoint for BRDFusion rendering.")
    parser.add_argument("--checkpoint", required=True, help="Source checkpoint .pth file.")
    parser.add_argument("--dataset", required=True, choices=["waymo", "self"], help="Dataset family.")
    parser.add_argument("--cams", required=True, type=int, choices=[1, 3], help="Camera preset.")
    parser.add_argument("--scene_idx", required=True, help="Waymo scene id or self scene name.")
    parser.add_argument("--num_frames", required=True, type=int, help="Number of frames represented by the checkpoint config.")
    parser.add_argument("--path_id", default=None, help="Self path id, e.g. 1 for data/self/path1_fixed_tree_gamma_full.")
    parser.add_argument("--data_root", default=None, help="Override dataset root. Defaults by dataset/path_id.")
    parser.add_argument("--output_root", default="work_dirs", help="Output root for the staged run folder.")
    parser.add_argument("--project", default=None, help="Override project/run folder name.")
    parser.add_argument("--start_timestep", type=int, default=0, help="First timestep in the staged config.")
    parser.add_argument("--test_image_stride", type=int, default=10, help="Test image stride for render/eval splits.")
    parser.add_argument("--config_file", default="configs/omnire.yaml", help="Base config file.")
    parser.add_argument(
        "--config_overlay",
        action="append",
        default=[],
        help="Config overlay. Defaults to configs/stage/stage3_finetune.yaml when omitted.",
    )
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--copy", action="store_true", help="Copy checkpoint into the staged run folder. This is the default.")
    placement.add_argument("--symlink", action="store_true", help="Symlink checkpoint instead of copying it.")
    parser.add_argument("--overwrite_config", action="store_true", help="Replace an existing target config.yaml.")
    parser.add_argument("--overwrite_checkpoint", action="store_true", help="Replace an existing staged checkpoint.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional OmegaConf CLI overrides for config.yaml.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = stage_checkpoint(args)
    except Exception as exc:
        raise SystemExit(f"[ERR] {exc}") from None

    print(f"Staged checkpoint run: {result['stage_dir']}")
    print(f"  checkpoint: {result['checkpoint']}")
    print(f"  config:     {result['config']}")
    print(f"  scene:      {result['scene_path']}")
    print("Render with:")
    print(f"  CKPT={result['checkpoint']} scripts/render/render_checkpoint.sh")


if __name__ == "__main__":
    main()

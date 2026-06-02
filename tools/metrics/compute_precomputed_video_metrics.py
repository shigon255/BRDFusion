#!/usr/bin/env python3
"""Compute metrics for videos staged under precomputed_videos/.

This script is an orchestrator around tools/compute_video_metrics.py. It keeps
the metric implementation in one place, but handles the precomputed layout used
for BRDFusion and external baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_METHODS = ["brdfusion", "gen3c_dr", "invrgbl", "urbanir"]
DEFAULT_WAYMO_SCENES = ["003", "019", "114", "172", "703"]
DEFAULT_SELF_PATH_IDS = [1, 2, 3, 4, 5, 6]
DEFAULT_SELF_RECON_SCENE = "qwantani_moon_noon_puresky_4k_shifted"
DEFAULT_SELF_RELIGHT_SCENES = [
    "citrus_orchard_4k_shifted",
    "qwantani_moon_noon_puresky_4k_rot90_shifted",
    "the_sky_is_on_fire_4k_shifted",
]

VIDEO_NAMES = ["pbr_rgb.mp4", "albedo.mp4", "normal.mp4", "normalized_depth.mp4", "roughness.mp4", "metallic.mp4"]
INTRINSIC_SPECS = {
    "normal": ("normal_mae", "train_normal", "test_normal", "mae_deg"),
    "albedo": ("albedo_si_psnr", "train_albedo", "test_albedo", "si_psnr"),
    "roughness": ("roughness_rmse", "train_roughness", "test_roughness", "rmse"),
    "metallic": ("metallic_rmse", "train_metallic", "test_metallic", "rmse"),
}
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class MetricRecord:
    dataset: str
    method: str
    variant: str
    kind: str
    output_json: Path
    scene: Optional[str] = None
    path_id: Optional[int] = None
    relight_scene: Optional[str] = None
    status: str = "pending"
    reason: str = ""

    @property
    def label(self) -> str:
        return self.method if self.variant == "default" else f"{self.method}/{self.variant}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("precomputed_videos"))
    parser.add_argument("--datasets", nargs="+", choices=["waymo", "self"], default=["waymo", "self"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--waymo-scenes", nargs="+", default=DEFAULT_WAYMO_SCENES)
    parser.add_argument("--self-path-ids", type=int, nargs="+", default=DEFAULT_SELF_PATH_IDS)
    parser.add_argument("--self-recon-scene", default=DEFAULT_SELF_RECON_SCENE)
    parser.add_argument("--self-relight-scenes", nargs="+", default=DEFAULT_SELF_RELIGHT_SCENES)
    parser.add_argument("--config-file", default="configs/omnire.yaml")
    parser.add_argument("--waymo-dataset", default="waymo/brdfusion_1cam")
    parser.add_argument("--self-dataset", default="self/brdfusion_1cam")
    parser.add_argument("--waymo-data-root", default="data/waymo/processed/training")
    parser.add_argument("--self-root-template", default="data/self/path{path_id}_fixed_tree_gamma_full")
    parser.add_argument("--self-shifted-root-template", default="data/self/path{path_id}-3-calib_fixed_tree_gamma_full")
    parser.add_argument("--start-timestep", type=int, default=0)
    parser.add_argument("--end-timestep", type=int, default=50)
    parser.add_argument("--test-image-stride", type=int, default=10)
    parser.add_argument("--cam-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--summary-split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--force", action="store_true", help="Recompute metrics even when the output JSON already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned metric jobs without running them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a metric subprocess fails.")
    args = parser.parse_args()
    if not args.root.is_absolute():
        args.root = (REPO_ROOT / args.root).resolve()
    if not Path(args.config_file).is_absolute():
        args.config_file = str((REPO_ROOT / args.config_file).resolve())
    return args


def strip_shifted(scene_name: str) -> str:
    return scene_name[: -len("_shifted")] if scene_name.endswith("_shifted") else scene_name


def scene_as_int_text(scene: str) -> str:
    try:
        return str(int(scene))
    except ValueError:
        return scene


def replace_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    src = src.resolve()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_video_folder(src_root: Path, dst_root: Path, cam_ids: Iterable[int], required: Iterable[str]) -> None:
    missing = [name for name in required if not (src_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required video(s) under {src_root}: {missing}")
    for name in VIDEO_NAMES:
        src = src_root / name
        if not src.is_file():
            continue
        for cam_id in cam_ids:
            replace_or_link(src, dst_root / str(cam_id) / name)


def stage_single_video(src_video: Path, dst_root: Path, cam_ids: Iterable[int], video_name: str = "pbr_rgb.mp4") -> None:
    if not src_video.is_file():
        raise FileNotFoundError(f"Missing video: {src_video}")
    for cam_id in cam_ids:
        replace_or_link(src_video, dst_root / str(cam_id) / video_name)


def metric_loader_opts() -> list[str]:
    return [
        "data.pixel_source.load_priors=false",
        "data.pixel_source.load_materials=false",
        "data.pixel_source.load_objects=false",
        "data.pixel_source.load_smpl=false",
        "data.pixel_source.load_dynamic_mask=false",
        "data.lidar_source.load_lidar=false",
    ]


def self_external_opts(args: argparse.Namespace, path_id: int, source_scene: str) -> list[str]:
    primary_root = args.self_root_template.format(path_id=path_id)
    shifted_root = args.self_shifted_root_template.format(path_id=path_id)
    primary_scene = strip_shifted(args.self_recon_scene)
    external_scene = strip_shifted(source_scene)
    return [
        f"data.data_root={primary_root}",
        f"data.scene_idx={primary_scene}",
        "data.pixel_source.external_source.enable=true",
        f"data.pixel_source.external_source.data_root={shifted_root}",
        f"data.pixel_source.external_source.scene_idx={external_scene}",
        "data.pixel_source.external_source.active_source=external",
        "data.pixel_source.load_sky_mask=false",
        "data.pixel_source.load_gt_depth=false",
    ]


def base_metric_cmd(args: argparse.Namespace, dataset: str, cam_ids: Iterable[int]) -> list[str]:
    return [
        args.python_bin,
        "tools/compute_video_metrics.py",
        "--config_file",
        str((REPO_ROOT / args.config_file).resolve()) if not Path(args.config_file).is_absolute() else args.config_file,
        "--start_timestep",
        str(args.start_timestep),
        "--end_timestep",
        str(args.end_timestep),
        "--test_image_stride",
        str(args.test_image_stride),
        "--dataset",
        dataset,
        "--cam_ids",
        *[str(c) for c in cam_ids],
    ]


def run_metric(
    args: argparse.Namespace,
    record: MetricRecord,
    video_root: Path,
    dataset: str,
    opts: list[str],
    mode: str,
    disabled_intrinsics: Optional[list[str]] = None,
) -> None:
    if record.output_json.exists() and not args.force:
        record.status = "exists"
        return

    cmd = base_metric_cmd(args, dataset, args.cam_ids)
    if mode == "image":
        cmd += ["--video_root", str(video_root), "--video_name", "pbr_rgb.mp4", "--image_output_json", str(record.output_json)]
    elif mode == "relight":
        cmd += [
            "--relight_video_root",
            str(video_root),
            "--relight_video_name",
            "pbr_rgb.mp4",
            "--relight_output_json",
            str(record.output_json),
        ]
    elif mode == "intrinsic":
        cmd += ["--intrinsic_video_root", str(video_root), "--intrinsic_output_json", str(record.output_json)]
        for name in disabled_intrinsics or []:
            cmd.append(f"--disable_{name}")
    else:
        raise ValueError(f"Unsupported metric mode: {mode}")
    cmd += opts

    if args.dry_run:
        print("[dry-run]", " ".join(cmd))
        record.status = "planned"
        return

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}:{pythonpath}"
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    except subprocess.CalledProcessError as exc:
        record.status = "failed"
        record.reason = f"exit code {exc.returncode}"
        if not args.keep_going:
            raise
    else:
        record.status = "computed"


def compute_from_folder(
    args: argparse.Namespace,
    record: MetricRecord,
    src_root: Path,
    dataset: str,
    opts: list[str],
    mode: str,
    required: Iterable[str],
    disabled_intrinsics: Optional[list[str]] = None,
) -> None:
    if not src_root.is_dir():
        record.status = "missing"
        record.reason = f"missing directory: {src_root}"
        return
    with tempfile.TemporaryDirectory(prefix="brdfusion_precomputed_metrics_") as tmp:
        staged = Path(tmp) / "videos"
        try:
            stage_video_folder(src_root, staged, args.cam_ids, required)
        except Exception as exc:
            record.status = "missing"
            record.reason = str(exc)
            return
        run_metric(args, record, staged, dataset, opts, mode, disabled_intrinsics)


def compute_from_video(
    args: argparse.Namespace,
    record: MetricRecord,
    src_video: Path,
    dataset: str,
    opts: list[str],
    mode: str,
) -> None:
    if not src_video.is_file():
        record.status = "missing"
        record.reason = f"missing video: {src_video}"
        return
    with tempfile.TemporaryDirectory(prefix="brdfusion_precomputed_metrics_") as tmp:
        staged = Path(tmp) / "videos"
        try:
            stage_single_video(src_video, staged, args.cam_ids)
        except Exception as exc:
            record.status = "missing"
            record.reason = str(exc)
            return
        run_metric(args, record, staged, dataset, opts, mode)


def available_disabled_intrinsics(src_root: Path) -> tuple[bool, list[str]]:
    required_albedo = src_root / "albedo.mp4"
    if not required_albedo.is_file():
        return False, []
    disabled = []
    for name in ["normal", "roughness", "metallic"]:
        if not (src_root / f"{name}.mp4").is_file():
            disabled.append(name)
    return True, disabled


def waymo_external_video(root: Path, method: str, scene: str) -> Path:
    base = root / method / "waymo" / scene
    if method == "gen3c_dr":
        return base / f"waymo_{scene}_nvs_composite.mp4"
    if method == "invrgbl":
        return base / "full_set_30000_rendered_pbr.mp4"
    return base / "pbr_rgb.mp4"


def append_record(records: list[MetricRecord], **kwargs) -> MetricRecord:
    record = MetricRecord(**kwargs)
    records.append(record)
    return record


def compute_waymo(args: argparse.Namespace, records: list[MetricRecord]) -> None:
    opts_base = metric_loader_opts() + [f"data.data_root={args.waymo_data_root}"]
    for scene in args.waymo_scenes:
        opts = opts_base + [f"data.scene_idx={scene_as_int_text(scene)}"]
        if "brdfusion" in args.methods:
            for variant in ["raw_render", "refined_render_w0.5_sky"]:
                src_root = args.root / "brdfusion" / "waymo" / scene / variant
                record = append_record(
                    records,
                    dataset="waymo",
                    method="brdfusion",
                    variant="raw" if variant == "raw_render" else "refined",
                    kind="image",
                    scene=scene,
                    output_json=src_root / "image_metrics.json",
                )
                compute_from_folder(args, record, src_root, args.waymo_dataset, opts, "image", ["pbr_rgb.mp4"])
        for method in [m for m in args.methods if m != "brdfusion"]:
            src_video = waymo_external_video(args.root, method, scene)
            record = append_record(
                records,
                dataset="waymo",
                method=method,
                variant="default",
                kind="image",
                scene=scene,
                output_json=src_video.parent / "image_metrics.json",
            )
            compute_from_video(args, record, src_video, args.waymo_dataset, opts, "image")


def compute_self(args: argparse.Namespace, records: list[MetricRecord]) -> None:
    for path_id in args.self_path_ids:
        if "brdfusion" in args.methods:
            recon_opts = metric_loader_opts() + self_external_opts(args, path_id, args.self_recon_scene)
            for variant in ["raw_render", "refined_render_w0.5_sky"]:
                src_root = args.root / "brdfusion" / "self" / f"path{path_id}_fixed_tree_gamma_full" / args.self_recon_scene / variant
                record = append_record(
                    records,
                    dataset="self",
                    method="brdfusion",
                    variant="raw" if variant == "raw_render" else "refined",
                    kind="image",
                    path_id=path_id,
                    output_json=src_root / "image_metrics.json",
                )
                compute_from_folder(args, record, src_root, args.self_dataset, recon_opts, "image", ["pbr_rgb.mp4"])
                if variant == "refined_render_w0.5_sky":
                    intr_ok, disabled = available_disabled_intrinsics(src_root)
                    intr_record = append_record(
                        records,
                        dataset="self",
                        method="brdfusion",
                        variant="refined",
                        kind="intrinsic",
                        path_id=path_id,
                        output_json=src_root / "intrinsic_metrics.json",
                    )
                    if intr_ok:
                        intr_opts = recon_opts + ["data.pixel_source.load_gt_sky_mask=true", "data.pixel_source.load_gt_intrinsic=true"]
                        compute_from_folder(
                            args,
                            intr_record,
                            src_root,
                            args.self_dataset,
                            intr_opts,
                            "intrinsic",
                            ["albedo.mp4"],
                            disabled,
                        )
                    else:
                        intr_record.status = "missing"
                        intr_record.reason = f"missing albedo.mp4 under {src_root}"

            for relight_scene in args.self_relight_scenes:
                relight_opts = (
                    metric_loader_opts()
                    + self_external_opts(args, path_id, relight_scene)
                    + [
                        "data.pixel_source.load_relighted_rgb=true",
                        f"data.pixel_source.relighted_scene_idx={strip_shifted(relight_scene)}",
                        f"data.pixel_source.external_source.relighted_scene_idx={strip_shifted(relight_scene)}",
                    ]
                )
                for variant in ["raw_render", "refined_render_w0.5_sky"]:
                    src_root = args.root / "brdfusion" / "self" / f"path{path_id}_fixed_tree_gamma_full" / relight_scene / variant
                    record = append_record(
                        records,
                        dataset="self",
                        method="brdfusion",
                        variant="raw" if variant == "raw_render" else "refined",
                        kind="relight",
                        path_id=path_id,
                        relight_scene=relight_scene,
                        output_json=src_root / "relight_metrics.json",
                    )
                    compute_from_folder(args, record, src_root, args.self_dataset, relight_opts, "relight", ["pbr_rgb.mp4"])

        for method in [m for m in args.methods if m != "brdfusion"]:
            recon_root = args.root / method / "self" / f"path{path_id}_fixed_tree_gamma_full" / args.self_recon_scene
            recon_opts = metric_loader_opts() + self_external_opts(args, path_id, args.self_recon_scene)
            record = append_record(
                records,
                dataset="self",
                method=method,
                variant="default",
                kind="image",
                path_id=path_id,
                output_json=recon_root / "image_metrics.json",
            )
            compute_from_folder(args, record, recon_root, args.self_dataset, recon_opts, "image", ["pbr_rgb.mp4"])

            intr_ok, disabled = available_disabled_intrinsics(recon_root)
            intr_record = append_record(
                records,
                dataset="self",
                method=method,
                variant="default",
                kind="intrinsic",
                path_id=path_id,
                output_json=recon_root / "intrinsic_metrics.json",
            )
            if intr_ok:
                intr_opts = recon_opts + ["data.pixel_source.load_gt_sky_mask=true", "data.pixel_source.load_gt_intrinsic=true"]
                compute_from_folder(
                    args,
                    intr_record,
                    recon_root,
                    args.self_dataset,
                    intr_opts,
                    "intrinsic",
                    ["albedo.mp4"],
                    disabled,
                )
            else:
                intr_record.status = "missing"
                intr_record.reason = f"missing albedo.mp4 under {recon_root}"

            for relight_scene in args.self_relight_scenes:
                relight_root = args.root / method / "self" / f"path{path_id}_fixed_tree_gamma_full" / relight_scene
                relight_opts = (
                    metric_loader_opts()
                    + self_external_opts(args, path_id, relight_scene)
                    + [
                        "data.pixel_source.load_relighted_rgb=true",
                        f"data.pixel_source.relighted_scene_idx={strip_shifted(relight_scene)}",
                        f"data.pixel_source.external_source.relighted_scene_idx={strip_shifted(relight_scene)}",
                    ]
                )
                record = append_record(
                    records,
                    dataset="self",
                    method=method,
                    variant="default",
                    kind="relight",
                    path_id=path_id,
                    relight_scene=relight_scene,
                    output_json=relight_root / "relight_metrics.json",
                )
                compute_from_folder(args, record, relight_root, args.self_dataset, relight_opts, "relight", ["pbr_rgb.mp4"])


def load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_metric(data: Optional[dict], train_key: str, test_key: str, metric_names: Iterable[str], split: str) -> dict[str, float]:
    out = {name: math.nan for name in metric_names}
    if data is None:
        return out
    metrics = data.get("metrics", {})
    train = metrics.get(train_key)
    test = metrics.get(test_key)
    if split == "train":
        src = train
    elif split == "test":
        src = test
    else:
        src = None
    if src is not None:
        for name in metric_names:
            out[name] = float(src.get(name, math.nan))
        return out
    if train is None or test is None:
        return out
    total = float(train.get("count", 0) + test.get("count", 0))
    if total <= 0:
        return out
    for name in metric_names:
        tv = train.get(name, math.nan)
        vv = test.get(name, math.nan)
        if not math.isnan(float(tv)) and not math.isnan(float(vv)):
            out[name] = (float(tv) * float(train.get("count", 0)) + float(vv) * float(test.get("count", 0))) / total
    return out


def image_values(record: MetricRecord, split: str) -> dict[str, float]:
    return split_metric(load_json(record.output_json), "train", "test", ["psnr", "ssim", "lpips"], split)


def relight_values(record: MetricRecord, split: str) -> dict[str, float]:
    return split_metric(load_json(record.output_json), "train_relight", "test_relight", ["psnr", "ssim", "lpips"], split)


def intrinsic_values(record: MetricRecord, split: str) -> dict[str, float]:
    data = load_json(record.output_json)
    out = {spec[0]: math.nan for spec in INTRINSIC_SPECS.values()}
    if data is None:
        return out
    metrics = data.get("metrics", {})
    for _, (out_key, train_key, test_key, value_key) in INTRINSIC_SPECS.items():
        values = split_metric({"metrics": metrics}, train_key, test_key, [value_key], split)
        out[out_key] = values[value_key]
    return out


def mean_metric(items: list[dict[str, float]], key: str) -> float:
    vals = [item.get(key, math.nan) for item in items]
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    return math.nan if not vals else sum(vals) / len(vals)


def fmt(value: float) -> str:
    return "NA" if math.isnan(float(value)) else f"{float(value):.4f}"


def print_triplet_table(title: str, rows: list[tuple[str, list[dict[str, float]], int]]) -> None:
    print(f"\n{title}")
    print("row\tpsnr\tssim\tlpips\tmissing")
    for label, values, missing in rows:
        print(
            f"{label}\t{fmt(mean_metric(values, 'psnr'))}\t"
            f"{fmt(mean_metric(values, 'ssim'))}\t{fmt(mean_metric(values, 'lpips'))}\t{missing}"
        )


def print_intrinsic_table(title: str, rows: list[tuple[str, list[dict[str, float]], int]]) -> None:
    print(f"\n{title}")
    print("row\tnormal_mae\talbedo_si_psnr\troughness_rmse\tmetallic_rmse\tmissing")
    for label, values, missing in rows:
        print(
            f"{label}\t{fmt(mean_metric(values, 'normal_mae'))}\t"
            f"{fmt(mean_metric(values, 'albedo_si_psnr'))}\t"
            f"{fmt(mean_metric(values, 'roughness_rmse'))}\t{fmt(mean_metric(values, 'metallic_rmse'))}\t{missing}"
        )


def summarize(records: list[MetricRecord], split: str) -> None:
    completed = [r for r in records if r.output_json.is_file()]
    planned = [r for r in records if r.status == "planned"]

    def unavailable_count(rows: list[MetricRecord]) -> int:
        return len([r for r in rows if not r.output_json.is_file() and r.status != "planned"])

    waymo_rows = []
    for label in sorted({r.label for r in records if r.dataset == "waymo" and r.kind == "image"}):
        rows = [r for r in records if r.dataset == "waymo" and r.kind == "image" and r.label == label]
        waymo_rows.append((label, [image_values(r, split) for r in rows if r.output_json.is_file()], unavailable_count(rows)))
    if waymo_rows:
        print_triplet_table(f"Waymo NVS ({split})", waymo_rows)

    nvs_rows = []
    for label in sorted({r.label for r in records if r.dataset == "self" and r.kind == "image"}):
        rows = [r for r in records if r.dataset == "self" and r.kind == "image" and r.label == label]
        nvs_rows.append((label, [image_values(r, split) for r in rows if r.output_json.is_file()], unavailable_count(rows)))
    if nvs_rows:
        print_triplet_table(f"Self Shifted NVS ({split})", nvs_rows)

    intr_rows = []
    for label in sorted({r.label for r in records if r.dataset == "self" and r.kind == "intrinsic"}):
        rows = [r for r in records if r.dataset == "self" and r.kind == "intrinsic" and r.label == label]
        intr_rows.append((label, [intrinsic_values(r, split) for r in rows if r.output_json.is_file()], unavailable_count(rows)))
    if intr_rows:
        print_intrinsic_table(f"Self Shifted Intrinsic ({split})", intr_rows)

    relight_scenes = sorted({r.relight_scene for r in records if r.dataset == "self" and r.kind == "relight" and r.relight_scene})
    for relight_scene in relight_scenes:
        relight_rows = []
        labels = sorted({r.label for r in records if r.dataset == "self" and r.kind == "relight" and r.relight_scene == relight_scene})
        for label in labels:
            rows = [
                r
                for r in records
                if r.dataset == "self" and r.kind == "relight" and r.relight_scene == relight_scene and r.label == label
            ]
            relight_rows.append((label, [relight_values(r, split) for r in rows if r.output_json.is_file()], unavailable_count(rows)))
        print_triplet_table(f"Self Shifted Relight {relight_scene} ({split}, full image)", relight_rows)

    relight_avg_rows = []
    for label in sorted({r.label for r in records if r.dataset == "self" and r.kind == "relight"}):
        rows = [r for r in records if r.dataset == "self" and r.kind == "relight" and r.label == label]
        relight_avg_rows.append((label, [relight_values(r, split) for r in rows if r.output_json.is_file()], unavailable_count(rows)))
    if relight_avg_rows:
        print_triplet_table(f"Self Shifted Relight Average ({split}, full image)", relight_avg_rows)

    missing = [r for r in records if not r.output_json.is_file() and r.status != "planned"]
    failed = [r for r in records if r.status == "failed"]
    print(f"\nMetric JSONs available: {len(completed)}/{len(records)}")
    if planned:
        print(f"Dry-run planned jobs: {len(planned)}")
    if failed:
        print(f"Failed jobs: {len(failed)}")
    if missing:
        print(f"Missing/skipped jobs: {len(missing)}")


def main() -> None:
    args = parse_args()
    records: list[MetricRecord] = []
    if "waymo" in args.datasets:
        compute_waymo(args, records)
    if "self" in args.datasets:
        compute_self(args, records)
    summarize(records, args.summary_split)


if __name__ == "__main__":
    main()

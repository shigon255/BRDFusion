import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_PATH_IDS = [1, 2, 3, 4, 5, 6]
DEFAULT_WAYMO_IDS = [3, 19, 114, 172, 245, 703]
DEFAULT_SCENE_IDX = "qwantani_moon_noon_puresky_4k"
DEFAULT_RELIGHT_SCENE_IDX = f"{DEFAULT_SCENE_IDX}_rot90"
DEFAULT_RELIGHT_FOLDERS = ["dr_eval_raw", "dr_eval_relight"]
DEFAULT_SEEDS = [1000, 1001, 1002, 1003, 1004]


def warn_missing(path: Path) -> None:
    print(f"[missing] {path}", file=sys.stderr)


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        warn_missing(path)
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_existing_path(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def relight_folder_candidates(folder: str, relight_tag: str, seeds: Optional[Sequence[int]] = None) -> List[str]:
    if seeds is None:
        return [
            f"{folder}_{relight_tag}",
            folder,
        ]
    out = []
    for seed in seeds:
        out.extend([
            f"{folder}_{relight_tag}_seed_{seed}",
            f"{folder}_seed_{seed}",
        ])
    return out


def resolve_relight_metric(exp_root: Path, folders: Sequence[str], relight_tag: str) -> Path:
    folder_candidates = []
    for folder in folders:
        folder_candidates.extend(relight_folder_candidates(folder, relight_tag))
    candidates = [exp_root / folder / "relight" / "image_metrics.json" for folder in folder_candidates]
    return resolve_existing_path(candidates)


def resolve_seed_relight_metrics(exp_root: Path, folder: str, relight_tag: str, seeds: Sequence[int]) -> List[Path]:
    paths = []
    for seed in seeds:
        candidates = [
            exp_root / f"{folder}_{relight_tag}_seed_{seed}" / "relight" / "image_metrics.json",
            exp_root / f"{folder}_seed_{seed}" / "relight" / "image_metrics.json",
        ]
        paths.append(resolve_existing_path(candidates))
    return paths


def discover_relight_scene_idxs(exp_root: Path, folders: Sequence[str]) -> List[str]:
    if not exp_root.exists():
        return []
    scene_idxs = set()
    for folder in folders:
        prefix = f"{folder}_relight_"
        for child in exp_root.iterdir():
            if not child.is_dir() or not child.name.startswith(prefix):
                continue
            scene_idx = child.name[len(prefix):]
            scene_idx = re.sub(r"_seed_\d+$", "", scene_idx)
            if scene_idx:
                scene_idxs.add(scene_idx)
    return sorted(scene_idxs)


def discover_relight_scene_idxs_for_exps(exp_roots: Sequence[Path], folders: Sequence[str]) -> List[str]:
    scene_idxs = set()
    for exp_root in exp_roots:
        scene_idxs.update(discover_relight_scene_idxs(exp_root, folders))
    return sorted(scene_idxs)


def resolve_relight_scene_idxs(exp_root: Path, folders: Sequence[str], requested: Sequence[str]) -> List[str]:
    if len(requested) == 1 and requested[0] == "all":
        discovered = discover_relight_scene_idxs(exp_root, folders)
        return discovered if discovered else [DEFAULT_RELIGHT_SCENE_IDX]
    return list(requested)


def resolve_relight_scene_idxs_for_exps(
    exp_roots: Sequence[Path],
    folders: Sequence[str],
    requested: Sequence[str],
) -> List[str]:
    if len(requested) == 1 and requested[0] == "all":
        discovered = discover_relight_scene_idxs_for_exps(exp_roots, folders)
        return discovered if discovered else [DEFAULT_RELIGHT_SCENE_IDX]
    return list(requested)


def relight_metric_paths_for_scene(
    exp_root: Path,
    folders: Sequence[str],
    scene_idx: str,
    seed: Optional[int] = None,
    metric_filename: str = "image_metrics.json",
) -> List[Path]:
    relight_tag = f"relight_{scene_idx}"
    paths = []
    for folder in folders:
        if seed is None:
            path = exp_root / f"{folder}_{relight_tag}" / "relight" / metric_filename
        else:
            path = exp_root / f"{folder}_{relight_tag}_seed_{seed}" / "relight" / metric_filename
        if path.exists():
            paths.append(path)
    if paths:
        return paths

    # Backward-compatible fallback only when a specific scene is requested.
    if scene_idx == DEFAULT_RELIGHT_SCENE_IDX:
        for folder in folders:
            if seed is None:
                path = exp_root / folder / "relight" / metric_filename
            else:
                path = exp_root / f"{folder}_seed_{seed}" / "relight" / metric_filename
            if path.exists():
                paths.append(path)
    return paths


def weighted_named(train: Dict[str, float], test: Dict[str, float], key: str) -> float:
    total = train["count"] + test["count"]
    if total == 0:
        return math.nan
    return (train[key] * train["count"] + test[key] * test["count"]) / total


def image_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {"psnr": math.nan, "ssim": math.nan, "lpips": math.nan}
    if "train" in data["metrics"] and "test" in data["metrics"]:
        return {
            "psnr": weighted_named(data["metrics"]["train"], data["metrics"]["test"], "psnr"),
            "ssim": weighted_named(data["metrics"]["train"], data["metrics"]["test"], "ssim"),
            "lpips": weighted_named(data["metrics"]["train"], data["metrics"]["test"], "lpips"),
        }
    test = data["metrics"]["test"]
    return {"psnr": test["psnr"], "ssim": test["ssim"], "lpips": test["lpips"]}


def intrinsic_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {
            "normal_mae": math.nan,
            "albedo_si_psnr": math.nan,
            "roughness_rmse": math.nan,
            "metallic_rmse": math.nan,
        }
    metrics = data["metrics"]
    out = {
        "albedo_si_psnr": weighted_named(metrics["train_albedo"], metrics["test_albedo"], "si_psnr"),
        "normal_mae": math.nan,
        "roughness_rmse": math.nan,
        "metallic_rmse": math.nan,
    }
    if "train_normal" in metrics and "test_normal" in metrics:
        out["normal_mae"] = weighted_named(metrics["train_normal"], metrics["test_normal"], "mae_deg")
    if "train_roughness" in metrics and "test_roughness" in metrics:
        out["roughness_rmse"] = weighted_named(metrics["train_roughness"], metrics["test_roughness"], "rmse")
    if "train_metallic" in metrics and "test_metallic" in metrics:
        out["metallic_rmse"] = weighted_named(metrics["train_metallic"], metrics["test_metallic"], "rmse")
    return out


def relight_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {"psnr": math.nan, "ssim": math.nan, "lpips": math.nan}
    return {
        "psnr": weighted_named(data["metrics"]["train_relight"], data["metrics"]["test_relight"], "psnr"),
        "ssim": weighted_named(data["metrics"]["train_relight"], data["metrics"]["test_relight"], "ssim"),
        "lpips": weighted_named(data["metrics"]["train_relight"], data["metrics"]["test_relight"], "lpips"),
    }


def average_metric_dicts(values: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for value in values for key in value.keys()})
    return {key: nanmean([value.get(key, math.nan) for value in values]) for key in keys}


def relight_metrics_for_scenes(
    exp_root: Path,
    folders: Sequence[str],
    scene_idxs: Sequence[str],
    metric_filename: str = "image_metrics.json",
) -> Dict[str, float]:
    values = []
    for scene_idx in scene_idxs:
        paths = relight_metric_paths_for_scene(exp_root, folders, scene_idx, metric_filename=metric_filename)
        if paths:
            values.append(average_metric_dicts([relight_metrics(load_json(path)) for path in paths]))
        else:
            values.append(relight_metrics(None))
    return average_metric_dicts(values)


def relight_metrics_for_scene(
    exp_root: Path,
    folders: Sequence[str],
    scene_idx: str,
    metric_filename: str = "image_metrics.json",
) -> Dict[str, float]:
    paths = relight_metric_paths_for_scene(exp_root, folders, scene_idx, metric_filename=metric_filename)
    if not paths:
        warn_missing(exp_root / f"<{'|'.join(folders)}>_relight_{scene_idx}" / "relight" / metric_filename)
        return relight_metrics(None)
    return average_metric_dicts([relight_metrics(load_json(path)) for path in paths])


def seed_relight_metrics_for_scenes(
    exp_root: Path,
    folder: str,
    scene_idxs: Sequence[str],
    seeds: Sequence[int],
    metric_filename: str = "image_metrics.json",
) -> List[Dict[str, float]]:
    values_by_seed = []
    for seed in seeds:
        scene_values = []
        for scene_idx in scene_idxs:
            paths = relight_metric_paths_for_scene(exp_root, [folder], scene_idx, seed=seed, metric_filename=metric_filename)
            if paths:
                scene_values.append(average_metric_dicts([relight_metrics(load_json(path)) for path in paths]))
            else:
                warn_missing(exp_root / f"{folder}_relight_{scene_idx}_seed_{seed}" / "relight" / metric_filename)
                scene_values.append(relight_metrics(None))
        values_by_seed.append(average_metric_dicts(scene_values))
    return values_by_seed


def seed_relight_metrics_for_scene(
    exp_root: Path,
    folders: Sequence[str],
    scene_idx: str,
    seeds: Sequence[int],
    metric_filename: str = "image_metrics.json",
) -> List[Dict[str, float]]:
    values_by_seed = []
    for seed in seeds:
        paths = relight_metric_paths_for_scene(exp_root, folders, scene_idx, seed=seed, metric_filename=metric_filename)
        if paths:
            values_by_seed.append(average_metric_dicts([relight_metrics(load_json(path)) for path in paths]))
        else:
            warn_missing(exp_root / f"<{'|'.join(folders)}>_relight_{scene_idx}_seed_{seed}" / "relight" / metric_filename)
            values_by_seed.append(relight_metrics(None))
    return values_by_seed


def mean(values: List[float]) -> float:
    if any(math.isnan(v) for v in values):
        return math.nan
    return math.nan if not values else sum(values) / len(values)


def nanmean(values: List[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return math.nan if not valid else sum(valid) / len(valid)


def nanstd(values: List[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return math.nan
    avg = sum(valid) / len(valid)
    return math.sqrt(sum((v - avg) ** 2 for v in valid) / len(valid))


def fmt(value: float) -> str:
    return "NA" if math.isnan(value) else f"{value:.6f}"


def prune_all_nan(metrics: Dict[str, List[float]]) -> Dict[str, List[float]]:
    return {k: v for k, v in metrics.items() if any(not math.isnan(x) for x in v)}


def print_table(title: str, rows: List[str], metrics: Dict[str, List[float]]) -> None:
    metrics = prune_all_nan(metrics)
    if not metrics:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}\n(no available metrics)")
        return
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    headers = list(metrics.keys())
    print("id\t" + "\t".join(headers))
    for i, row in enumerate(rows):
        vals = [fmt(metrics[h][i]) for h in headers]
        print(f"{row}\t" + "\t".join(vals))
    avg_vals = [fmt(mean(metrics[h])) for h in headers]
    print("avg\t" + "\t".join(avg_vals))


def print_seed_table(title: str, rows: List[str], metrics: Dict[str, Dict[str, List[float]]]) -> None:
    flattened = {}
    for metric_name, parts in metrics.items():
        means = parts.get("mean", [])
        stds = parts.get("std", [])
        if any(not math.isnan(x) for x in means):
            flattened[f"{metric_name}_mean"] = means
            flattened[f"{metric_name}_std"] = stds
    print_table(title, rows, flattened)


def append_metric_row(container: Dict[str, List[float]], values: Dict[str, float]) -> None:
    for key, value in values.items():
        container.setdefault(key, []).append(value)


def append_seed_metric_row(container: Dict[str, Dict[str, List[float]]], values_by_seed: List[Dict[str, float]]) -> None:
    keys = sorted({key for values in values_by_seed for key in values.keys()})
    for key in keys:
        vals = [values.get(key, math.nan) for values in values_by_seed]
        container.setdefault(key, {"mean": [], "std": []})
        container[key]["mean"].append(nanmean(vals))
        container[key]["std"].append(nanstd(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate current DR-only evaluation metrics.")
    parser.add_argument("--self-root", type=Path, default=Path("dr_eval_output"))
    parser.add_argument("--shifted-root", type=Path, default=Path("dr_eval_output_shifted_3"))
    parser.add_argument("--unirelight-root", type=Path, default=Path("unirelight_eval_output"))
    parser.add_argument("--unirelight-shifted-root", type=Path, default=Path("unirelight_eval_output_shifted_3"))
    parser.add_argument("--scene-idx", type=str, default=DEFAULT_SCENE_IDX)
    parser.add_argument(
        "--relight-scene-idx",
        type=str,
        nargs="+",
        default=["all"],
        help='Relight scene idxs to average. Use "all" to discover relight-scene-dependent folders.',
    )
    parser.add_argument("--path-ids", type=int, nargs="+", default=DEFAULT_PATH_IDS)
    parser.add_argument("--waymo-ids", type=int, nargs="+", default=DEFAULT_WAYMO_IDS)
    parser.add_argument("--self-relight-folders", type=str, nargs="+", default=DEFAULT_RELIGHT_FOLDERS)
    parser.add_argument("--shifted-relight-folders", type=str, nargs="+", default=DEFAULT_RELIGHT_FOLDERS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--multiseed", action="store_true", help="Aggregate DR relight metrics across seed-specific folders.")
    parser.add_argument("--include-unirelight", action="store_true", help="Aggregate single-seed UniRelight metrics.")
    parser.add_argument("--unirelight-folder", type=str, default="unirelight_eval")
    args = parser.parse_args()

    # waymo_rows = [f"waymo_{wid}" for wid in args.waymo_ids]
    # waymo_out = {"psnr": [], "ssim": [], "lpips": []}
    # for wid in args.waymo_ids:
    #     path = args.self_root / f"dr_only_1cam_{wid}" / "drivestudio" / "dr_eval_raw" / "image_metrics.json"
    #     append_metric_row(waymo_out, image_metrics(load_json(path)))
    # print_table("DR Waymo Image Metrics", waymo_rows, waymo_out)

    path_rows = [f"path{pid}" for pid in args.path_ids]
    
    # self_image = {"psnr": [], "ssim": [], "lpips": []}
    # self_intrinsic = {"normal_mae": [], "albedo_si_psnr": [], "roughness_rmse": [], "metallic_rmse": []}
    # self_relight = {"psnr": [], "ssim": [], "lpips": []}
    # for pid in args.path_ids:
    #     exp = args.self_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
    #     scene_idxs = resolve_relight_scene_idxs(exp, args.self_relight_folders, args.relight_scene_idx)
    #     append_metric_row(self_image, image_metrics(load_json(exp / "dr_eval_raw" / "image_metrics.json")))
    #     append_metric_row(self_intrinsic, intrinsic_metrics(load_json(exp / "dr_eval_raw" / "intrinsic" / "intrinsic_metrics.json")))
    #     append_metric_row(self_relight, relight_metrics_for_scenes(exp, args.self_relight_folders, scene_idxs))
    # print_table("DR Self Image Metrics", path_rows, self_image)
    # print_table("DR Self Intrinsic Metrics", path_rows, self_intrinsic)
    # print_table("DR Self Relight Metrics", path_rows, self_relight)

    if args.multiseed:
        self_exps = [
            args.self_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
            for pid in args.path_ids
        ]
        self_scene_idxs = resolve_relight_scene_idxs_for_exps(self_exps, args.self_relight_folders, args.relight_scene_idx)
        self_relight_seed = {"psnr": {"mean": [], "std": []}, "ssim": {"mean": [], "std": []}, "lpips": {"mean": [], "std": []}}
        for pid in args.path_ids:
            exp = args.self_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
            append_seed_metric_row(
                self_relight_seed,
                seed_relight_metrics_for_scenes(exp, args.self_relight_folders[0], self_scene_idxs, args.seeds),
            )

        for scene_idx in self_scene_idxs:
            scene_seed = {"psnr": {"mean": [], "std": []}, "ssim": {"mean": [], "std": []}, "lpips": {"mean": [], "std": []}}
            for pid in args.path_ids:
                exp = args.self_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
                append_seed_metric_row(
                    scene_seed,
                    seed_relight_metrics_for_scene(exp, args.self_relight_folders, scene_idx, args.seeds),
                )
            print_seed_table(f"DR Self Multiseed Relight Metrics: {scene_idx}", path_rows, scene_seed)
        print_seed_table("DR Self Multiseed Relight Metrics: scene avg", path_rows, self_relight_seed)

        self_gt_sky_seed = {"psnr": {"mean": [], "std": []}, "ssim": {"mean": [], "std": []}, "lpips": {"mean": [], "std": []}}
        for pid in args.path_ids:
            exp = args.self_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
            append_seed_metric_row(
                self_gt_sky_seed,
                seed_relight_metrics_for_scenes(
                    exp,
                    args.self_relight_folders[0],
                    self_scene_idxs,
                    args.seeds,
                    metric_filename="image_metrics_gt_sky.json",
                ),
            )
        for scene_idx in self_scene_idxs:
            scene_gt_sky_seed = {"psnr": {"mean": [], "std": []}, "ssim": {"mean": [], "std": []}, "lpips": {"mean": [], "std": []}}
            for pid in args.path_ids:
                exp = args.self_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
                append_seed_metric_row(
                    scene_gt_sky_seed,
                    seed_relight_metrics_for_scene(
                        exp,
                        args.self_relight_folders,
                        scene_idx,
                        args.seeds,
                        metric_filename="image_metrics_gt_sky.json",
                    ),
                )
            print_seed_table(f"DR Self Multiseed GT-Sky Relight Metrics: {scene_idx}", path_rows, scene_gt_sky_seed)
        print_seed_table("DR Self Multiseed GT-Sky Relight Metrics: scene avg", path_rows, self_gt_sky_seed)

    shifted_intrinsic = {"normal_mae": [], "albedo_si_psnr": [], "roughness_rmse": [], "metallic_rmse": []}
    shifted_exps = [
        args.shifted_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
        for pid in args.path_ids
    ]
    shifted_scene_idxs = resolve_relight_scene_idxs_for_exps(
        shifted_exps, args.shifted_relight_folders, args.relight_scene_idx
    )
    shifted_relight_all_scenes = {"psnr": [], "ssim": [], "lpips": []}
    shifted_relight_by_scene = {scene_idx: {"psnr": [], "ssim": [], "lpips": []} for scene_idx in shifted_scene_idxs}
    shifted_ground_relight_all_scenes = {"psnr": [], "ssim": [], "lpips": []}
    shifted_ground_relight_by_scene = {
        scene_idx: {"psnr": [], "ssim": [], "lpips": []} for scene_idx in shifted_scene_idxs
    }
    shifted_gt_sky_relight_all_scenes = {"psnr": [], "ssim": [], "lpips": []}
    shifted_gt_sky_relight_by_scene = {
        scene_idx: {"psnr": [], "ssim": [], "lpips": []} for scene_idx in shifted_scene_idxs
    }
    for pid in args.path_ids:
        exp = args.shifted_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
        append_metric_row(shifted_intrinsic, intrinsic_metrics(load_json(exp / "dr_eval_raw" / "intrinsic" / "intrinsic_metrics.json")))
        append_metric_row(
            shifted_relight_all_scenes,
            relight_metrics_for_scenes(exp, args.shifted_relight_folders, shifted_scene_idxs),
        )
        append_metric_row(
            shifted_ground_relight_all_scenes,
            relight_metrics_for_scenes(
                exp,
                args.shifted_relight_folders,
                shifted_scene_idxs,
                metric_filename="image_metrics_ground.json",
            ),
        )
        append_metric_row(
            shifted_gt_sky_relight_all_scenes,
            relight_metrics_for_scenes(
                exp,
                args.shifted_relight_folders,
                shifted_scene_idxs,
                metric_filename="image_metrics_gt_sky.json",
            ),
        )
        for scene_idx in shifted_scene_idxs:
            append_metric_row(
                shifted_relight_by_scene[scene_idx],
                relight_metrics_for_scene(exp, args.shifted_relight_folders, scene_idx),
            )
            append_metric_row(
                shifted_ground_relight_by_scene[scene_idx],
                relight_metrics_for_scene(
                    exp,
                    args.shifted_relight_folders,
                    scene_idx,
                    metric_filename="image_metrics_ground.json",
                ),
            )
            append_metric_row(
                shifted_gt_sky_relight_by_scene[scene_idx],
                relight_metrics_for_scene(
                    exp,
                    args.shifted_relight_folders,
                    scene_idx,
                    metric_filename="image_metrics_gt_sky.json",
                ),
            )
    print_table("DR Shifted Intrinsic Metrics", path_rows, shifted_intrinsic)
    for scene_idx, metrics in shifted_relight_by_scene.items():
        print_table(f"DR Shifted Relight Metrics: {scene_idx}", path_rows, metrics)
    print_table("DR Shifted Relight Metrics: scene avg", path_rows, shifted_relight_all_scenes)
    for scene_idx, metrics in shifted_ground_relight_by_scene.items():
        print_table(f"DR Shifted Ground Relight Metrics: {scene_idx}", path_rows, metrics)
    print_table("DR Shifted Ground Relight Metrics: scene avg", path_rows, shifted_ground_relight_all_scenes)
    for scene_idx, metrics in shifted_gt_sky_relight_by_scene.items():
        print_table(f"DR Shifted GT-Sky Relight Metrics: {scene_idx}", path_rows, metrics)
    print_table("DR Shifted GT-Sky Relight Metrics: scene avg", path_rows, shifted_gt_sky_relight_all_scenes)

    if args.multiseed:
        shifted_seed_all = {"psnr": {"mean": [], "std": []}, "ssim": {"mean": [], "std": []}, "lpips": {"mean": [], "std": []}}
        for pid in args.path_ids:
            exp = args.shifted_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
            seed_values = []
            for seed in args.seeds:
                scene_values = []
                for scene_idx in shifted_scene_idxs:
                    paths = relight_metric_paths_for_scene(exp, args.shifted_relight_folders, scene_idx, seed=seed)
                    if paths:
                        scene_values.append(average_metric_dicts([relight_metrics(load_json(path)) for path in paths]))
                    else:
                        warn_missing(exp / f"<{'|'.join(args.shifted_relight_folders)}>_relight_{scene_idx}_seed_{seed}" / "relight" / "image_metrics.json")
                        scene_values.append(relight_metrics(None))
                seed_values.append(average_metric_dicts(scene_values))
            append_seed_metric_row(shifted_seed_all, seed_values)
        for scene_idx in shifted_scene_idxs:
            shifted_seed = {"psnr": {"mean": [], "std": []}, "ssim": {"mean": [], "std": []}, "lpips": {"mean": [], "std": []}}
            for pid in args.path_ids:
                exp = args.shifted_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
                append_seed_metric_row(
                    shifted_seed,
                    seed_relight_metrics_for_scene(exp, args.shifted_relight_folders, scene_idx, args.seeds),
                )
            print_seed_table(f"DR Shifted Multiseed Relight Metrics: {scene_idx}", path_rows, shifted_seed)
        print_seed_table("DR Shifted Multiseed Relight Metrics: scene avg", path_rows, shifted_seed_all)

        shifted_gt_sky_seed_all = {"psnr": {"mean": [], "std": []}, "ssim": {"mean": [], "std": []}, "lpips": {"mean": [], "std": []}}
        for pid in args.path_ids:
            exp = args.shifted_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
            seed_values = []
            for seed in args.seeds:
                scene_values = []
                for scene_idx in shifted_scene_idxs:
                    paths = relight_metric_paths_for_scene(
                        exp,
                        args.shifted_relight_folders,
                        scene_idx,
                        seed=seed,
                        metric_filename="image_metrics_gt_sky.json",
                    )
                    if paths:
                        scene_values.append(average_metric_dicts([relight_metrics(load_json(path)) for path in paths]))
                    else:
                        warn_missing(exp / f"<{'|'.join(args.shifted_relight_folders)}>_relight_{scene_idx}_seed_{seed}" / "relight" / "image_metrics_gt_sky.json")
                        scene_values.append(relight_metrics(None))
                seed_values.append(average_metric_dicts(scene_values))
            append_seed_metric_row(shifted_gt_sky_seed_all, seed_values)
        for scene_idx in shifted_scene_idxs:
            shifted_gt_sky_seed = {"psnr": {"mean": [], "std": []}, "ssim": {"mean": [], "std": []}, "lpips": {"mean": [], "std": []}}
            for pid in args.path_ids:
                exp = args.shifted_root / f"dr_only_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
                append_seed_metric_row(
                    shifted_gt_sky_seed,
                    seed_relight_metrics_for_scene(
                        exp,
                        args.shifted_relight_folders,
                        scene_idx,
                        args.seeds,
                        metric_filename="image_metrics_gt_sky.json",
                    ),
                )
            print_seed_table(f"DR Shifted Multiseed GT-Sky Relight Metrics: {scene_idx}", path_rows, shifted_gt_sky_seed)
        print_seed_table("DR Shifted Multiseed GT-Sky Relight Metrics: scene avg", path_rows, shifted_gt_sky_seed_all)

    if args.include_unirelight:
        unirelight_exps = [
            args.unirelight_root / f"unirelight_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
            for pid in args.path_ids
        ]
        unirelight_scene_idxs = resolve_relight_scene_idxs_for_exps(
            unirelight_exps, [args.unirelight_folder], args.relight_scene_idx
        )
        unirelight_relight = {"psnr": [], "ssim": [], "lpips": []}
        unirelight_albedo = {"albedo_si_psnr": []}
        unirelight_relight_by_scene = {
            scene_idx: {"psnr": [], "ssim": [], "lpips": []} for scene_idx in unirelight_scene_idxs
        }
        unirelight_albedo_by_scene = {scene_idx: {"albedo_si_psnr": []} for scene_idx in unirelight_scene_idxs}
        for pid in args.path_ids:
            exp = args.unirelight_root / f"unirelight_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
            relight_values = []
            albedo_values = []
            for scene_idx in unirelight_scene_idxs:
                root = exp / f"{args.unirelight_folder}_relight_{scene_idx}"
                relight_value = relight_metrics(load_json(root / "relight" / "image_metrics.json"))
                albedo_value = intrinsic_metrics(load_json(root / "intrinsic" / "albedo_metrics.json"))
                relight_values.append(relight_value)
                albedo_values.append(albedo_value)
                append_metric_row(unirelight_relight_by_scene[scene_idx], relight_value)
                append_metric_row(unirelight_albedo_by_scene[scene_idx], albedo_value)
            append_metric_row(unirelight_relight, average_metric_dicts(relight_values))
            append_metric_row(unirelight_albedo, average_metric_dicts(albedo_values))
        for scene_idx in unirelight_scene_idxs:
            print_table(f"UniRelight Self Relight Metrics: {scene_idx}", path_rows, unirelight_relight_by_scene[scene_idx])
            print_table(f"UniRelight Self Albedo Metrics: {scene_idx}", path_rows, unirelight_albedo_by_scene[scene_idx])
        print_table("UniRelight Self Relight Metrics: scene avg", path_rows, unirelight_relight)
        print_table("UniRelight Self Albedo Metrics: scene avg", path_rows, unirelight_albedo)

        unirelight_shifted_exps = [
            args.unirelight_shifted_root / f"unirelight_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
            for pid in args.path_ids
        ]
        unirelight_shifted_scene_idxs = resolve_relight_scene_idxs_for_exps(
            unirelight_shifted_exps, [args.unirelight_folder], args.relight_scene_idx
        )
        unirelight_shifted_relight = {"psnr": [], "ssim": [], "lpips": []}
        unirelight_shifted_albedo = {"albedo_si_psnr": []}
        unirelight_shifted_relight_by_scene = {
            scene_idx: {"psnr": [], "ssim": [], "lpips": []} for scene_idx in unirelight_shifted_scene_idxs
        }
        unirelight_shifted_albedo_by_scene = {
            scene_idx: {"albedo_si_psnr": []} for scene_idx in unirelight_shifted_scene_idxs
        }
        for pid in args.path_ids:
            exp = args.unirelight_shifted_root / f"unirelight_1cam_path{pid}_fixed_tree_gamma_full_{args.scene_idx}" / "drivestudio"
            relight_values = []
            albedo_values = []
            for scene_idx in unirelight_shifted_scene_idxs:
                root = exp / f"{args.unirelight_folder}_relight_{scene_idx}"
                relight_value = relight_metrics(load_json(root / "relight" / "image_metrics.json"))
                albedo_value = intrinsic_metrics(load_json(root / "intrinsic" / "albedo_metrics.json"))
                relight_values.append(relight_value)
                albedo_values.append(albedo_value)
                append_metric_row(unirelight_shifted_relight_by_scene[scene_idx], relight_value)
                append_metric_row(unirelight_shifted_albedo_by_scene[scene_idx], albedo_value)
            append_metric_row(unirelight_shifted_relight, average_metric_dicts(relight_values))
            append_metric_row(unirelight_shifted_albedo, average_metric_dicts(albedo_values))
        for scene_idx in unirelight_shifted_scene_idxs:
            print_table(f"UniRelight Shifted Relight Metrics: {scene_idx}", path_rows, unirelight_shifted_relight_by_scene[scene_idx])
            print_table(f"UniRelight Shifted Albedo Metrics: {scene_idx}", path_rows, unirelight_shifted_albedo_by_scene[scene_idx])
        print_table("UniRelight Shifted Relight Metrics: scene avg", path_rows, unirelight_shifted_relight)
        print_table("UniRelight Shifted Albedo Metrics: scene avg", path_rows, unirelight_shifted_albedo)


if __name__ == "__main__":
    main()

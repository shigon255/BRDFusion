import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_RELIGHT_SCENES = [
    "qwantani_moon_noon_puresky_4k_rot90",
    "citrus_orchard_4k",
    "the_sky_is_on_fire_4k",
]
DEFAULT_PATH_IDS = [1, 2, 3, 4, 5, 6]
DEFAULT_SCENE_IDX = "qwantani_moon_noon_puresky_4k"
DEFAULT_INTRINSIC_TYPES = ["normal", "albedo", "roughness", "metallic"]
DEFAULT_LAYOUT = "auto"

INTRINSIC_SPECS = {
    "normal": ("normal_mae", "train_normal", "test_normal", "mae_deg"),
    "albedo": ("albedo_si_psnr", "train_albedo", "test_albedo", "si_psnr"),
    "roughness": ("roughness_rmse", "train_roughness", "test_roughness", "rmse"),
    "metallic": ("metallic_rmse", "train_metallic", "test_metallic", "rmse"),
}


def warn_missing(path: Path) -> None:
    print(f"[missing] {path}", file=sys.stderr)


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        warn_missing(path)
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)




def resolve_metric_path(base: Path, metric_kind: str, relight_scene: Optional[str], layout: str) -> Path:
    candidates = []
    if metric_kind == "image":
        candidates_by_layout = {
            "external": base / "image_metrics_external.json",
            "shifted_3": base / "image_metrics_shifted_3.json",
        }
    elif metric_kind == "intrinsic":
        candidates_by_layout = {
            "external": base / "intrinsic_metrics_external.json",
            "shifted_3": base / "intrinsic_metrics_shifted_3.json",
        }
    elif metric_kind == "relight":
        assert relight_scene is not None
        candidates_by_layout = {
            "external": base / f"relight_{relight_scene}_metrics_external.json",
            "shifted_3": base / f"relight_{relight_scene}_metrics_shifted_3.json",
        }
    elif metric_kind == "relight_ground":
        assert relight_scene is not None
        candidates_by_layout = {
            "external": base / f"relight_{relight_scene}_metrics_external_ground.json",
            "shifted_3": base / f"relight_{relight_scene}_metrics_shifted_3_ground.json",
        }
    else:
        raise ValueError(f"Unknown metric kind: {metric_kind}")

    if layout == "auto":
        candidates = [candidates_by_layout["external"], candidates_by_layout["shifted_3"]]
    else:
        candidates = [candidates_by_layout[layout]]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def weighted_named(train: Dict[str, float], test: Dict[str, float], key: str) -> float:
    total = train["count"] + test["count"]
    if total == 0:
        return math.nan
    return (train[key] * train["count"] + test[key] * test["count"]) / total


def weighted_image_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {"psnr": math.nan, "ssim": math.nan, "lpips": math.nan}
    train = data["metrics"]["train"]
    test = data["metrics"]["test"]
    total = train["count"] + test["count"]
    if total == 0:
        return {"psnr": math.nan, "ssim": math.nan, "lpips": math.nan}
    return {
        "psnr": (train["psnr"] * train["count"] + test["psnr"] * test["count"]) / total,
        "ssim": (train["ssim"] * train["count"] + test["ssim"] * test["count"]) / total,
        "lpips": (train["lpips"] * train["count"] + test["lpips"] * test["count"]) / total,
    }


def weighted_intrinsic_metrics(data: Optional[Dict], intrinsic_types: List[str]) -> Dict[str, float]:
    out = {INTRINSIC_SPECS[name][0]: math.nan for name in intrinsic_types}
    if data is None:
        return out
    metrics = data["metrics"]
    for name in intrinsic_types:
        out_key, train_key, test_key, value_key = INTRINSIC_SPECS[name]
        if train_key in metrics and test_key in metrics:
            out[out_key] = weighted_named(metrics[train_key], metrics[test_key], value_key)
    return out


def weighted_relight_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {"psnr": math.nan, "ssim": math.nan, "lpips": math.nan}
    train = data["metrics"]["train_relight"]
    test = data["metrics"]["test_relight"]
    total = train["count"] + test["count"]
    if total == 0:
        return {"psnr": math.nan, "ssim": math.nan, "lpips": math.nan}
    return {
        "psnr": (train["psnr"] * train["count"] + test["psnr"] * test["count"]) / total,
        "ssim": (train["ssim"] * train["count"] + test["ssim"] * test["count"]) / total,
        "lpips": (train["lpips"] * train["count"] + test["lpips"] * test["count"]) / total,
    }


def mean(values: List[float]) -> float:
    if any(math.isnan(v) for v in values):
        return math.nan
    return math.nan if not values else sum(values) / len(values)


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


def append_metric_row(container: Dict[str, List[float]], values: Dict[str, float]) -> None:
    for key, value in values.items():
        container.setdefault(key, []).append(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate external-path self evaluation metrics across paths.")
    parser.add_argument("--root", type=Path, default=Path("urbanir_eval_result/self"))
    parser.add_argument("--scene-idx", type=str, default=DEFAULT_SCENE_IDX)
    parser.add_argument("--path-ids", type=int, nargs="+", default=DEFAULT_PATH_IDS)
    parser.add_argument("--relight-scenes", type=str, nargs="+", default=DEFAULT_RELIGHT_SCENES)
    parser.add_argument("--intrinsic-types", type=str, nargs="+", choices=list(INTRINSIC_SPECS.keys()), default=DEFAULT_INTRINSIC_TYPES, help="Intrinsic channels to aggregate. Example: --intrinsic-types normal albedo")
    parser.add_argument("--layout", choices=["auto", "external", "shifted_3"], default=DEFAULT_LAYOUT, help="Metric filename layout to use. auto tries *_external.json first, then *_shifted_3.json.")
    args = parser.parse_args()

    path_labels = [f"path{pid}" for pid in args.path_ids]
    image_metrics = {"psnr": [], "ssim": [], "lpips": []}
    intrinsic_metrics = {INTRINSIC_SPECS[name][0]: [] for name in args.intrinsic_types}
    relight_by_scene = {
        scene: {"psnr": [], "ssim": [], "lpips": []} for scene in args.relight_scenes
    }
    ground_relight_by_scene = {
        scene: {"psnr": [], "ssim": [], "lpips": []} for scene in args.relight_scenes
    }

    for pid in args.path_ids:
        base = args.root / f"path{pid}_fixed_tree_gamma_full" / args.scene_idx
        append_metric_row(image_metrics, weighted_image_metrics(load_json(resolve_metric_path(base, "image", None, args.layout))))
        append_metric_row(intrinsic_metrics, weighted_intrinsic_metrics(load_json(resolve_metric_path(base, "intrinsic", None, args.layout)), args.intrinsic_types))
        for scene in args.relight_scenes:
            append_metric_row(
                relight_by_scene[scene],
                weighted_relight_metrics(load_json(resolve_metric_path(base, "relight", scene, args.layout))),
            )
            append_metric_row(
                ground_relight_by_scene[scene],
                weighted_relight_metrics(load_json(resolve_metric_path(base, "relight_ground", scene, args.layout))),
            )

    print_table("External Self Image Metrics", path_labels, image_metrics)
    print_table("External Self Intrinsic Metrics", path_labels, intrinsic_metrics)

    avg_relight = {"psnr": [], "ssim": [], "lpips": []}
    for scene in args.relight_scenes:
        print_table(f"External Self Relight Metrics: {scene}", path_labels, relight_by_scene[scene])
    for i in range(len(args.path_ids)):
        for key in avg_relight:
            avg_relight[key].append(mean([relight_by_scene[scene][key][i] for scene in args.relight_scenes]))
    print_table("External Self Relight Metrics: Average Across Scenes", path_labels, avg_relight)

    avg_ground_relight = {"psnr": [], "ssim": [], "lpips": []}
    for scene in args.relight_scenes:
        print_table(f"External Self Ground Relight Metrics: {scene}", path_labels, ground_relight_by_scene[scene])
    for i in range(len(args.path_ids)):
        for key in avg_ground_relight:
            avg_ground_relight[key].append(mean([ground_relight_by_scene[scene][key][i] for scene in args.relight_scenes]))
    print_table("External Self Ground Relight Metrics: Average Across Scenes", path_labels, avg_ground_relight)


if __name__ == "__main__":
    main()

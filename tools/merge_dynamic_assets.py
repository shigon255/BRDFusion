import argparse
import os
from typing import Dict, List

import torch


def _load_asset(path: str) -> Dict[str, object]:
    asset = torch.load(path, map_location="cpu")
    if "classes" not in asset:
        raise KeyError(f"Asset missing 'classes': {path}")
    if not isinstance(asset["classes"], dict) or not asset["classes"]:
        raise ValueError(f"Asset has empty or invalid 'classes': {path}")
    return asset


def _clone_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_value(v) for v in value]
    return value


def merge_assets(asset_paths: List[str]) -> Dict[str, object]:
    merged_classes: Dict[str, Dict[str, object]] = {}
    merged_sources: List[Dict[str, object]] = []

    for asset_path in asset_paths:
        abs_path = os.path.abspath(asset_path)
        asset = _load_asset(abs_path)
        asset_classes = asset["classes"]

        for class_name, class_payload in asset_classes.items():
            if class_name in merged_classes:
                raise ValueError(
                    "Merging duplicate classes is not supported yet. "
                    f"Class '{class_name}' appears more than once, including asset: {abs_path}"
                )
            merged_classes[class_name] = _clone_value(class_payload)

        merged_sources.append(
            {
                "asset_path": abs_path,
                "source_checkpoint": asset.get("source_checkpoint", None),
                "classes": sorted(asset_classes.keys()),
                "format_version": asset.get("format_version", None),
            }
        )

    return {
        "format_version": 1,
        "asset_type": "dynamic_nodes",
        "source_checkpoint": None,
        "merged_from_assets": merged_sources,
        "classes": merged_classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple exported dynamic asset packages into one insertable package."
    )
    parser.add_argument(
        "--asset_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more dynamic asset .pth files to merge",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output merged asset .pth path",
    )
    args = parser.parse_args()

    package = merge_assets(args.asset_paths)
    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(package, output_path)

    print(f"[OK] Saved merged dynamic asset package: {output_path}")
    for class_name, payload in package["classes"].items():
        instance_ids = payload.get("instance_ids", [])
        num_points = payload.get("num_points", None)
        print(
            f"[OK] {class_name}: instances={len(instance_ids)}, "
            f"points={num_points}, ids={instance_ids}"
        )


if __name__ == "__main__":
    main()

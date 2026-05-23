import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

from omegaconf import OmegaConf


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _iter_selected(items: Iterable[Dict], selected: set) -> Iterable[Dict]:
    for item in items:
        item_id = str(item["id"])
        if not selected or item_id in selected:
            yield item


def _link_scene(source: Path, target: Path, force: bool, dry_run: bool) -> str:
    if target.is_symlink():
        current = Path(os.readlink(target))
        if current == source:
            return "exists"
        if not force:
            raise FileExistsError(f"{target} already links to {current}; use --force to replace it")
        if not dry_run:
            target.unlink()
    elif target.exists():
        raise FileExistsError(f"{target} exists and is not a symlink")

    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, target, target_is_directory=True)
    return "linked"


def link_datasets(args: argparse.Namespace) -> List[Dict]:
    repo_root = _repo_root()
    cfg = OmegaConf.to_container(OmegaConf.load(args.layout), resolve=True)
    layout_root = _as_path(repo_root, cfg.get("root", "data/brdfusion"))
    selected_datasets = set(args.dataset or [])
    selected_scenes = set(args.scene or [])

    source_overrides = {
        "waymo": args.waymo_source_root,
        "self": args.self_source_root,
    }
    records = []

    for dataset_name, dataset_cfg in cfg["datasets"].items():
        if selected_datasets and dataset_name not in selected_datasets:
            continue

        scene_root = layout_root / dataset_cfg.get("scene_root", f"{dataset_name}/scenes")
        source_root_value = source_overrides.get(dataset_name) or dataset_cfg["source_root"]
        source_root = _as_path(repo_root, source_root_value)

        for scene in _iter_selected(dataset_cfg.get("scenes", []), selected_scenes):
            scene_id = str(scene["id"])
            source = _as_path(source_root, str(scene.get("source", scene_id)))
            target = scene_root / scene_id
            if args.require_source and not source.exists():
                raise FileNotFoundError(f"Source scene does not exist: {source}")
            status = _link_scene(source, target, force=args.force, dry_run=args.dry_run)
            records.append(
                {
                    "dataset": dataset_name,
                    "id": scene_id,
                    "role": scene.get("role"),
                    "source": str(source),
                    "target": str(target),
                    "source_exists": source.exists(),
                    "status": status,
                }
            )

    if not args.dry_run:
        manifest_path = layout_root / "dataset_links.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    return records


def main() -> None:
    parser = argparse.ArgumentParser("Create BRDFusion-local dataset symlinks.")
    parser.add_argument("--layout", default="configs/data/brdfusion_layout.yaml")
    parser.add_argument("--dataset", action="append", choices=["waymo", "self"])
    parser.add_argument("--scene", action="append", help="Only link this canonical scene id.")
    parser.add_argument("--waymo_source_root", default=None)
    parser.add_argument("--self_source_root", default=None)
    parser.add_argument("--require_source", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    records = link_datasets(args)
    for record in records:
        print(
            f"{record['status']}: {record['dataset']}:{record['id']} -> "
            f"{record['target']} => {record['source']}"
        )
    if not records:
        print("No dataset links selected")


if __name__ == "__main__":
    main()

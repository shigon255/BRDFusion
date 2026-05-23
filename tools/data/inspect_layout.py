import argparse
import os
from pathlib import Path
from typing import Dict, List


MODALITIES = [
    "images",
    "sky_masks",
    "road_masks",
    "dynamic_masks",
    "fine_dynamic_masks",
    "lidar",
    "ego_pose",
    "extrinsics",
    "intrinsics",
    "depth",
    "normal",
    "albedo",
    "roughness",
    "metallic",
    "diffusion_renderer_normal",
    "diffusion_renderer_depth",
    "diffusion_renderer_albedo",
    "diffusion_renderer_roughness",
    "diffusion_renderer_metallic",
]


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    count = 0
    for _, _, files in os.walk(path):
        count += len(files)
    return count


def inspect_scene(scene: Path) -> Dict:
    resolved = scene.resolve() if scene.exists() else scene
    modalities = {}
    for name in MODALITIES:
        subdir = scene / name
        if subdir.exists():
            modalities[name] = _count_files(subdir)
    return {
        "scene": str(scene),
        "resolved": str(resolved),
        "is_symlink": scene.is_symlink(),
        "exists": scene.exists(),
        "modalities": modalities,
    }


def _scene_dirs(root: Path, max_scenes: int) -> List[Path]:
    if not root.exists():
        return []
    scenes = [path for path in sorted(root.iterdir()) if path.is_dir() or path.is_symlink()]
    if max_scenes > 0:
        scenes = scenes[:max_scenes]
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser("Inspect BRDFusion dataset scene folders.")
    parser.add_argument("roots", nargs="+", help="Scene root(s), for example data/brdfusion/self/scenes.")
    parser.add_argument("--max_scenes", type=int, default=20)
    args = parser.parse_args()

    for root_value in args.roots:
        root = Path(root_value)
        print(f"# {root}")
        scenes = _scene_dirs(root, args.max_scenes)
        if not scenes:
            print("  no scenes found")
            continue
        for scene in scenes:
            info = inspect_scene(scene)
            link = " -> " + info["resolved"] if info["is_symlink"] else ""
            print(f"  {scene.name}{link}")
            for name, count in sorted(info["modalities"].items()):
                print(f"    {name}: {count}")


if __name__ == "__main__":
    main()

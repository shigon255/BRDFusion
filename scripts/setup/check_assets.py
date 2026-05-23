import argparse
import hashlib
import os
from typing import Dict, Iterable

from omegaconf import OmegaConf


def _md5(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _iter_assets(cfg: Dict, groups: Iterable[str]):
    assets = cfg.get("assets", {})
    for group in groups:
        if group == "all":
            for name, item in assets.items():
                yield name, item
            return
        if group not in assets:
            raise KeyError(f"Unknown asset group '{group}'. Known groups: {sorted(assets)}")
        yield group, assets[group]


def main() -> None:
    parser = argparse.ArgumentParser("Check BRDFusion local asset placement.")
    parser.add_argument("--manifest", default="configs/assets/weights.yaml")
    parser.add_argument("--group", action="append", default=None, help="Asset group to check.")
    parser.add_argument("--verify_md5", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.manifest), resolve=True)
    missing = []
    mismatched = []

    groups = args.group or ["all"]
    for group_name, group in _iter_assets(cfg, groups):
        root = group["root"]
        print(f"[{group_name}] root={root}")
        for item in group.get("required", []):
            rel_path = item["path"]
            path = os.path.join(root, rel_path)
            if not os.path.exists(path):
                missing.append(path)
                print(f"  missing: {rel_path}")
                continue
            if "size_bytes" in item and os.path.isfile(path):
                size = os.path.getsize(path)
                if int(item["size_bytes"]) != size:
                    mismatched.append(f"{path}: size {size} != {item['size_bytes']}")
                else:
                    print(f"  ok: {rel_path}")
            else:
                print(f"  ok: {rel_path}")
            if args.verify_md5 and "md5" in item and os.path.isfile(path):
                value = _md5(path)
                if value != item["md5"]:
                    mismatched.append(f"{path}: md5 {value} != {item['md5']}")

    if missing or mismatched:
        if missing:
            print("\nMissing files/directories:")
            print("\n".join(missing))
        if mismatched:
            print("\nMismatched files:")
            print("\n".join(mismatched))
        raise SystemExit(1)
    print("Asset check OK")


if __name__ == "__main__":
    main()

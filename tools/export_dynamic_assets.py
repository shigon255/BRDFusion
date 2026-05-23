import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw


POINT_ID_KEYS = ("points_ids", "point_ids")
TIME_INSTANCE_KEYS = {
    "instances_fv",
    "instances_trans",
    "instances_quats",
    "smpl_qauts",
}
SUPPORTED_CLASSES = ("RigidNodes", "SMPLNodes", "DeformableNodes")
SH_C0 = 0.28209479177387814


def parse_instance_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        out.append(int(token))
    return sorted(set(out))


def parse_classes(raw: str) -> List[str]:
    classes = [x.strip() for x in raw.split(",") if x.strip()]
    if not classes:
        raise ValueError("No classes provided.")
    bad = [c for c in classes if c not in SUPPORTED_CLASSES]
    if bad:
        raise ValueError(f"Unsupported classes: {bad}. Supported: {SUPPORTED_CLASSES}")
    return classes


def get_point_id_key(state: Dict[str, torch.Tensor]) -> str:
    for k in POINT_ID_KEYS:
        if k in state:
            return k
    raise KeyError(f"State dict missing point id key: expected one of {POINT_ID_KEYS}")


def get_num_instances(state: Dict[str, torch.Tensor]) -> int:
    if "instances_fv" in state:
        return int(state["instances_fv"].shape[1])
    if "instances_size" in state:
        return int(state["instances_size"].shape[0])
    pid_key = get_point_id_key(state)
    pids = state[pid_key].reshape(-1).long()
    return int(torch.max(pids).item()) + 1 if pids.numel() > 0 else 0


def get_num_timesteps(state: Dict[str, torch.Tensor]) -> Optional[int]:
    for key in ("instances_trans", "instances_fv", "instances_quats", "smpl_qauts"):
        val = state.get(key, None)
        if torch.is_tensor(val) and val.ndim >= 1:
            return int(val.shape[0])
    return None


def _clone_state_to_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = v
    return out


def subset_state_by_instances(
    state: Dict[str, torch.Tensor],
    selected_ids: Optional[Sequence[int]],
) -> Tuple[Dict[str, torch.Tensor], List[int], int]:
    state_cpu = _clone_state_to_cpu(state)

    pid_key = get_point_id_key(state_cpu)
    pids = state_cpu[pid_key].reshape(-1).long()
    num_points = int(pids.shape[0])
    num_instances = get_num_instances(state_cpu)

    if selected_ids is None:
        keep_ids = list(range(num_instances))
    else:
        keep_ids = sorted(set(int(x) for x in selected_ids))
        for x in keep_ids:
            if x < 0 or x >= num_instances:
                raise ValueError(
                    f"Selected instance id {x} out of range [0, {num_instances - 1}]"
                )

    keep_tensor = torch.tensor(keep_ids, dtype=torch.long)
    if keep_tensor.numel() == 0:
        raise ValueError("No instances selected.")

    point_mask = torch.isin(pids, keep_tensor)
    id_remap = {old: new for new, old in enumerate(keep_ids)}

    out: Dict[str, torch.Tensor] = {}
    for key, val in state_cpu.items():
        if not torch.is_tensor(val):
            out[key] = val
            continue

        if key == pid_key:
            remapped = val[point_mask].clone().long()
            for old, new in id_remap.items():
                remapped[remapped == old] = new
            out[key] = remapped
            continue

        if val.ndim >= 1 and val.shape[0] == num_points:
            out[key] = val[point_mask]
            continue

        if key in TIME_INSTANCE_KEYS and val.ndim >= 2 and val.shape[1] == num_instances:
            out[key] = val[:, keep_ids]
            continue

        if val.ndim >= 1 and val.shape[0] == num_instances:
            out[key] = val[keep_ids]
            continue

        if key.startswith("template.") and val.ndim >= 1 and val.shape[0] == num_instances:
            out[key] = val[keep_ids]
            continue

        out[key] = val

    return out, keep_ids, int(point_mask.sum().item())


def _class_ids_from_args(class_name: str, args: argparse.Namespace) -> Optional[List[int]]:
    if class_name == "RigidNodes":
        return parse_instance_ids(args.rigid_ids)
    if class_name == "SMPLNodes":
        return parse_instance_ids(args.smpl_ids)
    if class_name == "DeformableNodes":
        return parse_instance_ids(args.deformable_ids)
    return None


def _tensor_to_python(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_tensor_to_python(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _tensor_to_python(v) for k, v in value.items()}
    return value


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


def _select_anchor_frame(state: Dict[str, torch.Tensor], requested_frame: int) -> int:
    num_timesteps = get_num_timesteps(state)
    if num_timesteps is None or num_timesteps <= 0:
        return 0
    return max(0, min(int(requested_frame), num_timesteps - 1))


def _get_instance_visibility(state: Dict[str, torch.Tensor], frame_idx: int) -> Optional[bool]:
    instances_fv = state.get("instances_fv", None)
    if not torch.is_tensor(instances_fv) or instances_fv.numel() == 0:
        return None
    if frame_idx >= instances_fv.shape[0]:
        return None
    value = instances_fv[frame_idx]
    if value.numel() == 0:
        return None
    return bool(value.reshape(-1)[0].item())


def _get_anchor_center(state: Dict[str, torch.Tensor], frame_idx: int) -> Optional[List[float]]:
    trans = state.get("instances_trans", None)
    if not torch.is_tensor(trans) or trans.numel() == 0:
        return None
    if frame_idx >= trans.shape[0]:
        return None
    center = trans[frame_idx]
    if center.ndim == 2:
        center = center[0]
    return [float(x) for x in center.detach().cpu().tolist()]


def _estimate_extent_xyz(state: Dict[str, torch.Tensor]) -> Optional[List[float]]:
    means = state.get("_means", None)
    if not torch.is_tensor(means) or means.numel() == 0:
        return None
    mins = means.float().min(dim=0).values
    maxs = means.float().max(dim=0).values
    extent = (maxs - mins).detach().cpu().tolist()
    return [float(x) for x in extent]


def _get_preview_colors(state: Dict[str, torch.Tensor], num_points: int) -> List[Tuple[int, int, int]]:
    features_dc = state.get("_features_dc", None)
    if torch.is_tensor(features_dc) and features_dc.ndim >= 2 and features_dc.shape[0] == num_points and features_dc.shape[1] >= 3:
        rgb = torch.clamp(features_dc[:, :3].float() * SH_C0 + 0.5, 0.0, 1.0)
        return [
            (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
            for r, g, b in rgb.tolist()
        ]
    return [(40, 120, 240)] * num_points


def _project_points(points_2d: torch.Tensor, size: int, pad: int) -> List[Tuple[int, int]]:
    if points_2d.numel() == 0:
        return []
    pts = points_2d.float()
    mins = pts.min(dim=0).values
    maxs = pts.max(dim=0).values
    spans = torch.clamp(maxs - mins, min=1e-6)
    scale = float((size - 2 * pad) / max(float(spans[0].item()), float(spans[1].item())))
    center = (mins + maxs) * 0.5
    mapped = (pts - center) * scale
    out: List[Tuple[int, int]] = []
    for x, y in mapped.tolist():
        px = int(round(x + size * 0.5))
        py = int(round(size * 0.5 - y))
        px = max(0, min(size - 1, px))
        py = max(0, min(size - 1, py))
        out.append((px, py))
    return out


def _draw_view(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    size: int,
    points_2d: List[Tuple[int, int]],
    point_colors: List[Tuple[int, int, int]],
    title: str,
) -> None:
    draw.rectangle([x0, y0, x0 + size - 1, y0 + size - 1], outline=(210, 210, 210), width=1)
    if points_2d:
        for (x, y), color in zip(points_2d, point_colors):
            px = x0 + int(x)
            py = y0 + int(y)
            draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=color)
    draw.text((x0 + 8, y0 + 6), title, fill=(40, 40, 40))


def save_object_preview(
    state: Dict[str, torch.Tensor],
    class_name: str,
    source_instance_id: int,
    output_path: str,
    preview_size: int,
) -> None:
    means = state.get("_means", None)
    canvas_h = preview_size + 52
    canvas_w = preview_size * 2 + 12
    image = Image.new("RGB", (canvas_w, canvas_h), color=(250, 250, 250))
    draw = ImageDraw.Draw(image)

    if torch.is_tensor(means) and means.numel() > 0:
        pts = means.detach().cpu().float()
        point_colors = _get_preview_colors(state, pts.shape[0])
        top = _project_points(pts[:, [0, 1]], preview_size, pad=12)
        front = _project_points(pts[:, [0, 2]], preview_size, pad=12)
        _draw_view(draw, 0, 36, preview_size, top, point_colors, "top x-y")
        _draw_view(draw, preview_size + 12, 36, preview_size, front, point_colors, "front x-z")
    else:
        draw.text((12, preview_size // 2), "No _means available", fill=(20, 20, 180))

    title = f"{class_name} #{source_instance_id}"
    draw.text((10, 10), title, fill=(15, 15, 15))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path)


def build_asset_package(
    source_ckpt: str,
    class_name: str,
    subset_state: Dict[str, torch.Tensor],
    kept_ids: List[int],
    kept_points: int,
    anchor_frame: int,
) -> Dict[str, object]:
    metadata = {
        "class_name": class_name,
        "source_instance_ids": kept_ids,
        "source_instance_id": kept_ids[0] if len(kept_ids) == 1 else None,
        "num_instances": len(kept_ids),
        "num_points": int(kept_points),
        "anchor_frame": int(anchor_frame),
        "anchor_visible": _get_instance_visibility(subset_state, anchor_frame),
        "anchor_center": _get_anchor_center(subset_state, anchor_frame),
        "extent_xyz": _estimate_extent_xyz(subset_state),
        "num_timesteps": get_num_timesteps(subset_state),
    }
    return {
        "format_version": 2,
        "asset_type": "dynamic_nodes",
        "source_checkpoint": os.path.abspath(source_ckpt),
        "metadata": metadata,
        "classes": {
            class_name: {
                "state_dict": subset_state,
                "instance_ids": kept_ids,
                "num_points": kept_points,
            }
        },
    }


def export_combined_assets(
    args: argparse.Namespace,
    classes: List[str],
) -> None:
    ckpt = torch.load(args.resume_from, map_location="cpu")
    if "models" not in ckpt:
        raise KeyError("Checkpoint does not contain 'models'.")

    models = ckpt["models"]
    exported: Dict[str, Dict[str, object]] = {}

    for class_name in classes:
        if class_name not in models:
            print(f"[WARN] {class_name} missing in source checkpoint; skipping.")
            continue

        class_state = models[class_name]
        selected_ids = _class_ids_from_args(class_name, args)
        subset_state, kept_ids, kept_points = subset_state_by_instances(class_state, selected_ids)

        exported[class_name] = {
            "state_dict": subset_state,
            "instance_ids": kept_ids,
            "num_points": kept_points,
        }
        print(
            f"[OK] {class_name}: instances={len(kept_ids)}, points={kept_points}, "
            f"ids={kept_ids[:8]}{'...' if len(kept_ids) > 8 else ''}"
        )

    if not exported:
        raise ValueError("No classes were exported. Check class selection and source checkpoint.")

    package = {
        "format_version": 1,
        "asset_type": "dynamic_nodes",
        "source_checkpoint": os.path.abspath(args.resume_from),
        "classes": exported,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(package, args.output_path)
    print(f"[OK] Saved dynamic asset package: {args.output_path}")


def export_per_object_assets(
    args: argparse.Namespace,
    classes: List[str],
) -> None:
    ckpt = torch.load(args.resume_from, map_location="cpu")
    if "models" not in ckpt:
        raise KeyError("Checkpoint does not contain 'models'.")

    models = ckpt["models"]
    output_dir = os.path.abspath(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "format_version": 1,
        "export_mode": "per_object",
        "source_checkpoint": os.path.abspath(args.resume_from),
        "output_dir": output_dir,
        "classes": classes,
        "items": [],
    }
    total_written = 0

    for class_name in classes:
        if class_name not in models:
            print(f"[WARN] {class_name} missing in source checkpoint; skipping.")
            continue

        class_state = models[class_name]
        selected_ids = _class_ids_from_args(class_name, args)
        all_ids = selected_ids
        if all_ids is None:
            all_ids = list(range(get_num_instances(class_state)))

        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for source_instance_id in all_ids:
            subset_state, kept_ids, kept_points = subset_state_by_instances(class_state, [source_instance_id])
            anchor_frame = _select_anchor_frame(subset_state, args.preview_anchor_timestep)
            asset_package = build_asset_package(
                source_ckpt=args.resume_from,
                class_name=class_name,
                subset_state=subset_state,
                kept_ids=kept_ids,
                kept_points=kept_points,
                anchor_frame=anchor_frame,
            )

            stem = _safe_filename(f"{class_name}_{source_instance_id:04d}")
            asset_path = os.path.join(class_dir, f"{stem}.pth")
            preview_path = os.path.join(class_dir, f"{stem}.png")
            torch.save(asset_package, asset_path)
            save_object_preview(
                state=subset_state,
                class_name=class_name,
                source_instance_id=source_instance_id,
                output_path=preview_path,
                preview_size=args.preview_size,
            )

            record = {
                "asset_id": stem,
                "class_name": class_name,
                "source_instance_id": int(source_instance_id),
                "source_instance_ids": [int(x) for x in kept_ids],
                "num_points": int(kept_points),
                "anchor_frame": int(anchor_frame),
                "asset_path": os.path.relpath(asset_path, output_dir),
                "preview_path": os.path.relpath(preview_path, output_dir),
                "metadata": _tensor_to_python(asset_package["metadata"]),
            }
            manifest["items"].append(record)
            total_written += 1
            print(
                f"[OK] {class_name} id={source_instance_id}: points={kept_points}, "
                f"asset={record['asset_path']}, preview={record['preview_path']}"
            )

    if total_written == 0:
        raise ValueError("No per-object assets were exported. Check class selection and source checkpoint.")

    manifest_path = os.path.join(output_dir, args.manifest_name)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[OK] Wrote manifest: {manifest_path}")
    print(f"[OK] Exported {total_written} objects into {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export dynamic asset package(s) from a checkpoint."
    )
    parser.add_argument("--resume_from", type=str, required=True, help="Path to source checkpoint .pth")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output asset .pth path for combined export, or output directory for per_object export",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="RigidNodes,SMPLNodes",
        help="Comma-separated classes to export. Supported: RigidNodes,SMPLNodes,DeformableNodes",
    )
    parser.add_argument("--rigid_ids", type=str, default=None, help="Comma-separated rigid instance IDs")
    parser.add_argument("--smpl_ids", type=str, default=None, help="Comma-separated SMPL instance IDs")
    parser.add_argument(
        "--deformable_ids",
        type=str,
        default=None,
        help="Comma-separated deformable instance IDs",
    )
    parser.add_argument(
        "--export_mode",
        type=str,
        choices=("combined", "per_object"),
        default="combined",
        help="combined: one asset package, per_object: one asset package per object plus preview/manifest",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="manifest.json",
        help="Manifest filename when --export_mode=per_object",
    )
    parser.add_argument(
        "--preview_size",
        type=int,
        default=320,
        help="Square size in pixels for each preview panel when --export_mode=per_object",
    )
    parser.add_argument(
        "--preview_anchor_timestep",
        type=int,
        default=0,
        help="Frame index used for preview metadata fields when --export_mode=per_object",
    )
    args = parser.parse_args()

    classes = parse_classes(args.classes)
    if args.export_mode == "combined":
        export_combined_assets(args, classes)
    else:
        export_per_object_assets(args, classes)


if __name__ == "__main__":
    main()

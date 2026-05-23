import argparse
import os
from typing import List, Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def parse_instance_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(int(tok))
    return out


def sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return sh0 * c0 + 0.5


def safe_make_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def to_activated_scales(scales_raw: torch.Tensor) -> torch.Tensor:
    scales = torch.exp(scales_raw)
    if scales.shape[1] == 1:
        scales = scales.repeat(1, 3)
    elif scales.shape[1] == 2:
        pad = torch.zeros_like(scales[:, :1])
        scales = torch.cat([scales, pad], dim=1)
    elif scales.shape[1] != 3:
        raise ValueError(f"Unsupported scale shape: {tuple(scales.shape)}")
    return scales


def normalize_quat(quats_raw: torch.Tensor) -> torch.Tensor:
    return quats_raw / (torch.norm(quats_raw, dim=-1, keepdim=True) + 1e-12)


def write_rigid_ply(
    save_path: str,
    xyz: np.ndarray,
    normals: np.ndarray,
    albedos: np.ndarray,
    roughnesses: np.ndarray,
    metallics: np.ndarray,
    rgb: np.ndarray,
    f_dc: np.ndarray,
    f_rest: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    instance_ids: np.ndarray,
) -> None:
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("albedo_0", "f4"), ("albedo_1", "f4"), ("albedo_2", "f4"),
        ("roughness", "f4"),
        ("metallic", "f4"),
        ("red", "f4"), ("green", "f4"), ("blue", "f4"),
    ]
    dtype += [(f"f_dc_{i}", "f4") for i in range(f_dc.shape[1])]
    dtype += [(f"f_rest_{i}", "f4") for i in range(f_rest.shape[1])]
    dtype += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("instance_id", "i4"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attrs = np.concatenate(
        [
            xyz,
            normals,
            albedos,
            roughnesses,
            metallics,
            rgb,
            f_dc,
            f_rest,
            opacity,
            scales,
            quats,
            instance_ids[:, None],
        ],
        axis=1,
    )
    elements[:] = list(map(tuple, attrs))
    safe_make_dir(save_path)
    PlyData([PlyElement.describe(elements, "vertex")]).write(save_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export canonical RigidNodes gaussians from a training checkpoint."
    )
    parser.add_argument("--resume_from", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/rigid_canonical/rigid_all.ply",
        help="Merged PLY output path.",
    )
    parser.add_argument(
        "--instance_ids",
        type=str,
        default=None,
        help="Comma-separated instance ids to export. Default: all.",
    )
    parser.add_argument(
        "--split_by_instance",
        action="store_true",
        help="Also write one PLY per rigid instance.",
    )
    parser.add_argument(
        "--alpha_thresh",
        type=float,
        default=0.0,
        help="Opacity threshold after sigmoid activation.",
    )
    parser.add_argument(
        "--color_source",
        type=str,
        choices=["albedo", "sh0"],
        default="albedo",
        help="Color source in exported PLY.",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.resume_from, map_location="cpu")
    if "models" not in ckpt:
        raise KeyError("Checkpoint does not contain 'models'.")
    if "RigidNodes" not in ckpt["models"]:
        raise KeyError("Checkpoint does not contain 'RigidNodes'.")
    rigid = ckpt["models"]["RigidNodes"]

    required = ["_means", "_scales", "_quats", "_opacities"]
    for k in required:
        if k not in rigid:
            raise KeyError(f"RigidNodes state missing required key: {k}")

    point_ids = rigid.get("points_ids", rigid.get("point_ids", None))
    if point_ids is None:
        raise KeyError("RigidNodes state missing 'points_ids'/'point_ids'.")
    point_ids = point_ids.reshape(-1).long()

    xyz = rigid["_means"].float()
    scales = to_activated_scales(rigid["_scales"].float())
    quats = normalize_quat(rigid["_quats"].float())
    opacity = torch.sigmoid(rigid["_opacities"].float())
    normals = torch.nn.functional.normalize(rigid["_normals"].float(), dim=-1, eps=1e-3)
    albedos = torch.sigmoid(rigid["_albedos"].float())
    roughnesses = torch.sigmoid(rigid["_roughnesses"].float())
    metallics = torch.sigmoid(rigid["_metallics"].float())
    f_dc = rigid["_features_dc"].float()
    f_rest = rigid["_features_rest"].float().transpose(1, 2).flatten(start_dim=1)

    if args.color_source == "albedo" and "_albedos" in rigid:
        rgb = albedos
    elif "_features_dc" in rigid:
        rgb = sh0_to_rgb(f_dc)
    else:
        rgb = torch.full_like(xyz, 0.7)

    rgb = torch.clamp(rgb, 0.0, 1.0)

    selected_ids = parse_instance_ids(args.instance_ids)
    if selected_ids is None:
        selected_ids = sorted(torch.unique(point_ids).cpu().tolist())
    selected_set = set(selected_ids)

    instance_keep = torch.tensor(
        [int(i.item()) in selected_set for i in point_ids],
        dtype=torch.bool,
    )
    alpha_keep = opacity[:, 0] > args.alpha_thresh
    keep = instance_keep & alpha_keep

    xyz_k = xyz[keep].cpu().numpy().astype(np.float32)
    normals_k = normals[keep].cpu().numpy().astype(np.float32)
    albedos_k = albedos[keep].cpu().numpy().astype(np.float32)
    roughnesses_k = roughnesses[keep].cpu().numpy().astype(np.float32)
    metallics_k = metallics[keep].cpu().numpy().astype(np.float32)
    rgb_k = rgb[keep].cpu().numpy().astype(np.float32)
    f_dc_k = f_dc[keep].cpu().numpy().astype(np.float32)
    f_rest_k = f_rest[keep].cpu().numpy().astype(np.float32)
    opacity_k = opacity[keep].cpu().numpy().astype(np.float32)
    scales_k = scales[keep].cpu().numpy().astype(np.float32)
    quats_k = quats[keep].cpu().numpy().astype(np.float32)
    ids_k = point_ids[keep].cpu().numpy().astype(np.int32)

    if xyz_k.shape[0] == 0:
        raise ValueError("No points left after instance/opacity filtering.")

    write_rigid_ply(
        save_path=args.output_path,
        xyz=xyz_k,
        normals=normals_k,
        albedos=albedos_k,
        roughnesses=roughnesses_k,
        metallics=metallics_k,
        rgb=rgb_k,
        f_dc=f_dc_k,
        f_rest=f_rest_k,
        opacity=opacity_k,
        scales=scales_k,
        quats=quats_k,
        instance_ids=ids_k,
    )
    print(f"[OK] merged rigid PLY: {args.output_path} (points={xyz_k.shape[0]})")

    if args.split_by_instance:
        base, ext = os.path.splitext(args.output_path)
        for ins_id in selected_ids:
            mask = ids_k == ins_id
            if not np.any(mask):
                continue
            out_path = f"{base}_ins{ins_id:03d}{ext or '.ply'}"
            write_rigid_ply(
                save_path=out_path,
                xyz=xyz_k[mask],
                normals=normals_k[mask],
                albedos=albedos_k[mask],
                roughnesses=roughnesses_k[mask],
                metallics=metallics_k[mask],
                rgb=rgb_k[mask],
                f_dc=f_dc_k[mask],
                f_rest=f_rest_k[mask],
                opacity=opacity_k[mask],
                scales=scales_k[mask],
                quats=quats_k[mask],
                instance_ids=ids_k[mask],
            )
            print(f"[OK] instance {ins_id}: {out_path} (points={int(mask.sum())})")


if __name__ == "__main__":
    main()

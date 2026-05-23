import argparse
import os
from typing import Dict, Tuple

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from datasets.driving_dataset import DrivingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export dataset camera intrinsics/poses to COLMAP text format."
    )
    parser.add_argument("--config_file", type=str, required=True, help="Path to config file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for COLMAP txt files.")
    parser.add_argument(
        "--image_dir_rel",
        type=str,
        default="images",
        help="Relative image folder name written into COLMAP image names.",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default="jpg",
        help="Image extension used when composing image names in images.txt.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional config overrides.")
    return parser.parse_args()


def load_cfg(config_file: str, opts) -> OmegaConf:
    cfg = OmegaConf.load(config_file)
    args_from_cli = OmegaConf.from_cli(opts)

    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")

    assert "dataset" in cfg or "data" in cfg, "Please specify dataset in config or via overrides."

    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(os.path.join("configs", "datasets", f"{dataset_type}.yaml"))
        cfg = OmegaConf.merge(cfg, dataset_cfg)

    cfg = OmegaConf.merge(cfg, args_from_cli)
    OmegaConf.register_new_resolver("div", lambda a, b: a / b)
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
    return cfg


def _camera_model_and_params(K: np.ndarray) -> Tuple[str, Tuple[float, ...]]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    if abs(fx - fy) < 1e-10:
        return "SIMPLE_PINHOLE", (fx, cx, cy)
    return "PINHOLE", (fx, fy, cx, cy)


def export_colmap(dataset: DrivingDataset, output_dir: str, image_dir_rel: str, image_ext: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    num_cams = dataset.pixel_source.num_cams
    num_timesteps = dataset.num_img_timesteps
    camera_list = [int(x) for x in dataset.pixel_source.camera_list]

    cameras: Dict[Tuple, int] = {}
    camera_rows = []
    image_rows = []
    next_camera_id = 1
    next_image_id = 1

    for timestep in tqdm(range(num_timesteps), desc="Exporting COLMAP txt"):
        for cam_slot in range(num_cams):
            idx = timestep * num_cams + cam_slot
            _, cam_infos = dataset.full_image_set.get_image(idx=idx, camera_downscale=1.0)

            K = cam_infos["intrinsics"].cpu().numpy()
            c2w = cam_infos["camera_to_world"].cpu().numpy()
            width = int(cam_infos["width"].item())
            height = int(cam_infos["height"].item())

            model_name, params = _camera_model_and_params(K)
            camera_key = (
                model_name,
                width,
                height,
                tuple(np.round(np.asarray(params), decimals=12).tolist()),
            )
            if camera_key not in cameras:
                cameras[camera_key] = next_camera_id
                camera_rows.append((next_camera_id, model_name, width, height, params))
                next_camera_id += 1
            camera_id = cameras[camera_key]

            w2c = np.linalg.inv(c2w)
            rot_cw = w2c[:3, :3]
            trans_cw = w2c[:3, 3]
            quat_xyzw = R.from_matrix(rot_cw).as_quat()  # x, y, z, w
            qw = float(quat_xyzw[3])
            qx = float(quat_xyzw[0])
            qy = float(quat_xyzw[1])
            qz = float(quat_xyzw[2])
            tx, ty, tz = map(float, trans_cw.tolist())

            cam_id_raw = camera_list[cam_slot]
            image_name = f"{image_dir_rel}/{timestep:03d}_{cam_id_raw}.{image_ext}"
            image_rows.append((next_image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name))
            next_image_id += 1

    cameras_path = os.path.join(output_dir, "cameras.txt")
    images_path = os.path.join(output_dir, "images.txt")
    points_path = os.path.join(output_dir, "points3D.txt")

    with open(cameras_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(camera_rows)}\n")
        for camera_id, model_name, width, height, params in camera_rows:
            param_str = " ".join(f"{p:.12f}" for p in params)
            f.write(f"{camera_id} {model_name} {width} {height} {param_str}\n")

    with open(images_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_rows)}\n")
        for row in image_rows:
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name = row
            f.write(
                f"{image_id} {qw:.15f} {qx:.15f} {qy:.15f} {qz:.15f} "
                f"{tx:.15f} {ty:.15f} {tz:.15f} {camera_id} {image_name}\n"
            )
            f.write("\n")

    with open(points_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")

    print(f"Saved COLMAP files to: {output_dir}")
    print(f"  - {cameras_path}")
    print(f"  - {images_path}")
    print(f"  - {points_path}")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_file, args.opts)
    dataset = DrivingDataset(data_cfg=cfg.data)
    export_colmap(
        dataset=dataset,
        output_dir=args.output_dir,
        image_dir_rel=args.image_dir_rel,
        image_ext=args.image_ext,
    )


if __name__ == "__main__":
    main()

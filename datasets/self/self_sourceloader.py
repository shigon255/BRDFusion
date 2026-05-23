import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
import pyexr
import imageio
from omegaconf import OmegaConf
from torch import Tensor

from datasets.base.pixel_source import CameraData, ScenePixelSource
from datasets.dataset_meta import DATASETS_CONFIG

logger = logging.getLogger()


def _quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _load_pose_file(path: str) -> Tuple[List[int], List[np.ndarray]]:
    frame_ids = []
    cam_to_worlds = []
    seen_frame_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            # Some exported pose files may contain duplicated frame rows.
            # Keep first occurrence to preserve a 1:1 frame-to-pose mapping.
            if frame_id in seen_frame_ids:
                continue
            seen_frame_ids.add(frame_id)
            tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:8])
            rot = _quat_to_rotmat(qx, qy, qz, qw)
            cam_to_world = np.eye(4, dtype=np.float32)
            cam_to_world[:3, :3] = rot
            cam_to_world[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
            frame_ids.append(frame_id)
            cam_to_worlds.append(cam_to_world)
    if not frame_ids:
        raise ValueError(f"No pose data found in {path}")
    return frame_ids, cam_to_worlds


class SelfCameraData(CameraData):
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        cam_id: int,
        pose_path: str,
        world_from_scene: np.ndarray,
        start_timestep: int = 0,
        end_timestep: int = None,
        downscale_when_loading: float = 1.0,
        undistort: bool = False,
        buffer_downscale: float = 1.0,
        device: torch.device = torch.device("cpu"),
        load_sky_mask: bool = False,
        load_gt_sky_mask: bool = False,
        gt_sky_mask_dir: str = "gt_sky_mask",
        gt_sky_mask_ext: str = "png",
        load_shadow_mask: bool = False,
        load_priors: bool = True,
        load_materials: bool = True,
        load_gt_intrinsic: bool = False,
        gt_intrinsic_dirs: dict = None,
        gt_intrinsic_exts: dict = None,
        load_relighted_rgb: bool = False,
        relighted_data_path: str = None,
        load_gt_depth: bool = False,
        gt_depth_dir: str = "depth",
        gt_depth_ext: str = "exr",
        gt_depth_naming: str = "frame_camid",
        prior_type: str = "dr",
        prior_dirs: dict = None,
        prior_exts: dict = None,
        load_only_calibrations: bool = False,
        load_images_only: bool = False,
        source_name: str = "primary",
    ):
        self.dataset_name = dataset_name
        self.cam_id = cam_id
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.undistort = undistort
        self.buffer_downscale = buffer_downscale
        self.device = device

        self.cam_name = DATASETS_CONFIG[dataset_name][cam_id]["camera_name"]
        self.original_size = DATASETS_CONFIG[dataset_name][cam_id]["original_size"]
        self.load_size = [
            int(self.original_size[0] / downscale_when_loading),
            int(self.original_size[1] / downscale_when_loading),
        ]

        frame_ids, cam_to_worlds = _load_pose_file(pose_path)
        if self.end_timestep is None:
            self.end_timestep = len(frame_ids)
        frame_ids = frame_ids[self.start_timestep : self.end_timestep]
        cam_to_worlds = cam_to_worlds[self.start_timestep : self.end_timestep]

        self.frame_ids = frame_ids
        self._cam_to_worlds = [
            (world_from_scene @ pose).astype(np.float32) for pose in cam_to_worlds
        ]

        self.load_sky_mask = load_sky_mask
        self.load_gt_sky_mask = load_gt_sky_mask
        self.gt_sky_mask_dir = gt_sky_mask_dir
        self.gt_sky_mask_ext = gt_sky_mask_ext
        self.load_shadow_mask = load_shadow_mask
        self.load_priors = load_priors
        self.load_materials = load_materials
        self.load_gt_intrinsic = load_gt_intrinsic
        self.gt_intrinsic_dirs = gt_intrinsic_dirs or {}
        self.gt_intrinsic_exts = gt_intrinsic_exts or {}
        self.load_relighted_rgb = load_relighted_rgb
        self.relighted_data_path = relighted_data_path
        self.load_gt_depth = load_gt_depth
        # ensure optional mask attributes exist before any `.to()` calls
        self.shadow_masks = None
        self.shading_maps = None
        self.gt_depth_dir = gt_depth_dir
        self.gt_depth_ext = gt_depth_ext
        self.gt_depth_naming = gt_depth_naming
        self.prior_type = prior_type
        self.prior_dirs = prior_dirs or {}
        self.prior_exts = prior_exts or {}
        self.load_only_calibrations = load_only_calibrations
        self.load_images_only = load_images_only
        self.source_name = source_name
        self.external_camera: Optional["SelfCameraData"] = None
        self.active_source = "primary"

        if not self.load_only_calibrations:
            self.create_all_filelist()
        self.load_calibrations()

        if self.load_only_calibrations:
            self.images = None
            self.normals = None
            self.albedos = None
            self.metallics = None
            self.roughnesses = None
            self.mono_depths = None
            self.relighted_images = None
            self.blender_normals = None
            self.blender_albedos = None
            self.blender_metallics = None
            self.blender_roughnesses = None
            self.egocar_mask = None
            self.shadow_masks = None
            self.shading_maps = None
            self.road_masks = None
            self.dynamic_masks = None
            self.human_masks = None
            self.vehicle_masks = None
            self.sky_masks = None
            self.gt_sky_masks = None
            self.lidar_depth_maps = None
            self.image_error_maps = None
        else:
            self.load_images()
            if not self.load_images_only:
                if self.load_relighted_rgb:
                    self.load_relighted_images()
                if self.load_sky_mask:
                    self.load_sky_masks()
                if self.load_gt_sky_mask:
                    self.load_gt_sky_masks()
                if self.load_shadow_mask:
                    self.load_shadow_masks()
                if self.load_priors:
                    self.load_normals()
                    if self.load_materials:
                        self.load_albedos()
                        self.load_metallics()
                        self.load_roughnesses()
                    else:
                        self.albedos = None
                        self.metallics = None
                        self.roughnesses = None
                    self.load_mono_depths()
                if self.load_gt_intrinsic:
                    self.load_blender_intrinsics()
                if self.load_gt_depth:
                    self.load_gt_depths()
            else:
                self.normals = None
                self.albedos = None
                self.metallics = None
                self.roughnesses = None
                self.mono_depths = None
                self.relighted_images = None
                self.blender_normals = None
                self.blender_albedos = None
                self.blender_metallics = None
                self.blender_roughnesses = None
                self.sky_masks = None
                self.gt_sky_masks = None
                self.shadow_masks = None
                self.shading_maps = None
                self.road_masks = None
                self.dynamic_masks = None
                self.human_masks = None
                self.vehicle_masks = None
                self.lidar_depth_maps = None

            if not self.load_priors:
                self.normals = None
                self.albedos = None
                self.metallics = None
                self.roughnesses = None
                self.mono_depths = None
            if not self.load_gt_intrinsic:
                self.blender_normals = None
                self.blender_albedos = None
                self.blender_metallics = None
                self.blender_roughnesses = None
            if not self.load_relighted_rgb:
                self.relighted_images = None
            self.egocar_mask = None
            self.shading_maps = None
            self.road_masks = None
            self.dynamic_masks = None
            self.human_masks = None
            self.vehicle_masks = None
            if not self.load_sky_mask:
                self.sky_masks = None
            if not self.load_gt_sky_mask:
                self.gt_sky_masks = None
            if not self.load_gt_depth:
                self.lidar_depth_maps = None
            self.image_error_maps = None

        self.to(self.device)
        self.downscale_factor = 1.0

    def attach_external_camera(self, external_camera: "SelfCameraData") -> None:
        if len(self.frame_ids) != len(external_camera.frame_ids):
            raise ValueError(
                f"Frame count mismatch between primary ({len(self.frame_ids)}) and "
                f"external ({len(external_camera.frame_ids)}) sources."
            )
        if self.frame_ids != external_camera.frame_ids:
            raise ValueError("Frame id mismatch between primary and external sources.")
        if self.load_size != external_camera.load_size:
            raise ValueError(
                f"Image size mismatch between primary {self.load_size} and external "
                f"{external_camera.load_size}."
            )
        self.external_camera = external_camera

    def set_active_source(self, source: str) -> None:
        if source not in ("primary", "external"):
            raise ValueError(f"Unsupported source '{source}'. Expected 'primary' or 'external'.")
        if source == "external" and self.external_camera is None:
            raise ValueError("Requested external source, but no external camera is attached.")
        self.active_source = source

    def set_downscale_factor(self, downscale_factor: float):
        super().set_downscale_factor(downscale_factor)
        if self.external_camera is not None:
            self.external_camera.set_downscale_factor(downscale_factor)

    def to(self, device: torch.device):
        super().to(device)
        if self.external_camera is not None:
            self.external_camera.to(device)

    def get_image(self, frame_idx: int) -> Dict[str, Tensor]:
        if self.active_source == "external":
            if self.external_camera is None:
                raise ValueError("No external camera attached.")
            return self.external_camera.get_image(frame_idx)
        return super().get_image(frame_idx)

    def create_all_filelist(self):
        def resolve_data_subdir(path_str: str) -> str:
            path_str = os.path.expanduser(os.path.expandvars(path_str))
            if os.path.isabs(path_str):
                return path_str
            return os.path.join(self.data_path, path_str)

        img_filepaths = []
        sky_mask_filepaths = []
        gt_sky_mask_filepaths = []
        shadow_mask_filepaths = []
        normal_filepaths = []
        mono_depth_filepaths = []
        albedo_filepaths = []
        metallic_filepaths = []
        roughness_filepaths = []
        relighted_img_filepaths = []
        blender_normal_filepaths = []
        blender_albedo_filepaths = []
        blender_metallic_filepaths = []
        blender_roughness_filepaths = []
        gt_depth_filepaths = []
        for frame_id in self.frame_ids:
            img_filepaths.append(
                os.path.join(self.data_path, "images", f"{frame_id:03d}_{self.cam_id}.png")
            )
            if self.load_relighted_rgb:
                relighted_img_filepaths.append(
                    os.path.join(self.relighted_data_path, "images", f"{frame_id:03d}_{self.cam_id}.png")
                )
            if self.load_sky_mask:
                sky_mask_filepaths.append(
                    os.path.join(
                        self.data_path,
                        "sky_masks",
                        f"{frame_id:03d}_{self.cam_id}.png",
                    )
                )
            if self.load_gt_sky_mask:
                gt_sky_mask_filepaths.append(
                    os.path.join(
                        self.data_path,
                        self.gt_sky_mask_dir,
                        f"{frame_id:03d}_{self.cam_id}.{self.gt_sky_mask_ext}",
                    )
                )
            if self.load_shadow_mask:
                shadow_mask_filepaths.append(
                    os.path.join(
                        self.data_path,
                        "shadow_masks",
                        f"{frame_id:03d}_{self.cam_id}.png",
                    )
                )
            if self.load_priors:
                if self.prior_type != "dr":
                    raise ValueError(f"Unsupported prior_type for self dataset: {self.prior_type}")
                normal_dir = self.prior_dirs.get("normal", "diffusion_renderer_normal")
                mono_depth_dir = self.prior_dirs.get("mono_depth", "diffusion_renderer_depth")
                albedo_dir = self.prior_dirs.get("albedo", "diffusion_renderer_albedo")
                metallic_dir = self.prior_dirs.get("metallic", "diffusion_renderer_metallic")
                roughness_dir = self.prior_dirs.get("roughness", "diffusion_renderer_roughness")
                normal_ext = self.prior_exts.get("normal", "jpg")
                mono_depth_ext = self.prior_exts.get("mono_depth", "jpg")
                albedo_ext = self.prior_exts.get("albedo", "jpg")
                metallic_ext = self.prior_exts.get("metallic", "jpg")
                roughness_ext = self.prior_exts.get("roughness", "jpg")

                normal_dir_path = resolve_data_subdir(normal_dir)
                mono_depth_dir_path = resolve_data_subdir(mono_depth_dir)
                albedo_dir_path = resolve_data_subdir(albedo_dir)
                metallic_dir_path = resolve_data_subdir(metallic_dir)
                roughness_dir_path = resolve_data_subdir(roughness_dir)

                normal_filepaths.append(
                    os.path.join(
                        normal_dir_path,
                        f"{frame_id:03d}_{self.cam_id}.{normal_ext}",
                    )
                )
                mono_depth_filepaths.append(
                    os.path.join(
                        mono_depth_dir_path,
                        f"{frame_id:03d}_{self.cam_id}.{mono_depth_ext}",
                    )
                )
                if self.load_materials:
                    albedo_filepaths.append(
                        os.path.join(
                            albedo_dir_path,
                            f"{frame_id:03d}_{self.cam_id}.{albedo_ext}",
                        )
                    )
                    metallic_filepaths.append(
                        os.path.join(
                            metallic_dir_path,
                            f"{frame_id:03d}_{self.cam_id}.{metallic_ext}",
                        )
                    )
                    roughness_filepaths.append(
                        os.path.join(
                            roughness_dir_path,
                            f"{frame_id:03d}_{self.cam_id}.{roughness_ext}",
                        )
                    )
            if self.load_gt_depth:
                depth_name = f"{frame_id:03d}_{self.cam_id}.{self.gt_depth_ext}"
                gt_depth_dir_path = resolve_data_subdir(self.gt_depth_dir)
                gt_depth_filepaths.append(
                    os.path.join(gt_depth_dir_path, depth_name)
                )
            if self.load_gt_intrinsic:
                normal_dir = resolve_data_subdir(self.gt_intrinsic_dirs.get("normal", "normal"))
                albedo_dir = resolve_data_subdir(self.gt_intrinsic_dirs.get("albedo", "albedo"))
                metallic_dir = resolve_data_subdir(self.gt_intrinsic_dirs.get("metallic", "metallic"))
                roughness_dir = resolve_data_subdir(self.gt_intrinsic_dirs.get("roughness", "roughness"))
                normal_ext = self.gt_intrinsic_exts.get("normal", "exr")
                albedo_ext = self.gt_intrinsic_exts.get("albedo", "exr")
                metallic_ext = self.gt_intrinsic_exts.get("metallic", "exr")
                roughness_ext = self.gt_intrinsic_exts.get("roughness", "exr")

                blender_normal_filepaths.append(
                    os.path.join(normal_dir, f"{frame_id:03d}_{self.cam_id}.{normal_ext}")
                )
                blender_albedo_filepaths.append(
                    os.path.join(albedo_dir, f"{frame_id:03d}_{self.cam_id}.{albedo_ext}")
                )
                blender_metallic_filepaths.append(
                    os.path.join(metallic_dir, f"{frame_id:03d}_{self.cam_id}.{metallic_ext}")
                )
                blender_roughness_filepaths.append(
                    os.path.join(roughness_dir, f"{frame_id:03d}_{self.cam_id}.{roughness_ext}")
                )
        self.img_filepaths = np.array(img_filepaths)
        if self.load_sky_mask:
            self.sky_mask_filepaths = np.array(sky_mask_filepaths)
        if self.load_gt_sky_mask:
            self.gt_sky_mask_filepaths = np.array(gt_sky_mask_filepaths)
        if self.load_shadow_mask:
            self.shadow_mask_filepaths = np.array(shadow_mask_filepaths)
        if self.load_priors:
            self.normal_filepaths = np.array(normal_filepaths)
            self.mono_depth_filepaths = np.array(mono_depth_filepaths)
            if self.load_materials:
                self.albedo_filepaths = np.array(albedo_filepaths)
                self.metallic_filepaths = np.array(metallic_filepaths)
                self.roughness_filepaths = np.array(roughness_filepaths)
        if self.load_gt_depth:
            self.gt_depth_filepaths = np.array(gt_depth_filepaths)
        if self.load_relighted_rgb:
            self.relighted_img_filepaths = np.array(relighted_img_filepaths)
        if self.load_gt_intrinsic:
            self.blender_normal_filepaths = np.array(blender_normal_filepaths)
            self.blender_albedo_filepaths = np.array(blender_albedo_filepaths)
            self.blender_metallic_filepaths = np.array(blender_metallic_filepaths)
            self.blender_roughness_filepaths = np.array(blender_roughness_filepaths)

    def load_calibrations(self):
        intrinsic = np.loadtxt(
            os.path.join(self.data_path, "intrinsics", f"{self.cam_name}.txt")
        )
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
        fx, fy = (
            fx * self.load_size[1] / self.original_size[1],
            fy * self.load_size[0] / self.original_size[0],
        )
        cx, cy = (
            cx * self.load_size[1] / self.original_size[1],
            cy * self.load_size[0] / self.original_size[0],
        )
        _intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        _distortions = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

        intrinsics = [_intrinsics for _ in self._cam_to_worlds]
        distortions = [_distortions for _ in self._cam_to_worlds]

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(self._cam_to_worlds, axis=0)).float()
    
    def load_gt_depths(self):
        gt_depths = []

        from scipy.ndimage import zoom

        def resize_depth_scipy(depth, new_h, new_w, order=1):
            assert depth.ndim == 2, "Depth must be (H, W)"
            h, w = depth.shape
            zoom_factors = (new_h / h, new_w / w)
            return zoom(depth, zoom_factors, order=order)

        for ix, fname in enumerate(self.gt_depth_filepaths):
            depth = pyexr.read(fname).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            assert depth.ndim == 2, "Depth must be (H, W)"
            if depth.shape[0] != self.load_size[0] or depth.shape[1] != self.load_size[1]:
                depth = resize_depth_scipy(depth, self.load_size[0], self.load_size[1], order=1)
            if self.undistort:
                if ix == 0:
                    print("undistorting GT depth")
                depth = cv2.undistort(
                    depth,
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            # map invalid/sky depth to 0 so depth loss ignores it
            depth = np.where(depth < 0, 0.0, depth)
            gt_depths.append(depth)
        self.lidar_depth_maps = torch.from_numpy(np.stack(gt_depths, axis=0)).to(torch.float32)

    def load_relighted_images(self):
        relighted_images = []
        for ix, fname in enumerate(self.relighted_img_filepaths):
            rgb = imageio.imread(fname)
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            if rgb.shape[0] != self.load_size[0] or rgb.shape[1] != self.load_size[1]:
                rgb = cv2.resize(
                    rgb,
                    (self.load_size[1], self.load_size[0]),
                    interpolation=cv2.INTER_AREA,
                )
            if self.undistort:
                rgb = cv2.undistort(
                    rgb,
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            relighted_images.append(rgb)
        self.relighted_images = torch.from_numpy(np.stack(relighted_images, axis=0)).to(torch.float32) / 255.0

    def _load_scalar_or_rgb(self, fname: str) -> np.ndarray:
        if fname.endswith(".exr"):
            arr = pyexr.read(fname).astype(np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            return arr
        arr = imageio.imread(fname).astype(np.float32)
        if arr.ndim == 2:
            return arr / 255.0
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr / 255.0

    def _resize_map(self, arr: np.ndarray, is_scalar: bool) -> np.ndarray:
        if arr.shape[0] == self.load_size[0] and arr.shape[1] == self.load_size[1]:
            return arr
        if is_scalar:
            return cv2.resize(arr, (self.load_size[1], self.load_size[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(arr, (self.load_size[1], self.load_size[0]), interpolation=cv2.INTER_LINEAR)

    def load_blender_intrinsics(self):
        blender_normals = []
        blender_albedos = []
        blender_metallics = []
        blender_roughnesses = []

        for ix, fname in enumerate(self.blender_normal_filepaths):
            normal = pyexr.read(fname).astype(np.float32)
            if normal.ndim == 2:
                normal = np.stack([normal] * 3, axis=-1)
            if normal.shape[-1] > 3:
                normal = normal[..., :3]
            # Convention alignment: flip GT normal z-axis at load time.
            normal[..., 2] = -normal[..., 2]
            normal = self._resize_map(normal, is_scalar=False)
            if self.undistort:
                normal = cv2.undistort(normal, self.intrinsics[ix].numpy(), self.distortions[ix].numpy())
            blender_normals.append(normal)

        for ix, fname in enumerate(self.blender_albedo_filepaths):
            albedo = self._load_scalar_or_rgb(fname)
            if albedo.ndim == 2:
                albedo = np.stack([albedo] * 3, axis=-1)
            if albedo.shape[-1] > 3:
                albedo = albedo[..., :3]
            albedo = self._resize_map(albedo, is_scalar=False)
            if self.undistort:
                albedo = cv2.undistort(albedo, self.intrinsics[ix].numpy(), self.distortions[ix].numpy())
            blender_albedos.append(albedo)

        for ix, fname in enumerate(self.blender_metallic_filepaths):
            metallic = self._load_scalar_or_rgb(fname)
            if metallic.ndim == 3:
                metallic = metallic[..., :3].mean(axis=-1)
            metallic = self._resize_map(metallic, is_scalar=True)
            if self.undistort:
                metallic = cv2.undistort(metallic, self.intrinsics[ix].numpy(), self.distortions[ix].numpy())
            blender_metallics.append(metallic)

        for ix, fname in enumerate(self.blender_roughness_filepaths):
            roughness = self._load_scalar_or_rgb(fname)
            if roughness.ndim == 3:
                roughness = roughness[..., :3].mean(axis=-1)
            roughness = self._resize_map(roughness, is_scalar=True)
            if self.undistort:
                roughness = cv2.undistort(roughness, self.intrinsics[ix].numpy(), self.distortions[ix].numpy())
            blender_roughnesses.append(roughness)

        self.blender_normals = torch.from_numpy(np.stack(blender_normals, axis=0)).to(torch.float32)
        self.blender_albedos = torch.from_numpy(np.stack(blender_albedos, axis=0)).to(torch.float32)
        self.blender_metallics = torch.from_numpy(np.stack(blender_metallics, axis=0)).to(torch.float32)
        self.blender_roughnesses = torch.from_numpy(np.stack(blender_roughnesses, axis=0)).to(torch.float32)

    def load_gt_sky_masks(self):
        gt_sky_masks = []
        for ix, fname in enumerate(self.gt_sky_mask_filepaths):
            gt_sky_mask = imageio.imread(fname)
            if gt_sky_mask.ndim == 3:
                gt_sky_mask = gt_sky_mask[..., 0]
            if gt_sky_mask.shape[0] != self.load_size[0] or gt_sky_mask.shape[1] != self.load_size[1]:
                gt_sky_mask = cv2.resize(
                    gt_sky_mask,
                    (self.load_size[1], self.load_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            if self.undistort:
                if ix == 0:
                    print("undistorting GT sky mask")
                gt_sky_mask = cv2.undistort(
                    np.array(gt_sky_mask),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            gt_sky_masks.append(np.array(gt_sky_mask) > 0)
        self.gt_sky_masks = torch.from_numpy(np.stack(gt_sky_masks, axis=0)).float()

    def load_shadow_masks(self):
        shadow_masks = []
        for ix, fname in enumerate(self.shadow_mask_filepaths):
            shadow_mask = imageio.imread(fname)
            if shadow_mask.ndim == 2:
                shadow_mask = np.stack([shadow_mask] * 3, axis=-1)
            if shadow_mask.shape[-1] == 4:
                shadow_mask = shadow_mask[..., :3]
            if shadow_mask.shape[0] != self.load_size[0] or shadow_mask.shape[1] != self.load_size[1]:
                shadow_mask = cv2.resize(
                    shadow_mask,
                    (self.load_size[1], self.load_size[0]),
                    interpolation=cv2.INTER_AREA,
                )
            if self.undistort:
                if ix == 0:
                    print("undistorting shadow_mask")
                shadow_mask = cv2.undistort(
                    shadow_mask,
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            shadow_masks.append(shadow_mask)
        self.shadow_masks = torch.from_numpy(np.stack(shadow_masks, axis=0)).to(torch.float32) / 255.0


class SelfPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.camera_data = {}
        self.load_data()

    def load_cameras(self):
        self._timesteps = torch.arange(self.start_timestep, self.end_timestep)
        self.register_normalized_timestamps()
        external_cfg = self.data_cfg.get("external_source", None)
        use_external_source = bool(external_cfg is not None and external_cfg.get("enable", False))
        active_source = "primary"
        if use_external_source:
            active_source = external_cfg.get("active_source", "primary")
            if active_source not in ("primary", "external"):
                raise ValueError(
                    f"Unsupported external_source.active_source='{active_source}'. "
                    "Expected 'primary' or 'external'."
                )

        def _resolve_scene_path(data_root: str, scene_idx) -> str:
            try:
                return os.path.join(data_root, f"{int(scene_idx):03d}")
            except Exception:
                return os.path.join(data_root, str(scene_idx))

        load_relighted_rgb = self.data_cfg.get("load_relighted_rgb", False)
        relighted_scene_idx = self.data_cfg.get("relighted_scene_idx", None)
        relighted_data_path = None
        if load_relighted_rgb:
            if relighted_scene_idx is None:
                raise ValueError("load_relighted_rgb=True requires data.pixel_source.relighted_scene_idx")
            relighted_data_path = os.path.join(os.path.dirname(self.data_path), str(relighted_scene_idx))
            if not os.path.exists(relighted_data_path):
                raise FileNotFoundError(f"Relighted scene path not found: {relighted_data_path}")

        external_data_path = None
        external_relighted_data_path = None
        if use_external_source:
            external_data_root = external_cfg.get("data_root", None)
            external_scene_idx = external_cfg.get("scene_idx", None)
            if external_data_root is None or external_scene_idx is None:
                raise ValueError(
                    "external_source.enable=True requires both external_source.data_root and "
                    "external_source.scene_idx."
                )
            external_data_path = _resolve_scene_path(external_data_root, external_scene_idx)
            if not os.path.exists(external_data_path):
                raise FileNotFoundError(f"External scene path not found: {external_data_path}")
            if load_relighted_rgb:
                external_relighted_scene_idx = external_cfg.get("relighted_scene_idx", relighted_scene_idx)
                if external_relighted_scene_idx is None:
                    raise ValueError(
                        "load_relighted_rgb=True with external source requires either "
                        "external_source.relighted_scene_idx or data.pixel_source.relighted_scene_idx."
                    )
                external_relighted_data_path = os.path.join(
                    os.path.dirname(external_data_path), str(external_relighted_scene_idx)
                )
                if not os.path.exists(external_relighted_data_path):
                    raise FileNotFoundError(
                        f"External relighted scene path not found: {external_relighted_data_path}"
                    )

        ref_cam_name = DATASETS_CONFIG[self.dataset_name][self.camera_list[0]]["camera_name"]
        ref_pose_path = os.path.join(self.data_path, f"{ref_cam_name}_poses.txt")
        ref_frame_ids, ref_poses = _load_pose_file(ref_pose_path)
        ref_pose = ref_poses[0]
        world_from_scene = np.linalg.inv(ref_pose)
        logger.info("Using first pose from %s as world origin", ref_cam_name)

        for idx, cam_id in enumerate(self.camera_list):
            cam_name = DATASETS_CONFIG[self.dataset_name][cam_id]["camera_name"]
            pose_path = os.path.join(self.data_path, f"{cam_name}_poses.txt")
            logger.info("Loading camera %s", cam_name)
            camera = SelfCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                pose_path=pose_path,
                world_from_scene=world_from_scene,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                undistort=self.data_cfg.undistort,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
                load_sky_mask=self.data_cfg.load_sky_mask,
                load_gt_sky_mask=self.data_cfg.get("load_gt_sky_mask", False),
                gt_sky_mask_dir=self.data_cfg.get("gt_sky_mask_dir", "gt_sky_mask"),
                gt_sky_mask_ext=self.data_cfg.get("gt_sky_mask_ext", "png"),
                load_shadow_mask=self.data_cfg.get("load_shadow_mask", False),
                load_priors=self.data_cfg.get("load_priors", True),
                load_materials=self.data_cfg.get("load_materials", True),
                load_gt_intrinsic=self.data_cfg.get("load_gt_intrinsic", False),
                gt_intrinsic_dirs=self.data_cfg.get("gt_intrinsic_dirs", None),
                gt_intrinsic_exts=self.data_cfg.get("gt_intrinsic_exts", None),
                load_relighted_rgb=load_relighted_rgb,
                relighted_data_path=relighted_data_path,
                load_gt_depth=self.data_cfg.get("load_gt_depth", False),
                gt_depth_dir=self.data_cfg.get("gt_depth_dir", "depth"),
                gt_depth_ext=self.data_cfg.get("gt_depth_ext", "exr"),
                gt_depth_naming=self.data_cfg.get("gt_depth_naming", "frame_camid"),
                prior_type=self.data_cfg.get("prior_type", "dr"),
                prior_dirs=self.data_cfg.get("prior_dirs", None),
                prior_exts=self.data_cfg.get("prior_exts", None),
                load_only_calibrations=self.data_cfg.get("load_only_calibrations", False),
                load_images_only=self.data_cfg.get("load_images_only", False),
                source_name="primary",
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + idx
            camera.set_unique_ids(
                unique_cam_idx=idx,
                unique_img_idx=unique_img_idx,
            )
            if use_external_source:
                external_cam_name = DATASETS_CONFIG[self.dataset_name][cam_id]["camera_name"]
                external_pose_path = os.path.join(external_data_path, f"{external_cam_name}_poses.txt")
                external_camera = SelfCameraData(
                    dataset_name=self.dataset_name,
                    data_path=external_data_path,
                    cam_id=cam_id,
                    pose_path=external_pose_path,
                    world_from_scene=world_from_scene,
                    start_timestep=self.start_timestep,
                    end_timestep=self.end_timestep,
                    downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                    undistort=self.data_cfg.undistort,
                    buffer_downscale=self.buffer_downscale,
                    device=self.device,
                    load_sky_mask=self.data_cfg.load_sky_mask,
                    load_gt_sky_mask=self.data_cfg.get("load_gt_sky_mask", False),
                    gt_sky_mask_dir=self.data_cfg.get("gt_sky_mask_dir", "gt_sky_mask"),
                    gt_sky_mask_ext=self.data_cfg.get("gt_sky_mask_ext", "png"),
                    load_shadow_mask=self.data_cfg.get("load_shadow_mask", False),
                    load_priors=self.data_cfg.get("load_priors", True),
                    load_materials=self.data_cfg.get("load_materials", True),
                    load_gt_intrinsic=self.data_cfg.get("load_gt_intrinsic", False),
                    gt_intrinsic_dirs=self.data_cfg.get("gt_intrinsic_dirs", None),
                    gt_intrinsic_exts=self.data_cfg.get("gt_intrinsic_exts", None),
                    load_relighted_rgb=load_relighted_rgb,
                    relighted_data_path=external_relighted_data_path,
                    load_gt_depth=self.data_cfg.get("load_gt_depth", False),
                    gt_depth_dir=self.data_cfg.get("gt_depth_dir", "depth"),
                    gt_depth_ext=self.data_cfg.get("gt_depth_ext", "exr"),
                    gt_depth_naming=self.data_cfg.get("gt_depth_naming", "frame_camid"),
                    prior_type=self.data_cfg.get("prior_type", "dr"),
                    prior_dirs=self.data_cfg.get("prior_dirs", None),
                    prior_exts=self.data_cfg.get("prior_exts", None),
                    load_only_calibrations=self.data_cfg.get("load_only_calibrations", False),
                    load_images_only=self.data_cfg.get("load_images_only", False),
                    source_name="external",
                )
                external_camera.load_time(self.normalized_time)
                external_camera.set_unique_ids(
                    unique_cam_idx=idx,
                    unique_img_idx=unique_img_idx,
                )
                camera.attach_external_camera(external_camera)
                camera.set_active_source(active_source)
            else:
                camera.set_active_source("primary")
            self.camera_data[cam_id] = camera

        if use_external_source:
            logger.info(
                "SelfPixelSource external_source enabled: active_source=%s, primary=%s, external=%s",
                active_source,
                self.data_path,
                external_data_path,
            )

    def load_objects(self) -> None:
        num_frames = len(self._timesteps)
        self.instances_pose = torch.zeros((num_frames, 0, 4, 4), device=self.device)
        self.instances_size = torch.zeros((0, 3), device=self.device)
        self.instances_true_id = torch.zeros((0,), device=self.device)
        self.instances_model_types = torch.zeros((0,), device=self.device)
        self.per_frame_instance_mask = torch.zeros((num_frames, 0), device=self.device)
        self.smpl_human_all = {}

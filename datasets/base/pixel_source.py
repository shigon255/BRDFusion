from typing import Dict, Tuple, List, Callable
from omegaconf import OmegaConf
import os
import abc
import cv2
import random
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
import pyexr

import torch
from torch import Tensor
from datasets.dataset_meta import DATASETS_CONFIG

logger = logging.getLogger()

def idx_to_3d(idx, H, W):
    """
    Converts a 1D index to a 3D index (img_id, row_id, col_id)

    Args:
        idx (int): The 1D index to convert.
        H (int): The height of the 3D space.
        W (int): The width of the 3D space.

    Returns:
        tuple: A tuple containing the 3D index (i, j, k),
                where i is the image index, j is the row index,
                and k is the column index.
    """
    i = idx // (H * W)
    j = (idx % (H * W)) // W
    k = idx % W
    return i, j, k


def get_rays(
    x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        x: the horizontal coordinates of the pixels, shape: (num_rays,)
        y: the vertical coordinates of the pixels, shape: (num_rays,)
        c2w: the camera-to-world matrices, shape: (num_cams, 4, 4)
        intrinsic: the camera intrinsic matrices, shape: (num_cams, 3, 3)
    Returns:
        origins: the ray origins, shape: (num_rays, 3)
        viewdirs: the ray directions, shape: (num_rays, 3)
        direction_norm: the norm of the ray directions, shape: (num_rays, 1)
    """
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    if len(c2w.shape) == 2:
        c2w = c2w[None, :, :]
    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
                (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [num_rays, 3]

    # rotate the camera rays w.r.t. the camera pose
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    # TODO: not sure if we still need direction_norm
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    # normalize the ray directions
    viewdirs = directions / (direction_norm + 1e-8)
    return origins, viewdirs, direction_norm

def sparse_lidar_map_downsampler(lidar_depth_map, downscale_factor):
    # NOTE(ziyu): Important! 
    # This is the correct way to downsample the sparse lidar depth map.  
    raw_avg = torch.nn.functional.interpolate(
        lidar_depth_map.unsqueeze(0).unsqueeze(0),
        scale_factor=downscale_factor,
        mode="area",
    ).squeeze(0).squeeze(0)
    raw_mask = torch.nn.functional.interpolate(
        (lidar_depth_map > 1e-3).float().unsqueeze(0).unsqueeze(0),
        scale_factor=downscale_factor,
        mode="area",
    ).squeeze(0).squeeze(0)
    downsampled_lidar_map = torch.zeros_like(raw_avg)
    downsampled_lidar_map[raw_mask > 0] = raw_avg[raw_mask > 0] / raw_mask[raw_mask > 0]
    return downsampled_lidar_map

class CameraData(object):
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        cam_id: int,
        # the start timestep to load
        start_timestep: int = 0,
        # the end timestep to load
        end_timestep: int = None,
        # whether to load the dynamic masks
        load_dynamic_mask: bool = False,
        # whether to load the sky masks
        load_sky_mask: bool = False,
        # whether to load the shading maps,
        load_shading_map: bool = False,
        # whether to load the shadow masks,
        load_shadow_mask: bool = False,
        # whether to load the road masks
        load_road_mask: bool = False,
        # the size to load the images
        downscale_when_loading: float = 1.0,
        # whether to undistort the images
        undistort: bool = False,
        # whether to use buffer sampling
        buffer_downscale: float = 1.0,
        # the device to move the camera to
        device: torch.device = torch.device("cpu"),
        # the prior used for the camera. Currently supports "rgbx" and "dr" (diffusion renderer)
        prior_type: str = "dr",
        # optional overrides for prior folder names/extensions
        prior_dirs: dict = None,
        prior_exts: dict = None,
        # whether to load only calibrations (skip images, normals, masks, etc.)
        load_only_calibrations: bool = False,
        # whether to load materials (albedo, metallic, roughness)
        load_materials: bool = True,
        # whether to load only images (skip normals, mono_depths, materials, masks)
        load_images_only: bool = False,
    ):
        self.dataset_name = dataset_name
        self.cam_id = cam_id
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.undistort = undistort
        self.buffer_downscale = buffer_downscale
        self.device = device
        self.prior_dirs = prior_dirs or {}
        self.prior_exts = prior_exts or {}
        
        self.cam_name = DATASETS_CONFIG[dataset_name][cam_id]["camera_name"]
        self.original_size = DATASETS_CONFIG[dataset_name][cam_id]["original_size"]
        self.load_size = [
            int(self.original_size[0] / downscale_when_loading),
            int(self.original_size[1] / downscale_when_loading),
        ]

        # Load calibrations (always loaded)
        if not load_only_calibrations:
            # Load the images, dynamic masks, sky masks, etc.
            self.create_all_filelist(prior_type=prior_type)
        self.load_calibrations()

        # Skip all image/mask/depth loading if only calibrations are needed
        if not load_only_calibrations:
            self.load_images()

            # If load_images_only is True, skip all priors (normals, mono_depths, materials, masks)
            if load_images_only:
                self.normals = None
                self.albedos = None
                self.metallics = None
                self.roughnesses = None
                self.mono_depths = None
                self.egocar_mask = None
                self.shading_maps = None
                self.shadow_masks = None
                self.road_masks = None
                self.dynamic_masks = None
                self.human_masks = None
                self.vehicle_masks = None
                self.sky_masks = None
                self.lidar_depth_maps = None
                self.image_error_maps = None
            else:
                self.load_normals()
                if load_materials:
                    self.load_albedos()
                    self.load_metallics()
                    self.load_roughnesses()
                else:
                    self.albedos = None
                    self.metallics = None
                    self.roughnesses = None
                self.load_mono_depths()
                self.load_egocar_mask()
                self.shading_maps = None
                if load_shading_map:
                    self.load_shading_maps()
                self.shadow_masks = None
                if load_shadow_mask:
                    self.load_shadow_masks()
                self.road_masks = None
                if load_road_mask:
                    self.load_road_masks()
                self.dynamic_masks = None
                self.human_masks = None
                self.vehicle_masks = None
                if load_dynamic_mask:
                    self.load_dynamic_masks()
                self.sky_masks = None
                if load_sky_mask:
                    self.load_sky_masks()
                self.lidar_depth_maps = None # will be loaded by: self.load_depth()
                self.image_error_maps = None # will be built by: self.build_image_error_buffer()
        else:
            # Initialize empty placeholders when only loading calibrations
            self.images = None
            self.normals = None
            self.albedos = None
            self.metallics = None
            self.roughnesses = None
            self.mono_depths = None
            self.egocar_mask = None
            self.shading_maps = None
            self.shadow_masks = None
            self.road_masks = None
            self.dynamic_masks = None
            self.human_masks = None
            self.vehicle_masks = None
            self.sky_masks = None
            self.lidar_depth_maps = None
            self.image_error_maps = None

        self.to(self.device)
        self.downscale_factor = 1.0
        
    @property
    def num_frames(self) -> int:
        return self.cam_to_worlds.shape[0]
    
    @property
    def HEIGHT(self) -> int:
        return self.load_size[0]

    @property
    def WIDTH(self) -> int:
        return self.load_size[1]
    
    def __len__(self):
        return self.num_frames
    
    def set_downscale_factor(self, downscale_factor: float):
        self.downscale_factor = downscale_factor
        
    def set_unique_ids(self, unique_cam_idx: int, unique_img_idx: Tensor):
        """
        unique id is the compact order of the camera index and frame index
        for example camera idx is [0, 2, 4]
        the camera index is [0, 1, 2]
        """
        self.unique_cam_idx = unique_cam_idx
        self.unique_img_idx = unique_img_idx.to(self.device)
        
    def load_calibrations(
        self,
        unique_img_idx,
        cam_to_worlds,
        intrinsics,
        distortions,
    ):
        self.unique_img_idx = unique_img_idx
        # Load the camera intrinsics, extrinsics, timestamps, etc.
        # Compute the camera-to-world matrices, ego-to-world matrices, etc.
        self.cam_to_worlds = cam_to_worlds # (num_frames, 4, 4)
        self.intrinsics = intrinsics # (num_frames, 3, 3)
        self.distortions = distortions # (num_frames, 5)

    def create_all_filelist(self, prior_type="dr"):
        """
        Create file lists for all data files.
        e.g., img files, feature files, etc.
        """
        def resolve_data_subdir(path_str: str) -> str:
            path_str = os.path.expanduser(os.path.expandvars(path_str))
            if os.path.isabs(path_str):
                return path_str
            return os.path.join(self.data_path, path_str)

        # ---- define filepaths ---- #
        img_filepaths = []
        dynamic_mask_filepaths, sky_mask_filepaths = [], []
        human_mask_filepaths, vehicle_mask_filepaths = [], []
        normal_filepaths = []
        mono_depth_filepaths = []
        albedo_filepaths = []
        metallic_filepaths = []
        roughness_filepaths = []
        shading_map_filepaths = []
        shadow_mask_filepaths = []
        road_mask_filepaths = []
        
        fine_mask_path = os.path.join(self.data_path, "fine_dynamic_masks")
        if os.path.exists(fine_mask_path):
            dynamic_mask_dir = "fine_dynamic_masks"
            logger.info("Using fine dynamic masks")
        else:
            dynamic_mask_dir = "dynamic_masks"
            logger.info("Using coarse dynamic masks")

        # Note: we assume all the files in waymo dataset are synchronized
        print(f"Using prior type: {prior_type}")
        for t in range(self.start_timestep, self.end_timestep):
            img_filepaths.append(
                os.path.join(self.data_path, "images", f"{t:03d}_{self.cam_id}.jpg")
            )
            dynamic_mask_filepaths.append(
                os.path.join(
                    self.data_path, dynamic_mask_dir, "all", f"{t:03d}_{self.cam_id}.png"
                )
            )
            human_mask_filepaths.append(
                os.path.join(
                    self.data_path, dynamic_mask_dir, "human", f"{t:03d}_{self.cam_id}.png"
                )
            )
            vehicle_mask_filepaths.append(
                os.path.join(
                    self.data_path, dynamic_mask_dir, "vehicle", f"{t:03d}_{self.cam_id}.png"
                )
            )
            sky_mask_filepaths.append(
                os.path.join(self.data_path, "sky_masks", f"{t:03d}_{self.cam_id}.png")
            )
            shading_map_dir = "shading_maps"
            shadow_mask_dir = os.path.join(self.data_path, "shadow_masks")
            shadow_mask_ext = "png"
            if os.path.exists(shadow_mask_dir):
                probe_png = os.path.join(shadow_mask_dir, f"{self.start_timestep:03d}_{self.cam_id}.png")
                probe_jpg = os.path.join(shadow_mask_dir, f"{self.start_timestep:03d}_{self.cam_id}.jpg")
                if not os.path.exists(probe_png) and os.path.exists(probe_jpg):
                    shadow_mask_ext = "jpg"
            shading_map_filepaths.append(
                os.path.join(self.data_path, shading_map_dir, f"{t:03d}_{self.cam_id}.jpg")
            )
            shadow_mask_filepaths.append(
                os.path.join(self.data_path, "shadow_masks", f"{t:03d}_{self.cam_id}.{shadow_mask_ext}")
            )
            road_mask_filepaths.append(
                os.path.join(self.data_path, "road_masks", f"{t:03d}_{self.cam_id}.png")
            )
            
            
            # allow overriding prior folder names/extensions from config
            prior_dirs = self.prior_dirs
            prior_exts = self.prior_exts

            # mono_depth_folder = 'pi3_depth'
            # mono_depth_ext = 'npy'

            mono_depth_folder = prior_dirs.get("mono_depth", "diffusion_renderer_depth")
            mono_depth_ext = prior_exts.get("mono_depth", "jpg")

            # mono_depth_folder = 'depth_vipe'
            # mono_depth_ext = 'exr'
            
            mono_depth_dir_path = resolve_data_subdir(mono_depth_folder)
            mono_depth_filepaths.append(
                os.path.join(mono_depth_dir_path, f"{t:03d}_{self.cam_id}.{mono_depth_ext}")
            )
        
            if prior_type == "rgbx":
                normal_dir = prior_dirs.get("normal", "normals")
                albedo_dir = prior_dirs.get("albedo", "albedos")
                metallic_dir = prior_dirs.get("metallic", "metallics")
                roughness_dir = prior_dirs.get("roughness", "roughnesses")
                normal_ext = prior_exts.get("normal", "jpg")
                albedo_ext = prior_exts.get("albedo", "jpg")
                metallic_ext = prior_exts.get("metallic", "jpg")
                roughness_ext = prior_exts.get("roughness", "jpg")
            elif prior_type == "dr":
                normal_dir = prior_dirs.get("normal", "diffusion_renderer_normal")
                albedo_dir = prior_dirs.get("albedo", "diffusion_renderer_albedo")
                metallic_dir = prior_dirs.get("metallic", "diffusion_renderer_metallic")
                roughness_dir = prior_dirs.get("roughness", "diffusion_renderer_roughness")
                normal_ext = prior_exts.get("normal", "jpg")
                albedo_ext = prior_exts.get("albedo", "jpg")
                metallic_ext = prior_exts.get("metallic", "jpg")
                roughness_ext = prior_exts.get("roughness", "jpg")
            else:
                raise ValueError(f"Unknown prior type: {prior_type}. Supported types are 'rgbx' and 'dr'.")

            normal_dir_path = resolve_data_subdir(normal_dir)
            albedo_dir_path = resolve_data_subdir(albedo_dir)
            metallic_dir_path = resolve_data_subdir(metallic_dir)
            roughness_dir_path = resolve_data_subdir(roughness_dir)

            normal_filepaths.append(
                os.path.join(normal_dir_path, f"{t:03d}_{self.cam_id}.{normal_ext}")
            )
            albedo_filepaths.append(
                os.path.join(albedo_dir_path, f"{t:03d}_{self.cam_id}.{albedo_ext}")
            )
            metallic_filepaths.append(
                os.path.join(metallic_dir_path, f"{t:03d}_{self.cam_id}.{metallic_ext}")
            )
            roughness_filepaths.append(
                os.path.join(roughness_dir_path, f"{t:03d}_{self.cam_id}.{roughness_ext}")
            )
            
        self.img_filepaths = np.array(img_filepaths)
        self.dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
        self.human_mask_filepaths = np.array(human_mask_filepaths)
        self.vehicle_mask_filepaths = np.array(vehicle_mask_filepaths)
        self.sky_mask_filepaths = np.array(sky_mask_filepaths)
        self.shading_map_filepaths = np.array(shading_map_filepaths)
        self.shadow_mask_filepaths = np.array(shadow_mask_filepaths)
        self.road_mask_filepaths = np.array(road_mask_filepaths)
        self.normal_filepaths = np.array(normal_filepaths)
        self.mono_depth_filepaths = np.array(mono_depth_filepaths)
        self.albedo_filepaths = np.array(albedo_filepaths)
        self.metallic_filepaths = np.array(metallic_filepaths)
        self.roughness_filepaths = np.array(roughness_filepaths)
        
        
    def load_images(self):
        images = []
        for ix, fname in tqdm(
            enumerate(self.img_filepaths),
            desc="Loading images",
            dynamic_ncols=True,
            total=len(self.img_filepaths),
        ):
            rgb = Image.open(fname).convert("RGB")
            # resize them to the load_size
            rgb = rgb.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            # undistort the images
            if self.undistort:
                if ix == 0:
                    print("undistorting rgb")
                rgb = cv2.undistort(
                    np.array(rgb),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            images.append(rgb)
        # normalize the images to [0, 1]
        self.images = images = torch.from_numpy(np.stack(images, axis=0)) / 255
    
    def load_normals(self):
        normals = []
        for ix, fname in tqdm(
            enumerate(self.normal_filepaths),
            desc="Loading normals",
            dynamic_ncols=True,
            total=len(self.normal_filepaths),
        ):
            normal = Image.open(fname).convert("RGB")
            # resize them to the load_size
            normal = normal.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            # undistort the images
            if self.undistort:
                if ix == 0:
                    print("undistorting normal")
                normal = cv2.undistort(
                    np.array(normal),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            normals.append(normal)
        # normalize the images to [0, 1]
        self.normals = normals = torch.from_numpy(np.stack(normals, axis=0)) / 255
        # renormalize the normal to [-1, 1]
        normals = normals * 2.0 - 1.0
        self.normals = self.normals * 2.0 - 1.0
        # normalize to unit length
        self.normals = torch.nn.functional.normalize(
            self.normals, dim=-1, eps=1e-8
        )
    
    def load_mono_depths(self):        
        mono_depths = []
        
        from scipy.ndimage import zoom
        import imageio
        def resize_depth_scipy(depth, new_h, new_w, order=1):
            assert depth.ndim == 2, "Depth must be (H, W)"
            h, w = depth.shape
            zoom_factors = (new_h / h, new_w / w)
            return zoom(depth, zoom_factors, order=order)

        
        for ix, fname in tqdm(
            enumerate(self.mono_depth_filepaths),
            desc="Loading monocular depths",
            dynamic_ncols=True,
            total=len(self.mono_depth_filepaths),
        ):
            if fname.endswith("npy"):
                mono_depth = np.load(fname) # (H, W), in meters
                assert len(mono_depth.shape) == 2, "Depth must be (H, W)"
                assert mono_depth.dtype == np.float32 or mono_depth.dtype == np.float64, "Depth must be float32 or float64"
                mono_depth = resize_depth_scipy(mono_depth, self.load_size[0], self.load_size[1], order=1)
            elif fname.endswith("jpg"):
                mono_depth = imageio.imread(fname)
                w, h = self.load_size[1], self.load_size[0]
                mono_depth = np.array(Image.fromarray(mono_depth).resize((w, h), resample=Image.BILINEAR)).astype(np.float32) / 255.0
                mono_depth = np.mean(mono_depth, axis=-1)
            elif fname.endswith("exr"):
                mono_depth = pyexr.read(fname).astype(np.float32)
                mono_depth = mono_depth.squeeze(-1)
                assert len(mono_depth.shape) == 2, "Depth must be (H, W)"
                if mono_depth.shape[0] != self.load_size[0] or mono_depth.shape[1] != self.load_size[1]:
                    mono_depth = resize_depth_scipy(mono_depth, self.load_size[0], self.load_size[1], order=1)
            
            if self.undistort:
                if ix == 0:
                    print("undistorting monocular depth")
                mono_depth = cv2.undistort(
                    mono_depth,
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            mono_depths.append(mono_depth)
        self.mono_depths = mono_depths = torch.from_numpy(np.stack(mono_depths, axis=0)).to(torch.float32)
                
    
    # NOTE: albedo from Diffusion Renderer is in sRGB space
    def load_albedos(self):
        albedos = []
        for ix, fname in tqdm(
            enumerate(self.albedo_filepaths),
            desc="Loading images",
            dynamic_ncols=True,
            total=len(self.albedo_filepaths),
        ):
            albedo = Image.open(fname).convert("RGB")
            # resize them to the load_size
            albedo = albedo.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            # undistort the images
            if self.undistort:
                if ix == 0:
                    print("undistorting albedo")
                albedo = cv2.undistort(
                    np.array(albedo),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            albedos.append(albedo)
        # normalize the images to [0, 1]
        self.albedos = albedos = torch.from_numpy(np.stack(albedos, axis=0)) / 255
    
    def load_metallics(self):
        metallics = []
        for ix, fname in tqdm(
            enumerate(self.metallic_filepaths),
            desc="Loading images",
            dynamic_ncols=True,
            total=len(self.metallic_filepaths),
        ):
            metallic = Image.open(fname).convert("RGB")
            # resize them to the load_size
            metallic = metallic.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            # undistort the images
            if self.undistort:
                if ix == 0:
                    print("undistorting metallic")
                metallic = cv2.undistort(
                    np.array(metallic),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            metallics.append(metallic)
        # normalize the images to [0, 1]
        self.metallics = metallics = torch.from_numpy(np.stack(metallics, axis=0)).to(torch.float32).mean(dim=-1) / 255
    
    def load_roughnesses(self):
        roughnesses = []
        for ix, fname in tqdm(
            enumerate(self.roughness_filepaths),
            desc="Loading images",
            dynamic_ncols=True,
            total=len(self.roughness_filepaths),
        ):
            roughness = Image.open(fname).convert("RGB")
            # resize them to the load_size
            roughness = roughness.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            # undistort the images
            if self.undistort:
                if ix == 0:
                    print("undistorting roughness")
                roughness = cv2.undistort(
                    np.array(roughness),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            roughnesses.append(roughness)
        # normalize the images to [0, 1]
        self.roughnesses = roughnesses = torch.from_numpy(np.stack(roughnesses, axis=0)).to(torch.float32).mean(dim=-1) / 255
    
    # NOTE: shading maps (shading) is in sRGB space
    def load_shading_maps(self):
        shading_maps = []
        for ix, fname in tqdm(
            enumerate(self.shading_map_filepaths),
            desc="Loading images",
            dynamic_ncols=True,
            total=len(self.shading_map_filepaths),
        ):
            shading_map = Image.open(fname).convert("RGB")
            # resize them to the load_size
            shading_map = shading_map.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            # undistort the images
            if self.undistort:
                if ix == 0:
                    print("undistorting shading_map")
                shading_map = cv2.undistort(
                    np.array(shading_map),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            shading_maps.append(shading_map)
        # normalize the images to [0, 1] (float mask with 3-channel)
        self.shading_maps = shading_maps = torch.from_numpy(np.stack(shading_maps, axis=0)).to(torch.float32) / 255

    def load_shadow_masks(self):
        shadow_masks = []
        for ix, fname in tqdm(
            enumerate(self.shadow_mask_filepaths),
            desc="Loading shadow masks",
            dynamic_ncols=True,
            total=len(self.shadow_mask_filepaths),
        ):
            shadow_mask = Image.open(fname).convert("L")
            # resize them to the load_size
            shadow_mask = shadow_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            # undistort the images
            if self.undistort:
                if ix == 0:
                    print("undistorting shadow_mask")
                shadow_mask = cv2.undistort(
                    np.array(shadow_mask),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            shadow_masks.append(shadow_mask)
        # normalize the images to [0, 1] (float mask with 1-channel)
        self.shadow_masks = shadow_masks = torch.from_numpy(np.stack(shadow_masks, axis=0)).to(torch.float32) / 255
    
    def load_road_masks(self):
        road_masks = []
        for ix, fname in tqdm(
            enumerate(self.road_mask_filepaths),
            desc="Loading road masks",
            dynamic_ncols=True,
            total=len(self.road_mask_filepaths),
        ):  
            road_mask = Image.open(fname).convert("L")
            # resize them to the load_size
            road_mask = road_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            if self.undistort:
                if ix == 0:
                    print("undistorting road mask")
                road_mask = cv2.undistort(
                    np.array(road_mask),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            road_masks.append(np.array(road_mask) > 0)
        self.road_masks = torch.from_numpy(np.stack(road_masks, axis=0)).float()
    
    def load_egocar_mask(self):
        """
        Since in some datasets, the ego car body is visible in the images,
        we need to load the ego car mask to mask out the ego car body.
        """
        egocar_mask = os.path.join("data", "ego_masks", self.dataset_name, f"{self.cam_id}.png")
        if os.path.exists(egocar_mask):
            egocar_mask = Image.open(egocar_mask).convert("L")
            # resize them to the load_size
            egocar_mask = egocar_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            if self.undistort:
                egocar_mask = cv2.undistort(
                    np.array(egocar_mask),
                    self.intrinsics[0].numpy(),
                    self.distortions[0].numpy(),
                )
            self.egocar_mask = torch.from_numpy(np.array(egocar_mask) > 0).float()
        else:
            self.egocar_mask = None
        
    def load_dynamic_masks(self):
        dynamic_masks = []
        for ix, fname in tqdm(
            enumerate(self.dynamic_mask_filepaths),
            desc="Loading dynamic masks",
            dynamic_ncols=True,
            total=len(self.dynamic_mask_filepaths),
        ):  
            dyn_mask = Image.open(fname).convert("L")
            # resize them to the load_size
            dyn_mask = dyn_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            if self.undistort:
                if ix == 0:
                    print("undistorting dynamic mask")
                dyn_mask = cv2.undistort(
                    np.array(dyn_mask),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            dynamic_masks.append(np.array(dyn_mask) > 0)
        self.dynamic_masks = torch.from_numpy(np.stack(dynamic_masks, axis=0)).float()
        
        human_masks = []
        for ix, fname in tqdm(
            enumerate(self.human_mask_filepaths),
            desc="Loading human masks",
            dynamic_ncols=True,
            total=len(self.human_mask_filepaths),
        ):  
            human_mask = Image.open(fname).convert("L")
            # resize them to the load_size
            human_mask = human_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            if self.undistort:
                if ix == 0:
                    print("undistorting human mask")
                human_mask = cv2.undistort(
                    np.array(human_mask),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            human_masks.append(np.array(human_mask) > 0)
        self.human_masks = torch.from_numpy(np.stack(human_masks, axis=0)).float()
        
        vehicle_masks = []
        for ix, fname in tqdm(
            enumerate(self.vehicle_mask_filepaths),
            desc="Loading vehicle masks",
            dynamic_ncols=True,
            total=len(self.vehicle_mask_filepaths),
        ):  
            vehicle_mask = Image.open(fname).convert("L")
            # resize them to the load_size
            vehicle_mask = vehicle_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.BILINEAR
            )
            if self.undistort:
                if ix == 0:
                    print("undistorting vehicle mask")
                vehicle_mask = cv2.undistort(
                    np.array(vehicle_mask),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            vehicle_masks.append(np.array(vehicle_mask) > 0)
        self.vehicle_masks = torch.from_numpy(np.stack(vehicle_masks, axis=0)).float()
        
    def load_sky_masks(self):
        sky_masks = []
        for ix, fname in tqdm(
            enumerate(self.sky_mask_filepaths),
            desc="Loading sky masks",
            dynamic_ncols=True,
            total=len(self.sky_mask_filepaths),
        ):
            sky_mask = Image.open(fname).convert("L")
            # resize them to the load_size
            sky_mask = sky_mask.resize(
                (self.load_size[1], self.load_size[0]), Image.NEAREST
            )
            if self.undistort:
                if ix == 0:
                    print("undistorting sky mask")
                sky_mask = cv2.undistort(
                    np.array(sky_mask),
                    self.intrinsics[ix].numpy(),
                    self.distortions[ix].numpy(),
                )
            sky_masks.append(np.array(sky_mask) > 0)
        self.sky_masks = torch.from_numpy(np.stack(sky_masks, axis=0)).float()
        
    def load_depth(
        self,
        lidar_depth_maps: Tensor,
    ):
        self.lidar_depth_maps = lidar_depth_maps.to(self.device)
        
    def load_time(
        self,
        normalized_time: Tensor,
    ):
        self.normalized_time = normalized_time.to(self.device)
        
    def build_image_error_buffer(self) -> None:
        """
        Build the image error buffer.
        """
        # Tensor (num_frames, H // buffer_downscale, W // buffer_downscale)
        self.image_error_maps = torch.ones(
            (
                self.num_frames,
                self.HEIGHT // self.buffer_downscale,
                self.WIDTH // self.buffer_downscale,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        
    def get_image_error_video(self) -> List[np.ndarray]:
        """
        Get the pixel sample weights video.
        Returns:
            frames: the pixel sample weights video.
                shape: (num_frames, H, W, 3)
        """
        # normalize the image error buffer to [0, 1]
        image_error_maps = (
            self.image_error_maps - self.image_error_maps.min()
        ) / (self.image_error_maps.max() - self.image_error_maps.min())
        
        maps = []
        loss_maps = (
            image_error_maps.detach()
            .cpu()
            .numpy()
            .reshape(
                self.num_frames,
                self.HEIGHT // self.buffer_downscale,
                self.WIDTH // self.buffer_downscale,
            )
        )
        for i in range(self.num_frames):
            maps.append(loss_maps[i])
        return maps
    
    def update_image_error_maps(self, render_results: Dict[str, Tensor]) -> None:
        """
        Update the image error buffer with the given render results.
        """
        gt_rgbs = render_results["gt_rgbs"]
        pred_rgbs = render_results["rgbs"]
        gt_rgbs = torch.from_numpy(np.stack(gt_rgbs, axis=0))
        pred_rgbs = torch.from_numpy(np.stack(pred_rgbs, axis=0))
        image_error_maps = torch.abs(gt_rgbs - pred_rgbs).mean(dim=-1)
        assert image_error_maps.shape == self.image_error_maps.shape
        if "Dynamic_opacities" in render_results:
            if len(render_results["Dynamic_opacities"]) > 0:
                dynamic_opacity = render_results["Dynamic_opacities"]
                dynamic_opacity = torch.from_numpy(np.stack(dynamic_opacity, axis=0))
                # we prioritize the dynamic objects by multiplying the error by 5
                image_error_maps[dynamic_opacity > 0.1] *= 5
        # update the image error buffer
        self.image_error_maps: Tensor = image_error_maps.to(self.device)
        logger.info(f"Updated image error buffer for camera {self.cam_id}.")

    def to(self, device: torch.device):
        """
        Move the camera to the given device.
        Args:
            device: the device to move the camera to.
        """
        self.cam_to_worlds = self.cam_to_worlds.to(device)
        self.intrinsics = self.intrinsics.to(device)
        if self.distortions is not None:
            self.distortions = self.distortions.to(device)
        if self.images is not None:
            self.images = self.images.to(device)
        if self.egocar_mask is not None:
            self.egocar_mask = self.egocar_mask.to(device)
        if self.dynamic_masks is not None:
            self.dynamic_masks = self.dynamic_masks.to(device)
        if self.human_masks is not None:
            self.human_masks = self.human_masks.to(device)
        if self.vehicle_masks is not None:
            self.vehicle_masks = self.vehicle_masks.to(device)
        if self.sky_masks is not None:
            self.sky_masks = self.sky_masks.to(device)
        if getattr(self, "gt_sky_masks", None) is not None:
            self.gt_sky_masks = self.gt_sky_masks.to(device)
        if self.shading_maps is not None:
            self.shading_maps = self.shading_maps.to(device)
        if self.shadow_masks is not None:
            self.shadow_masks = self.shadow_masks.to(device)
        if self.road_masks is not None:
            self.road_masks = self.road_masks.to(device)
        if self.lidar_depth_maps is not None:
            self.lidar_depth_maps = self.lidar_depth_maps.to(device)
        if self.image_error_maps is not None:
            self.image_error_maps = self.image_error_maps.to(device)
        if self.mono_depths is not None:
            self.mono_depths = self.mono_depths.to(device)
        if self.normals is not None:
            self.normals = self.normals.to(device)
        if self.albedos is not None:
            self.albedos = self.albedos.to(device)
        if self.metallics is not None:
            self.metallics = self.metallics.to(device)
        if self.roughnesses is not None:
            self.roughnesses = self.roughnesses.to(device)
        if getattr(self, "relighted_images", None) is not None:
            self.relighted_images = self.relighted_images.to(device)
        if getattr(self, "blender_normals", None) is not None:
            self.blender_normals = self.blender_normals.to(device)
        if getattr(self, "blender_albedos", None) is not None:
            self.blender_albedos = self.blender_albedos.to(device)
        if getattr(self, "blender_metallics", None) is not None:
            self.blender_metallics = self.blender_metallics.to(device)
        if getattr(self, "blender_roughnesses", None) is not None:
            self.blender_roughnesses = self.blender_roughnesses.to(device)
    
    def get_image(self, frame_idx: int) -> Dict[str, Tensor]:
        """
        Get the rays for rendering the given frame index.
        Args:
            frame_idx: the frame index.
        Returns:
            a dict containing the rays for rendering the given frame index.
        """
        rgb, sky_mask = None, None
        gt_sky_mask = None
        dynamic_mask, human_mask, vehicle_mask = None, None, None
        pixel_coords, normalized_time = None, None
        mono_depth = None
        normal = None
        egocar_mask = None
        shading_map = None
        shadow_mask = None
        road_mask = None
        albedo, metallic, roughness = None, None, None
        blender_normal, blender_albedo = None, None
        blender_metallic, blender_roughness = None, None
        relighted_pixels = None
        
        # in case images are not loaded
        if self.downscale_factor != 1.0:
            img_width = int(self.WIDTH * self.downscale_factor)
            img_height = int(self.HEIGHT * self.downscale_factor)
        else:
            img_width = self.WIDTH
            img_height = self.HEIGHT
        
        if self.images is not None:
            rgb = self.images[frame_idx]
            if self.downscale_factor != 1.0:
                rgb = (
                    torch.nn.functional.interpolate(
                        rgb.unsqueeze(0).permute(0, 3, 1, 2),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                img_height, img_width = rgb.shape[:2]
            else:
                img_height, img_width = self.HEIGHT, self.WIDTH

        if self.mono_depths is not None:
            mono_depth = self.mono_depths[frame_idx]
            if self.downscale_factor != 1.0:
                mono_depth = (
                    torch.nn.functional.interpolate(
                        mono_depth.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

        if self.normals is not None:
            normal = self.normals[frame_idx]
            if self.downscale_factor != 1.0:
                normal = (
                    torch.nn.functional.interpolate(
                        normal.unsqueeze(0).permute(0, 3, 1, 2),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
        
        if self.albedos is not None:
            albedo = self.albedos[frame_idx]
            if self.downscale_factor != 1.0:
                albedo = (
                    torch.nn.functional.interpolate(
                        albedo.unsqueeze(0).permute(0, 3, 1, 2),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
        
        if self.metallics is not None:
            metallic = self.metallics[frame_idx]
            if self.downscale_factor != 1.0:
                metallic = (
                    torch.nn.functional.interpolate(
                        metallic.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
        
        if self.roughnesses is not None:
            roughness = self.roughnesses[frame_idx]
            if self.downscale_factor != 1.0:
                roughness = (
                    torch.nn.functional.interpolate(
                        roughness.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

        relighted_images_all = getattr(self, "relighted_images", None)
        if relighted_images_all is not None:
            relighted_pixels = relighted_images_all[frame_idx]
            if self.downscale_factor != 1.0:
                relighted_pixels = (
                    torch.nn.functional.interpolate(
                        relighted_pixels.unsqueeze(0).permute(0, 3, 1, 2),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
        
        blender_normals_all = getattr(self, "blender_normals", None)
        if blender_normals_all is not None:
            blender_normal = blender_normals_all[frame_idx]
            if self.downscale_factor != 1.0:
                blender_normal = (
                    torch.nn.functional.interpolate(
                        blender_normal.unsqueeze(0).permute(0, 3, 1, 2),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
        
        blender_albedos_all = getattr(self, "blender_albedos", None)
        if blender_albedos_all is not None:
            blender_albedo = blender_albedos_all[frame_idx]
            if self.downscale_factor != 1.0:
                blender_albedo = (
                    torch.nn.functional.interpolate(
                        blender_albedo.unsqueeze(0).permute(0, 3, 1, 2),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
        
        blender_metallics_all = getattr(self, "blender_metallics", None)
        if blender_metallics_all is not None:
            blender_metallic = blender_metallics_all[frame_idx]
            if self.downscale_factor != 1.0:
                blender_metallic = (
                    torch.nn.functional.interpolate(
                        blender_metallic.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
        
        blender_roughnesses_all = getattr(self, "blender_roughnesses", None)
        if blender_roughnesses_all is not None:
            blender_roughness = blender_roughnesses_all[frame_idx]
            if self.downscale_factor != 1.0:
                blender_roughness = (
                    torch.nn.functional.interpolate(
                        blender_roughness.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

        if self.shading_maps is not None:
            shading_map = self.shading_maps[frame_idx]
            if self.downscale_factor != 1.0:
                shading_map = (
                    torch.nn.functional.interpolate(
                        shading_map.unsqueeze(0).permute(0, 3, 1, 2),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
        if self.shadow_masks is not None:
            shadow_mask = self.shadow_masks[frame_idx]
            if self.downscale_factor != 1.0:
                shadow_mask = (
                    torch.nn.functional.interpolate(
                        shadow_mask.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

        x, y = torch.meshgrid(
            torch.arange(img_width),
            torch.arange(img_height),
            indexing="xy",
        )
        x, y = x.flatten(), y.flatten()
        x, y = x.to(self.device), y.to(self.device)
        # pixel coordinates
        pixel_coords = (
            torch.stack([y / img_height, x / img_width], dim=-1)
            .float()
            .reshape(img_height, img_width, 2)
        )
        if self.egocar_mask is not None:
            egocar_mask = self.egocar_mask
            if self.downscale_factor != 1.0:
                egocar_mask = (
                    torch.nn.functional.interpolate(
                        egocar_mask.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
        
        if self.sky_masks is not None:
            sky_mask = self.sky_masks[frame_idx]
            if self.downscale_factor != 1.0:
                sky_mask = (
                    torch.nn.functional.interpolate(
                        sky_mask.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

        if getattr(self, "gt_sky_masks", None) is not None:
            gt_sky_mask = self.gt_sky_masks[frame_idx]
            if self.downscale_factor != 1.0:
                gt_sky_mask = (
                    torch.nn.functional.interpolate(
                        gt_sky_mask.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
        
        if self.road_masks is not None:
            road_mask = self.road_masks[frame_idx]
            if self.downscale_factor != 1.0:
                road_mask = (
                    torch.nn.functional.interpolate(
                        road_mask.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
        
        if self.dynamic_masks is not None:
            dynamic_mask = self.dynamic_masks[frame_idx]
            if self.downscale_factor != 1.0:
                dynamic_mask = (
                    torch.nn.functional.interpolate(
                        dynamic_mask.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
        if self.human_masks is not None:
            human_mask = self.human_masks[frame_idx]
            if self.downscale_factor != 1.0:
                human_mask = (
                    torch.nn.functional.interpolate(
                        human_mask.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
        if self.vehicle_masks is not None:
            vehicle_mask = self.vehicle_masks[frame_idx]
            if self.downscale_factor != 1.0:
                vehicle_mask = (
                    torch.nn.functional.interpolate(
                        vehicle_mask.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            
        lidar_depth_map = None
        if self.lidar_depth_maps is not None:
            lidar_depth_map = self.lidar_depth_maps[frame_idx]
            if self.downscale_factor != 1.0:
                # BUG: cannot use, need futher investigation
                # if self.data_cfg.denser_lidar_times > 1:
                #     lidar_supervision_downsample = 1 / self.data_cfg.denser_lidar_times * self.downscale_factor
                #     lidar_depth_map = sparse_lidar_map_downsampler(lidar_depth_map, lidar_supervision_downsample)
                    
                #     # resize back
                #     lidar_depth_map = (
                #         torch.nn.functional.interpolate(
                #             lidar_depth_map.unsqueeze(0).unsqueeze(0),
                #             scale_factor=self.data_cfg.denser_lidar_times,
                #             mode="bicubic",
                #         )
                #         .squeeze(0)
                #         .squeeze(0)
                #     )
                # else:
                lidar_depth_map = sparse_lidar_map_downsampler(lidar_depth_map, self.downscale_factor)

        if self.normalized_time is not None:
            normalized_time = torch.full(
                (img_height, img_width),
                self.normalized_time[frame_idx],
                dtype=torch.float32,
            )
        camera_id = torch.full(
            (img_height, img_width),
            self.cam_id,
            dtype=torch.long,
        )
        image_id = torch.full(
            (img_height, img_width),
            self.unique_img_idx[frame_idx],
            dtype=torch.long,
        )
        frame_id = torch.full(
            (img_height, img_width),
            frame_idx,
            dtype=torch.long,
        )
        c2w = self.cam_to_worlds[frame_idx]
        intrinsics = self.intrinsics[frame_idx] * self.downscale_factor
        intrinsics[2, 2] = 1.0
        origins, viewdirs, direction_norm = get_rays(x, y, c2w, intrinsics)
        origins = origins.reshape(img_height, img_width, 3)
        viewdirs = viewdirs.reshape(img_height, img_width, 3)
        direction_norm = direction_norm.reshape(img_height, img_width, 1)
        _image_infos = {
            "origins": origins,
            "viewdirs": viewdirs,
            "direction_norm": direction_norm,
            "pixel_coords": pixel_coords,
            "normed_time": normalized_time,
            "img_idx": image_id,
            "frame_idx": frame_id,
            "pixels": rgb,
            "sky_masks": sky_mask,
            "gt_sky_masks": gt_sky_mask,
            "dynamic_masks": dynamic_mask,
            "human_masks": human_mask,
            "vehicle_masks": vehicle_mask,
            "egocar_masks": egocar_mask,
            "shading_maps": shading_map,
            "shadow_masks": shadow_mask,
            "road_masks": road_mask,
            "lidar_depth_map": lidar_depth_map,
            "mono_depths": mono_depth,
            "normals": normal,
            "albedos": albedo,
            "metallics": metallic,
            "roughnesses": roughness,
            "blender_normal": blender_normal,
            "blender_albedo": blender_albedo,
            "blender_metallic": blender_metallic,
            "blender_roughness": blender_roughness,
            "relighted_pixels": relighted_pixels,
        }
        image_infos = {k: v for k, v in _image_infos.items() if v is not None}
        
        cam_infos = {
            "cam_id": camera_id,
            "cam_name": self.cam_name,
            "camera_to_world": c2w,
            "height": torch.tensor(img_height, dtype=torch.long, device=c2w.device),
            "width": torch.tensor(img_width, dtype=torch.long, device=c2w.device),
            "intrinsics": intrinsics,
        }
        return image_infos, cam_infos

class ScenePixelSource(abc.ABC):
    """
    The base class for all pixel sources of a scene.
    """
    # define a transformation matrix to convert the opencv camera coordinate system to the dataset camera coordinate system
    data_cfg: OmegaConf = None
    # the dataset name, choose from ["waymo", "kitti", "nuscenes", "pandaset", "argoverse"]
    dataset_name: str = None
    # the dict of camera data
    camera_data: Dict[int, CameraData] = {}
    # the normalized time of all images (normalized to [0, 1]), shape: (num_frames,)
    _normalized_time: Tensor = None
    # timestep indices of frames, shape: (num_frames,)
    _timesteps: Tensor = None
    # image error buffer, (num_images, )
    image_error_buffer: Tensor = None
    # whether the buffer is computed
    image_error_buffered: bool = False
    # the downscale factor of the error buffer
    buffer_downscale: float = 1.0
    # per-image ray error cache for ray-level sampling, shape: (num_imgs, H_train, W_train)
    ray_error_maps: Tensor = None
    
    # -------------- object annotations
    # (num_frame, num_instances, 4, 4)
    instances_pose: Tensor = None
    # (num_instances, 3) 
    instances_size: Tensor = None
    # (num_instances, )
    instances_true_id: Tensor = None
    # (num_instances, )
    instances_model_types: Tensor = None
    # (num_frame, num_instances)
    per_frame_instance_mask: Tensor = None

    def __init__(
        self, dataset_name, pixel_data_config: OmegaConf, device: torch.device = torch.device("cpu")
    ) -> None:
        # hold the config of the pixel data
        self.dataset_name = dataset_name
        self.data_cfg = pixel_data_config
        self.device = device
        self._downscale_factor = 1 / pixel_data_config.downscale
        self._old_downscale_factor = []

    @abc.abstractmethod
    def load_cameras(self) -> None:
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Load the images, dynamic masks, sky masks, etc.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_objects(self) -> None:
        """
        Load the object annotations.
        """
        raise NotImplementedError

    def load_data(self) -> None:
        """
        A general function to load all data.
        """
        self.load_cameras()
        self.build_image_error_buffer()
        logger.info("[Pixel] All Pixel Data loaded.")
        
        if self.data_cfg.load_objects:
            self.load_objects()
            logger.info("[Pixel] All Object Annotations loaded.")
        
        # set initial downscale factor
        for cam_id in self.camera_list:
            self.camera_data[cam_id].set_downscale_factor(self._downscale_factor)

    def to(self, device: torch.device) -> "ScenePixelSource":
        """
        Move the dataset to the given device.
        Args:
            device: the device to move the dataset to.
        """
        self.device = device
        if self._timesteps is not None:
            self._timesteps = self._timesteps.to(device)
        if self._normalized_time is not None:
            self._normalized_time = self._normalized_time.to(device)
        if self.instances_pose is not None:
            self.instances_pose = self.instances_pose.to(device)
        if self.instances_size is not None:
            self.instances_size = self.instances_size.to(device)
        if self.per_frame_instance_mask is not None:
            self.per_frame_instance_mask = self.per_frame_instance_mask.to(device)
        if self.instances_model_types is not None:
            self.instances_model_types = self.instances_model_types.to(device)
        if self.ray_error_maps is not None:
            self.ray_error_maps = self.ray_error_maps.to(device)
        return self

    def get_aabb(self) -> Tensor:
        """
        Returns:
            aabb_min, aabb_max: the min and max of the axis-aligned bounding box of the scene
        Note:
            We compute the coarse aabb by using the front camera positions / trajectories. We then
            extend this aabb by 40 meters along horizontal directions and 20 meters up and 5 meters
            down along vertical directions.
        """
        front_camera_trajectory = self.front_camera_trajectory

        # compute the aabb
        aabb_min = front_camera_trajectory.min(dim=0)[0]
        aabb_max = front_camera_trajectory.max(dim=0)[0]

        # extend aabb by 40 meters along forward direction and 40 meters along the left/right direction
        # aabb direction: x, y, z: front, left, up
        aabb_max[0] += 40
        aabb_max[1] += 40
        # when the car is driving uphills
        aabb_max[2] = min(aabb_max[2] + 20, 20)

        # for waymo, there will be a lot of waste of space because we don't have images in the back,
        # it's more reasonable to extend the aabb only by a small amount, e.g., 5 meters
        # we use 40 meters here for a more general case
        aabb_min[0] -= 40
        aabb_min[1] -= 40
        # when a car is driving downhills
        aabb_min[2] = max(aabb_min[2] - 5, -5)
        aabb = torch.tensor([*aabb_min, *aabb_max])
        logger.info(f"[Pixel] Auto AABB from camera: {aabb}")
        return aabb
    
    @property
    def front_camera_trajectory(self) -> Tensor:
        """
        Returns:
            the front camera trajectory.
        """
        front_camera = self.camera_data[0]
        assert (
            front_camera.cam_to_worlds is not None
        ), "Camera poses not loaded, cannot compute front camera trajectory."
        return front_camera.cam_to_worlds[:, :3, 3]
    
    def parse_img_idx(self, img_idx: int) -> Tuple[int, int]:
        """
        Parse the image index to the camera index and frame index.
        Args:
            img_idx: the image index.
        Returns:
            cam_idx: the camera index.
            frame_idx: the frame index.
        """
        unique_cam_idx = img_idx % self.num_cams
        frame_idx = img_idx // self.num_cams
        return unique_cam_idx, frame_idx

    def get_image(self, img_idx: int) -> Dict[str, Tensor]:
        """
        Get the rays for rendering the given image index.
        Args:
            img_idx: the image index.
        Returns:
            a dict containing the rays for rendering the given image index.
        """
        unique_cam_idx, frame_idx = self.parse_img_idx(img_idx)
        for cam_id in self.camera_list:
            if unique_cam_idx == self.camera_data[cam_id].unique_cam_idx:
                return self.camera_data[cam_id].get_image(frame_idx)

    @property
    def camera_list(self) -> List[int]:
        """
        Returns:
            the list of camera indices
        """
        return self.data_cfg.cameras
    
    @property
    def num_cams(self) -> int:
        """
        Returns:
            the number of cameras in the dataset
        """
        return len(self.data_cfg.cameras)
    
    @property
    def num_frames(self) -> int:
        """
        Returns:
            the number of frames in the dataset
        """
        return len(self._timesteps)
    
    @property
    def num_timesteps(self) -> int:
        """
        Returns:
            the number of image timesteps in the dataset
        """
        return len(self._timesteps)

    @property
    def num_imgs(self) -> int:
        """
        Returns:
            the number of images in the dataset
        """
        return self.num_cams * self.num_frames

    @property
    def timesteps(self) -> Tensor:
        """
        Returns:
            the integer timestep indices of all images,
            shape: (num_imgs,)
        Note:
            the difference between timestamps and timesteps is that
            timestamps are the actual timestamps (minus 1e9) of images
            while timesteps are the integer timestep indices of images.
        """
        return self._timesteps

    @property
    def normalized_time(self) -> Tensor:
        """
        Returns:
            the normalized timestamps of all images
            (normalized to the range [0, 1]),
            shape: (num_imgs,)
        """
        return self._normalized_time

    def register_normalized_timestamps(self) -> None:
        # normalized timestamps are between 0 and 1
        normalized_time = (self._timesteps - self._timesteps.min()) / (
            self._timesteps.max() - self._timesteps.min()
        )
        
        if self._timesteps.max() == self._timesteps.min():
            logger.warning(
                "[Pixel] All timesteps are the same, normalized time will be all zeros."
            )
            normalized_time = torch.zeros_like(self._timesteps, dtype=torch.float32)
        
        self._normalized_time = normalized_time.to(self.device)
        self._unique_normalized_timestamps = self._normalized_time.unique()

    def find_closest_timestep(self, normed_timestamp: float) -> int:
        """
        Find the closest timestep to the given timestamp.
        Args:
            normed_timestamp: the normalized timestamp to find the closest timestep for.
        Returns:
            the closest timestep to the given timestamp.
        """
        return torch.argmin(
            torch.abs(self._normalized_time - normed_timestamp)
        )
    
    def propose_training_image(
        self,
        candidate_indices: Tensor = None,
    ) -> Dict[str, Tensor]:
        if random.random() < self.buffer_ratio and self.image_error_buffered:
            # sample according to the image error buffer
            image_mean_error = self.image_error_buffer[candidate_indices]
            start_enhance_weight = self.data_cfg.sampler.get('start_enhance_weight', 1)
            if start_enhance_weight > 1:
                # increase the error of the first 10% frames
                frame_num = int(self.num_imgs / self.num_cams)
                error_weight = torch.cat((
                    torch.linspace(start_enhance_weight, 1, int(frame_num * 0.1)),
                    torch.ones(frame_num - int(frame_num * 0.1))
                ))
                error_weight = error_weight[..., None].repeat(1, self.num_cams).reshape(-1)
                error_weight = error_weight[candidate_indices].to(self.device)
                
                image_mean_error = image_mean_error * error_weight
            idx = torch.multinomial(
                image_mean_error, 1, replacement=False
            ).item()
            img_idx = candidate_indices[idx]
        else:
            # random sample one from candidate_indices
            img_idx = random.choice(candidate_indices)
            
        return img_idx
        
    def build_image_error_buffer(self) -> None:
        """
        Build the image error buffer.
        """
        if self.buffer_ratio > 0:
            for cam_id in self.camera_list:
                self.camera_data[cam_id].build_image_error_buffer()
        else:
            logger.info("Not building image error buffer because buffer_ratio <= 0.")

    def ensure_ray_error_maps(self, height: int, width: int) -> None:
        """
        Ensure ray-level error maps exist at the current training resolution.
        The cache is resized when training resolution changes.
        """
        if self.ray_error_maps is None:
            self.ray_error_maps = torch.ones(
                (self.num_imgs, height, width),
                dtype=torch.float32,
                device=self.device,
            )
            logger.info(f"Initialized ray error maps with shape {tuple(self.ray_error_maps.shape)}")
            return

        if self.ray_error_maps.shape[1] == height and self.ray_error_maps.shape[2] == width:
            return

        self.ray_error_maps = torch.nn.functional.interpolate(
            self.ray_error_maps.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        logger.info(f"Resized ray error maps to shape {tuple(self.ray_error_maps.shape)}")

    def get_ray_error_map(self, img_idx: int) -> Tensor:
        """
        Get ray-level error map for one image index.
        """
        if self.ray_error_maps is None:
            raise RuntimeError("Ray error maps are not initialized. Call ensure_ray_error_maps first.")
        return self.ray_error_maps[img_idx]

    @torch.no_grad()
    def bootstrap_ray_error_maps_from_render(
        self,
        render_results: Dict[str, Tensor],
        img_indices: List[int],
        use_pbr: bool = True,
        min_error: float = 1e-6,
    ) -> None:
        """
        Initialize ray-level error maps from a full render pass.
        """
        if "gt_rgbs" not in render_results:
            raise KeyError("render_results must include 'gt_rgbs' for ray error bootstrap.")

        pred_key = "pbr_colors" if use_pbr and "pbr_colors" in render_results else "rgbs"
        if pred_key not in render_results:
            raise KeyError(f"render_results must include '{pred_key}' for ray error bootstrap.")

        gt_rgbs = torch.from_numpy(np.stack(render_results["gt_rgbs"], axis=0)).float().to(self.device)
        pred_rgbs = torch.from_numpy(np.stack(render_results[pred_key], axis=0)).float().to(self.device)
        if gt_rgbs.shape != pred_rgbs.shape:
            raise ValueError(
                f"Shape mismatch for bootstrap: gt={tuple(gt_rgbs.shape)}, pred={tuple(pred_rgbs.shape)}"
            )
        if gt_rgbs.shape[0] != len(img_indices):
            raise ValueError(
                f"Bootstrap img count mismatch: rendered={gt_rgbs.shape[0]}, img_indices={len(img_indices)}"
            )

        error_maps = torch.abs(gt_rgbs - pred_rgbs).mean(dim=-1).clamp_min(min_error)
        _, h, w = error_maps.shape
        self.ensure_ray_error_maps(height=h, width=w)
        index_tensor = torch.as_tensor(img_indices, dtype=torch.long, device=self.device)
        self.ray_error_maps[index_tensor] = error_maps

    @torch.no_grad()
    def update_ray_error_map(
        self,
        image_infos: Dict[str, Tensor],
        outputs: Dict[str, Tensor],
        ema_decay: float = 0.9,
        min_error: float = 1e-6,
    ) -> None:
        """
        Update ray-level PBR error map for the current training image.
        Only sampled PBR pixels (outputs['pbr_mask']) are updated.
        """
        if "pbr_color" not in outputs or "pixels" not in image_infos:
            return
        if "img_idx" not in image_infos:
            return

        gt_rgb = image_infos["pixels"]
        pred_rgb = outputs["pbr_color"]
        if gt_rgb is None or pred_rgb is None:
            return

        height, width = gt_rgb.shape[:2]
        self.ensure_ray_error_maps(height=height, width=width)
        img_idx = int(image_infos["img_idx"].flatten()[0].item())

        error_map = torch.abs(gt_rgb - pred_rgb).mean(dim=-1)
        update_mask = outputs.get("pbr_mask", None)
        if update_mask is None:
            update_mask = torch.ones_like(error_map, dtype=torch.bool)
        else:
            update_mask = update_mask.bool()

        cached = self.ray_error_maps[img_idx]
        cached_vals = cached[update_mask]
        new_vals = error_map[update_mask]
        cached[update_mask] = ema_decay * cached_vals + (1.0 - ema_decay) * new_vals
        cached.clamp_(min=min_error)
        
    def update_image_error_maps(self, render_results: Dict[str, Tensor]) -> None:
        """
        Update the image error buffer with the given render results for each camera.
        """
        # (img_num, )
        image_error_buffer = torch.zeros(self.num_imgs, device=self.device)
        image_cam_id = torch.from_numpy(np.stack(render_results["cam_ids"], axis=0))
        for cam_id in self.camera_list:
            cam_name = self.camera_data[cam_id].cam_name
            gt_rgbs, pred_rgbs = [], []
            Dynamic_opacities = []
            for img_idx, img_cam in enumerate(render_results["cam_names"]):
                if img_cam == cam_name:
                    gt_rgbs.append(render_results["gt_rgbs"][img_idx])
                    pred_rgbs.append(render_results["rgbs"][img_idx])
                    if "Dynamic_opacities" in render_results:
                        Dynamic_opacities.append(render_results["Dynamic_opacities"][img_idx])
            
            camera_results = {
                "gt_rgbs": gt_rgbs,
                "rgbs": pred_rgbs,
            }
            if len(Dynamic_opacities) > 0:
                camera_results["Dynamic_opacities"] = Dynamic_opacities
            self.camera_data[cam_id].update_image_error_maps(
                camera_results
            )
            
            # update the image error buffer
            image_error_buffer[image_cam_id == cam_id] = \
                self.camera_data[cam_id].image_error_maps.mean(dim=(1, 2))
            
                
        self.image_error_buffer = image_error_buffer
        self.image_error_buffered = True
        logger.info("Successfully updated image error buffer")

    def get_image_error_video(self, layout: Callable) -> List[np.ndarray]:
        """
        Get the image error buffer video.
        Returns:
            frames: the pixel sample weights video.
        """
        per_cam_video = {}
        for cam_id in self.camera_list:
            per_cam_video[cam_id] = self.camera_data[cam_id].get_image_error_video()
        
        all_error_images = []
        all_cam_names = []
        for frame_id in range(self.num_frames):
            for cam_id in self.camera_list:
                all_error_images.append(per_cam_video[cam_id][frame_id])
                all_cam_names.append(self.camera_data[cam_id].cam_name)
                
        merged_list = []
        for i in range(len(all_error_images) // self.num_cams):
            frames = all_error_images[
                i
                * self.num_cams : (i + 1)
                * self.num_cams
            ]
            frames = [
                np.stack([frame, frame, frame], axis=-1) for frame in frames
            ]
            cam_names = all_cam_names[
                i
                * self.num_cams : (i + 1)
                * self.num_cams
            ]
            tiled_img = layout(frames, cam_names)
            merged_list.append(tiled_img)
        
        merged_video = np.stack(merged_list, axis=0)
        merged_video -= merged_video.min()
        merged_video /= merged_video.max()
        merged_video = np.clip(merged_video * 255, 0, 255).astype(np.uint8)
        return merged_video

    @property
    def downscale_factor(self) -> float:
        """
        Returns:
            downscale_factor: the downscale factor of the images
        """
        return self._downscale_factor

    def update_downscale_factor(self, downscale: float) -> None:
        """
        Args:
            downscale: the new downscale factor
        Updates the downscale factor
        """
        self._old_downscale_factor.append(self._downscale_factor)
        self._downscale_factor = downscale
        for cam_id in self.camera_list:
            self.camera_data[cam_id].set_downscale_factor(self._downscale_factor)

    def reset_downscale_factor(self) -> None:
        """
        Resets the downscale factor to the original value
        """
        assert len(self._old_downscale_factor) > 0, "No downscale factor to reset to"
        self._downscale_factor = self._old_downscale_factor.pop()
        for cam_id in self.camera_list:
            self.camera_data[cam_id].set_downscale_factor(self._downscale_factor)

    @property
    def buffer_ratio(self) -> float:
        """
        Returns:
            buffer_ratio: the ratio of the rays sampled from the image error buffer
        """
        return self.data_cfg.sampler.buffer_ratio
    
    @property
    def buffer_downscale(self) -> float:
        """
        Returns:
            buffer_downscale: the downscale factor of the image error buffer
        """
        return self.data_cfg.sampler.buffer_downscale
    
    def prepare_novel_view_render_data(self, dataset_type: str, traj: torch.Tensor) -> list:
        """
        Prepare all necessary elements for novel view rendering.

        Args:
            dataset_type (str): Type of dataset
            traj (torch.Tensor): Novel view trajectory, shape (N, 4, 4)

        Returns:
            list: List of dicts, each containing elements required for rendering a single frame:
                - cam_infos: Camera information (extrinsics, intrinsics, image dimensions)
                - image_infos: Image-related information (indices, normalized time, viewdirs, etc.)
        """
        if dataset_type == "argoverse":
            cam_id = 1  # Use cam_id 1 for Argoverse dataset
        else:
            cam_id = 0  # Use cam_id 0 for other datasets
        
        intrinsics = self.camera_data[cam_id].intrinsics[0]  # Assume intrinsics are constant across frames
        H, W = self.camera_data[cam_id].HEIGHT, self.camera_data[cam_id].WIDTH
        
        original_frame_count = self.num_frames
        scaled_indices = torch.linspace(0, original_frame_count - 1, len(traj))
        normed_time = torch.linspace(0, 1, len(traj))
        
        render_data = []
        for i in range(len(traj)):
            c2w = traj[i]
            
            # Generate ray origins and directions
            x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
            x, y = x.to(self.device), y.to(self.device)
            
            origins, viewdirs, direction_norm = get_rays(x.flatten(), y.flatten(), c2w, intrinsics)
            origins = origins.reshape(H, W, 3)
            viewdirs = viewdirs.reshape(H, W, 3)
            direction_norm = direction_norm.reshape(H, W, 1)
            
            cam_infos = {
                "camera_to_world": c2w,
                "intrinsics": intrinsics,
                "height": torch.tensor([H], dtype=torch.long, device=self.device),
                "width": torch.tensor([W], dtype=torch.long, device=self.device),
            }
            
            image_infos = {
                "origins": origins,
                "viewdirs": viewdirs,
                "direction_norm": direction_norm,
                "img_idx": torch.full((H, W), i, dtype=torch.long, device=self.device),
                "frame_idx": torch.full((H, W), scaled_indices[i].round().long(), device=self.device),
                "normed_time": torch.full((H, W), normed_time[i], dtype=torch.float32, device=self.device),
                "pixel_coords": torch.stack(
                    [y.float() / H, x.float() / W], dim=-1
                ),  # [H, W, 2]
            }
            
            render_data.append({
                "cam_infos": cam_infos,
                "image_infos": image_infos,
            })
        
        return render_data

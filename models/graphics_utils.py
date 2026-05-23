import numpy as np
import torch
import torch.nn.functional as F
from models.gaussians.basics import *

import torch

def compute_adaptive_weight_map(envmap, lambda_param=1.5, k=10.0):
    """
    PyTorch version for training loops.
    Input: envmap [H, W, 3]
    Output: weight_map [H, W]
    """
    
    # 1. 關鍵：停止梯度傳播！
    # 我們只希望 Weight 當作一個"常數係數"來用，不希望優化器去改變 envmap 來操弄 weight。
    envmap_detached = envmap.detach() 
    
    intensity = envmap_detached.mean(dim=-1)  # [H, W] 平均 RGB 通道得到強度圖

    # 3. 計算統計量 (Global Mean & Std)
    # 針對 H, W 維度做平均，保留 Batch 維度
    mu = intensity.mean(dim=(0, 1), keepdim=True)  # [1, 1]
    std = intensity.std(dim=(0, 1), keepdim=True) + 1e-6  # [1, 1] 加小常數避免除零

    # 4. 計算 Z-Score
    z_score = (intensity - mu) / std

    # 5. 生成 Sigmoid Weight
    # 公式: sigmoid( k * (lambda - z) )
    # 當 z < lambda (背景) -> exponent > 0 -> weight -> 1
    # 當 z > lambda (強光) -> exponent < 0 -> weight -> 0
    exponent = k * (lambda_param - z_score)
    
    # 使用 sigmoid 函數
    weight = torch.sigmoid(exponent)

    return weight



def soft_clip(x, tau=0.9):
    """
    clip to [0, 1]
    smaller or equal than tau: the same
    larger than tau: 1-(1-tau)*exp(-(x - tau)/(1 - tau))
    """
    clipped = torch.where(
        x <= tau,
        x,
        1.0 - (1.0 - tau) * torch.exp(-(x - tau) / (1.0 - tau))
    )
    return clipped

def find_closest_factors(n):
    best = (1, n)
    min_diff = n
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            j = n // i
            if abs(i - j) < min_diff:
                best = (i, j)
                min_diff = abs(i - j)
    return best

def generate_subpixel_rays(c2w, K, H, W, num_sample_rays, device='cuda'):
    """
    Generate num_sample_ray rays for each subpixel in world space for a given camera-to-world matrix and intrinsic matrix.
    Args:
        c2w (torch.Tensor): (4, 4) camera-to-world transformation matrix.
        K (torch.Tensor): (3, 3) camera intrinsic matrix.
        H (int): height of the image.
        W (int): width of the image.
        num_sample_rays (int): number of rays to sample.
        device (str): device to run computation on.
    Returns:
        dirs (torch.Tensor): (H, W, num_sample_ray, 3) sampled directions in world space.
        sH (int): height scaling factor.
        sW (int): width scaling factor.
    """

    sH, sW = find_closest_factors(num_sample_rays)
    new_H = H * sH
    new_W = W * sW
    assert sH * sW == num_sample_rays, f"num_sample_rays {num_sample_rays} is not a product of two factors"

    # Generate pixel grid in subpixel resolution
    y_coords, x_coords = torch.meshgrid(
        torch.arange(new_H, dtype=torch.float32, device=device),
        torch.arange(new_W, dtype=torch.float32, device=device),
        indexing='ij',
    )

    # Pixel centers
    x = (x_coords + 0.5).view(-1)
    y = (y_coords + 0.5).view(-1)

    pix_h = torch.stack([x, y, torch.ones_like(x, device=device)], dim=1)  # (N, 3)

    # Backproject to camera space
    K_inv = torch.inverse(K)
    cam_dirs = (K_inv @ pix_h.T).T  # (N, 3)

    # Normalize directions
    dirs = torch.nn.functional.normalize(cam_dirs, dim=-1)

    # Rotate to world space
    world_dirs = dirs @ c2w[:3, :3].T  # (N, 3)
    world_dirs = F.normalize(world_dirs, dim=-1)
    world_dirs = world_dirs.reshape(H, sH, W, sW, 3)  # (H, sH, W, sW, 3)
    world_dirs = world_dirs.permute(0, 2, 1, 3, 4).reshape(H, W, num_sample_rays, 3)

    return world_dirs, sH, sW

def deproject_depth(depth_map, K, c2w, img_width, img_height, device='cuda'):
    """
    Deprojects a depth map into 3D world coordinates

    Args:
        depth_map (torch.Tensor): (H, W) depth map
        K (torch.Tensor): (3, 3) camera intrinsic matrix
        c2w (torch.Tensor): (4, 4) camera-to-world transformation matrix
        img_width (int): width of the image
        img_height (int): height of the image
        device (str): device to run computation on

    Returns:
        torch.Tensor: (N, 3) 3D points in world coordinates
    """

    # if img_width and img_height is torch tensor, convert to int
    # this will happen when rendering novel views
    if isinstance(img_width, torch.Tensor):
        img_width = img_width.item()
    if isinstance(img_height, torch.Tensor):
        img_height = img_height.item()

    # Step 1: Create meshgrid of pixel coordinates
    x, y = torch.meshgrid(
        torch.arange(img_width, device=device),
        torch.arange(img_height, device=device),
        indexing="xy"
    )
    x = x.flatten()  # (N,)
    y = y.flatten()  # (N,)
    ones = torch.ones_like(x)

    pixel_coords = torch.stack([x, y, ones], dim=0)  # (3, N)
    pixel_coords = pixel_coords.to(torch.float32)  # Ensure float32 type

    # Step 2: Inverse intrinsics
    K_inv = torch.inverse(K)

    # Step 3: Backproject to camera coordinates
    depth = depth_map.flatten()  # (N,)
    cam_coords = K_inv @ pixel_coords  # (3, N)
    cam_coords = cam_coords * depth.unsqueeze(0)  # (3, N)

    # Step 4: Convert to homogeneous coordinates (4, N)
    ones_row = torch.ones((1, cam_coords.shape[1]), device=device)
    cam_coords_hom = torch.cat([cam_coords, ones_row], dim=0)  # (4, N)

    # Step 5: Transform to world coordinates
    world_coords_hom = c2w @ cam_coords_hom  # (4, N)
    world_coords = world_coords_hom[:3, :].T  # (N, 3)

    return world_coords

def get_img_grad_weight(img, beta=2.0):
    """
    Accepts image in either (C, H, W) or (H, W, C) or (H, W) and returns a (H, W) gradient weight map.
    """
    # Normalize input layout to (C, H, W)
    if img.ndim == 3 and img.shape[2] == 3:
        img_t = img.permute(2, 0, 1).contiguous()
    elif img.ndim == 3 and img.shape[0] == 3:
        img_t = img
    elif img.ndim == 2:
        img_t = img.unsqueeze(0)
    else:
        raise ValueError("Unsupported image shape. Expect (H,W,3), (3,H,W) or (H,W).")

    img_t = img_t.to(torch.float32)
    _, hd, wd = img_t.shape

    # If too small to compute central differences, return ones
    if hd < 3 or wd < 3:
        return torch.ones((hd, wd), device=img_t.device, dtype=img_t.dtype)

    bottom_point = img_t[..., 2:hd,   1:wd-1]
    top_point    = img_t[..., 0:hd-2, 1:wd-1]
    right_point  = img_t[..., 1:hd-1, 2:wd]
    left_point   = img_t[..., 1:hd-1, 0:wd-2]

    grad_img_x = torch.mean(torch.abs(right_point - left_point), dim=0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), dim=0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)

    # normalize safely
    mn = grad_img.min()
    mx = grad_img.max()
    if (mx - mn) > 1e-8:
        grad_img = (grad_img - mn) / (mx - mn)
    else:
        grad_img = torch.zeros_like(grad_img)

    # pad to original size with constant 1.0 (as original)
    grad_img = torch.nn.functional.pad(grad_img[None, None], (1, 1, 1, 1), mode='constant', value=1.0).squeeze()

    return grad_img

def depth_to_normal(c2w, K, depth_map, img_width, img_height, device='cuda'):
    """
    Compute world-space normal map from a depth map.

    Args:
        c2w (torch.Tensor): (4,4) camera-to-world matrix.
        K (torch.Tensor): (3,3) intrinsics.
        depth_map (torch.Tensor): (H, W) or (1, H, W) depth map.
        img_width (int|Tensor): image width.
        img_height (int|Tensor): image height.
        device (str): device.

    Returns:
        normal_map (torch.Tensor): (H, W, 3) world-space normals (zero on borders).
        points_world (torch.Tensor): (H, W, 3) world-space points.
    """
    # Normalize depth shape
    if depth_map.ndim == 3 and depth_map.shape[0] == 1:
        depth = depth_map.squeeze(0).to(device)
    else:
        depth = depth_map.to(device)

    H = int(depth.shape[0])
    W = int(depth.shape[1])

    # Deproject to world points (N,3) then reshape to (H, W, 3)
    pts_world_flat = deproject_depth(depth, K, c2w, img_width, img_height, device=device)  # (N,3)
    points_world = pts_world_flat.reshape(H, W, 3)

    # Prepare output normal map (zeros on borders)
    normal_map = torch.zeros_like(points_world, device=device, dtype=points_world.dtype)

    if H > 2 and W > 2:
        # central differences:
        # dx: vertical differences (along rows)
        dx = points_world[2:, 1:-1, :] - points_world[:-2, 1:-1, :]   # (H-2, W-2, 3)
        # dy: horizontal differences (along cols)
        dy = points_world[1:-1, 2:, :] - points_world[1:-1, :-2, :]   # (H-2, W-2, 3)

        # normal = normalize(cross(dx, dy))
        n = torch.cross(dx, dy, dim=-1)
        n = F.normalize(n, dim=-1, eps=1e-6)

        normal_map[1:-1, 1:-1, :] = n

    return normal_map

def sample_depth_at_world_points(
    depth_map: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    points_world: torch.Tensor,   # (N, 3) points in WORLD space
    img_width,
    img_height,
    scale: int = 1,
    device: str = "cuda",
):
    """
    Sample depth values from a depth map at the pixel projections of world-space points.

    Args:
        depth_map (torch.Tensor): (H, W) depth map.
        K (torch.Tensor): (3, 3) camera intrinsic matrix.
        c2w (torch.Tensor): (4, 4) camera-to-world matrix (extrinsics).
        points_world (torch.Tensor): (N, 3) points in world coordinates.
        img_width (int|Tensor): image width.
        img_height (int|Tensor): image height.
        scale (int): optional downsampling factor for the depth map and projections.
        device (str): device for computation.

    Returns:
        map_z (torch.Tensor): (1, N) sampled depths at projected locations.
        mask (torch.Tensor): (N,) boolean mask of valid, in-bounds, in-front points.
    """

    # Normalize width/height types (match your deproject_depth convention)
    if isinstance(img_width, torch.Tensor):
        img_width = int(img_width.item())
    if isinstance(img_height, torch.Tensor):
        img_height = int(img_height.item())

    # ----- 1) Prepare (optionally downsampled) depth view -----
    # Start offset mimicking original function
    st = max(int(scale / 2) - 1, 0)
    # depth_view: (1, 1, H', W') for grid_sample
    depth_view = depth_map[st::scale, st::scale].unsqueeze(0).unsqueeze(0)
    W_ds = int(img_width / scale)
    H_ds = int(img_height / scale)
    depth_view = depth_view[:, :, :H_ds, :W_ds]

    # ----- 2) Transform world points -> camera space -----
    # w2c is inverse of c2w
    w2c = torch.linalg.inv(c2w)
    N = points_world.shape[0]
    ones = torch.ones((N, 1), device=device, dtype=torch.float32)
    points_world_h = torch.cat([points_world, ones], dim=1)          # (N, 4)
    points_cam_h = (w2c @ points_world_h.T).T                        # (N, 4)
    points_cam = points_cam_h[:, :3]                                 # (N, 3)
    Xc, Yc, Zc = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # ----- 3) Project to pixel coords using intrinsics -----
    # u = (K @ [Xc, Yc, Zc])_x / Zc ; v = ..._y / Zc
    cam_xy1 = (K @ points_cam.T).T                                   # (N, 3) = K * [Xc, Yc, Zc]^T
    Zc_safe = Zc.clamp(min=1e-8)
    u = cam_xy1[:, 0] / Zc_safe
    v = cam_xy1[:, 1] / Zc_safe

    # Downsampled pixel coordinates
    u_ds = u / scale
    v_ds = v / scale

    # ----- 4) Validity mask (in-bounds on DS grid and in front of camera) -----
    mask = (u_ds > 0) & (u_ds < W_ds) & (v_ds > 0) & (v_ds < H_ds) & (Zc > 0.1) # (N,)

    # ----- 5) Normalize to [-1, 1] for grid_sample (align_corners=True) -----
    # x_norm = u_ds / ((W_ds-1)/2) - 1 ; y_norm = v_ds / ((H_ds-1)/2) - 1
    # Handle degenerate tiny dims safely
    denom_x = ((W_ds - 1) / 2) if W_ds > 1 else 1.0
    denom_y = ((H_ds - 1) / 2) if H_ds > 1 else 1.0
    x_norm = (u_ds / denom_x) - 1.0
    y_norm = (v_ds / denom_y) - 1.0

    # Any remaining non-finite grid values can poison grid_sample; clamp
    x_norm = torch.nan_to_num(x_norm, neginf=-2.0, posinf=2.0)
    y_norm = torch.nan_to_num(y_norm, neginf=-2.0, posinf=2.0)

    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)   # (1, N, 1, 2)

    # ----- 6) Bilinear sample depths at those coordinates -----
    sampled = F.grid_sample(
        input=depth_view,                # (1, 1, H', W')
        grid=grid,                       # (1, N, 1, 2)
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )                                    # -> (1, 1, N, 1)

    map_z = sampled[0, :, :, 0]          # (1, N), matches original function's shape

    return map_z, mask

def reproject_world_points_via_depth(
    points_world: torch.Tensor,   # (N, 3) input points in WORLD coordinates
    map_z: torch.Tensor,          # (N,) or (1, N) depths along the nearest-cam rays
    K_near: torch.Tensor,         # (3, 3) intrinsics of nearest camera
    c2w_near: torch.Tensor,       # (4, 4) camera-to-world of nearest camera
    K_view: torch.Tensor,         # (3, 3) intrinsics of viewpoint camera
    c2w_view: torch.Tensor,       # (4, 4) camera-to-world of viewpoint camera
    device: str = "cuda",
):
    """
    Reproject world-space points to a viewpoint camera using per-point depths
    measured from a nearest camera.

    Steps:
      1) World -> nearest-cam: get camera-space coords and ray directions (x/z, y/z, 1)
      2) Deproject with map_z to obtain true 3D points in nearest-cam coords
      3) Nearest-cam -> world (via c2w_near)
      4) World -> viewpoint-cam (via w2c_view)
      5) Project with K_view to pixel coords

    Args:
        points_world: (N, 3)
        map_z: (N,) or (1, N)
        K_near: (3, 3)
        c2w_near: (4, 4)
        K_view: (3, 3)
        c2w_view: (4, 4)
        device: str

    Returns:
        pts_proj_view:   (N, 2) pixel coordinates in viewpoint camera
        pts_view_cam:    (N, 3) 3D points in viewpoint camera coordinates
        pts_world_depth: (N, 3) 3D points in world coordinates reconstructed via map_z
        mask:            (N,)   bool mask: valid depth & in front of viewpoint (Z>0)
    """
    # --- Move & cast ---
    if map_z.ndim == 0:
        map_z = map_z.unsqueeze(0)

    N = points_world.shape[0]
    ones = torch.ones((N, 1), device=device, dtype=torch.float32)
    points_world_h = torch.cat([points_world, ones], dim=1)              # (N, 4)

    # --- 1) World -> nearest camera (column-vector convention) ---
    w2c_near = torch.inverse(c2w_near)                                    # (4, 4)
    pts_near_cam_h = (w2c_near @ points_world_h.T).T                      # (N, 4)
    pts_near_cam = pts_near_cam_h[:, :3]                                  # (N, 3)
    Xn, Yn, Zn = pts_near_cam[:, 0], pts_near_cam[:, 1], pts_near_cam[:, 2]

    # Ray directions in nearest cam: (x/z, y/z, 1)
    Zn_safe = Zn.clamp(min=1e-8)
    rays_near = torch.stack([Xn / Zn_safe, Yn / Zn_safe, torch.ones_like(Zn_safe)], dim=-1)  # (N, 3)

    # --- 2) Deproject with per-point depth map_z to get true 3D in nearest-cam coords ---
    map_z = map_z.reshape(-1)                                             # (N,)
    pts_near_cam_depth = rays_near * map_z[:, None]                       # (N, 3)

    # --- 3) Nearest cam -> World (via c2w_near) ---
    pts_near_cam_depth_h = torch.cat([pts_near_cam_depth, ones], dim=1)   # (N, 4)
    pts_world_depth_h = (c2w_near @ pts_near_cam_depth_h.T).T             # (N, 4)
    pts_world_depth = pts_world_depth_h[:, :3]                             # (N, 3)

    # --- 4) World -> viewpoint camera (via w2c_view) ---
    w2c_view = torch.inverse(c2w_view)                                    # (4, 4)
    pts_world_depth_h2 = torch.cat([pts_world_depth, ones], dim=1)        # (N, 4)
    pts_view_cam_h = (w2c_view @ pts_world_depth_h2.T).T                  # (N, 4)
    pts_view_cam = pts_view_cam_h[:, :3]                                  # (N, 3)

    # --- 5) Project with K_view ---
    Xv, Yv, Zv = pts_view_cam[:, 0], pts_view_cam[:, 1], pts_view_cam[:, 2]
    Zv_safe = Zv.clamp(min=1e-8)
    # u = fx*X/Z + cx ; v = fy*Y/Z + cy
    fx, fy, cx, cy = K_view[0, 0], K_view[1, 1], K_view[0, 2], K_view[1, 2]
    u = fx * (Xv / Zv_safe) + cx
    v = fy * (Yv / Zv_safe) + cy
    pts_proj_view = torch.stack([u, v], dim=-1)                           # (N, 2)

    # Validity mask: positive depth in the viewpoint and valid (positive) map_z
    mask = (map_z > 0.1) & (Zv > 0.1) & (Zn > 0.1)                             # (N,)

    return pts_proj_view, pts_view_cam, pts_world_depth, mask


def transform_to_world(local_dirs, normals):
    """
    Transforms local-space directions to world-space using the given normal vectors.

    Args:
        local_dirs: [B, 3] — sampled directions in local space (z-up).
        normals: [B, 3] — surface normals in world space.

    Returns:
        world_dirs: [B, 3]
    """
    
    # Handle degenerate normals by picking a fallback axis
    B = normals.shape[0]
    up = torch.tensor([0.0, 0.0, 1.0], device=normals.device).expand(B, 3)
    right = torch.cross(up, normals)
    mask = torch.norm(right, dim=1) < 1e-3

    # Recompute right vector for near-parallel cases
    if mask.any():
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=normals.device).expand(B, 3)
        right[mask] = torch.cross(alt_up[mask], normals[mask])

    right = F.normalize(right, dim=1)
    up = F.normalize(torch.cross(normals, right), dim=1)

    # Construct local-to-world transformation matrix [B, 3, 3]
    T = torch.stack([right, up, normals], dim=2)

    # Transform directions: [B, 3] = [B, 3, 3] @ [B, 3, 1]
    world_dirs = torch.bmm(T, local_dirs.unsqueeze(2)).squeeze(2)
    
    # normalize again
    world_dirs = F.normalize(world_dirs, dim=1)
    
    return world_dirs

# From Relightable 3D Gaussian: https://github.com/NJU-3DV/Relightable3DGaussian
def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):
    """
    Sample `sample_num` directions on a Fibonacci sphere around the normals.
    Args:
        normals (Tensor): [N, 3] — surface normals.
        sample_num (int): number of samples per normal.
        random_rotate (bool): whether to apply random rotation to the samples.
    Returns:
        incident_dirs (Tensor): [N, sample_num, 3] — sampled directions in world space.
        incident_pdfs (Tensor): [N, sample_num] — PDF values for each sample.
    """
    
    normals = F.normalize(normals, dim=-1)
    N = normals.shape[0]
    
    delta = np.pi * (3.0 - np.sqrt(5.0))
    # fibonacci sphere sample around z axis
    idx = torch.arange(sample_num, dtype=torch.float, device='cuda')
    z = (1 - 2 * idx / (2 * sample_num - 1)).clamp_min(np.sin(10/180*np.pi))
    rad = torch.sqrt(1 - z ** 2)
    theta = delta * idx
    theta = theta.unsqueeze(0).expand(N, -1)  # [N, sample_num]
    if random_rotate:
        theta = torch.rand(N, sample_num, device='cuda') * 2 * np.pi + theta
    y = torch.cos(theta) * rad
    x = torch.sin(theta) * rad
    z_samples = torch.stack([x, y, z.expand_as(y)], dim=-1) # [N, sample_num, 3]

    incident_dirs = transform_to_world(
        local_dirs=z_samples.reshape(-1, 3),
        normals=normals.repeat_interleave(sample_num, dim=0),
    )
    # [N, sample_num, 3]
    incident_dirs = F.normalize(incident_dirs, dim=-1).reshape(-1, sample_num, 3)
    # [N, sample_num]
    incident_pdfs = torch.ones_like(incident_dirs)[..., 0] * (1.0 / (2 * np.pi))
    
    return incident_dirs, incident_pdfs


# From Relightable 3D Gaussian: https://github.com/NJU-3DV/Relightable3DGaussian
def GGX_specular(
        normal, # [nrays, 3]
        pts2c, # [nrays, 3]
        pts2l, # [nrays, nlights, 3]
        roughness, # [nrays, 3]
        fresnel # float
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec




def sample_reflect(normals, viewdirs):
    """
    Compute perfect reflection vectors for a set of surface normals and view directions.

    Args:
        normals (Tensor): [N, 3] — normalized surface normal vectors.
        viewdirs (Tensor): [N, 3] — normalized view directions (pointing away from surface).

    Returns:
        reflect_dirs (Tensor): [N, 1, 3] — reflection directions in world space.
        pdfs (None): [N, 1] — PDF values (not used for perfect reflection).
    """
    # Ensure inputs are normalized
    normals = F.normalize(normals, dim=1)
    viewdirs = F.normalize(viewdirs, dim=1)

    # Reflect: r = v - 2(n ⋅ v) n
    dot_nv = torch.sum(normals * viewdirs, dim=1, keepdim=True)  # [N, 1]
    reflect_dirs = viewdirs - 2 * dot_nv * normals  # [N, 3]

    # Optional: normalize for safety
    reflect_dirs = F.normalize(reflect_dirs, dim=1)
    reflect_dirs = reflect_dirs.unsqueeze(1)  # [N, 1, 3]

    return reflect_dirs, None

def sample_cosine(normals, num_sample_rays, device='cuda'):
    """
    Sample `num` cosine-weighted directions per normal vector in world space.

    Args:
        normals (Tensor): shape [N, 3], surface normal vectors.
        num_sample_rays (int): number of samples per normal.
        device (str): PyTorch device.

    Returns:
        dirs_world: [N, num, 3] sampled directions in world space.
        pdfs: [N, num] cosine-weighted PDF values for each sample.
    """
    N = normals.shape[0]

    # Step 1: Sample local cosine-weighted hemisphere directions [N * num, 3]
    total = N * num_sample_rays
    u1 = torch.rand(total, device=device)
    u2 = torch.rand(total, device=device)

    r = torch.sqrt(u1)
    theta = 2 * math.pi * u2

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.sqrt(1 - u1)

    dirs_local = torch.stack([x, y, z], dim=1)  # [N * num, 3]
    pdfs = z / math.pi  # [N * num]

    # Step 2: Expand normals to match samples
    normals = F.normalize(normals, dim=1)
    normals_expanded = normals[:, None, :].expand(N, num_sample_rays, 3).reshape(-1, 3)  # [N * num, 3]

    # Step 3: Transform each local direction to world space using the per-normal basis
    dirs_world = transform_to_world(dirs_local, normals_expanded)  # [N * num, 3]
    dirs_world = F.normalize(dirs_world, dim=1)  

    # Step 4: Reshape back to [N, num, 3] and [N, num]
    dirs_world = dirs_world.reshape(N, num_sample_rays, 3)
    pdfs = pdfs.reshape(N, num_sample_rays)



    return dirs_world, pdfs

def pdf_cosine(normals, sampled_dirs, device='cuda'):
    """
    Compute the PDF of cosine-weighted sampling for given directions.

    Args:
        normals (Tensor): [B, 3], surface normal vectors.
        sampled_dirs (Tensor): [B, 3], sampled directions in world space.
        device (str): PyTorch device.

    Returns:
        pdfs: [B, 1], cosine-weighted PDF values for each direction.
    """
    normals = F.normalize(normals, dim=1)
    sampled_dirs = F.normalize(sampled_dirs, dim=1)

    # Compute the cosine of the angle between normals and sampled directions
    cos_theta = torch.sum(normals * sampled_dirs, dim=1, keepdim=True).clamp(min=1e-6)

    # Compute the PDF
    pdfs = cos_theta / math.pi  # [B, 1]

    return pdfs


def sample_GGX(normals, roughness, viewdirs, num, device='cuda', use_vndf=False):
    """
    Sample GGX-distributed directions per normal using importance sampling,
    with individual viewing directions for each normal.

    Args:
        normals (Tensor): [N, 3], surface normal vectors.
        roughness (Tensor or float): [N] or scalar, roughness values.
        viewdirs (Tensor): [N, 3], per-normal view direction vectors in world space. (dir: object -> camera)
        num (int): number of samples per normal.
        device (str): PyTorch device.

    Returns:
        dirs_world: [N, num, 3], sampled incoming light directions in world space.
        pdfs: [N, num], GGX sampling PDF values for each direction.
                        if the reflection is not in the hemisphere, pdfs will be -1.0
    
    Reference:
        FIPT: https://github.com/lwwu2/fipt/blob/main/model/brdf.py
        Heitz 2018: https://jcgt.org/published/0007/04/01/paper.pdf
        BRDF implementation crash course: https://github.com/boksajak/brdf/blob/master/brdf.h
    """
    N = normals.shape[0]
    eps = 1e-6
    normals = F.normalize(normals, dim=1, eps=eps)
    viewdirs = F.normalize(viewdirs, dim=1, eps=eps)
    V = viewdirs[:, None, :].expand(N, num, 3).reshape(-1, 3)  # [N * num, 3]

    # Broadcast roughness to [N]
    if isinstance(roughness, float) or isinstance(roughness, int):
        roughness = torch.full((N,), roughness, device=device)
    alpha = roughness ** 2  # GGX alpha

    # Step 1: Sample microfacet normals h from GGX NDF in tangent space
    total = N * num
    u1 = torch.rand(total, device=device)
    u2 = torch.rand(total, device=device)
    a = alpha.repeat_interleave(num)     # [N * num]
    if not use_vndf:
        # Same as FIPT
        phi = 2 * math.pi * u1
        cos_theta = torch.sqrt(torch.clamp((1 - u2) / (1 + (a ** 2 - 1) * u2), min=eps, max=1.0))
        sin_theta = torch.sqrt(1 - cos_theta ** 2)

        hx = sin_theta * torch.cos(phi)
        hy = sin_theta * torch.sin(phi)
        hz = cos_theta
        h_local = torch.stack([hx, hy, hz], dim=1)  # [N * num, 3]
    else:
        # VNDF to sample microfacet normals
        # TODO: justify the computation
        # Step 1: Stretch view vector
        Vh = F.normalize(torch.stack([a * V[:, 0], a * V[:, 1], V[:, 2]], dim=1), dim=1, eps=eps)

        # Step 2: Construct orthonormal basis around stretched view vector
        def tangent_space_basis(v):
            # assumes v is normalized
            # NOTE: this part is slightly different but equivalent to Heitz's implementation
            # lensq < 0 == abs(z) > 0.999
            # cross(up, v) == (-Vh.y, Vh.x, 0)
            up = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(v)
            cond = (torch.abs(v[:, 2]) > 0.999)
            tangent = torch.where(cond[:, None], 
                                torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(v), 
                                F.normalize(torch.cross(up, v, dim=1), dim=1, eps=eps))
            bitangent = torch.cross(v, tangent, dim=1)
            return tangent, bitangent, v

        T, B, N_ = tangent_space_basis(Vh)  # Each [N * num, 3]

        # Step 3: Sample from cosine-weighted hemisphere
        r     = torch.sqrt(u1)
        phi   = 2.0 * math.pi * u2
        t1    = r * torch.cos(phi)
        t2    = r * torch.sin(phi)
        s     = 0.5 * (1.0 + Vh[:, 2])                         
        t2    = (1.0 - s) * torch.sqrt(torch.clamp(1.0 - t1*t1, min=0.0)) + s * t2

        # Step 4: Transform to hemisphere aligned with Vh
        Nh = t1.unsqueeze(-1) * T \
           + t2.unsqueeze(-1) * B \
           + torch.sqrt(torch.clamp(1.0 - t1*t1 - t2*t2, min=0.0)).unsqueeze(-1) * Vh

        # Step 5: Unstretch
        h_local = F.normalize(torch.stack([
            a * Nh[:, 0],
            a * Nh[:, 1],
            Nh[:, 2]
        ], dim=1), dim=1, eps=eps)  # [N * num, 3]
        

    # Step 2: Rotate halfway vectors to world space using normal basis
    normals_expanded = normals[:, None, :].expand(N, num, 3).reshape(-1, 3)  # [N * num, 3]
    h_world = transform_to_world(h_local, normals_expanded)  # [N * num, 3]
    h_world = F.normalize(h_world, dim=1, eps=eps)  

    # Step 3: Use per-normal view direction to compute reflection l = reflect(-v, h)
    l_world = 2 * torch.sum(h_world * V, dim=1, keepdim=True) * h_world - V  # reflect
    l_world = F.normalize(l_world, dim=1, eps=eps) 

    # Step 4: Reshape to [N, num, 3]
    dirs_world = l_world.reshape(N, num, 3)

    # Step 5: Compute GGX sampling PDF
    NoH = torch.sum(normals_expanded * h_world, dim=1).clamp(min=0.0) + eps  # [N * num]
    VoH = torch.sum(V * h_world, dim=1).clamp(min=0.0) + eps  # [N * num]
    NoV = torch.sum(normals_expanded * V, dim=1).clamp(min=0.0) + eps  # [N * num]
    alpha2 = a ** 2
    D = alpha2 / (math.pi * ((NoH ** 2) * (alpha2 - 1) + 1) ** 2 + eps)
    if not use_vndf:
        # Walter PDF
        pdf = D * NoH / ((4 * VoH).clamp(min=eps))  
    else:
        # Heitz 2018
        # NOTE: VoNi is always positive, so max(0, VoNi) = VoNi
        # Which cancel out the VoZ term in the denominator
        k = (a + 1) ** 2 / 8.0  # [N * num], a = roughness repeated for each sample
        denom = NoV * (1 - k) + k  # [N * num]
        G1 = NoV / denom           # [N * num]
        pdf = D * G1 / (4 * NoV)   # [N * num]
    
    pdfs = pdf.reshape(N, num)

    # # handle the case that the reflection is not in the hemisphere
    # l_dot_n = torch.sum(normals_expanded * l_world, dim=1) # [N * num]
    # pdfs[l_dot_n.reshape(N, num) < 0] = -1.0
    
    # Useless
    # # handle the case that the sampled half vector is toward the opposite direction of the viewdir
    # # note: if using pdf_GGX on the sampled directions and the viewdirs, the result will be different
    # # since it doesn't sample the half vectors.
    # pdfs[VoH.reshape(N, num) < 0] = -1.0

    return dirs_world, pdfs

def pdf_GGX(normals, roughness, viewdirs, lightdirs, device='cuda', use_vndf=False):
    """
    Compute GGX sampling PDF for given directions.

    Args:
        normals (Tensor): [B, 3], surface normals.
        roughness (Tensor or float): [B] or scalar, roughness values.
        viewdirs (Tensor): [B, 3], view directions. (dir: object -> camera)
        lightdirs (Tensor): [B, 3], sampled light directions.
        device (str): device to run computation on.

    Returns:
        pdfs: [B, 1], GGX importance sampling PDF values.
                        if the reflection is not in the hemisphere, pdfs will be -1.0
    Reference:
        FIPT: https://github.com/lwwu2/fipt/blob/main/model/brdf.py
    """
    B = normals.shape[0]
    eps=1e-6
    normals = F.normalize(normals, dim=1)
    viewdirs = F.normalize(viewdirs, dim=1)
    lightdirs = F.normalize(lightdirs, dim=1)

    if isinstance(roughness, float) or isinstance(roughness, int):
        roughness = torch.full((B,), roughness, device=device)

    alpha = roughness ** 2
    alpha2 = (alpha ** 2).view(B, 1)  # Ensure shape is [B, 1]

    h = F.normalize(viewdirs + lightdirs, dim=1, eps=eps)  # [B, 3]

    NoH = torch.sum(normals * h, dim=1, keepdim=True).clamp(min=0.0) + eps  # [B, 1]
    VoH = torch.sum(viewdirs * h, dim=1, keepdim=True).clamp(min=0.0) + eps  # [B, 1]
    NoV = torch.sum(normals * viewdirs, dim=1, keepdim=True).clamp(min=0.0) + eps  # [B, 1]

    D = alpha2 / (math.pi * ((NoH ** 2) * (alpha2 - 1) + 1) ** 2 + eps)  # [B, 1]
    if not use_vndf:
        pdfs = D * NoH / ((4.0 * VoH) + 1e-5)  # [B, 1]
    else:
        # Heitz 2018 VNDF PDF
        k = (roughness + 1) ** 2 / 8.0
        denom = NoV * (1 - k) + k
        G1 = NoV / denom
        pdfs = D * G1 / (4 * NoV + eps)

    # # Clamp below horizon
    # l_dot_n = torch.sum(lightdirs * normals, dim=1, keepdim=True)
    # pdfs[l_dot_n < 0] = -1.0

    return pdfs

def sample_envmap(normals, envmap, num, device='cuda', training=False, pdf_differentiable=False):
    """
    Sample `num` light directions from an environment map for each normal vector.
    Args:
        normals (Tensor): [B, 3], surface normal vectors.
        envmap (EnvironmentMap): environment map object with a sample_light_directions method.
        num (int): number of samples per normal.
        device (str): PyTorch device.
        training (bool): whether to use training mode for sampling.
    Returns:
        lightdirs (Tensor): [B, num, 3], sampled light directions in world space.
        pdfs (Tensor): [B, num], PDF values for each sampled direction.
    Reference:
        FIPT: https://github.com/lwwu2/fipt/blob/main/model/brdf.py
    """
    B = normals.shape[0]
    lightdirs, pdfs = envmap.sample_light_directions(
        B=B,
        sample_num=num,
        training=training,
        pdf_differentiable=pdf_differentiable
    )
    pdfs = pdfs.reshape(B, num)  # [B, num]
    
    lightdirs = F.normalize(lightdirs, dim=-1)  # [B, num, 3]
    
    # l_dot_n = torch.sum(normals[:, None, :] * lightdirs, dim=-1)  # [B, num]
    # pdfs[l_dot_n.reshape(B, num) < 0] = -1.0  # Clamp below horizon
    
    return lightdirs, pdfs

def pdf_envmap(normals, lightdirs, envmap):
    """
    Compute the PDF of sampling light directions from an environment map.
    Args:
        normals (Tensor): [N, 3], surface normals.
        lightdirs (Tensor): [N, 3], sampled light directions in world space.
        envmap (EnvironmentMap): environment map object with a pdf_light_directions method.
    Returns:
        pdfs (Tensor): [N, 1], PDF values for each sampled direction.
    """
    N = normals.shape[0]
    pdfs = envmap.light_pdf(lightdirs.reshape(N, 1, 3)).reshape(N, 1)  # [N, 1]
    
    # # Clamp below horizon
    # l_dot_n = torch.sum(lightdirs * normals, dim=1, keepdim=True)
    # pdfs[l_dot_n < 0] = -1.0
    
    return pdfs

def brdf_diffuse(albedos, lightdirs):
    """
    Compute the diffuse BRDF value for sampled rays.
    Args:
        albedos (Tensor): [N, 3], diffuse albedo values.
        lightdirs (Tensor): [N, num, 3], sampled light directions.
    Returns:    
        brdf_vals (Tensor): [N, num, 3], RGB diffuse BRDF values for each ray.
    """
    num_sample_rays = lightdirs.shape[1]
    brdf_vals = albedos[:, None, :] / math.pi  # [N, 1, 3]
    brdf_vals = brdf_vals.expand(-1, num_sample_rays, -1)  # [N, num, 3]
    return brdf_vals

def brdf_GGX(normals, viewdirs, lightdirs, roughness, fresnel=0.04, 
             verbose=False):
    """
    Compute the GGX microfacet BRDF value for sampled rays.

    Args:
        normals (Tensor): [N, 3], surface normals.
        viewdirs (Tensor): [N, 3], view directions.
        lightdirs (Tensor): [N, num, 3], sampled light directions.
        roughness (Tensor or float): [N] or scalar, surface roughness.
        fresnel (Tensor float): [N, 3] or scalar, Fresnel reflectance values.

    Returns:
        brdf_vals (Tensor): [N, num, 3], RGB BRDF values for each ray.
    
    Ref:
        FIPT: https://github.com/lwwu2/fipt/blob/main/model/brdf.py 
    """
    N, num = lightdirs.shape[:2]
    device = normals.device
    eps = 1e-6

    normals = F.normalize(normals, dim=1)
    viewdirs = F.normalize(viewdirs, dim=1)
    lightdirs = F.normalize(lightdirs, dim=2)

    if isinstance(roughness, float) or isinstance(roughness, int):
        roughness = torch.full((N,), roughness, device=device)
    alpha = roughness ** 2  # GGX alpha

    # Expand normals/viewdirs to match lightdirs
    normals_exp = normals[:, None, :].expand(N, num, 3)      # [N, num, 3]
    viewdirs_exp = viewdirs[:, None, :].expand(N, num, 3)    # [N, num, 3]
    alpha_exp = alpha[:, None].expand(N, num).unsqueeze(-1)  # [N, num, 1]
    roughness_exp = roughness[:, None].expand(N, num).unsqueeze(-1)  # [N, num, 1]
    if isinstance(fresnel, torch.Tensor):
        fresnel = fresnel[:, None, :].expand(N, num, 3)          # [N, num, 3]

    # Half vector
    half = F.normalize(viewdirs_exp + lightdirs, dim=2, eps=eps)  # [N, num, 3]

    # All [N, num, 1]
    NoV = torch.sum(normals_exp * viewdirs_exp, dim=2, keepdim=True).clamp(min=0.0) + eps
    NoL = torch.sum(normals_exp * lightdirs, dim=2, keepdim=True).clamp(min=0.0) + eps
    NoH = torch.sum(normals_exp * half, dim=2, keepdim=True).clamp(min=0.0) + eps
    VoH = torch.sum(viewdirs_exp * half, dim=2, keepdim=True).clamp(min=0.0) + eps

    alpha2 = alpha_exp ** 2
        
    # GGX NDF
    # [N, num, 1]
    D = alpha2 / ((math.pi * (((NoH ** 2) * (alpha2 - 1) + 1) ** 2)) + eps)

    # # GGX Smith G term (derived from Walter 2007)
    # def G1(NoX, alpha):
    #     tan2 = (1 - NoX ** 2) / (NoX ** 2 + 1e-6)
    #     return 2 / (1 + torch.sqrt(1 + alpha ** 2 * tan2))
    # Gv = G1(NoV, alpha_exp)
    # Gl = G1(NoL, alpha_exp)

    # Schick's approximation for G1
    def G1(NoX, roughness):
        k = (roughness + 1.0) ** 2 / 8.0
        denom = NoX * (1.0 - k) + k
        return NoX / denom

    Gv = G1(NoV, roughness_exp)
    Gl = G1(NoL, roughness_exp)
    
    G = Gv * Gl  # [N, num, 1]
    
    # Fresnel (Schlick's approximation)
    # scalar or [N, num, 3]
    # Fr = fresnel + (1 - fresnel) * ((1 - VoH) ** 5)
    Fr = fresnel + (1 - fresnel) * torch.pow(torch.clamp(1 - VoH, min=0.0, max=1.0), 5)

    # Final BRDF
    # [N, num, 1] if fresnel is scalar
    # [N, num, 3] if fresnel is [N, num, 3] tensor
    demon = (4 * NoV * NoL).clamp_(min=eps) 
    spec = (D * G * Fr) / demon

    # (alpha2 / NoH^4) * ((1-VoH)**5)

    # Convert to RGB [N, num, 3] if needed
    if isinstance(fresnel, float) or isinstance(fresnel, int):
        # [N, num, 1] -> [N, num, 3]
        spec = spec.expand(N, num, 3)

    if torch.isnan(spec).any() or torch.isinf(spec).any():
        print("GGX BRDF: NaN or Inf detected. Check inputs.")
        import pdb; pdb.set_trace()

    # # Clamp directions below surface to 0. Seems to be unnecessary
    # spec = spec.masked_fill(NoL < 1e-4, 0.0)
    
    extra_info = None
    if verbose:
        # For debugging, return additional information
        extra_info = {
            'NoV': NoV, # [N, num, 1]
            'NoL': NoL, # [N, num, 1]
            'NoH': NoH, # [N, num, 1]
            'VoH': VoH, # [N, num, 1]
            'D': D, # [N, num, 1]
            'Gv': Gv, # [N, num, 1]
            'Gl': Gl, # [N, num, 1]
            'Fr': Fr # [N, num, 1] or [N, num, 3]
        }
    
    return spec, extra_info

def sample_BRDF(normals, viewdirs, roughness, 
               diffuse_weight=0.5, specular_weight=0.5, num_sample_rays=64, device='cuda', use_vndf=False):
    """
    Perform multiple importance sampling (MIS) combining cosine and GGX sampling to sample on simplified Disney BRDF,
    and return exactly `num_sample_rays` samples per normal.

    Args:
        normals (Tensor): [N, 3] surface normals.
        viewdirs (Tensor): [N, 3] view directions. (dir: object -> camera)
        roughness (Tensor): [N].
        diffuse_weight (float or Tensor): 1 or [N] weight for cosine sampling.
        specular_weight (float or Tensor): 1 or [N] weight for GGX sampling.
        num_sample_rays (int): number of samples to return per normal.
        device (str): PyTorch device.

    Returns:
        directions: [N, num_sample_rays, 3] sampled directions.
        pdfs: [N, num_sample_rays] combined PDFs.
        
    Note that pdfs here is the mixture PDF of the two sampling methods.
        
    """
    N = normals.shape[0]
    if isinstance(diffuse_weight, float) or isinstance(diffuse_weight, int):
        diffuse_weight = torch.full((N,), diffuse_weight, device=device)
    if isinstance(specular_weight, float) or isinstance(specular_weight, int):
        specular_weight = torch.full((N,), specular_weight, device=device)
    
    # [N]
    diffuse_probs = diffuse_weight / (diffuse_weight + specular_weight)
    specular_probs = specular_weight / (diffuse_weight + specular_weight)

    # 1. Cosine sampling
    dirs_cos, pdf_cos = sample_cosine(normals, num_sample_rays, device=device)  # [N, num_sample_rays, 3], [N, num_sample_rays]
    # ggx pdf of the sampled directions from cosine sampling
    pdf_g_cos = pdf_GGX(
        normals.repeat_interleave(num_sample_rays, 0),
        roughness.repeat_interleave(num_sample_rays),
        viewdirs.repeat_interleave(num_sample_rays, 0),
        dirs_cos.reshape(-1, 3)
    ).reshape(N, num_sample_rays)  # [N, num_sample_rays]

    # 2. GGX sampling
    dirs_ggx, pdf_ggx = sample_GGX(normals, roughness, viewdirs, num_sample_rays, device=device, use_vndf=use_vndf)  # [N, num_sample_rays, 3], [N, num_sample_rays]
    # cosine pdf of the sampled directions from GGX sampling
    pdf_c_ggx = pdf_cosine(
        normals.repeat_interleave(num_sample_rays, 0),
        dirs_ggx.reshape(-1, 3)
    ).reshape(N, num_sample_rays)  # [N, num_sample_rays]

    # Shape: [N, num_sample_rays]
    rand_vals = torch.rand(N, num_sample_rays, device=device)
    diffuse_mask = rand_vals < diffuse_probs[:, None]  # bool mask

    # Initialize output arrays
    directions = torch.zeros((N, num_sample_rays, 3), device=device)
    pdfs = torch.zeros((N, num_sample_rays), device=device)

    # Assign directions using the mask
    directions[diffuse_mask] = dirs_cos[diffuse_mask]
    directions[~diffuse_mask] = dirs_ggx[~diffuse_mask]

    # PDF from both methods
    pdf_cos_weighted = pdf_cos * diffuse_probs[:, None] + pdf_g_cos * specular_probs[:, None]
    pdf_ggx_weighted = pdf_ggx * specular_probs[:, None] + pdf_c_ggx * diffuse_probs[:, None]

    # Mix PDFs using the same mask
    pdfs[diffuse_mask] = pdf_cos_weighted[diffuse_mask]
    pdfs[~diffuse_mask] = pdf_ggx_weighted[~diffuse_mask]

    return directions, pdfs, diffuse_mask

def sample_GGX_envmap(normals, viewdirs, roughness, envmap, 
               GGX_weight=0.5, envmap_weight=0.5, num_sample_rays=64, device='cuda', use_vndf=False):
    """
    Perform multiple importance sampling (MIS) combining GGX sampling and environment map sampling,
    and return exactly `num_sample_rays` samples per normal.
    Args:
        normals (Tensor): [N, 3] surface normals.
        viewdirs (Tensor): [N, 3] view directions. (dir: object -> camera)
        roughness (Tensor): [N].
        GGX_weight (float or Tensor): 1 or [N] weight for GGX sampling.
        envmap_weight (float or Tensor): 1 or [N] weight for environment map sampling.
        num_sample_rays (int): number of samples to return per normal.
        device (str): PyTorch device.
        use_vndf (bool): whether to use VNDF for GGX sampling.
    Returns:
        directions: [N, num_sample_rays, 3] sampled directions.
        pdfs: [N, num_sample_rays] combined PDFs.
        GGX_mask: [N, num_sample_rays] boolean mask indicating which samples are from GGX sampling.
    Note that pdfs here is the mixture PDF of the two sampling methods.
    """
    
    N = normals.shape[0]
    if isinstance(GGX_weight, float) or isinstance(GGX_weight, int):
        GGX_weight = torch.full((N,), GGX_weight, device=device)
    if isinstance(envmap_weight, float) or isinstance(envmap_weight, int):
        envmap_weight = torch.full((N,), envmap_weight, device=device)

    # [N]
    GGX_probs = GGX_weight / (GGX_weight + envmap_weight)
    envmap_probs = envmap_weight / (GGX_weight + envmap_weight)

    with torch.no_grad():
        # 1. GGX sampling
        dirs_ggx, pdf_ggx = sample_GGX(normals, roughness, viewdirs, num_sample_rays, device=device, use_vndf=use_vndf)  # [N, num_sample_rays, 3], [N, num_sample_rays]
        # envmap pdf of the sampled directions from GGX sampling
        pdf_e_ggx = pdf_envmap(
            normals.repeat_interleave(num_sample_rays, 0),
            dirs_ggx.reshape(-1, 3),
            envmap=envmap
        ).reshape(N, num_sample_rays)  # [N, num_sample_rays]

        # 2. envmap sampling
        dirs_envmap, pdf_envmap_ = sample_envmap(normals, envmap, num_sample_rays, device=device)  # [N, num_sample_rays, 3], [N, num_sample_rays]
        # GGX pdf of the sampled directions from envmap sampling
        pdf_g_envmap = pdf_GGX(
            normals.repeat_interleave(num_sample_rays, 0),
            roughness.repeat_interleave(num_sample_rays),
            viewdirs.repeat_interleave(num_sample_rays, 0),
            dirs_envmap.reshape(-1, 3)
        ).reshape(N, num_sample_rays)  # [N, num_sample_rays]

        # Shape: [N, num_sample_rays]
        rand_vals = torch.rand(N, num_sample_rays, device=device)
        GGX_mask = rand_vals < GGX_probs[:, None]  # bool mask

        # Initialize output arrays
        directions = torch.zeros((N, num_sample_rays, 3), device=device)
        pdfs = torch.zeros((N, num_sample_rays), device=device)

        # Assign directions using the mask
        directions[GGX_mask] = dirs_ggx[GGX_mask]
        directions[~GGX_mask] = dirs_envmap[~GGX_mask]

        # PDF from both methods
        pdf_ggx_weighted = pdf_ggx * GGX_probs[:, None] + pdf_e_ggx * envmap_probs[:, None]
        pdf_envmap_weighted = pdf_envmap_ * envmap_probs[:, None] + pdf_g_envmap * GGX_probs[:, None]

        # Mix PDFs using the same mask
        pdfs[GGX_mask] = pdf_ggx_weighted[GGX_mask]
        pdfs[~GGX_mask] = pdf_envmap_weighted[~GGX_mask]

    return directions, pdfs, GGX_mask

def sample_uniform_envmap(normals, envmap, 
                          uniform_weight=0.5, envmap_weight=0.5, 
                          num_sample_rays=64, device='cuda', random_rotate=True):
    """
    Perform multiple importance sampling (MIS) combining uniform (Fibonacci) sampling 
    and environment map sampling, returning `num_sample_rays` samples per normal.
    
    Args:
        normals (Tensor): [N, 3], surface normals.
        envmap (object): environment map with `sample_light_directions` and `light_pdf`.
        uniform_weight (float or Tensor): relative weight for uniform sampling.
        envmap_weight (float or Tensor): relative weight for envmap sampling.
        num_sample_rays (int): number of samples per normal.
        device (str): PyTorch device.
        random_rotate (bool): whether to rotate the Fibonacci samples randomly.
        
    Returns:
        directions: [N, num_sample_rays, 3] sampled directions.
        pdfs: [N, num_sample_rays] combined PDFs.
        uniform_mask: [N, num_sample_rays] boolean mask indicating uniform sampling.
    """
    N = normals.shape[0]

    if isinstance(uniform_weight, (float, int)):
        uniform_weight = torch.full((N,), uniform_weight, device=device)
    if isinstance(envmap_weight, (float, int)):
        envmap_weight = torch.full((N,), envmap_weight, device=device)

    uniform_probs = uniform_weight / (uniform_weight + envmap_weight)
    envmap_probs = envmap_weight / (uniform_weight + envmap_weight)

    # 1. Uniform (Fibonacci) sampling
    dirs_uniform, pdf_uniform = fibonacci_sphere_sampling(normals, num_sample_rays, random_rotate=random_rotate)
    # Envmap pdf for the uniform-sampled directions
    pdf_env_uniform = pdf_envmap(
        normals.repeat_interleave(num_sample_rays, 0),
        dirs_uniform.reshape(-1, 3),
        envmap=envmap
    ).reshape(N, num_sample_rays)

    # 2. Environment map sampling
    dirs_envmap, pdf_envmap_ = sample_envmap(normals, envmap, num_sample_rays, device=device)
    # Uniform pdf for envmap-sampled directions (always 1 / 2π on hemisphere)
    pdf_uniform_envmap = torch.full_like(pdf_envmap_, 1.0 / (2 * math.pi))

    # Decide sampling origin per direction
    rand_vals = torch.rand(N, num_sample_rays, device=device)
    uniform_mask = rand_vals < uniform_probs[:, None]

    # Initialize output buffers
    directions = torch.zeros((N, num_sample_rays, 3), device=device)
    pdfs = torch.zeros((N, num_sample_rays), device=device)

    # Combine directions
    directions[uniform_mask] = dirs_uniform[uniform_mask]
    directions[~uniform_mask] = dirs_envmap[~uniform_mask]

    # Combine PDFs using MIS
    pdf_uniform_weighted = pdf_uniform * uniform_probs[:, None] + pdf_env_uniform * envmap_probs[:, None]
    pdf_envmap_weighted = pdf_envmap_ * envmap_probs[:, None] + pdf_uniform_envmap * uniform_probs[:, None]

    pdfs[uniform_mask] = pdf_uniform_weighted[uniform_mask]
    pdfs[~uniform_mask] = pdf_envmap_weighted[~uniform_mask]

    return directions, pdfs, uniform_mask

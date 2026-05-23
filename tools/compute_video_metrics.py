import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from datasets.driving_dataset import DrivingDataset
from utils.config import resolve_config

logger = logging.getLogger(__name__)


def _resolve_cfg(
    config_file: str,
    opts: List[str],
    dataset_override: Optional[str] = None,
    overlays: Optional[List[str]] = None,
) -> OmegaConf:
    return resolve_config(
        config_file=config_file,
        opts=opts,
        dataset_override=dataset_override,
        overlays=overlays,
    )


def _build_split_timesteps(num_timesteps: int, stride: int) -> Tuple[List[int], List[int]]:
    if stride > 0:
        test_local = list(np.arange(stride, num_timesteps, stride))
    else:
        test_local = []
    test_set = set(test_local)
    train_local = [i for i in range(num_timesteps) if i not in test_set]
    return train_local, test_local


def _find_cam_video(video_root: str, cam_id: int, video_name: str) -> str:
    candidates = [
        os.path.join(video_root, str(cam_id), video_name),
        os.path.join(video_root, f"{cam_id}.mp4"),
        os.path.join(video_root, f"cam_{cam_id}.mp4"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find video for cam_id={cam_id}. Tried: {candidates}")


def _prepare_render_frame(frame: np.ndarray, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    render_np = frame.astype(np.float32) / 255.0
    render_t = torch.from_numpy(render_np).to(device=device, dtype=torch.float32)
    render_t = render_t.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    if render_t.shape[-2] != h or render_t.shape[-1] != w:
        render_t = F.interpolate(render_t, size=(h, w), mode="bilinear", align_corners=False)
    render_t = render_t.clamp(0.0, 1.0)
    render_np_resized = render_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return render_t, render_np_resized


def _compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt)
    return (-10 * torch.log10(mse)).item()


def _compute_masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask_hw: torch.Tensor, eps: float = 1e-8) -> float:
    # pred/gt: [1, C, H, W], mask_hw: [H, W] bool/float
    mask = (mask_hw > 0.5).float()[None, None, ...]  # [1,1,H,W]
    denom = (mask.sum() * pred.shape[1]).item()
    if denom <= 0:
        return -1.0
    mse = (((pred - gt) ** 2) * mask).sum() / (mask.sum() * pred.shape[1] + eps)
    return (-10 * torch.log10(mse + eps)).item()


def _compute_masked_rmse(pred_hw: torch.Tensor, gt_hw: torch.Tensor, mask_hw: torch.Tensor, eps: float = 1e-8) -> float:
    mask = (mask_hw > 0.5)
    if mask.sum().item() == 0:
        return -1.0
    diff2 = (pred_hw - gt_hw) ** 2
    rmse = torch.sqrt(diff2[mask].mean() + eps)
    return rmse.item()


def _compute_masked_lpips(lpips_fn, pred_bchw: torch.Tensor, gt_bchw: torch.Tensor, mask_hw: torch.Tensor) -> float:
    num_masked = (mask_hw > 0.5).sum().item()
    if num_masked == 0:
        return -1.0
    h, w = mask_hw.shape[-2], mask_hw.shape[-1]
    num_total = h * w
    mask_bchw = (mask_hw > 0.5).float()[None, None, ...]
    raw = float(lpips_fn(
        (pred_bchw * mask_bchw).clamp(0.0, 1.0),
        (gt_bchw   * mask_bchw).clamp(0.0, 1.0),
    ).item())
    return raw * (num_total / num_masked)


def _srgb_to_linear(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    return x.clamp(0.0, 1.0) ** gamma


def _aggregate_values(values: List[float], name: str) -> Dict[str, float]:
    valid = [v for v in values if v >= 0]
    if len(valid) == 0:
        return {"count": 0, name: -1.0}
    return {"count": len(valid), name: float(np.mean(valid))}


def _aggregate_triplet(psnrs: List[float], ssims: List[float], lpipss: List[float]) -> Dict[str, float]:
    p = [v for v in psnrs if v >= 0]
    s = [v for v in ssims if v >= 0]
    l = [v for v in lpipss if v >= 0]
    return {
        "count": len(p),
        "psnr": float(np.mean(p)) if len(p) > 0 else -1.0,
        "ssim": float(np.mean(s)) if len(s) > 0 else -1.0,
        "lpips": float(np.mean(l)) if len(l) > 0 else -1.0,
    }


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()
    return x


def _solve_albedo_scale(pred_lin_bchw: torch.Tensor, gt_lin_bchw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
    # Return scaled pred in linear space, same shape [1,3,H,W]
    pred_hwc = pred_lin_bchw.squeeze(0).permute(1, 2, 0)
    gt_hwc = gt_lin_bchw.squeeze(0).permute(1, 2, 0)
    valid = mask_hw > 0.5
    scaled = pred_hwc.clone()
    eps = 1e-8
    for c in range(3):
        p = pred_hwc[..., c][valid]
        g = gt_hwc[..., c][valid]
        if p.numel() == 0:
            s = torch.tensor(1.0, device=pred_hwc.device, dtype=pred_hwc.dtype)
        else:
            s = (p * g).sum() / (p * p).sum().clamp_min(eps)
        scaled[..., c] = pred_hwc[..., c] * s
    return scaled.permute(2, 0, 1).unsqueeze(0)


def affine_color_correct(gt_img: np.ndarray, pred_img: np.ndarray) -> np.ndarray:
    """Per-image, per-channel affine color correction (Zip-NeRF / Mip-NeRF 360 protocol).

    Fits a scale+bias transform pred→gt per channel via least-squares, then applies
    it to pred so that PSNR/SSIM/LPIPS reflect geometry, not photometric drift.
    """
    corrected = pred_img.astype(np.float64).copy()
    for c in range(pred_img.shape[2]):
        x = pred_img[:, :, c].astype(np.float64).ravel()  # pred channel
        y = gt_img[:, :, c].astype(np.float64).ravel()    # gt channel
        # Least-squares: y ≈ a*x + b
        A = np.stack([x, np.ones_like(x)], axis=1)
        result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = result
        corrected[:, :, c] = a * pred_img[:, :, c].astype(np.float64) + b
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _apply_color_correction(pred_np_01: np.ndarray, gt_np_01: np.ndarray, device: torch.device):
    """Apply affine_color_correct on [0,1] float arrays; return corrected (np float32 [0,1], torch [1,C,H,W])."""
    pred_u8 = (pred_np_01 * 255.0).clip(0, 255).astype(np.uint8)
    gt_u8 = (gt_np_01 * 255.0).clip(0, 255).astype(np.uint8)
    corrected_u8 = affine_color_correct(gt_u8, pred_u8)
    corrected_np = corrected_u8.astype(np.float32) / 255.0
    corrected_t = torch.from_numpy(corrected_np).to(device=device, dtype=torch.float32)
    corrected_bchw = corrected_t.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
    return corrected_np, corrected_bchw


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser("Compute image/relighting/intrinsic metrics from rendered camera videos.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument(
        "--config_overlay",
        action="append",
        default=[],
        help=(
            "optional overlay config; can be repeated. Merge order is "
            "base -> dataset -> overlays -> CLI opts"
        ),
    )
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--start_timestep", type=int, required=True)
    parser.add_argument("--end_timestep", type=int, required=True, help="Inclusive end timestep")
    parser.add_argument("--test_image_stride", type=int, required=True)
    parser.add_argument("--video_name", type=str, default="pbr_rgb.mp4")
    parser.add_argument("--relight_video_root", type=str, default=None)
    parser.add_argument("--relight_video_name", type=str, default="relight.mp4")
    parser.add_argument("--intrinsic_video_root", type=str, default=None)
    parser.add_argument("--albedo_name", type=str, default="albedo.mp4")
    parser.add_argument("--normal_name", type=str, default="normal.mp4")
    parser.add_argument("--roughness_name", type=str, default="roughness.mp4")
    parser.add_argument("--metallic_name", type=str, default="metallic.mp4")
    parser.add_argument("--disable_normal", action="store_true", help="Skip normal intrinsic evaluation.")
    parser.add_argument("--disable_roughness", action="store_true", help="Skip roughness intrinsic evaluation.")
    parser.add_argument("--disable_metallic", action="store_true", help="Skip metallic intrinsic evaluation.")
    parser.add_argument("--image_output_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None) # alias for backward compatibility; maps to image_output_json if provided and image_output_json is not set.
    parser.add_argument("--relight_output_json", type=str, default=None)
    parser.add_argument("--intrinsic_output_json", type=str, default=None)
    parser.add_argument("--rou_met_srgb", action="store_true", help="Whether to apply sRGB->linear conversion for roughness/metallic videos before computing metrics.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override, e.g. self/3cams")
    parser.add_argument("--dataset_source", type=str, default=None, choices=["primary", "external"], help="For self dataset, choose GT source branch")
    parser.add_argument("--color_correction", action="store_true", help="Apply per-image per-channel affine color correction (LSQ scale+bias) to predictions before computing metrics.")
    parser.add_argument("--paste_gt_sky_for_image", action="store_true", help="For image RGB metrics, replace predicted sky pixels with GT RGB using the dataset sky mask before computing metrics.")
    parser.add_argument("--ground_only_relight", action="store_true", help="Compute relight RGB metrics on non-sky pixels only.")
    parser.add_argument("--paste_gt_sky_for_relight", action="store_true", help="For relight RGB metrics, replace predicted sky pixels with GT relighted RGB using gt_sky_masks.")
    parser.add_argument("--cam_ids", type=int, nargs="+", default=None, help="Optional camera ids to evaluate. Default: all dataset cameras.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None, help="Extra OmegaConf overrides")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    assert args.end_timestep >= args.start_timestep, "end_timestep must be >= start_timestep"
    assert args.test_image_stride >= 0, "test_image_stride must be >= 0"
    # Backward compatibility: --output_json maps to image output if provided.
    if args.image_output_json is None and args.output_json is not None:
        args.image_output_json = args.output_json

    enable_image = args.video_root is not None and args.image_output_json is not None
    enable_relight = args.relight_video_root is not None and args.relight_output_json is not None
    enable_intrinsic = args.intrinsic_video_root is not None and args.intrinsic_output_json is not None

    if not (enable_image or enable_relight or enable_intrinsic):
        raise ValueError(
            "Nothing to evaluate. Provide at least one complete set:\n"
            "  image: --video_root + --image_output_json (or --output_json)\n"
            "  relight: --relight_video_root + --relight_output_json\n"
            "  intrinsic: --intrinsic_video_root + --intrinsic_output_json"
        )

    if enable_image:
        assert os.path.isdir(args.video_root), f"video_root does not exist: {args.video_root}"
    if enable_relight:
        assert os.path.isdir(args.relight_video_root), f"relight_video_root does not exist: {args.relight_video_root}"
    if enable_intrinsic:
        assert os.path.isdir(args.intrinsic_video_root), f"intrinsic_video_root does not exist: {args.intrinsic_video_root}"

    cfg = _resolve_cfg(
        args.config_file,
        args.opts,
        dataset_override=args.dataset,
        overlays=args.config_overlay,
    )
    cfg.data.start_timestep = int(args.start_timestep)
    cfg.data.end_timestep = int(args.end_timestep)
    if args.dataset_source is not None:
        if cfg.data.dataset != "self":
            raise ValueError("--dataset_source is only supported for self dataset.")
        if "external_source" not in cfg.data.pixel_source:
            cfg.data.pixel_source.external_source = OmegaConf.create({})
        cfg.data.pixel_source.external_source.active_source = args.dataset_source
        logger.info("Using self dataset source for metric GT: %s", args.dataset_source)
    if "lidar_source" in cfg.data and "load_lidar" in cfg.data.lidar_source:
        cfg.data.lidar_source.load_lidar = False

    logger.info("Building dataset...")
    dataset = DrivingDataset(data_cfg=cfg.data)
    num_cams = dataset.pixel_source.num_cams
    num_timesteps = dataset.num_img_timesteps
    logger.info("Dataset ready: %d timesteps, %d cams", num_timesteps, num_cams)
    if args.cam_ids is None:
        eval_cam_ids = list(range(num_cams))
    else:
        eval_cam_ids = [int(c) for c in args.cam_ids]
        bad = [c for c in eval_cam_ids if c < 0 or c >= num_cams]
        if bad:
            raise ValueError(f"Invalid cam_ids {bad} for dataset with num_cams={num_cams}")
        # Keep order stable, remove duplicates.
        eval_cam_ids = list(dict.fromkeys(eval_cam_ids))
    logger.info("Evaluating cam ids: %s", eval_cam_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    lpips_metric.eval()

    # Base image metric readers
    image_readers: Optional[Dict[int, imageio.core.format.Reader]] = None
    if enable_image:
        image_readers = {}
        for cam_id in eval_cam_ids:
            video_path = _find_cam_video(args.video_root, cam_id, args.video_name)
            logger.info("Using image cam %d video: %s", cam_id, video_path)
            image_readers[cam_id] = imageio.get_reader(video_path)

    # Optional relighting readers
    relight_readers: Optional[Dict[int, imageio.core.format.Reader]] = None
    if enable_relight:
        relight_readers = {}
        for cam_id in eval_cam_ids:
            video_path = _find_cam_video(args.relight_video_root, cam_id, args.relight_video_name)
            logger.info("Using relight cam %d video: %s", cam_id, video_path)
            relight_readers[cam_id] = imageio.get_reader(video_path)

    eval_normal = enable_intrinsic and not args.disable_normal
    eval_roughness = enable_intrinsic and not args.disable_roughness
    eval_metallic = enable_intrinsic and not args.disable_metallic

    # Optional intrinsic readers
    intrinsic_readers: Optional[Dict[str, Dict[int, imageio.core.format.Reader]]] = None
    if enable_intrinsic:
        intrinsic_readers = {"albedo": {}}
        if eval_normal:
            intrinsic_readers["normal"] = {}
        if eval_roughness:
            intrinsic_readers["roughness"] = {}
        if eval_metallic:
            intrinsic_readers["metallic"] = {}
        names = {
            "albedo": args.albedo_name,
        }
        if eval_normal:
            names["normal"] = args.normal_name
        if eval_roughness:
            names["roughness"] = args.roughness_name
        if eval_metallic:
            names["metallic"] = args.metallic_name
        for intr_key, vname in names.items():
            for cam_id in eval_cam_ids:
                video_path = _find_cam_video(args.intrinsic_video_root, cam_id, vname)
                logger.info("Using %s cam %d video: %s", intr_key, cam_id, video_path)
                intrinsic_readers[intr_key][cam_id] = imageio.get_reader(video_path)

    train_local, test_local = _build_split_timesteps(num_timesteps, args.test_image_stride)
    test_local_set = set(test_local)

    # Buckets
    image_metrics = (
        {"train": {"psnr": [], "ssim": [], "lpips": []}, "test": {"psnr": [], "ssim": [], "lpips": []}}
        if enable_image
        else None
    )
    relight = (
        {"train": {"psnr": [], "ssim": [], "lpips": []}, "test": {"psnr": [], "ssim": [], "lpips": []}}
        if enable_relight
        else None
    )
    intrinsic = (
        {
            "albedo": {
                "train": {"psnr": [], "lpips": [], "si_psnr": [], "si_lpips": []},
                "test": {"psnr": [], "lpips": [], "si_psnr": [], "si_lpips": []},
            },
        }
        if enable_intrinsic
        else None
    )
    if enable_intrinsic and eval_normal:
        intrinsic["normal"] = {"train": {"mae_deg": []}, "test": {"mae_deg": []}}
    if enable_intrinsic and eval_roughness:
        intrinsic["roughness"] = {"train": {"rmse": []}, "test": {"rmse": []}}
    if enable_intrinsic and eval_metallic:
        intrinsic["metallic"] = {"train": {"rmse": []}, "test": {"rmse": []}}

    per_frame = []
    total = num_timesteps * len(eval_cam_ids)
    for t_local in tqdm(range(num_timesteps), total=num_timesteps, desc="Timesteps", dynamic_ncols=True):
        split = "test" if t_local in test_local_set else "train"
        for cam_id in eval_cam_ids:
            global_idx = t_local * num_cams + cam_id
            image_infos, _ = dataset.full_image_set.get_image(global_idx, camera_downscale=1.0)
            gt_rgb = image_infos["pixels"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)  # [H,W,3]
            h, w = gt_rgb.shape[0], gt_rgb.shape[1]
            gt_rgb_bchw = gt_rgb.permute(2, 0, 1).unsqueeze(0)

            row = {
                "timestep_local": int(t_local),
                "timestep_abs": int(args.start_timestep + t_local),
                "cam_id": int(cam_id),
                "global_idx_local_dataset": int(global_idx),
                "split": split,
            }

            # Base image metrics
            if enable_image:
                image_frame = image_readers[cam_id].get_data(t_local)
                image_pred_bchw, image_pred_np = _prepare_render_frame(image_frame, h, w, device)
                gt_rgb_np = gt_rgb.detach().cpu().numpy()
                if args.paste_gt_sky_for_image:
                    sky_key = "gt_sky_masks" if "gt_sky_masks" in image_infos else "sky_masks"
                    if sky_key not in image_infos:
                        raise KeyError(
                            "Neither gt_sky_masks nor sky_masks found in image_infos; required for --paste_gt_sky_for_image."
                        )
                    sky_mask_hw = image_infos[sky_key].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                    sky_mask_bchw = sky_mask_hw[None, None, ...]
                    image_pred_bchw = image_pred_bchw * (1.0 - sky_mask_bchw) + gt_rgb_bchw * sky_mask_bchw
                    image_pred_np = image_pred_bchw.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                if args.color_correction:
                    image_pred_np, image_pred_bchw = _apply_color_correction(image_pred_np, gt_rgb_np, device)
                image_psnr = _compute_psnr(image_pred_bchw, gt_rgb_bchw)
                image_ssim = float(ssim(image_pred_np, gt_rgb_np, data_range=1.0, channel_axis=-1))
                image_lpips = float(lpips_metric(image_pred_bchw, gt_rgb_bchw).item())
                image_metrics[split]["psnr"].append(image_psnr)
                image_metrics[split]["ssim"].append(image_ssim)
                image_metrics[split]["lpips"].append(image_lpips)
                row["psnr"] = image_psnr
                row["ssim"] = image_ssim
                row["lpips"] = image_lpips

            # Relighting (RGB metrics)
            if enable_relight:
                if "relighted_pixels" not in image_infos:
                    raise KeyError(
                        "relighted_pixels not found in image_infos. "
                        "Enable data.pixel_source.load_relighted_rgb=true and set relighted_scene_idx."
                    )
                gt_relight = image_infos["relighted_pixels"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                gt_relight_bchw = gt_relight.permute(2, 0, 1).unsqueeze(0)
                gt_relight_np = gt_relight.detach().cpu().numpy()
                relight_frame = relight_readers[cam_id].get_data(t_local)
                relight_pred_bchw, relight_pred_np = _prepare_render_frame(relight_frame, h, w, device)
                if args.color_correction:
                    relight_pred_np, relight_pred_bchw = _apply_color_correction(relight_pred_np, gt_relight_np, device)
                if args.paste_gt_sky_for_relight:
                    if "gt_sky_masks" not in image_infos or image_infos["gt_sky_masks"] is None:
                        raise KeyError(
                            "gt_sky_masks not found in image_infos; "
                            "--paste_gt_sky_for_relight requires data.pixel_source.load_gt_sky_mask=true."
                        )
                    sky = image_infos["gt_sky_masks"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                    sky_bchw = sky[None, None, ...]
                    relight_pred_bchw = relight_pred_bchw * (1.0 - sky_bchw) + gt_relight_bchw * sky_bchw
                    relight_pred_np = relight_pred_bchw.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    r_psnr = _compute_psnr(relight_pred_bchw, gt_relight_bchw)
                    r_ssim = float(ssim(relight_pred_np, gt_relight_np, data_range=1.0, channel_axis=-1))
                    r_lpips = float(lpips_metric(relight_pred_bchw, gt_relight_bchw).item())
                elif args.ground_only_relight:
                    sky_key = "gt_sky_masks" if "gt_sky_masks" in image_infos else "sky_masks"
                    if sky_key not in image_infos:
                        raise KeyError(
                            "Neither gt_sky_masks nor sky_masks found in image_infos; required for --ground_only_relight."
                        )
                    sky = image_infos[sky_key].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                    ground_mask_hw = (1.0 - sky).clamp(0.0, 1.0)
                    ground_mask_np = ground_mask_hw.detach().cpu().numpy()[..., None]
                    r_psnr = _compute_masked_psnr(relight_pred_bchw, gt_relight_bchw, ground_mask_hw)
                    r_ssim = float(
                        ssim(
                            relight_pred_np * ground_mask_np,
                            gt_relight_np * ground_mask_np,
                            data_range=1.0,
                            channel_axis=-1,
                        )
                    )
                    r_lpips = _compute_masked_lpips(lpips_metric, relight_pred_bchw, gt_relight_bchw, ground_mask_hw)
                else:
                    r_psnr = _compute_psnr(relight_pred_bchw, gt_relight_bchw)
                    r_ssim = float(ssim(relight_pred_np, gt_relight_np, data_range=1.0, channel_axis=-1))
                    r_lpips = float(lpips_metric(relight_pred_bchw, gt_relight_bchw).item())
                relight[split]["psnr"].append(r_psnr)
                relight[split]["ssim"].append(r_ssim)
                relight[split]["lpips"].append(r_lpips)
                row["relight_psnr"] = r_psnr
                row["relight_ssim"] = r_ssim
                row["relight_lpips"] = r_lpips

            # Intrinsic metrics (ground-only)
            if enable_intrinsic:
                sky_key = "gt_sky_masks" if "gt_sky_masks" in image_infos else "sky_masks"
                if sky_key not in image_infos:
                    raise KeyError("Neither gt_sky_masks nor sky_masks found in image_infos; required for ground-only intrinsic metrics.")
                sky = image_infos[sky_key].to(device=device, dtype=torch.float32)
                ground_mask_hw = (1.0 - sky).clamp(0.0, 1.0)

                if eval_normal:
                    # --- normal ---
                    if "blender_normal" not in image_infos:
                        raise KeyError(
                            "blender_normal not found in image_infos. "
                            "Enable data.pixel_source.load_gt_intrinsic=true."
                        )
                    gt_normal = image_infos["blender_normal"].to(device=device, dtype=torch.float32)  # as-is from EXR
                    pred_normal_frame = intrinsic_readers["normal"][cam_id].get_data(t_local)
                    pred_normal_bchw, _ = _prepare_render_frame(pred_normal_frame, h, w, device)
                    pred_normal = pred_normal_bchw.squeeze(0).permute(1, 2, 0)  # [H,W,3]
                    pred_normal = pred_normal * 2.0 - 1.0  # [0,1] -> [-1,1]
                    pred_normal = F.normalize(pred_normal, dim=-1, eps=1e-8)
                    gt_normal = F.normalize(gt_normal, dim=-1, eps=1e-8)
                    valid = ground_mask_hw > 0.5
                    if valid.sum().item() > 0:
                        dot = (pred_normal * gt_normal).sum(dim=-1).clamp(-1.0, 1.0)
                        ang = torch.rad2deg(torch.acos(dot))
                        n_mae = float(ang[valid].mean().item())
                    else:
                        n_mae = -1.0
                    intrinsic["normal"][split]["mae_deg"].append(n_mae)
                    row["normal_mae_deg"] = n_mae

                # --- albedo ---
                gt_albedo = image_infos["blender_albedo"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                gt_albedo_bchw = gt_albedo.permute(2, 0, 1).unsqueeze(0)
                pred_albedo_frame = intrinsic_readers["albedo"][cam_id].get_data(t_local)
                pred_albedo_bchw, _ = _prepare_render_frame(pred_albedo_frame, h, w, device)
                # Assume GT albedo is EXR in linear space; predicted albedo video is sRGB.
                gt_albedo_lin = gt_albedo_bchw
                pred_albedo_lin = _srgb_to_linear(pred_albedo_bchw)
                a_psnr = _compute_masked_psnr(pred_albedo_lin, gt_albedo_lin, ground_mask_hw)
                # LPIPS with masked images (renormalized to mask area)
                a_lpips = _compute_masked_lpips(lpips_metric, pred_albedo_lin, gt_albedo_lin, ground_mask_hw)
                # SI metrics
                pred_albedo_si = _solve_albedo_scale(pred_albedo_lin, gt_albedo_lin, ground_mask_hw)
                a_si_psnr = _compute_masked_psnr(pred_albedo_si, gt_albedo_lin, ground_mask_hw)
                a_si_lpips = _compute_masked_lpips(lpips_metric, pred_albedo_si, gt_albedo_lin, ground_mask_hw)
                intrinsic["albedo"][split]["psnr"].append(a_psnr)
                intrinsic["albedo"][split]["lpips"].append(a_lpips)
                intrinsic["albedo"][split]["si_psnr"].append(a_si_psnr)
                intrinsic["albedo"][split]["si_lpips"].append(a_si_lpips)
                row["albedo_psnr"] = a_psnr
                row["albedo_lpips"] = a_lpips
                row["albedo_si_psnr"] = a_si_psnr
                row["albedo_si_lpips"] = a_si_lpips

                if eval_roughness:
                    # --- roughness ---
                    gt_rough = image_infos["blender_roughness"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                    pred_rough_frame = intrinsic_readers["roughness"][cam_id].get_data(t_local)
                    pred_rough_bchw, _ = _prepare_render_frame(pred_rough_frame, h, w, device)
                    pred_rough = pred_rough_bchw.squeeze(0).mean(dim=0)  # [H,W]
                    if args.rou_met_srgb:
                        pred_rough = _srgb_to_linear(pred_rough)
                    # else, roughness is assumed to be already in linear space
                    r_rmse = _compute_masked_rmse(pred_rough, gt_rough, ground_mask_hw)
                    intrinsic["roughness"][split]["rmse"].append(r_rmse)
                    row["roughness_rmse"] = r_rmse

                if eval_metallic:
                    # --- metallic ---
                    gt_metal = image_infos["blender_metallic"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                    pred_metal_frame = intrinsic_readers["metallic"][cam_id].get_data(t_local)
                    pred_metal_bchw, _ = _prepare_render_frame(pred_metal_frame, h, w, device)
                    pred_metal = pred_metal_bchw.squeeze(0).mean(dim=0)  # [H,W]
                    if args.rou_met_srgb:
                        pred_metal = _srgb_to_linear(pred_metal)
                    # else, metallic is assumed to be already in linear space
                    m_rmse = _compute_masked_rmse(pred_metal, gt_metal, ground_mask_hw)
                    intrinsic["metallic"][split]["rmse"].append(m_rmse)
                    row["metallic_rmse"] = m_rmse

            per_frame.append(row)

    # Close all readers
    if image_readers is not None:
        for reader in image_readers.values():
            reader.close()
    if relight_readers is not None:
        for reader in relight_readers.values():
            reader.close()
    if intrinsic_readers is not None:
        for intr in intrinsic_readers.values():
            for reader in intr.values():
                reader.close()

    train_abs = [args.start_timestep + t for t in train_local]
    test_abs = [args.start_timestep + t for t in test_local]

    metrics_out = {}
    if enable_image:
        metrics_out["train"] = _aggregate_triplet(
            image_metrics["train"]["psnr"], image_metrics["train"]["ssim"], image_metrics["train"]["lpips"]
        )
        metrics_out["test"] = _aggregate_triplet(
            image_metrics["test"]["psnr"], image_metrics["test"]["ssim"], image_metrics["test"]["lpips"]
        )

    if enable_relight:
        metrics_out["train_relight"] = _aggregate_triplet(
            relight["train"]["psnr"], relight["train"]["ssim"], relight["train"]["lpips"]
        )
        metrics_out["test_relight"] = _aggregate_triplet(
            relight["test"]["psnr"], relight["test"]["ssim"], relight["test"]["lpips"]
        )

    if enable_intrinsic:
        if eval_normal:
            metrics_out["train_normal"] = _aggregate_values(intrinsic["normal"]["train"]["mae_deg"], "mae_deg")
            metrics_out["test_normal"] = _aggregate_values(intrinsic["normal"]["test"]["mae_deg"], "mae_deg")
        if eval_roughness:
            metrics_out["train_roughness"] = _aggregate_values(intrinsic["roughness"]["train"]["rmse"], "rmse")
            metrics_out["test_roughness"] = _aggregate_values(intrinsic["roughness"]["test"]["rmse"], "rmse")
        if eval_metallic:
            metrics_out["train_metallic"] = _aggregate_values(intrinsic["metallic"]["train"]["rmse"], "rmse")
            metrics_out["test_metallic"] = _aggregate_values(intrinsic["metallic"]["test"]["rmse"], "rmse")
        metrics_out["train_albedo"] = {
            "count": len([v for v in intrinsic["albedo"]["train"]["psnr"] if v >= 0]),
            "psnr": float(np.mean([v for v in intrinsic["albedo"]["train"]["psnr"] if v >= 0])) if len([v for v in intrinsic["albedo"]["train"]["psnr"] if v >= 0]) > 0 else -1.0,
            "lpips": float(np.mean([v for v in intrinsic["albedo"]["train"]["lpips"] if v >= 0])) if len([v for v in intrinsic["albedo"]["train"]["lpips"] if v >= 0]) > 0 else -1.0,
            "si_psnr": float(np.mean([v for v in intrinsic["albedo"]["train"]["si_psnr"] if v >= 0])) if len([v for v in intrinsic["albedo"]["train"]["si_psnr"] if v >= 0]) > 0 else -1.0,
            "si_lpips": float(np.mean([v for v in intrinsic["albedo"]["train"]["si_lpips"] if v >= 0])) if len([v for v in intrinsic["albedo"]["train"]["si_lpips"] if v >= 0]) > 0 else -1.0,
        }
        metrics_out["test_albedo"] = {
            "count": len([v for v in intrinsic["albedo"]["test"]["psnr"] if v >= 0]),
            "psnr": float(np.mean([v for v in intrinsic["albedo"]["test"]["psnr"] if v >= 0])) if len([v for v in intrinsic["albedo"]["test"]["psnr"] if v >= 0]) > 0 else -1.0,
            "lpips": float(np.mean([v for v in intrinsic["albedo"]["test"]["lpips"] if v >= 0])) if len([v for v in intrinsic["albedo"]["test"]["lpips"] if v >= 0]) > 0 else -1.0,
            "si_psnr": float(np.mean([v for v in intrinsic["albedo"]["test"]["si_psnr"] if v >= 0])) if len([v for v in intrinsic["albedo"]["test"]["si_psnr"] if v >= 0]) > 0 else -1.0,
            "si_lpips": float(np.mean([v for v in intrinsic["albedo"]["test"]["si_lpips"] if v >= 0])) if len([v for v in intrinsic["albedo"]["test"]["si_lpips"] if v >= 0]) > 0 else -1.0,
        }

    base_meta = {
        "config_file": args.config_file,
        "start_timestep": int(args.start_timestep),
        "end_timestep": int(args.end_timestep),
        "num_timesteps": int(num_timesteps),
        "num_cams": int(num_cams),
        "evaluated_cam_ids": [int(c) for c in eval_cam_ids],
        "num_evaluated_cams": int(len(eval_cam_ids)),
        "total_evaluated_images": int(total),
        "test_image_stride": int(args.test_image_stride),
        "dataset_type": str(cfg.data.dataset),
        "dataset_source": str(
            cfg.data.pixel_source.get("external_source", {}).get("active_source", "primary")
        ),
        "scene_idx": str(cfg.data.scene_idx),
        "averaging": "per-frame-per-camera metrics averaged across selected samples",
    }
    splits = {
        "train_timestep_local": train_local,
        "test_timestep_local": test_local,
        "train_timestep_abs": train_abs,
        "test_timestep_abs": test_abs,
    }

    # Save image metrics json
    if enable_image:
        image_result = {
            "meta": {
                **base_meta,
                "video_root": args.video_root,
                "video_name": args.video_name,
                "paste_gt_sky_for_image": bool(args.paste_gt_sky_for_image),
            },
            "splits": splits,
            "metrics": {
                "train": metrics_out["train"],
                "test": metrics_out["test"],
            },
            "per_frame": [r for r in per_frame if "psnr" in r],
        }
        _ensure_parent_dir(args.image_output_json)
        with open(args.image_output_json, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(image_result), f, indent=2)
        logger.info("Saved image metrics json: %s", args.image_output_json)

    # Save relighting metrics json
    if enable_relight:
        if "train_relight" in metrics_out and "test_relight" in metrics_out:
            relight_per_frame = [r for r in per_frame if "relight_psnr" in r]
            relight_result = {
                "meta": {
                    **base_meta,
                    "relight_video_root": args.relight_video_root,
                    "relight_video_name": args.relight_video_name,
                    "ground_only_relight": bool(args.ground_only_relight),
                    "paste_gt_sky_for_relight": bool(args.paste_gt_sky_for_relight),
                    "sky_mask_source": "gt_sky_masks" if args.paste_gt_sky_for_relight else None,
                },
                "splits": splits,
                "metrics": {
                    "train_relight": metrics_out["train_relight"],
                    "test_relight": metrics_out["test_relight"],
                },
                "per_frame": relight_per_frame,
            }
            _ensure_parent_dir(args.relight_output_json)
            with open(args.relight_output_json, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(relight_result), f, indent=2)
            logger.info("Saved relighting metrics json: %s", args.relight_output_json)
        else:
            logger.warning(
                "relight_output_json is set but relighting metrics are unavailable. "
                "Provide --relight_video_root/--relight_video_name and relighting GT in dataset."
            )

    # Save intrinsic metrics json
    if enable_intrinsic:
        required_intrinsic_keys = ["train_albedo", "test_albedo"]
        if eval_normal:
            required_intrinsic_keys.extend(["train_normal", "test_normal"])
        if eval_roughness:
            required_intrinsic_keys.extend(["train_roughness", "test_roughness"])
        if eval_metallic:
            required_intrinsic_keys.extend(["train_metallic", "test_metallic"])
        has_intrinsic = all(k in metrics_out for k in required_intrinsic_keys)
        if has_intrinsic:
            intrinsic_metrics = {
                "train_albedo": metrics_out["train_albedo"],
                "test_albedo": metrics_out["test_albedo"],
            }
            if eval_normal:
                intrinsic_metrics["train_normal"] = metrics_out["train_normal"]
                intrinsic_metrics["test_normal"] = metrics_out["test_normal"]
            if eval_roughness:
                intrinsic_metrics["train_roughness"] = metrics_out["train_roughness"]
                intrinsic_metrics["test_roughness"] = metrics_out["test_roughness"]
            if eval_metallic:
                intrinsic_metrics["train_metallic"] = metrics_out["train_metallic"]
                intrinsic_metrics["test_metallic"] = metrics_out["test_metallic"]
            intrinsic_result = {
                "meta": {
                    **base_meta,
                    "intrinsic_video_root": args.intrinsic_video_root,
                    "albedo_name": args.albedo_name,
                    "normal_name": args.normal_name if eval_normal else None,
                    "roughness_name": args.roughness_name if eval_roughness else None,
                    "metallic_name": args.metallic_name if eval_metallic else None,
                    "eval_normal": bool(eval_normal),
                    "eval_roughness": bool(eval_roughness),
                    "eval_metallic": bool(eval_metallic),
                },
                "splits": splits,
                "metrics": intrinsic_metrics,
                "per_frame": [r for r in per_frame if "albedo_psnr" in r],
            }
            _ensure_parent_dir(args.intrinsic_output_json)
            with open(args.intrinsic_output_json, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(intrinsic_result), f, indent=2)
            logger.info("Saved intrinsic metrics json: %s", args.intrinsic_output_json)
        else:
            logger.warning(
                "intrinsic_output_json is set but intrinsic metrics are unavailable. "
                "Provide --intrinsic_video_root and GT intrinsic in dataset."
            )
    if "train" in metrics_out:
        logger.info("Train image metrics: %s", metrics_out["train"])
    if "test" in metrics_out:
        logger.info("Test image metrics:  %s", metrics_out["test"])


if __name__ == "__main__":
    main()

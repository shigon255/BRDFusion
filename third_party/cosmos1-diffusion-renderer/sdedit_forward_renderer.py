#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SDEdit-style refinement using the Forward Renderer.

This script takes an input RGB video, G-buffer intrinsics, and an environment map,
then refines the RGB video using the forward renderer's learned prior while
conditioning on the intrinsics and lighting.
"""

import argparse
import os
import torch
import imageio
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, Optional
import cv2
import torchvision.transforms as transforms
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.transform_utils import prepare_images_np

from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.utils_env_proj import process_environment_map
from cosmos_predict1.diffusion.model.model_diffusion_renderer import DiffusionRendererModel
from cosmos_predict1.utils import misc, log
from cosmos_predict1.utils.io import save_image_or_video


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDEdit refinement using Forward Renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python sdedit_forward_renderer.py \\
    --input_rgb_video data/rgb_video.mp4 \\
    --basecolor_video data/basecolor \\
    --normal_video data/normal \\
    --depth_video data/depth \\
    --roughness_video data/roughness \\
    --metallic_video data/metallic \\
    --env_map data/studio.hdr \\
    --output_video output/refined.mp4 \\
    --sdedit_strength 0.4 \\
    --guidance 7.5
        """
    )

    # Input files
    parser.add_argument(
        "--input_rgb_video",
        type=str,
        required=True,
        help="Path to input RGB video file or directory of frames"
    )
    parser.add_argument(
        "--basecolor_video",
        type=str,
        required=True,
        help="Path to basecolor/albedo video file or directory of frames"
    )
    parser.add_argument(
        "--normal_video",
        type=str,
        required=True,
        help="Path to normal map video file or directory of frames"
    )
    parser.add_argument(
        "--depth_video",
        type=str,
        required=True,
        help="Path to depth map video file or directory of frames"
    )
    parser.add_argument(
        "--roughness_video",
        type=str,
        required=True,
        help="Path to roughness map video file or directory of frames"
    )
    parser.add_argument(
        "--metallic_video",
        type=str,
        required=True,
        help="Path to metallic map video file or directory of frames"
    )
    parser.add_argument(
        "--env_map",
        type=str,
        required=True,
        help="Path to environment map (.hdr or .exr file)"
    )

    # Output
    parser.add_argument(
        "--output_video",
        type=str,
        default="output/sdedit_refined.mp4",
        help="Path to save the refined output video"
    )

    # Model settings
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Diffusion_Renderer_Forward_Cosmos_7B",
        help="Forward renderer model directory name"
    )
    parser.add_argument(
        "--scheduler_sigma_min",
        type=float,
        default=0.02,
        help="EDMEulerScheduler sigma_min set during scheduler initialization. Default: 0.02",
    )
    parser.add_argument(
        "--scheduler_sigma_max",
        type=float,
        default=80.0,
        help="EDMEulerScheduler sigma_max set during scheduler initialization. Default: 80.0",
    )
    parser.add_argument(
        "--scheduler_rho",
        type=float,
        default=7.0,
        help="EDMEulerScheduler rho set during scheduler initialization. Default: 7.0",
    )

    # SDEdit parameters
    parser.add_argument(
        "--sdedit_strength",
        type=float,
        default=None,
        help="(Deprecated) SDEdit strength (0.0=no change, 1.0=full generation). Use --sdedit_sigma instead."
    )
    parser.add_argument(
        "--sdedit_sigma",
        type=float,
        default=None,
        help="SDEdit start sigma (noise level). If set, start step is chosen by nearest scheduler sigma."
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--guidance_start",
        type=float,
        default=None,
        help="Start CFG guidance weight for linear scheduling during denoising. Must be paired with --guidance_end.",
    )
    parser.add_argument(
        "--guidance_end",
        type=float,
        default=None,
        help="End CFG guidance weight for linear scheduling during denoising. Must be paired with --guidance_start.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=35,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--s_churn",
        type=float,
        default=0.0,
        help="EDM stochasticity parameter passed to scheduler.step(...). Default: 0.0",
    )
    parser.add_argument(
        "--s_noise",
        type=float,
        default=1.0,
        help="EDM noise scale parameter passed to scheduler.step(...). Default: 1.0",
    )
    parser.add_argument(
        "--s_tmin",
        type=float,
        default=0.0,
        help="EDM lower sigma bound for stochasticity in scheduler.step(...). Default: 0.0",
    )
    parser.add_argument(
        "--s_tmax",
        type=float,
        default=float("inf"),
        help="EDM upper sigma bound for stochasticity in scheduler.step(...). Default: inf",
    )

    # Video settings
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Video height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=57,
        help="Number of frames to process (must be 57)"
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame index (inclusive) when slicing input videos/frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second"
    )
    parser.add_argument(
        "--rotate_light",
        action="store_true",
        help="Rotate the environment map across frames"
    )

    # Other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--offload_diffusion_transformer",
        action="store_true",
        help="Offload diffusion transformer to save GPU memory"
    )
    parser.add_argument(
        "--offload_tokenizer",
        action="store_true",
        help="Offload tokenizer to save GPU memory"
    )

    return parser.parse_args()


def count_input_frames(path: str) -> int:
    path_obj = Path(path)
    if path_obj.is_file():
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count > 0:
            return count
        # Fallback: count by decoding if metadata is missing.
        cap = cv2.VideoCapture(str(path))
        frames = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frames += 1
        cap.release()
        return frames
    if path_obj.is_dir():
        return len([
            f for f in path_obj.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.exr']
        ])
    raise ValueError(f"Path {path} is neither a file nor directory")


def _repeat_last_frame(frames: np.ndarray, max_frames: int) -> np.ndarray:
    if frames.shape[0] >= max_frames:
        return frames[:max_frames]
    last = frames[-1:]
    pad = np.repeat(last, max_frames - frames.shape[0], axis=0)
    return np.concatenate([frames, pad], axis=0)


def load_video_frames(path: str, max_frames: int = 57, height: int = 704, width: int = 1280, start_frame: int = 0) -> torch.Tensor:
    """Load video frames from file or directory with aspect-ratio preserving center crop.

    Args:
        path: Path to video file or directory of image frames
        max_frames: Maximum number of frames to load (clips longer videos)
        height: Target height
        width: Target width
        start_frame: Start frame index (inclusive) to skip before reading

    Returns:
        Tensor of shape [1, 3, T, H, W] in range [-1, 1]
    """
    path_obj = Path(path)

    # Set up transforms (matching forward renderer): resize first, then center-crop
    resize_transform = transforms.Resize((height, width),
                                         interpolation=transforms.InterpolationMode.BILINEAR,
                                         antialias=True)
    crop_transform = transforms.CenterCrop((height, width))

    if path_obj.is_file():
        # Load from video file using OpenCV (preserve behavior for videos)
        cap = cv2.VideoCapture(str(path))
        frames = []

        # Skip start_frame frames
        skipped = 0
        while skipped < start_frame:
            ret, _ = cap.read()
            if not ret:
                break
            skipped += 1

        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {path}")

        frames = np.stack(frames, axis=0)  # [T, H, W, C]
        frames = _repeat_last_frame(frames, max_frames)

    elif path_obj.is_dir():
        # Load from directory of frames using the same loader as forward renderer
        image_files = sorted([
            f for f in path_obj.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.exr']
        ])
        image_files = image_files[start_frame : start_frame + max_frames]

        if len(image_files) == 0:
            raise ValueError(f"No image files found in {path}")

        paths = [str(f) for f in image_files]
        # Use prepare_images_np to handle EXR alpha/compositing and uint8 conversion
        frames = prepare_images_np(paths, use_grayscale=False, mask_np=None, bg_color=(1., 1., 1.))
        # prepare_images_np returns [T, H, W, C]
        frames = _repeat_last_frame(frames, max_frames)

    else:
        raise ValueError(f"Path {path} is neither a file nor directory")

    # Convert to tensor [T, H, W, C] -> [T, C, H, W], ensure float in [0,1]
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0    
    
    frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2).contiguous()

    # Apply transforms in forward-renderer order: resize -> center-crop
    frames = resize_transform(frames)
    frames = crop_transform(frames)

    # Normalize to [-1, 1] (matching forward renderer)
    frames = frames * 2.0 - 1.0

    # Rearrange to [1, 3, T, H, W]
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]

    log.info(f"Loaded {frames.shape[2]} frames from {path} with shape {frames.shape}")

    return frames


def load_environment_map(
    env_path: str,
    num_frames: int,
    height: int,
    width: int,
    rotate_light: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load and process HDR environment map (matching forward renderer).

    Args:
        env_path: Path to .hdr or .exr file
        num_frames: Number of frames for the video
        height: Target height for env map (should match video height)
        width: Target width for env map (should match video width)

    Returns:
        Dictionary with env_ldr, env_log, env_nrm tensors
    """
    from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import envmap_vec

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process using 'proj' format (matching forward renderer)
    env_map_processed = process_environment_map(
        hdr_dir=env_path,
        resolution=(height, width),
        num_frames=num_frames,
        fixed_pose=True,
        rotate_envlight=rotate_light,
        env_format=['proj'],  # Use 'proj' format like forward renderer
        device=device,
    )

    # Extract and normalize environment maps (matching forward renderer)
    # Tensors from process_environment_map are [T, H, W, 3] in [0, 1]
    env_ldr = env_map_processed['env_ldr'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1  # [1, 3, T, H, W] in [-1, 1]
    env_log = env_map_processed['env_log'].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1  # [1, 3, T, H, W] in [-1, 1]

    # Generate environment map normal vectors
    env_nrm = envmap_vec([height, width], device=device)  # [H, W, 3]
    env_nrm = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(env_ldr)  # [1, 3, T, H, W]

    log.info(f"Loaded environment map from {env_path} with shapes: env_ldr={env_ldr.shape}, env_log={env_log.shape}, env_nrm={env_nrm.shape}")

    # Keep environment tensors on CPU between stages to reduce persistent VRAM residency.
    output = {
        'env_ldr': env_ldr.cpu(),
        'env_log': env_log.cpu(),
        'env_nrm': env_nrm.cpu(),
    }
    if device.type == "cuda":
        del env_ldr
        del env_log
        del env_nrm
        torch.cuda.empty_cache()
    return output


class SDEditForwardRenderer:
    """SDEdit-enabled Forward Renderer pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Initialize the pipeline
        log.info("Initializing DiffusionRenderer pipeline...")
        self.pipeline = DiffusionRendererPipeline(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.diffusion_transformer_dir,
            offload_network=args.offload_diffusion_transformer,
            offload_tokenizer=args.offload_tokenizer,
            guidance=args.guidance,
            num_steps=args.num_steps,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_video_frames=args.max_frames,
            seed=args.seed,
            scheduler_sigma_min=args.scheduler_sigma_min,
            scheduler_sigma_max=args.scheduler_sigma_max,
            scheduler_rho=args.scheduler_rho,
        )

        self.model: DiffusionRendererModel = self.pipeline.model

    @torch.inference_mode()
    def generate_sdedit(
        self,
        input_rgb: torch.Tensor,
        gbuffer_dict: Dict[str, torch.Tensor],
        env_map_dict: Optional[Dict[str, torch.Tensor]] = None,
        skip: bool = False,
        # skip: bool = True,
    ) -> torch.Tensor:
        """Run SDEdit refinement.

        Args:
            input_rgb: Input RGB video [1, 3, T, H, W]
            gbuffer_dict: Dictionary with G-buffer tensors
            env_map_dict: Optional dictionary with env_ldr, env_log, env_nrm tensors
            skip: If True, skip denoising and return the encoded latent directly

        Returns:
            Refined RGB video in latent space
        """

        # Keep large conditioning tensors on CPU and move them on demand inside condition prep.
        video_cuda = input_rgb.to(device="cuda", dtype=torch.bfloat16)
        data_batch = {
            'video': video_cuda,
            'basecolor': gbuffer_dict['basecolor'],
            'normal': gbuffer_dict['normal'],
            'depth': gbuffer_dict['depth'],
            'roughness': gbuffer_dict['roughness'],
            'metallic': gbuffer_dict['metallic'],
        }

        # Add environment maps if provided (matching forward renderer)
        if env_map_dict is not None:
            data_batch['env_ldr'] = env_map_dict['env_ldr']
            data_batch['env_log'] = env_map_dict['env_log']
            data_batch['env_nrm'] = env_map_dict['env_nrm']

        # Add dummy text embeddings (required by conditioner, matches forward renderer)
        # The conditioner expects these keys but they're not actually used for rendering
        # Shape matches VideoGBufferDataset but with batch dimension
        dummy_text_embedding = torch.zeros(1, 512, 1024)  # [batch=1, seq_len=512, embed_dim=1024]
        dummy_text_mask = torch.zeros(1, 512)  # [batch=1, seq_len=512]
        dummy_text_mask[0, 0] = 1

        # Add metadata fields that the conditioner expects (match inference_utils.py format)
        num_frames = input_rgb.shape[2]
        metadata_batch = {
            't5_text_embeddings': dummy_text_embedding,
            't5_text_mask': dummy_text_mask,
            'num_frames': torch.tensor([num_frames], dtype=torch.float),  # [1]
            'image_size': torch.tensor([[self.args.height, self.args.width]], dtype=torch.float),  # [1, 2]
            'fps': torch.tensor([self.args.fps], dtype=torch.float),  # [1]
            'padding_mask': torch.zeros(1, 1, self.args.height, self.args.width),  # [1, 1, H, W]
            'context_index': torch.LongTensor([0]),  # [1]
        }
        metadata_batch = misc.to(metadata_batch, device="cuda", dtype=torch.bfloat16)
        data_batch.update(metadata_batch)
        data_batch['is_preprocessed'] = True

        # Load models if offloaded
        if self.args.offload_diffusion_transformer:
            self.pipeline._load_network()
        if self.args.offload_tokenizer:
            self.pipeline._load_tokenizer()

        # 1. Get conditions (G-buffers) FIRST (to match forward renderer RNG state)
        log.info("Preparing conditions...")
        t_cond = time.time()
        condition, uncondition = self.model._get_conditions(data_batch, is_negative_prompt=False)
        log.info(f"Conditions prepared in {time.time() - t_cond:.1f}s")

        # 2. Set up timesteps
        self.model.scheduler.set_timesteps(self.args.num_steps)

        # 3. Calculate starting timestep based on sigma (preferred) or deprecated strength.
        return_encoded_without_denoise = False
        if self.args.sdedit_sigma is not None:
            if self.args.sdedit_sigma <= 1e-5:
                start_step_idx = 0
            else:
                sigmas = self.model.scheduler.sigmas
                diffs = (sigmas - self.args.sdedit_sigma).abs()
                start_step_idx = int(diffs.argmin().item())
                start_step_idx = min(start_step_idx, self.args.num_steps - 1)
            actual_sigma = float(self.model.scheduler.sigmas[start_step_idx].item())
            log.info(f"SDEdit sigma: {self.args.sdedit_sigma}")
            log.info(
                f"Starting from step {start_step_idx}/{self.args.num_steps}, actual sigma: {actual_sigma:.4f}"
            )
            start_from_pure_noise = self.args.sdedit_sigma <= 1e-5
        else:
            strength = 0.4 if self.args.sdedit_strength is None else self.args.sdedit_strength
            start_step_idx = int(self.args.num_steps * strength)
            if start_step_idx >= self.args.num_steps:
                start_step_idx = self.args.num_steps - 1
            log.warning(
                "Using deprecated --sdedit_strength. Please migrate to --sdedit_sigma for scheduler-invariant control."
            )
            log.info(f"SDEdit strength: {strength}")
            log.info(f"Starting from step {start_step_idx}/{self.args.num_steps}")
            start_from_pure_noise = strength <= 1e-5
            # For strength=1, treat SDEdit as pure reconstruction and skip scheduler denoising.
            return_encoded_without_denoise = strength >= 1.0 - 1e-5

        # 4. Encode input RGB to latent space (after conditions to preserve RNG state for noise)
        log.info("Encoding input RGB to latent space...")
        t_encode = time.time()
        with torch.no_grad():
            x0_latent = self.model.encode(video_cuda)
        log.info(f"Encode took {time.time() - t_encode:.1f}s")
        # Release large raw-condition tensors after condition objects and x0_latent are ready.
        del video_cuda
        del data_batch
        torch.cuda.empty_cache()

        log.info(f"Latent shape: {x0_latent.shape}")

        # If skip flag is enabled, or deprecated strength=1 is requested, return the encoded latent directly.
        if skip or return_encoded_without_denoise:
            if skip:
                log.info("Skip flag enabled: skipping denoising and returning encoded latent.")
            else:
                log.info("SDEdit strength is 1.0: skipping denoising and returning encoded latent.")
            del condition
            del uncondition
            if self.args.offload_diffusion_transformer:
                self.pipeline._offload_network()
            if self.args.offload_tokenizer:
                self.pipeline._offload_tokenizer()
            return x0_latent

        # 5. Add noise to the starting timestep
        timesteps_subset = self.model.scheduler.timesteps[start_step_idx:]

        # Use init_noise_sigma to match forward renderer exactly (instead of sigma_start)
        # Forward renderer: xt = torch.randn(...) * self.scheduler.init_noise_sigma
        log.info(f"Generating noise with init_noise_sigma={self.model.scheduler.init_noise_sigma:.4f}...")

        if start_from_pure_noise:
            # For strength=0, start from pure noise (matching forward renderer)
            print("SDEdit sigma/strength indicates pure-noise start.")
            noise = torch.randn_like(x0_latent)
            xt = noise * self.model.scheduler.init_noise_sigma
        else:
            # For strength>0, use standard SDEdit: add noise to x0_latent
            t_start = timesteps_subset[0]
            self.model.scheduler._init_step_index(t_start)
            sigma_start = self.model.scheduler.sigmas[self.model.scheduler.step_index]
            noise = torch.randn_like(x0_latent)
            xt = x0_latent + sigma_start.to(x0_latent.device) * noise

        # 6. Handle context parallelism if enabled
        from cosmos_predict1.diffusion.module.parallel import split_inputs_cp, cat_outputs_cp

        to_cp = self.model.net.is_context_parallel_enabled
        if to_cp:
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.model.net.cp_group)

        # 7. Denoise from t_start to final timestep
        log.info(f"Running denoising for {len(timesteps_subset)} steps...")
        if (self.args.guidance_start is None) != (self.args.guidance_end is None):
            raise ValueError(
                "Both --guidance_start and --guidance_end must be provided together, or neither."
            )
        use_guidance_schedule = self.args.guidance_start is not None and self.args.guidance_end is not None
        total_steps = len(timesteps_subset)

        t_denoise = time.time()
        for i, t in enumerate(timesteps_subset):
            if i % 5 == 0:
                elapsed = time.time() - t_denoise
                log.info(f"  Step {i}/{len(timesteps_subset)} ({elapsed:.1f}s elapsed)")

            xt = xt.to(**self.model.tensor_kwargs)
            xt_scaled = self.model.scheduler.scale_model_input(xt, timestep=t)

            # Predict with G-buffer conditioning
            t = t.to(**self.model.tensor_kwargs)
            net_output_cond = self.model.net(x=xt_scaled, timesteps=t, **condition.to_dict())

            net_output = net_output_cond
            if use_guidance_schedule:
                alpha = 0.0 if total_steps <= 1 else i / (total_steps - 1)
                current_guidance = self.args.guidance_start + alpha * (self.args.guidance_end - self.args.guidance_start)
            else:
                current_guidance = self.args.guidance
            if current_guidance != 0:
                net_output_uncond = self.model.net(x=xt_scaled, timesteps=t, **uncondition.to_dict())
                net_output = net_output_cond + current_guidance * (net_output_cond - net_output_uncond)

            # Step the scheduler
            xt = self.model.scheduler.step(
                net_output,
                t,
                xt,
                s_churn=self.args.s_churn,
                s_tmin=self.args.s_tmin,
                s_tmax=self.args.s_tmax,
                s_noise=self.args.s_noise,
            ).prev_sample

        samples = xt

        if to_cp:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.model.net.cp_group)

        log.info(f"Denoising {len(timesteps_subset)} steps took {time.time() - t_denoise:.1f}s")

        del condition
        del uncondition

        log.info("Denoising complete!")
        if self.args.offload_diffusion_transformer:
            self.pipeline._offload_network()

        return samples


def main():
    args = parse_arguments()
    if args.sdedit_sigma is not None and args.sdedit_strength is not None:
        raise ValueError("Provide only one of --sdedit_sigma or --sdedit_strength.")
    if args.sdedit_sigma is not None and args.sdedit_sigma < 0:
        raise ValueError("--sdedit_sigma must be >= 0.")
    if (args.guidance_start is None) != (args.guidance_end is None):
        raise ValueError("Both --guidance_start and --guidance_end must be provided together, or neither.")

    # Set random seed
    misc.set_random_seed(args.seed)

    t_total = time.time()
    if args.rotate_light:
        if args.max_frames != 57:
            raise ValueError(f"--max_frames must be 57 when --rotate_light is set, got {args.max_frames}")

    log.info("="*80)
    log.info("SDEdit Forward Renderer")
    log.info("="*80)
    log.info(f"Input RGB: {args.input_rgb_video}")
    log.info(f"Environment map: {args.env_map}")
    if args.sdedit_sigma is not None:
        log.info(f"SDEdit sigma: {args.sdedit_sigma}")
    else:
        log.info(f"SDEdit strength (deprecated): {args.sdedit_strength}")
    log.info(f"Guidance: {args.guidance}")
    log.info(f"Output: {args.output_video}")
    log.info("="*80)

    # Create output directory
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input videos
    log.info("\n" + "="*80)
    log.info("Loading input data...")
    log.info("="*80)
    t_load = time.time()

    if args.rotate_light:
        input_paths = [
            args.input_rgb_video,
            args.basecolor_video,
            args.normal_video,
            args.depth_video,
            args.roughness_video,
            args.metallic_video,
        ]
        for path in input_paths:
            input_count = count_input_frames(path)
            if input_count != 57:
                raise ValueError(
                    f"--rotate_light requires exactly 57 input frames, got {input_count} for {path}"
                )

    input_rgb = load_video_frames(
        args.input_rgb_video,
        max_frames=args.max_frames,
        height=args.height,
        width=args.width,
        start_frame=args.start_frame,
    )

    gbuffer_dict = {
        'basecolor': load_video_frames(
            args.basecolor_video,
            max_frames=args.max_frames,
            height=args.height,
            width=args.width,
            start_frame=args.start_frame,
        ),
        'normal': load_video_frames(
            args.normal_video,
            max_frames=args.max_frames,
            height=args.height,
            width=args.width,
            start_frame=args.start_frame,
        ),
        'depth': load_video_frames(
            args.depth_video,
            max_frames=args.max_frames,
            height=args.height,
            width=args.width,
            start_frame=args.start_frame,
        ),
        'roughness': load_video_frames(
            args.roughness_video,
            max_frames=args.max_frames,
            height=args.height,
            width=args.width,
            start_frame=args.start_frame,
        ),
        'metallic': load_video_frames(
            args.metallic_video,
            max_frames=args.max_frames,
            height=args.height,
            width=args.width,
            start_frame=args.start_frame,
        ),
    }

    # Load environment map (matching forward renderer)
    env_map_dict = load_environment_map(
        args.env_map,
        num_frames=args.max_frames,
        height=args.height,
        width=args.width,
        rotate_light=args.rotate_light,
    ) if args.env_map else None
    log.info(f"Data loading took {time.time() - t_load:.1f}s")

    # Initialize renderer
    log.info("\n" + "="*80)
    log.info("Initializing renderer...")
    log.info("="*80)
    t_init = time.time()
    renderer = SDEditForwardRenderer(args)
    log.info(f"Model initialization took {time.time() - t_init:.1f}s")

    # Run SDEdit
    log.info("\n" + "="*80)
    log.info("Running SDEdit refinement...")
    log.info("="*80)

    t_sdedit = time.time()
    refined_latent = renderer.generate_sdedit(
        input_rgb=input_rgb,
        gbuffer_dict=gbuffer_dict,
        env_map_dict=env_map_dict,
    )
    log.info(f"generate_sdedit took {time.time() - t_sdedit:.1f}s")

    # Decode to pixel space
    log.info("\n" + "="*80)
    log.info("Decoding to pixel space...")
    log.info("="*80)

    t_decode = time.time()
    if args.offload_tokenizer:
        renderer.pipeline._load_tokenizer()
    with torch.no_grad():
        refined_video = renderer.model.decode(refined_latent)
    log.info(f"Decode took {time.time() - t_decode:.1f}s")
    torch.cuda.empty_cache()
    if args.offload_tokenizer:
        renderer.pipeline._offload_tokenizer()

    # Convert to numpy and save (match forward renderer format)
    log.info(f"Saving output to {args.output_video}...")

    # Convert from [1, 3, T, H, W] to [T, H, W, 3] in uint8 format
    # Match the forward renderer's conversion process (diffusion_renderer_pipeline.py:175-176)
    refined_video = (1.0 + refined_video).clamp(0, 2) / 2  # Normalize to [0, 1]
    refined_video = (refined_video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

    # Save video
    t_save = time.time()
    save_image_or_video(
        video_save_path=args.output_video,
        video=refined_video,
        H=args.height,
        W=args.width,
        fps=args.fps,
        video_save_quality=9,
    )
    log.info(f"Save took {time.time() - t_save:.1f}s")

    log.info("\n" + "="*80)
    log.info(f"Total time: {time.time() - t_total:.1f}s")
    log.info("DONE! Output saved to: " + args.output_video)
    log.info("="*80)


if __name__ == "__main__":
    main()

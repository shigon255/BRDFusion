#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SDEdit-style refinement using the Inverse Renderer.

Given an input RGB video and initial G-buffer estimates, this script adds noise
and denoises with the inverse renderer's prior while conditioning on RGB.
"""

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import GBUFFER_INDEX_MAPPING
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.transform_utils import prepare_images_np
from cosmos_predict1.diffusion.model.model_diffusion_renderer import DiffusionRendererModel
from cosmos_predict1.utils import misc, log
from cosmos_predict1.utils.io import save_image_or_video


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDEdit refinement using Inverse Renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python sdedit_inverse_renderer.py \\
    --input_rgb_video data/rgb_video.mp4 \\
    --basecolor_video data/basecolor \\
    --normal_video data/normal \\
    --depth_video data/depth \\
    --roughness_video data/roughness \\
    --metallic_video data/metallic \\
    --output_dir output/ \\
    --sdedit_strength 0.4 \\
    --guidance 3.0
        """,
    )

    # Input files
    parser.add_argument(
        "--input_rgb_video",
        type=str,
        required=True,
        help="Path to input RGB video file or directory of frames",
    )
    parser.add_argument("--basecolor_video", type=str, help="Path to basecolor video/frames")
    parser.add_argument("--normal_video", type=str, help="Path to normal video/frames")
    parser.add_argument("--depth_video", type=str, help="Path to depth video/frames")
    parser.add_argument("--roughness_video", type=str, help="Path to roughness video/frames")
    parser.add_argument("--metallic_video", type=str, help="Path to metallic video/frames")

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/sdedit_inverse",
        help="Directory to save refined outputs",
    )
    parser.add_argument(
        "--save_image",
        type=str2bool,
        default=False,
        help="If True, save per-frame images in addition to videos",
    )

    # Model settings
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Diffusion_Renderer_Inverse_Cosmos_7B",
        help="Inverse renderer model directory name",
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
        help="(Deprecated) SDEdit strength (0.0=no change, 1.0=full generation). Use --sdedit_sigma instead.",
    )
    parser.add_argument(
        "--sdedit_sigma",
        type=float,
        default=None,
        help="SDEdit start sigma (noise level). If set, start step is chosen by nearest scheduler sigma.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
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
        help="Number of diffusion steps",
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
    parser.add_argument(
        "--normalize_normal",
        type=str2bool,
        default=False,
        help="If True, normalize normal-map outputs before saving",
    )

    # Video settings
    parser.add_argument("--height", type=int, default=704, help="Video height")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--max_frames", type=int, default=57, help="Number of frames to process")
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame index (inclusive) when slicing input videos/frames",
    )
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")

    # Other settings
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument(
        "--offload_diffusion_transformer",
        action="store_true",
        help="Offload diffusion transformer to save GPU memory",
    )
    parser.add_argument(
        "--offload_tokenizer",
        action="store_true",
        help="Offload tokenizer to save GPU memory",
    )

    parser.add_argument(
        "--inference_passes",
        type=str,
        nargs="+",
        default=["basecolor", "normal", "depth", "roughness", "metallic"],
        help="Which G-buffer passes to refine",
    )

    return parser.parse_args()


def _repeat_last_frame(frames: np.ndarray, max_frames: int) -> np.ndarray:
    if frames.shape[0] >= max_frames:
        return frames[:max_frames]
    last = frames[-1:]
    pad = np.repeat(last, max_frames - frames.shape[0], axis=0)
    return np.concatenate([frames, pad], axis=0)


def load_video_frames(
    path: str,
    max_frames: int,
    height: int,
    width: int,
    start_frame: int = 0,
) -> torch.Tensor:
    """Load video frames from file or directory with resize+center crop and normalize to [-1, 1]."""
    path_obj = Path(path)

    resize_transform = transforms.Resize(
        (height, width), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
    )
    crop_transform = transforms.CenterCrop((height, width))

    if path_obj.is_file():
        cap = cv2.VideoCapture(str(path))
        frames = []
        skipped = 0
        while skipped < start_frame:
            ret, _ = cap.read()
            if not ret:
                break
            skipped += 1

        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"No frames loaded from {path}")
        frames = np.stack(frames, axis=0)
        frames = _repeat_last_frame(frames, max_frames)
    elif path_obj.is_dir():
        image_files = sorted(
            [f for f in path_obj.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".exr"]]
        )
        if not image_files:
            raise ValueError(f"No image files found in {path}")
        image_files = image_files[start_frame : start_frame + max_frames]
        frames = prepare_images_np(
            [str(f) for f in image_files], use_grayscale=False, mask_np=None, bg_color=(1.0, 1.0, 1.0)
        )
        frames = _repeat_last_frame(frames, max_frames)
    else:
        raise ValueError(f"Path {path} is neither a file nor directory")

    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0

    if frames.ndim == 3:
        frames = frames[..., None]
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2).contiguous()
    frames = resize_transform(frames)
    frames = crop_transform(frames)
    frames = frames * 2.0 - 1.0

    # Match VideoFramesDataset EXR gamma handling
    if path_obj.is_dir() and image_files[0].suffix.lower() == ".exr":
        frames = (frames + 1) / 2
        frames = torch.pow(frames.clamp(0, 1), 1 / 2.2)
        frames = frames * 2 - 1

    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
    log.info(f"Loaded {frames.shape[2]} frames from {path} with shape {frames.shape}")
    return frames


def _postprocess_video(video: torch.Tensor, normalize_normal: bool) -> np.ndarray:
    if normalize_normal:
        norm = torch.norm(video, dim=1, p=2, keepdim=True)
        video_normalized = video / norm.clamp(min=1e-12)
        norm_threshold_upper = 0.4
        norm_threshold_lower = 0.2
        blend_ratio = torch.clip(
            (norm - norm_threshold_lower) / (norm_threshold_upper - norm_threshold_lower), 0, 1
        )
        video = video_normalized * blend_ratio + video * (1 - blend_ratio)

    video = (1.0 + video).clamp(0, 2) / 2
    video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
    return video


class SDEditInverseRenderer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
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
        input_gbuffer: torch.Tensor,
        gbuffer_pass: str,
    ) -> torch.Tensor:

        context_index = GBUFFER_INDEX_MAPPING[gbuffer_pass]
        rgb_cuda = input_rgb.to(device="cuda", dtype=torch.bfloat16)
        data_batch = {
            "video": rgb_cuda,
            "rgb": rgb_cuda,
            "is_preprocessed": True,
        }

        dummy_text_embedding = torch.zeros(1, 512, 1024)
        dummy_text_mask = torch.zeros(1, 512)
        dummy_text_mask[0, 0] = 1

        num_frames = input_rgb.shape[2]
        metadata_batch = {
            "t5_text_embeddings": dummy_text_embedding,
            "t5_text_mask": dummy_text_mask,
            "num_frames": torch.tensor([num_frames], dtype=torch.float),
            "image_size": torch.tensor([[self.args.height, self.args.width]], dtype=torch.float),
            "fps": torch.tensor([self.args.fps], dtype=torch.float),
            "padding_mask": torch.zeros(1, 1, self.args.height, self.args.width),
            "context_index": torch.LongTensor([context_index]),
        }
        metadata_batch = misc.to(metadata_batch, device="cuda", dtype=torch.bfloat16)
        data_batch.update(metadata_batch)

        if self.args.offload_diffusion_transformer:
            self.pipeline._load_network()
        if self.args.offload_tokenizer:
            self.pipeline._load_tokenizer()

        condition, uncondition = self.model._get_conditions(data_batch, is_negative_prompt=False)

        self.model.scheduler.set_timesteps(self.args.num_steps)
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
            # Deprecated strength-based start index for backward compatibility.
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

        with torch.no_grad():
            t_encode = time.time()
            input_gbuffer_cuda = input_gbuffer.to(device="cuda", dtype=torch.bfloat16)
            x0_latent = self.model.encode(input_gbuffer_cuda)
            log.info(f"Encode took {time.time() - t_encode:.1f}s")
        # Release large raw-condition tensors after condition objects and x0_latent are ready.
        del rgb_cuda
        del input_gbuffer_cuda
        del data_batch
        if self.args.offload_tokenizer:
            # Tokenizer is only needed for encode/decode; keep it off during denoising.
            self.pipeline._offload_tokenizer()
        torch.cuda.empty_cache()

        if return_encoded_without_denoise:
            log.info("SDEdit strength is 1.0: skipping denoising and returning encoded latent.")
            del condition
            del uncondition
            if self.args.offload_diffusion_transformer:
                self.pipeline._offload_network()
            return x0_latent

        if start_from_pure_noise:
            noise = torch.randn_like(x0_latent)
            xt = noise * self.model.scheduler.init_noise_sigma
        else:
            timesteps_subset = self.model.scheduler.timesteps[start_step_idx:]
            t_start = timesteps_subset[0]
            self.model.scheduler._init_step_index(t_start)
            sigma_start = self.model.scheduler.sigmas[self.model.scheduler.step_index]
            noise = torch.randn_like(x0_latent)
            xt = x0_latent + sigma_start.to(x0_latent.device) * noise

        from cosmos_predict1.diffusion.module.parallel import split_inputs_cp, cat_outputs_cp

        to_cp = self.model.net.is_context_parallel_enabled
        if to_cp:
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.model.net.cp_group)

        timesteps_subset = self.model.scheduler.timesteps[start_step_idx:]
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

            xt = self.model.scheduler.step(
                net_output,
                t,
                xt,
                s_churn=self.args.s_churn,
                s_tmin=self.args.s_tmin,
                s_tmax=self.args.s_tmax,
                s_noise=self.args.s_noise,
            ).prev_sample

        log.info(f"Denoising {len(timesteps_subset)} steps took {time.time() - t_denoise:.1f}s")

        samples = xt
        if to_cp:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.model.net.cp_group)

        del condition
        del uncondition

        if self.args.offload_diffusion_transformer:
            self.pipeline._offload_network()
        return samples


def _get_clip_name(path: str) -> str:
    path_obj = Path(path)
    return path_obj.stem if path_obj.is_file() else path_obj.name


def _collect_gbuffer_paths(args: argparse.Namespace) -> Dict[str, str]:
    mapping = {
        "basecolor": args.basecolor_video,
        "normal": args.normal_video,
        "depth": args.depth_video,
        "roughness": args.roughness_video,
        "metallic": args.metallic_video,
    }
    return mapping


def main():
    args = parse_arguments()
    if args.sdedit_sigma is not None and args.sdedit_strength is not None:
        raise ValueError("Provide only one of --sdedit_sigma or --sdedit_strength.")
    if args.sdedit_sigma is not None and args.sdedit_sigma < 0:
        raise ValueError("--sdedit_sigma must be >= 0.")
    if (args.guidance_start is None) != (args.guidance_end is None):
        raise ValueError("Both --guidance_start and --guidance_end must be provided together, or neither.")
    misc.set_random_seed(args.seed)

    t_total = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_name = _get_clip_name(args.input_rgb_video)

    log.info("=" * 80)
    log.info("SDEdit Inverse Renderer")
    log.info("=" * 80)
    log.info(f"Input RGB: {args.input_rgb_video}")
    if args.sdedit_sigma is not None:
        log.info(f"SDEdit sigma: {args.sdedit_sigma}")
    else:
        log.info(f"SDEdit strength (deprecated): {args.sdedit_strength}")
    log.info(f"Guidance: {args.guidance}")
    log.info(f"Output dir: {args.output_dir}")
    log.info("=" * 80)

    t_load = time.time()
    input_rgb = load_video_frames(
        args.input_rgb_video,
        max_frames=args.max_frames,
        height=args.height,
        width=args.width,
        start_frame=args.start_frame,
    )

    gbuffer_paths = _collect_gbuffer_paths(args)
    for gbuf in args.inference_passes:
        if gbuffer_paths.get(gbuf) is None:
            raise ValueError(f"Missing input path for pass '{gbuf}'")

    gbuffer_videos = {}
    for gbuf in args.inference_passes:
        gbuffer_videos[gbuf] = load_video_frames(
            gbuffer_paths[gbuf],
            max_frames=args.max_frames,
            height=args.height,
            width=args.width,
            start_frame=args.start_frame,
        )
    log.info(f"Data loading took {time.time() - t_load:.1f}s")

    t_init = time.time()
    renderer = SDEditInverseRenderer(args)
    log.info(f"Model initialization took {time.time() - t_init:.1f}s")

    for gbuf in args.inference_passes:
        log.info(f"Refining {gbuf}...")
        t_pass = time.time()
        refined_latent = renderer.generate_sdedit(
            input_rgb=input_rgb,
            input_gbuffer=gbuffer_videos[gbuf],
            gbuffer_pass=gbuf,
        )
        log.info(f"generate_sdedit ({gbuf}) took {time.time() - t_pass:.1f}s")

        if args.offload_tokenizer:
            renderer.pipeline._load_tokenizer()
        t_decode = time.time()
        with torch.no_grad():
            refined_video = renderer.model.decode(refined_latent)
        log.info(f"Decode ({gbuf}) took {time.time() - t_decode:.1f}s")
        if args.offload_tokenizer:
            renderer.pipeline._offload_tokenizer()
        del refined_latent

        normalize_normal = args.normalize_normal and gbuf == "normal"
        refined_video = _postprocess_video(refined_video, normalize_normal=normalize_normal)

        video_save_path = output_dir / f"{clip_name}.{gbuf}.mp4"
        t_save = time.time()
        save_image_or_video(
            video_save_path=str(video_save_path),
            video=refined_video,
            H=args.height,
            W=args.width,
            fps=args.fps,
            video_save_quality=5,
        )
        log.info(f"Saved video to {video_save_path} ({time.time() - t_save:.1f}s)")

        if args.save_image:
            frame_dir = output_dir / "gbuffer_frames" / clip_name / gbuf
            frame_dir.mkdir(parents=True, exist_ok=True)
            for ind in range(refined_video.shape[0]):
                frame_path = frame_dir / f"{ind:04d}.jpg"
                save_image_or_video(
                    video_save_path=str(frame_path),
                    video=refined_video[ind : ind + 1, ...],
                    H=args.height,
                    W=args.width,
                )
        del refined_video
        gc.collect()
        torch.cuda.empty_cache()

    log.info(f"Total time: {time.time() - t_total:.1f}s")
    log.info("DONE")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from typing import Any, Optional, Tuple, Dict

import torch
from cosmos_predict1.diffusion.inference.inference_utils import (
    load_model_by_config,
    load_network_model,
    load_tokenizer_model,
    read_video_or_image_into_frames_BCTHW,
)
from cosmos_predict1.utils import misc, log
from cosmos_predict1.diffusion.inference.world_generation_pipeline import DiffusionText2WorldGenerationPipeline
from cosmos_predict1.diffusion.model.model_diffusion_renderer import DiffusionRendererModel


class DiffusionRendererPipeline(DiffusionText2WorldGenerationPipeline):
    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_name: str,
        prompt_upsampler_dir: Optional[str] = None,
        enable_prompt_upsampler: bool = False,
        has_text_input: bool = False,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        disable_guardrail: bool = True,
        guidance: float = 0.0,
        num_steps: int = 15,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 57,
        seed: int = 1000,
        scheduler_sigma_min: float = 0.02,
        scheduler_sigma_max: float = 80.0,
        scheduler_rho: float = 7.0,
    ):
        """Initialize the diffusion renderer pipeline.
        
        Args:
            is_inverse: If True, uses the Inverse renderer (rgb -> other maps)
                      If False, uses the Forward renderer (maps -> rgb)
            Other args are inherited from DiffusionText2WorldGenerationPipeline
        """
        self.scheduler_sigma_min = scheduler_sigma_min
        self.scheduler_sigma_max = scheduler_sigma_max
        self.scheduler_rho = scheduler_rho
        super().__init__(
            inference_type='text2world',
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            prompt_upsampler_dir=prompt_upsampler_dir,
            enable_prompt_upsampler=enable_prompt_upsampler,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_prompt_upsampler=offload_prompt_upsampler,
            offload_guardrail_models=offload_guardrail_models,
            disable_guardrail=disable_guardrail,
            guidance=guidance,
            num_steps=num_steps,
            height=height,
            width=width,
            fps=fps,
            num_video_frames=num_video_frames,
            seed=seed,
        )
        

    def _load_model(self):
        """Load the appropriate renderer model based on is_inverse flag."""
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos_predict1/diffusion/config/diffusion_renderer_config.py",
            model_class=DiffusionRendererModel,
            config_overrides=[
                f"model.scheduler_sigma_min={self.scheduler_sigma_min}",
                f"model.scheduler_sigma_max={self.scheduler_sigma_max}",
                f"model.scheduler_rho={self.scheduler_rho}",
            ],
        )

    def _load_text_encoder_model(self):
        """Skip loading of the T5 text encoder model.
        """
        self.text_encoder = None

    def generate_video(
        self,
        data_batch: Dict[str, torch.Tensor],
        normalize_normal: bool = False,
        seed: int = None,
        guidance_start: float | None = None,
        guidance_end: float | None = None,
    ) -> np.ndarray:
        """Generate G-buffer maps from input video/image using inverse rendering.

        Args:
            data_batch: Dictionary containing:
                - video/rgb/basecolor/etc: Input tensor of shape [B, C, T, H, W]
                - context_index: Tensor indicating which G-buffer to generate
                Other optional keys may be present depending on the model configuration
            normalize_normal: Whether to normalize and blend normal map outputs. Only applies
                when generating normal maps.

        Returns:
            Generated video frames as uint8 np.ndarray of shape [T, H, W, C].
            For different G-buffers, C represents:
                - basecolor: RGB color map
                - normal: Surface normal vectors
                - depth: Depth map
                - roughness: Surface roughness map
                - metallic: Surface metallic map
        """
        # Generate video
        log.info("Run generation")

        if self.offload_network:
            self._load_network()

        if self.offload_tokenizer:
            self._load_tokenizer()

        # prepare data_batch
        if 'video' not in data_batch:
            for attributes in ['rgb', 'basecolor', 'normal', 'depth', 'roughness', 'metallic']:
                if attributes in data_batch:
                    data_batch['video'] = data_batch[attributes]
                    break

        data_batch = misc.to(data_batch, device="cuda", dtype=torch.bfloat16)  # move to GPU

        # prepare state_shape
        C = self.model.tokenizer.channel
        F = (data_batch['video'].shape[2] - 1) // 8 + 1
        H = data_batch['video'].shape[3] // self.model.tokenizer.spatial_compression_factor
        W = data_batch['video'].shape[4] // self.model.tokenizer.spatial_compression_factor
        state_shape = [C, F, H, W]

        # Generate video frames
        # Allow model-level tokenizer offload after condition preparation to reduce denoising peak VRAM.
        self.model._offload_tokenizer_after_conditions = bool(self.offload_tokenizer)
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=self.guidance,
            guidance_start=guidance_start,
            guidance_end=guidance_end,
            state_shape=state_shape,
            num_steps=self.num_steps,
            is_negative_prompt=False,
            seed=self.seed if seed is None else seed,
        )

        if self.offload_network:
            self._offload_network()

        if self.offload_tokenizer:
            self._load_tokenizer()

        video = self.model.decode(sample)

        # post-processing (surface normals)
        if normalize_normal:
            norm = torch.norm(video, dim=1, p=2, keepdim=True)  # (1, C, T, H, W) -> (1, 1, T, H, W)
            video_normalized = video / norm.clamp(min=1e-12)
            norm_threshold_upper = 0.4
            norm_threshold_lower = 0.2
            blend_ratio = torch.clip(
                (norm - norm_threshold_lower) / (norm_threshold_upper - norm_threshold_lower),
                0, 1
            )
            video = video_normalized * blend_ratio + video * (1 - blend_ratio)

        video = (1.0 + video).clamp(0, 2) / 2  # [B, 3, T, H, W]
        video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()


        if self.offload_tokenizer:
            self._offload_tokenizer()

        log.info("Finish generation")
        return video

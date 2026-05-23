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

import gc
from typing import Callable, Dict, Tuple, Union, Optional

import numpy as np
import torch
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.conditioner import VideoDiffusionRendererCondition
from cosmos_predict1.diffusion.model.model_t2w import DiffusionT2WModel, broadcast_condition, split_inputs_cp, cat_outputs_cp, EDMEulerScheduler
from cosmos_predict1.utils import distributed, log, misc

class DiffusionRendererModel(DiffusionT2WModel):
    def __init__(self, config):
        super().__init__(config)

        # Custom configs
        self.condition_keys = config.condition_keys
        self.condition_drop_rate = config.condition_drop_rate
        self.append_condition_mask = config.append_condition_mask

        # Update scheduler (configurable from model config; defaults preserve previous behavior).
        self.scheduler = EDMEulerScheduler(
            sigma_max=getattr(config, "scheduler_sigma_max", 80.0),
            sigma_min=getattr(config, "scheduler_sigma_min", 0.02),
            sigma_data=self.sigma_data,
            rho=getattr(config, "scheduler_rho", 7.0),
        )
        print(f"Using EDMEulerScheduler with sigma_max={self.scheduler.sigma_max}, sigma_min={self.scheduler.sigma_min}, rho={self.scheduler.rho}")
        # Optional kwargs forwarded to scheduler.step(...), e.g. EDM stochasticity controls.
        self.scheduler_step_kwargs: Dict[str, float] = {}

    def prepare_diffusion_renderer_latent_conditions(
            self, data_batch: dict[str, Tensor],
            condition_keys: list[str] = ["rgb"], condition_drop_rate: float = 0, append_condition_mask: bool = True,
            dtype: torch.dtype = None, device: torch.device = None,
            latent_shape: Union[Tuple[int, int, int, int, int], torch.Size] = None,
            mode="train",
    ) -> Tensor:
        if latent_shape is None:
            B, C, T, H, W = data_batch[condition_keys[0]].shape
            latent_shape = (B, 16, T // 8 + 1, H // 8, W // 8)
        if append_condition_mask:
            latent_mask_shape = (latent_shape[0], 1, latent_shape[2], latent_shape[3], latent_shape[4])
        if dtype is None:
            dtype = data_batch[condition_keys[0]].dtype
        if device is None:
            device = data_batch[condition_keys[0]].device

        latent_condition_list = []
        for cond_key in condition_keys:
            ## Note: relaxed this constraint so during training the model can also train with missing attributes.
            # if cond_key not in data_batch and mode == "train":
            #     raise KeyError(f"Condition key '{cond_key}' is missing in data_batch during 'train' mode. "
            #                                f"Expected keys: {condition_keys}")
            is_condition_dropped = condition_drop_rate > 0 and np.random.rand() < condition_drop_rate
            is_condition_skipped = cond_key not in data_batch
            if is_condition_dropped or is_condition_skipped:
                # Dropped or skipped condition
                condition_state = torch.zeros(latent_shape, dtype=dtype, device=device)
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    condition_mask = torch.zeros(latent_mask_shape, dtype=dtype, device=device)
                    latent_condition_list.append(condition_mask)
            else:
                # Valid condition
                condition_state = data_batch[cond_key].to(device=device, dtype=dtype)
                condition_state = self.encode(condition_state).contiguous()
                latent_condition_list.append(condition_state)
                if append_condition_mask:
                    condition_mask = torch.ones(latent_mask_shape, dtype=dtype, device=device)
                    latent_condition_list.append(condition_mask)

        return torch.cat(latent_condition_list, dim=1)

    def _get_conditions(
        self,
        data_batch: Dict,
        is_negative_prompt: bool = False,
    ):
        # Latent state
        raw_state = data_batch[self.input_data_key]
        with torch.no_grad():
            latent_condition = self.prepare_diffusion_renderer_latent_conditions(
                data_batch,
                condition_keys=self.condition_keys,
                condition_drop_rate=0,
                append_condition_mask=self.append_condition_mask,
                dtype=raw_state.dtype, device=raw_state.device, latent_shape=None, mode="inference",
            )

        data_batch["latent_condition"] = latent_condition
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

        return condition, uncondition

    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = 0.0,
        guidance_start: float | None = None,
        guidance_end: float | None = None,
        seed: int = 1000,
        state_shape: tuple | None = None,
        n_sample: int | None = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 15,
    ) -> Tensor:
        """Generate samples from a data batch using diffusion sampling.

        This function generates samples from either image or video data batches using diffusion sampling.
        It handles both conditional and unconditional generation with classifier-free guidance.

        Args:
            data_batch (dict): Raw data batch from the training data loader
            guidance (float, optional): Constant classifier-free guidance weight. Defaults to 1.5.
            guidance_start (float | None, optional): Start guidance for linear CFG scheduling.
            guidance_end (float | None, optional): End guidance for linear CFG scheduling.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            state_shape (tuple | None, optional): Shape of the state tensor. Uses self.state_shape if None. Defaults to None.
            n_sample (int | None, optional): Number of samples to generate. Defaults to 1.
            is_negative_prompt (bool, optional): Whether to use negative prompt for unconditional generation. Defaults to False.
            num_steps (int, optional): Number of diffusion sampling steps. Defaults to 35.

        Returns:
            Tensor: Generated samples after diffusion sampling
        """
        condition, uncondition = self._get_conditions(data_batch, is_negative_prompt)

        # Optional memory optimization for inference:
        # once latent_condition is prepared, tokenizer is no longer needed for denoising.
        if getattr(self, "_offload_tokenizer_after_conditions", False) and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

        self.scheduler.set_timesteps(num_steps)

        xt = torch.randn(size=(n_sample,) + tuple(state_shape)) * self.scheduler.init_noise_sigma
        to_cp = self.net.is_context_parallel_enabled
        if to_cp:
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.net.cp_group)

        if (guidance_start is None) != (guidance_end is None):
            raise ValueError(
                "Both guidance_start and guidance_end must be provided together, or neither."
            )
        use_guidance_schedule = guidance_start is not None and guidance_end is not None
        total_steps = len(self.scheduler.timesteps)

        for step_idx, t in enumerate(self.scheduler.timesteps):
            xt = xt.to(**self.tensor_kwargs)
            xt_scaled = self.scheduler.scale_model_input(xt, timestep=t)
            # Predict the noise residual
            t = t.to(**self.tensor_kwargs)
            net_output_cond = self.net(x=xt_scaled, timesteps=t, **condition.to_dict())
            net_output = net_output_cond
            if use_guidance_schedule:
                alpha = 0.0 if total_steps <= 1 else step_idx / (total_steps - 1)
                current_guidance = guidance_start + alpha * (guidance_end - guidance_start)
            else:
                current_guidance = guidance
            if current_guidance != 0:
                net_output_uncond = self.net(x=xt_scaled, timesteps=t, **uncondition.to_dict())
                net_output = net_output_cond + current_guidance * (net_output_cond - net_output_uncond)
            # Compute the previous noisy sample x_t -> x_t-1
            xt = self.scheduler.step(net_output, t, xt, **self.scheduler_step_kwargs).prev_sample
            # xt = self.scheduler.step(net_output, t, xt).prev_sample
        samples = xt

        if to_cp:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples

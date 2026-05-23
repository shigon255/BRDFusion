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

from typing import Optional, Tuple, Container, List

import torch
from einops import rearrange
from torch import nn
import torch.distributed as dist
from torchvision import transforms

from cosmos_predict1.diffusion.conditioner import DataType
from cosmos_predict1.diffusion.networks.general_dit import GeneralDIT, PatchEmbed, VideoRopePosition3DEmb, log
from cosmos_predict1.diffusion.training.module.position_embedding import SinCosPosEmbAxis


class DiffusionRendererGeneralDIT(GeneralDIT):
    def __init__(
        self,
        *args,
        additional_concat_ch: int = None,
        use_context_embedding: bool = True,
        **kwargs,
    ):
        self.additional_concat_ch = additional_concat_ch
        self.use_context_embedding = use_context_embedding
        super().__init__(*args, **kwargs)

        # init context embedding
        if self.use_context_embedding:
            self.context_embedding = torch.nn.Embedding(num_embeddings=16,
                                                        embedding_dim=kwargs["crossattn_emb_channels"])
            rng_state = torch.get_rng_state()   # Save the current random state
            torch.manual_seed(42)
            torch.nn.init.uniform_(self.context_embedding.weight, -0.3, 0.3)
            torch.set_rng_state(rng_state)  # Restore the original random state

    def build_pos_embed(self):
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        log.debug(f"Building positional embedding with {self.pos_emb_cls} class, impl {cls_type}")
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
        )
        self.pos_embedder = cls_type(
            **kwargs,
        )

        if self.extra_per_block_abs_pos_emb:
            assert self.extra_per_block_abs_pos_emb_type in [
                "sincos",
            ], f"Unknown extra_per_block_abs_pos_emb_type {self.extra_per_block_abs_pos_emb_type}"
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = SinCosPosEmbAxis(**kwargs)

    def build_patch_embed(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        if self.additional_concat_ch is None:
            self.additional_concat_ch = in_channels

        in_channels = in_channels + self.additional_concat_ch
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            bias=False,
        )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the `self.pos_embedder`
                with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """

        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, latent_condition], dim=1)
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        scalar_feature: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        latent_condition: Optional[torch.Tensor] = None,
        latent_condition_sigma: Optional[torch.Tensor] = None,
        condition_video_augment_sigma: Optional[torch.Tensor] = None,
        context_index = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Overrides the parent `forward` method. It handles the context_index, and passes all given arguments to the
        parent method and returns its result.

        Arguments:
            (The list of arguments corresponds to the parent class's method signature.)
            context_index: tensor for the index of the context embeddings

        Returns:
            Output of the parent class forward method: a tensor, list of tensors, or tuple.
        """
        if self.use_context_embedding:
            # Overwrite the context embedding
            input_context_emb = self.context_embedding(context_index.long())
            if input_context_emb.ndim == 2:
                input_context_emb = input_context_emb.unsqueeze(1).clone()

            input_context_emb = input_context_emb.repeat_interleave(crossattn_emb.shape[1], dim=1)
            input_context_emb = input_context_emb.to(device=crossattn_emb.device, dtype=crossattn_emb.dtype)
            crossattn_emb = input_context_emb

        return super().forward(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            crossattn_mask=crossattn_mask,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
            scalar_feature=scalar_feature,
            data_type=data_type,
            latent_condition=latent_condition,
            latent_condition_sigma=latent_condition_sigma,
            condition_video_augment_sigma=condition_video_augment_sigma,
            **kwargs,
        )


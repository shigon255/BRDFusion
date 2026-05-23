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

from hydra.core.config_store import ConfigStore

from cosmos_predict1.diffusion.networks.general_dit_diffusion_renderer import DiffusionRendererGeneralDIT
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict


cs = ConfigStore.instance()


num_frames = 57
Diffusion_Renderer_Inverse_Cosmos_7B: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
            {"override /conditioner": "video_diffusion_renderer_cond"},
            "_self_",
        ],
        job=dict(
            group="diffusion_renderer",
            name="Diffusion_Renderer_Inverse_Cosmos_7B",
        ),
        model=dict(
            latent_shape=[
                16,
                num_frames // 8 + 1,
                88,
                160,
            ],
            tokenizer=dict(
                video_vae=dict(
                    pixel_chunk_duration=num_frames,
                )
            ),
            net=L(DiffusionRendererGeneralDIT)(
                additional_concat_ch=16,
                use_context_embedding=True,
                max_img_h=240,
                max_img_w=240,
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=1,
                block_x_format="THWBD",
                patch_spatial=2,
                extra_per_block_abs_pos_emb=True,
            ),
            conditioner=dict(
                latent_condition=dict(
                    dropout_rate=0.1,
                )
            ),
            condition_keys=["rgb"],
            condition_drop_rate=0,
            append_condition_mask=False,
        ),
    )
)
cs.store(
    group="experiment",
    package="_global_",
    name="Diffusion_Renderer_Inverse_Cosmos_7B",
    node=Diffusion_Renderer_Inverse_Cosmos_7B,
)


Diffusion_Renderer_Forward_Cosmos_7B: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /net": "faditv2_7b"},
            {"override /tokenizer": "cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624"},
            {"override /conditioner": "video_diffusion_renderer_cond"},
            "_self_",
        ],
        job=dict(
            group="diffusion_renderer",
            name="Diffusion_Renderer_Forward_Cosmos_7B",
        ),
        model=dict(
            latent_shape=[
                16,
                num_frames // 8 + 1,
                88,
                160,
            ],
            tokenizer=dict(
                video_vae=dict(
                    pixel_chunk_duration=num_frames,
                )
            ),
            net=L(DiffusionRendererGeneralDIT)(
                additional_concat_ch=17 * 8,
                use_context_embedding=False,
                max_img_h=240,
                max_img_w=240,
                rope_h_extrapolation_ratio=1,
                rope_w_extrapolation_ratio=1,
                rope_t_extrapolation_ratio=1,
                block_x_format="THWBD",
                patch_spatial=2,
                extra_per_block_abs_pos_emb=True,
            ),
            conditioner=dict(
                latent_condition=dict(
                    dropout_rate=0.05,
                )
            ),
            condition_keys=['basecolor', 'normal', 'metallic', 'roughness', 'depth', 'env_ldr', 'env_log', 'env_nrm', ],
            condition_drop_rate=0.05,
            append_condition_mask=True,
        ),
    )
)
cs.store(
    group="experiment",
    package="_global_",
    name="Diffusion_Renderer_Forward_Cosmos_7B",
    node=Diffusion_Renderer_Forward_Cosmos_7B,
)

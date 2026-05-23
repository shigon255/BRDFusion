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

import argparse
# from threedgrut.render import Renderer
from threedgrut.render_ply import Renderer
import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])

@hydra.main(config_path="configs", version_base=None)
def main(conf: DictConfig) -> None:
    # please set:
    # - conf.import_ply.path
    # - conf.out_dir
    # - conf.path
    ply_path = conf.import_ply.path
    out_dir = conf.out_dir
    path = conf.path 
    
    renderer = Renderer.from_ply(
                    ply_path=ply_path,
                    conf=conf,
                    path=path,
                    out_dir=out_dir,
                    save_gt=False,
                    computes_extra_metrics=False)

    renderer.render_all()

if __name__ == "__main__":
    main()
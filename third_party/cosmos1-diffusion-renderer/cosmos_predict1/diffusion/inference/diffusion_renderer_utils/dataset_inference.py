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

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .inference_utils import (
    find_images_recursive,
    group_images_into_videos,
    split_list_with_overlap,
    base_plus_ext,
)
from .transform_utils import prepare_images


class VideoFramesDataset(Dataset):
    def __init__(
        self,
        root_dir="/lustre/fsw/portfolios/nvr/users/zianw/experiments/test_video_cosmos",
        sample_n_frames=57,
        overlap_n_frames=0,
        chunk_mode="first",
        image_extensions=None,
        resolution=(704, 1280),  # (height, width)
        resize_resolution=None,
        subsample_every_n_frames=1,
        group_mode="folder",
        bg_color=(1., 1., 1.),
        normalize_cond_img=True,
        world_size=None, rank=None,
    ):
        """
        Args:
            root_dir (str): Path to the root folder containing images.
            sample_n_frames (int): Number of frames per video clip.
            overlap_n_frames (int): Number of overlapping frames between chunks.
            chunk_mode (str): Mode to split the sequence into chunks.
            image_extensions (list[str]): List of image file extensions to consider.
            resolution (tuple[int, int]): Desired (height, width) to resize the images.
            subsample_every_n_frames (int): If > 1, subsample frames within a video.
            group_mode (str): How images are grouped into videos (e.g., by folder).
            use_grayscale (bool): Whether to convert images to grayscale.
            bg_color (tuple): Background color for compositing.
            normalize_cond_img (bool): Whether to normalize image pixel values to [-1, 1].
        """
        self.root_dir = root_dir
        self.sample_n_frames = sample_n_frames
        self.overlap_n_frames = overlap_n_frames
        self.chunk_mode = chunk_mode
        self.group_mode = group_mode

        self.resolution = (resolution, resolution) if isinstance(resolution, int) else list(resolution)
        self.resize_resolution = resize_resolution if resize_resolution else self.resolution
        self.bg_color = bg_color
        self.normalize_cond_img = normalize_cond_img

        if image_extensions is None:
            self.image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp', '.exr']
        else:
            self.image_extensions = [image_extensions] if isinstance(image_extensions, str) else image_extensions

        # Step 1: Find all image paths (stored as relative paths)
        self.all_image_paths = find_images_recursive(root_dir, image_extensions=self.image_extensions)

        # Step 2: Group image paths into videos (e.g., by folder)
        self.video_groups = group_images_into_videos(
            self.all_image_paths,
            image_group_mode=group_mode,
            subsample_every_n_frames=subsample_every_n_frames,
        )

        # Step 3: Split each video group into chunks of length inference_n_frames.
        self.chunks = []
        self.chunk_index_list = []
        for video_relative_paths in self.video_groups:
            video_chunks = split_list_with_overlap(
                video_relative_paths, self.sample_n_frames, self.overlap_n_frames, chunk_mode=self.chunk_mode
            )
            self.chunks.extend(video_chunks)
            self.chunk_index_list.extend(list(range(len(video_chunks))))

        # Set up the resize transform.
        self.to_tensor = transforms.ToTensor()
        self.resize_transform = transforms.Resize(self.resize_resolution,
                                                  interpolation=transforms.InterpolationMode.BILINEAR,
                                                  antialias=True)
        self.resize_transform_nearest = transforms.Resize(self.resize_resolution,
                                                          interpolation=transforms.InterpolationMode.NEAREST,
                                                          antialias=False)
        self.crop_transform = transforms.CenterCrop(self.resolution)

        # split
        if world_size is not None and rank is not None:
            sample_list = self.chunks
            start_ratio = float(rank) / float(world_size)
            end_ratio = float(rank + 1) / float(world_size)
            self.chunks = sample_list[int(start_ratio * len(sample_list)):int(end_ratio * len(sample_list))]

    def __len__(self):
        return len(self.chunks)

    def _prepare_dummy_data_i4(self):
        dummy_text_embedding = torch.zeros(512, 1024)
        dummy_text_mask = torch.zeros(512)
        dummy_text_mask[0] = 1
        return {"t5_text_embeddings": dummy_text_embedding, "t5_text_mask": dummy_text_mask}

    def __getitem__(self, idx):
        # Get the list of relative image paths for the current chunk.
        chunk_relative_paths = self.chunks[idx]

        # If the chunk is shorter than inference_n_frames, repeat the last frame.
        if len(chunk_relative_paths) < self.sample_n_frames:
            chunk_relative_paths = chunk_relative_paths + [chunk_relative_paths[-1]] * (
                    self.sample_n_frames - len(chunk_relative_paths)
            )
        # Convert relative paths to absolute paths.
        abs_paths = [os.path.join(self.root_dir, rel_path) for rel_path in chunk_relative_paths]

        # Use prepare_images to load, transform, and normalize the images.
        video_tensor = prepare_images(
            abs_paths,
            use_grayscale=False,
            mask_np=None,
            bg_color=self.bg_color,
            resize_transform=self.resize_transform,
            crop_transform=self.crop_transform,
            normalize_cond_img=self.normalize_cond_img,
        )   # [T, C, H, W]

        # gamma for exr files
        if abs_paths[0].endswith(".exr"):
            video_tensor = (video_tensor + 1) / 2 if self.normalize_cond_img else video_tensor
            video_tensor = torch.pow(video_tensor.clip(0, 1), 1/2.2)
            if self.normalize_cond_img:
                video_tensor = video_tensor * 2 - 1

        # formatting output samples
        fps = 24
        t5_embed_dummy = self._prepare_dummy_data_i4()  # "t5_text_embeddings", "t5_text_mask"
        out_example = {
            "clip_name": base_plus_ext(chunk_relative_paths[0], mode=self.group_mode)[0],
            "chunk_index": f"{self.chunk_index_list[idx]:04d}",
            "is_preprocessed": True,
            "num_frames": torch.tensor(self.sample_n_frames, dtype=torch.float),
            "image_size": torch.from_numpy(np.asarray(self.resolution)),
            "fps": torch.tensor(fps, dtype=torch.float),
            "padding_mask": torch.zeros(1, self.resolution[0], self.resolution[1]),
            "t5_text_embeddings": t5_embed_dummy["t5_text_embeddings"],
            "t5_text_mask": t5_embed_dummy["t5_text_mask"],
            "context_index": torch.LongTensor([0]),
            "rgb": video_tensor.permute(1, 0, 2, 3)     # (C, N, H, W),
        }
        return out_example


class VideoGBufferDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sample_n_frames=57,
        image_extensions=None,
        resolution=(704, 1280),  # (height, width)
        resize_resolution=None,
        subsample_every_n_frames=1,
        group_mode="webdataset",
        gbuf_labels=["basecolor", "normal", "depth", "roughness", "metallic"],
        bg_color=(1., 1., 1.),
        normalize_cond_img=True,
        world_size=None, rank=None,
    ):
        """
        Args:
            root_dir (str): Root folder containing g-buffer images.
            sample_n_frames (int): Number of frames per video clip.
            overlap_n_frames (int): Number of overlapping frames between chunks.
            image_extensions (list[str]): List of image file extensions to consider.
            resolution (tuple[int, int]): Desired (height, width) for the images.
            resize_resolution (tuple[int, int], optional): Resolution to use for resizing. Defaults to resolution.
            subsample_every_n_frames (int): If > 1, subsample frames within a video.
            group_mode (str): How images are grouped into videos (e.g., by folder).
            gbuf_labels (list[str]): List of g-buffer labels to look for (e.g., ["basecolor", "normal", ...]).
            bg_color (tuple): Background color for compositing.
            normalize_cond_img (bool): Whether to normalize image pixel values to [-1, 1].
        """
        self.root_dir = root_dir
        self.sample_n_frames = sample_n_frames

        self.resolution = resolution if isinstance(resolution, (tuple, list)) else (resolution, resolution)
        self.resize_resolution = resize_resolution if resize_resolution else self.resolution
        self.bg_color = bg_color
        self.normalize_cond_img = normalize_cond_img
        self.subsample_every_n_frames = subsample_every_n_frames
        self.group_mode = group_mode
        self.gbuf_labels = gbuf_labels

        if image_extensions is None:
            self.image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp']
        else:
            self.image_extensions = [image_extensions] if isinstance(image_extensions, str) else image_extensions

        # Step 1: Find all image paths (relative paths)
        self.all_image_paths = find_images_recursive(root_dir, image_extensions=self.image_extensions)

        # Step 2: Group image paths into videos (e.g., by webdataset)
        self.video_groups = group_images_into_videos(
            self.all_image_paths,
            image_group_mode=group_mode,
            subsample_every_n_frames=subsample_every_n_frames,
        )

        # Set up the resize transform.
        self.to_tensor = transforms.ToTensor()
        self.resize_transform = transforms.Resize(self.resize_resolution,
                                                  interpolation=transforms.InterpolationMode.BILINEAR,
                                                  antialias=True)
        self.resize_transform_nearest = transforms.Resize(self.resize_resolution,
                                                          interpolation=transforms.InterpolationMode.NEAREST,
                                                          antialias=False)
        self.crop_transform = transforms.CenterCrop(self.resolution)

        # split
        if world_size is not None and rank is not None:
            sample_list = self.video_groups
            start_ratio = float(rank) / float(world_size)
            end_ratio = float(rank + 1) / float(world_size)
            self.video_groups = sample_list[int(start_ratio * len(sample_list)):int(end_ratio * len(sample_list))]

    def __len__(self):
        return len(self.video_groups)

    def _prepare_dummy_data_i4(self):
        dummy_text_embedding = torch.zeros(512, 1024)
        dummy_text_mask = torch.zeros(512)
        dummy_text_mask[0] = 1
        return {"t5_text_embeddings": dummy_text_embedding, "t5_text_mask": dummy_text_mask}

    def __getitem__(self, idx):
        # Get the list of relative image paths for the current chunk.
        chunk_relative_paths = self.video_groups[idx]
        get_label_paths = lambda label: sorted([k for k in chunk_relative_paths if f'.{label}.' in k])

        # Process each g-buffer label.
        out_example = {}
        for label in self.gbuf_labels:
            # Filter the relative paths for the current g-buffer label.
            label_paths = get_label_paths(label)
            if len(label_paths) == 0:
                continue
            if len(label_paths) < self.sample_n_frames:
                label_paths = label_paths + [label_paths[-1]] * (self.sample_n_frames - len(label_paths))
            if len(label_paths) > self.sample_n_frames:
                # Note that the current loader will not do re-chunking, and only take first chunk
                label_paths = label_paths[:self.sample_n_frames]

            abs_paths = [os.path.join(self.root_dir, rel_path) for rel_path in label_paths]
            video_tensor = prepare_images(
                abs_paths,
                use_grayscale=False,
                mask_np=None,
                bg_color=self.bg_color,
                resize_transform=self.resize_transform,
                crop_transform=self.crop_transform,
                normalize_cond_img=self.normalize_cond_img,
            )
            out_example[label] = video_tensor.permute(1, 0, 2, 3)   # [C, T, H, W]

        # Format the output similar to VideoFramesDataset.
        fps = 24
        dummy = self._prepare_dummy_data_i4()
        out_example.update({
            "clip_name": base_plus_ext(chunk_relative_paths[0], mode=self.group_mode)[0],
            "is_preprocessed": True,
            "num_frames": torch.tensor(self.sample_n_frames, dtype=torch.float),
            "image_size": torch.from_numpy(np.asarray(self.resolution)),
            "fps": torch.tensor(fps, dtype=torch.float),
            "padding_mask": torch.zeros(1, self.resolution[0], self.resolution[1]),
            "t5_text_embeddings": dummy["t5_text_embeddings"],
            "t5_text_mask": dummy["t5_text_mask"],
            "context_index": torch.LongTensor([0]),
        })
        return out_example


# ----------------------------------------------------------------------------
# Example usage:
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = VideoFramesDataset(root_dir="path/to/your/images")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]  # Get the first video clip
    print(f"Sample shape: {sample.shape}")  # Expected: [num_frames, channels, height, width]

    dataset = VideoGBufferDataset(
        root_dir="path/to/your/gbuffer/images",
        sample_n_frames=57,
        gbuf_labels=["basecolor", "normal", "depth", "roughness", "metallic"],
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    for label, tensor in sample["gbufs"].items():
        print(f"{label} tensor shape: {tensor.shape}")
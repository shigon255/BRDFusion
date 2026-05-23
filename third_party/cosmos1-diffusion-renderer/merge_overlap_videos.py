#!/usr/bin/env python3
"""
Merge overlapping video chunks by averaging frames that appear in multiple chunks.

Usage:
    python merge_overlap_videos.py <input_dir> <output_dir> <weight> <camera> <seed> <buffer_type> [--chunk_size 57] [--step 7]

Example:
    python merge_overlap_videos.py \
        drivestudio_exp/3/refine_intrinsics_gtrgb_full_refine_overlap50 \
        drivestudio_exp/3/refine_intrinsics_gtrgb_full_refine_overlap50 \
        .5 1 1000 basecolor
"""

import argparse
import os
import glob
import re
import subprocess
import numpy as np
from PIL import Image
from collections import defaultdict
import tempfile
import shutil


def extract_frames(video_path, output_dir):
    """Extract all frames from a video to a directory."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vsync', '0',
        os.path.join(output_dir, 'frame_%04d.png'),
        '-y', '-loglevel', 'error'
    ]
    subprocess.run(cmd, check=True)

    # Return list of frame paths
    frames = sorted(glob.glob(os.path.join(output_dir, 'frame_*.png')))
    return frames


def encode_video(frame_pattern, output_path, fps=24):
    """Encode frames to video."""
    cmd = [
        'ffmpeg', '-framerate', str(fps),
        '-i', frame_pattern,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path,
        '-y', '-loglevel', 'error'
    ]
    subprocess.run(cmd, check=True)


def average_images(image_paths):
    """Average multiple images and return the result."""
    if len(image_paths) == 0:
        return None

    # Load the first image
    base_image = Image.open(image_paths[0]).convert("RGB")
    base_array = np.array(base_image, dtype=np.float64)

    # Add the rest
    for path in image_paths[1:]:
        img = Image.open(path).convert("RGB")
        base_array += np.array(img, dtype=np.float64)

    # Average
    avg_array = (base_array / len(image_paths)).astype(np.uint8)
    return Image.fromarray(avg_array)


def main():
    parser = argparse.ArgumentParser(description='Merge overlapping video chunks')
    parser.add_argument('input_dir', help='Directory containing sdedit_* folders')
    parser.add_argument('output_dir', help='Output directory for merged results')
    parser.add_argument('weight', help='Weight value (e.g., .5)')
    parser.add_argument('camera', help='Camera index (e.g., 1)')
    parser.add_argument('seed', help='Seed value (e.g., 1000)')
    parser.add_argument('buffer_type', help='Buffer type (e.g., basecolor, normal, depth, roughness, metallic)')
    parser.add_argument('--chunk_size', type=int, default=57, help='Number of frames per chunk')
    parser.add_argument('--step', type=int, default=7, help='Step between chunk starts')
    parser.add_argument('--fps', type=int, default=24, help='Output video FPS')

    args = parser.parse_args()

    # Find all matching folders
    pattern = f"sdedit_w{args.weight}_cam{args.camera}_seed{args.seed}_start*"
    folder_pattern = os.path.join(args.input_dir, pattern)
    folders = sorted(glob.glob(folder_pattern))

    if not folders:
        print(f"No folders found matching pattern: {folder_pattern}")
        return

    print(f"Found {len(folders)} chunks to merge")

    # Extract start indices from folder names
    start_indices = []
    for folder in folders:
        match = re.search(r'_start(\d+)$', folder)
        if match:
            start_indices.append(int(match.group(1)))

    # Determine total number of frames
    max_start = max(start_indices)
    # Check actual frame count in last chunk
    last_folder = [f for f in folders if f.endswith(f'_start{max_start}')][0]
    video_pattern = f"*{args.buffer_type}.mp4"
    last_videos = glob.glob(os.path.join(last_folder, video_pattern))
    if not last_videos:
        print(f"No video found matching {video_pattern} in {last_folder}")
        return

    # Get frame count from last video
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-count_packets', '-show_entries', 'stream=nb_read_packets',
           '-of', 'csv=p=0', last_videos[0]]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    last_chunk_frames = int(result.stdout.strip())

    total_frames = max_start + last_chunk_frames
    print(f"Total frames to merge: {total_frames}")

    # Create temporary directory for frame extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        # Dictionary to store frame paths: frame_idx -> list of image paths
        frame_sources = defaultdict(list)

        # Extract frames from each chunk
        for folder in folders:
            match = re.search(r'_start(\d+)$', folder)
            if not match:
                continue
            start_idx = int(match.group(1))

            # Find the video file
            videos = glob.glob(os.path.join(folder, f"*{args.buffer_type}.mp4"))
            if not videos:
                print(f"Warning: No {args.buffer_type} video in {folder}")
                continue

            video_path = videos[0]
            chunk_dir = os.path.join(tmpdir, f"chunk_start{start_idx}")

            print(f"Extracting frames from {os.path.basename(folder)}...")
            frames = extract_frames(video_path, chunk_dir)

            # Map extracted frames to global frame indices
            for local_idx, frame_path in enumerate(frames):
                global_idx = start_idx + local_idx
                frame_sources[global_idx].append(frame_path)

        # Create output directory for merged frames
        merged_frames_dir = os.path.join(tmpdir, 'merged')
        os.makedirs(merged_frames_dir, exist_ok=True)

        # Average frames
        print("Averaging overlapping frames...")
        for frame_idx in sorted(frame_sources.keys()):
            sources = frame_sources[frame_idx]
            avg_image = average_images(sources)
            if avg_image:
                output_path = os.path.join(merged_frames_dir, f'frame_{frame_idx:04d}.png')
                avg_image.save(output_path)
                if frame_idx % 20 == 0:
                    print(f"  Frame {frame_idx}: averaged {len(sources)} source(s)")

        # Create output directory
        output_folder = os.path.join(args.output_dir,
                                     f"sdedit_w{args.weight}_cam{args.camera}_seed{args.seed}")
        os.makedirs(output_folder, exist_ok=True)

        # Find the video filename pattern from first folder
        first_video = glob.glob(os.path.join(folders[0], f"*{args.buffer_type}.mp4"))[0]
        video_basename = os.path.basename(first_video)
        output_video_path = os.path.join(output_folder, video_basename)

        # Encode merged video
        print(f"Encoding merged video to {output_video_path}...")
        frame_pattern = os.path.join(merged_frames_dir, 'frame_%04d.png')
        encode_video(frame_pattern, output_video_path, fps=args.fps)

        print(f"Done! Merged video saved to {output_video_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Convert video files to gbuffer_frames format for forward renderer.

This script extracts frames from multiple videos (albedo, normal, depth, roughness, metallic)
and organizes them into the gbuffer_frames directory structure expected by the forward renderer.
"""

import argparse
import os
import cv2
from tqdm import tqdm


def extract_frames(video_path, output_dir, frame_type, prefix="0000", max_frames=None):
    """
    Extract frames from a video and save them with the gbuffer naming convention.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the frames
        frame_type: Type of frame (basecolor, depth, metallic, normal, roughness)
        prefix: Prefix for frame names (default: "0000")
    """
    if not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    print(f"Processing {frame_type}: {os.path.basename(video_path)} ({total_frames} frames)")

    with tqdm(total=total_frames, desc=f"  {frame_type}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames is not None and frame_idx >= max_frames:
                break

            # Generate filename: 0000.0000.basecolor.png, 0000.0001.basecolor.png, etc.
            frame_name = f"{prefix}.{frame_idx:04d}.{frame_type}.png"
            frame_path = os.path.join(output_dir, frame_name)

            # Save frame as PNG
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            frame_idx += 1
            pbar.update(1)

    cap.release()
    return frame_idx


def main():
    parser = argparse.ArgumentParser(
        description="Convert videos to gbuffer_frames format for forward renderer"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="drivestudio_exp",
        help="Directory containing input videos (default: drivestudio_exp)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="drivestudio_exp/gbuffer_frames/video1",
        help="Output directory for gbuffer frames (default: drivestudio_exp/gbuffer_frames/video1)"
    )
    parser.add_argument(
        "--albedo_video",
        type=str,
        default="albedo_resize.mp4",
        help="Albedo/basecolor video filename (default: albedo_resize.mp4)"
    )
    parser.add_argument(
        "--normal_video",
        type=str,
        default="normal_resize.mp4",
        help="Normal video filename (default: normal_resize.mp4)"
    )
    parser.add_argument(
        "--depth_video",
        type=str,
        default="normalized_depth_resize.mp4",
        help="Depth video filename (default: normalized_depth_resize.mp4)"
    )
    parser.add_argument(
        "--roughness_video",
        type=str,
        default="roughness_resize.mp4",
        help="Roughness video filename (default: roughness_resize.mp4)"
    )
    parser.add_argument(
        "--metallic_video",
        type=str,
        default="metallic_resize.mp4",
        help="Metallic video filename (default: metallic_resize.mp4)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}\n")

    # Define video-to-type mapping
    # extract
    
    video_mapping = [
        (args.albedo_video, "basecolor"),
        (args.normal_video, "normal"),
        (args.depth_video, "depth"),
        (args.roughness_video, "roughness"),
        (args.metallic_video, "metallic"),
    ]

    # Process each video
    frame_counts = {}
    for video_filename, frame_type in video_mapping:
        video_path = os.path.join(args.input_dir, video_filename)
        num_frames = extract_frames(video_path, args.output_dir, frame_type)
        frame_counts[frame_type] = num_frames

    # Summary
    print("\n" + "="*50)
    print("Conversion Complete!")
    print("="*50)
    print(f"Output directory: {args.output_dir}")
    print("\nFrames extracted per type:")
    for frame_type, count in frame_counts.items():
        print(f"  {frame_type:12s}: {count} frames")

    # Check if all videos have the same number of frames
    unique_counts = set(frame_counts.values())
    if len(unique_counts) > 1:
        print("\n⚠️  Warning: Videos have different numbers of frames!")
        print("This may cause issues with the forward renderer.")
    elif len(unique_counts) == 1 and list(unique_counts)[0] > 0:
        print(f"\n✓ All videos have {list(unique_counts)[0]} frames")

    print(f"\nYou can now run the forward renderer with:")
    print(f"  --dataset_path={args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""merge.py

Merge three videos horizontally (left, middle, right) into one video.

Usage:
  python merge.py left.mp4 middle.mp4 right.mp4 -o merged.mp4
"""

import argparse
import cv2
import os
import sys
import numpy as np


def merge_three_videos(left_path, middle_path, right_path, output_path):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    caps = [cv2.VideoCapture(p) for p in (left_path, middle_path, right_path)]

    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: cannot open video: { (left_path, middle_path, right_path)[i] }")
            for c in caps:
                c.release()
            return

    # Read properties from inputs
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
    widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]

    # Use the minimum height to avoid stretching
    target_h = min(heights)

    # Compute resized widths maintaining aspect ratio for the target height
    resized_widths = [int(w * (target_h / h)) if h != 0 else 0 for w, h in zip(widths, heights)]
    total_w = sum(resized_widths)

    if total_w <= 0 or target_h <= 0:
        print("Error: invalid video dimensions")
        for c in caps:
            c.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_w, target_h))

    frame_idx = 0
    while True:
        rets = []
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            rets.append(ret)
            frames.append(frame)

        # Stop when any of the videos ends
        if not all(rets):
            break

        # Resize each frame to target height
        resized = []
        for i, f in enumerate(frames):
            h, w = f.shape[:2]
            new_w = resized_widths[i]
            if h != target_h or w != new_w:
                f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA)
            resized.append(f)

        merged = np.hstack(resized)
        out.write(merged)
        frame_idx += 1

    for cap in caps:
        cap.release()
    out.release()

    print(f"Merged {frame_idx} frames into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge three videos horizontally")
    parser.add_argument("left", help="Left video path")
    parser.add_argument("middle", help="Middle video path")
    parser.add_argument("right", help="Right video path")
    parser.add_argument("-o", "--output", default="merged.mp4", help="Output video path")

    args = parser.parse_args()

    # Ensure input files exist
    for p in (args.left, args.middle, args.right):
        if not os.path.exists(p):
            print(f"Error: file not found: {p}")
            sys.exit(2)

    merge_three_videos(args.left, args.middle, args.right, args.output)


if __name__ == "__main__":
    main()

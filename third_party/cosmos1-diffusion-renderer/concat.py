#!/usr/bin/env python3
import argparse
import os
import shutil
import sys

import cv2

def choose_codec(output_path):
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".mp4", ".m4v", ".mov"):
        return "mp4v"
    if ext in (".avi",):
        return "XVID"
    if ext in (".mkv",):
        return "X264"
    return "mp4v"

def concat_opencv(inputs, output, verbose=False):
    if len(inputs) == 0:
        if verbose:
            print("No inputs provided.", file=sys.stderr)
        return False

    # Open first to get properties
    first = inputs[0]
    cap0 = cv2.VideoCapture(first)
    if not cap0.isOpened():
        if verbose:
            print(f"Failed to open {first}", file=sys.stderr)
        return False
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    cap0.release()

    codec = choose_codec(output)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
    if not writer.isOpened():
        if verbose:
            print(f"Failed to open VideoWriter for {output} with codec {codec}", file=sys.stderr)
        return False

    for path in inputs:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            if verbose:
                print(f"Failed to open {path}", file=sys.stderr)
            writer.release()
            return False
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # We use the fps of the first file; if other files have different fps, frames are written at target fps.
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                continue
            # Resize if needed
            if (src_w, src_h) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # Ensure 3 channels (color)
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
        cap.release()
        if verbose:
            print(f"Wrote frames from {path}")

    writer.release()
    if verbose:
        print(f"Saved concatenated video to {output}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Concatenate videos using OpenCV (no audio support).")
    parser.add_argument("inputs", nargs="+", help="Input video files (at least 1).")
    parser.add_argument("-o", "--output", required=True, help="Output file path.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    if len(args.inputs) == 0:
        print("No input files provided.", file=sys.stderr)
        sys.exit(2)

    # Single input: copy file
    if len(args.inputs) == 1:
        try:
            shutil.copyfile(args.inputs[0], args.output)
            if args.verbose:
                print("Copied single input to output.")
            sys.exit(0)
        except Exception as e:
            print("Failed to copy single input:", e, file=sys.stderr)
            sys.exit(1)

    success = concat_opencv(args.inputs, args.output, verbose=args.verbose)
    if success:
        # Note: audio is not preserved when using OpenCV
        sys.exit(0)
    else:
        print("Failed to concatenate videos (OpenCV).", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

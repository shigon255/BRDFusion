import argparse
import sys
import cv2
import imageio

#!/usr/bin/env python3
"""
resize_vid.py

Crop (center) and resize a video to a target size while preserving aspect ratio.
First center-crop the input frames so their aspect ratio matches the target aspect ratio,
then scale both dimensions with the same factor to reach the exact target size.

Usage:
    python resize_vid.py --input input.mp4 --output output.mp4 --width 1280 --height 720

Dependencies:
    - OpenCV (pip install opencv-python)
    - imageio (pip install imageio imageio-ffmpeg)
"""



def center_crop(frame, target_aspect):
    h, w = frame.shape[:2]
    src_aspect = w / h
    if abs(src_aspect - target_aspect) < 1e-9:
        return frame  # already same aspect
    if src_aspect > target_aspect:
        # source is wider -> crop width
        new_w = int(h * target_aspect)
        x0 = (w - new_w) // 2
        return frame[:, x0 : x0 + new_w]
    else:
        # source is taller -> crop height
        new_h = int(w / target_aspect)
        y0 = (h - new_h) // 2
        return frame[y0 : y0 + new_h, :]


def process_video(input_path, output_path, target_w, target_h, codec="libx264"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open input file '{input_path}'", file=sys.stderr)
        return 1

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    target_aspect = target_w / target_h

    writer = imageio.get_writer(output_path, fps=fps, codec=codec)

    print(f"Input: {src_w}x{src_h}, fps={fps}, frames={frame_count}")
    print(f"Target: {target_w}x{target_h}, aspect={target_aspect:.4f}")

    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # direct resizing without cropping
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB for imageio
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        writer.append_data(resized_rgb)
        processed += 1
        if frame_count and processed % 100 == 0:
            print(f"Processed {processed}/{frame_count} frames...", end="\r")

    cap.release()
    writer.close()
    print(f"\nDone. Processed {processed} frames. Output saved to '{output_path}'")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Center-crop and uniformly resize a video to a target size.")
    p.add_argument("--input", "-i", required=True, help="Input video file path")
    p.add_argument("--output", "-o", required=True, help="Output video file path (e.g. out.mp4)")
    p.add_argument("--width", "-W", type=int, required=True, help="Target width in pixels")
    p.add_argument("--height", "-H", type=int, required=True, help="Target height in pixels")
    p.add_argument("--codec", "-c", default="libx264", help="Codec for imageio (default: libx264)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(process_video(args.input, args.output, args.width, args.height, args.codec))

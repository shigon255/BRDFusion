import argparse
import os
import cv2

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    count = 0
    success, frame = cap.read()
    while success:
        img_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1
        success, frame = cap.read()
    cap.release()
    print(f"Extracted {count} frames to '{output_dir}'.")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from an MP4 video.")
    parser.add_argument("video", help="Path to the input MP4 video file.")
    parser.add_argument("output_dir", help="Directory to save extracted images.")
    args = parser.parse_args()
    extract_frames(args.video, args.output_dir)

if __name__ == "__main__":
    main()
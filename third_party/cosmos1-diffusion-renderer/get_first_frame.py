import cv2
import sys

def extract_frame(video_path, output_image_path, frame_index=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Check total frames if available
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    if frame_index < 0:
        print("Error: frame_index must be non-negative.")
        cap.release()
        return
    if total_frames is not None and frame_index >= total_frames:
        print(f"Error: frame_index {frame_index} out of range (total frames: {total_frames}).")
        cap.release()
        return

    # Seek to desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_index} saved as {output_image_path}")
    else:
        print(f"Error: Cannot read frame {frame_index} from video.")

    cap.release()

if __name__ == "__main__":
    if not (3 <= len(sys.argv) <= 4):
        print("Usage: python get_first_frame.py <input_video> <output_image> [frame_index (optional, zero-based)]")
    else:
        video = sys.argv[1]
        out = sys.argv[2]
        if len(sys.argv) == 4:
            try:
                idx = int(sys.argv[3])
            except ValueError:
                print("Error: frame_index must be an integer.")
                sys.exit(1)
        else:
            idx = 0
        extract_frame(video, out, idx)
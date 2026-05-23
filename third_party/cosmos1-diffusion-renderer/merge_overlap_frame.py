import argparse
import cv2
import numpy as np
from pathlib import Path

def stack_videos_vertically(video1_path, video2_path, video3_path, output_path, 
                           start_frames=None, end_frames=None):
    """Stack three videos vertically and save the result."""
    
    # Open video readers
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    cap3 = cv2.VideoCapture(video3_path)
    
    # Get video properties from first video
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set default start/end frames if not provided
    if start_frames is None:
        start_frames = [0, 0, 0]
    if end_frames is None:
        end_frames = [
            int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap3.get(cv2.CAP_PROP_FRAME_COUNT))
        ]
    
    # Calculate number of frames to process (minimum of all three)
    num_frames = min(
        end_frames[0] - start_frames[0],
        end_frames[1] - start_frames[1],
        end_frames[2] - start_frames[2]
    )
    
    # Seek to start frames
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frames[0])
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frames[1])
    cap3.set(cv2.CAP_PROP_POS_FRAMES, start_frames[2])
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height * 3))
    
    frame_count = 0
    while frame_count < num_frames:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        
        if not (ret1 and ret2 and ret3):
            break
        
        # Ensure all frames have same width
        frame2 = cv2.resize(frame2, (width, height))
        frame3 = cv2.resize(frame3, (width, height))
        
        # Stack vertically
        stacked = np.vstack((frame1, frame2, frame3))
        out.write(stacked)
        frame_count += 1
    
    cap1.release()
    cap2.release()
    cap3.release()
    out.release()
    print(f"Output saved to {output_path} ({frame_count} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack three videos vertically")
    parser.add_argument("video1", help="Path to first video")
    parser.add_argument("video2", help="Path to second video")
    parser.add_argument("video3", help="Path to third video")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video path")
    parser.add_argument("-s", "--start", type=int, nargs=3, metavar=("V1", "V2", "V3"),
                       help="Start frame for each video")
    parser.add_argument("-e", "--end", type=int, nargs=3, metavar=("V1", "V2", "V3"),
                       help="End frame for each video")
    
    args = parser.parse_args()
    
    stack_videos_vertically(args.video1, args.video2, args.video3, args.output,
                           args.start, args.end)
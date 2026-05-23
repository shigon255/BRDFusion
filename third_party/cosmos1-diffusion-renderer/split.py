import argparse
import cv2
import os

def split_video_by_width(input_video, output_dir):
    """Split a video into three videos by dividing width into three equal parts."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original video name without extension
    ori_vid_name = os.path.splitext(os.path.basename(input_video))[0]
    
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate split width
    split_width = width // 3
    
    # Create video writers for three output videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(os.path.join(output_dir, f'{ori_vid_name}_1.mp4'), fourcc, fps, (split_width, height))
    out2 = cv2.VideoWriter(os.path.join(output_dir, f'{ori_vid_name}_0.mp4'), fourcc, fps, (split_width, height))
    out3 = cv2.VideoWriter(os.path.join(output_dir, f'{ori_vid_name}_2.mp4'), fourcc, fps, (split_width, height))
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Split frame into three parts
        part1 = frame[:, :split_width]
        part2 = frame[:, split_width:2*split_width]
        part3 = frame[:, 2*split_width:]
        
        # Write to output videos
        out1.write(part1)
        out2.write(part2)
        out3.write(part3)
    
    # Release resources
    cap.release()
    out1.release()
    out2.release()
    out3.release()
    
    print(f"Successfully split video into three parts in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a video into three videos by width')
    parser.add_argument('input_video', help='Path to the input video file')
    
    args = parser.parse_args()
    
    # Get the directory of the input video
    output_dir = os.path.dirname(args.input_video) or '.'
    
    split_video_by_width(args.input_video, output_dir)
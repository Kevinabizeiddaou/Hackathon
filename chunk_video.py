import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def chunk_video_adaptive(video_path: str,
                         min_chunk: int = 4,
                         max_chunk: int = 24,
                         motion_threshold: float = 0.0375,
                         frame_sample_rate: int = 40):
    """
    Splits a video into variable-length chunks based on pixel-level motion.

    Args:
        video_path (str): Path to the input video file.
        min_chunk (int): Minimum chunk length in seconds.
        max_chunk (int): Maximum chunk length in seconds.
        motion_threshold (float): Normalized threshold for motion activity (0–1).
        frame_sample_rate (int): Process every Nth frame for efficiency.

    Returns:
        list: A list of VideoFileClip objects representing the chunks.
    """
    # Load full clip for later subclipping
    full_clip = VideoFileClip(video_path)
    fps = full_clip.fps
    total_duration = full_clip.duration

    # OpenCV video reader for frame analysis
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        raise ValueError("Could not read video file.")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_idx = 0
    current_start = 0.0
    chunks = []
    activity_window = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_sample_rate != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        prev_gray = gray

        # Normalized motion intensity
        motion_score = np.mean(diff) / 255.0
        activity_window.append(motion_score)

        # Convert current frame to seconds
        elapsed_time = frame_idx / fps
        chunk_duration = elapsed_time - current_start

        # Smooth average motion over recent frames
        avg_motion = np.mean(activity_window[-int(fps * min_chunk / frame_sample_rate):]) if activity_window else 0

        # Decide if we should start a new chunk
        if (chunk_duration >= min_chunk and avg_motion > motion_threshold) or chunk_duration >= max_chunk:
            # Create subclip for current segment
            end_time = min(elapsed_time, total_duration)
            chunk_clip = full_clip.subclip(current_start, end_time)
            chunks.append(chunk_clip)

            # Reset
            current_start = end_time
            activity_window = []

    # Handle remaining video tail
    if current_start < total_duration:
        chunk_clip = full_clip.subclip(current_start, total_duration)
        chunks.append(chunk_clip)

    cap.release()

    return chunks
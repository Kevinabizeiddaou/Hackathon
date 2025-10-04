"""
Chunker: motion-based, variable-length video splitting.

Main entry:
    chunk_video(
        video_path: str,
        min_chunk: int = 4,
        max_chunk: int = 24,
        motion_threshold: float = 0.0375,
        frame_sample_rate: int = 40,
    ) -> list[CVClip]

What it does:
    Reads a video and returns a list of subclips (CVClip) whose boundaries
    are decided by pixel-level motion (OpenCV). Chunks are at least `min_chunk`
    seconds, at most `max_chunk` seconds.

Inputs:
    video_path: path to the input video
    min_chunk: minimum chunk length in seconds
    max_chunk: maximum chunk length in seconds
    motion_threshold: normalized [0..1] motion trigger
    frame_sample_rate: process every Nth frame for efficiency

Output:
    A list of CVClip subclips. Remember to close them when done.

Usage:
    from chunker import chunk_video, close_video_chunks
    chunks = chunk_video("eagle.mp4", min_chunk=4, max_chunk=24)
    # ... use chunks ...
    close_video_chunks(chunks)  # free resources when done
"""

from typing import List
import numpy as np
import cv2

from cvclip import CVClip

def chunk_video(
    video_path: str,
    min_chunk: int = 4,
    max_chunk: int = 24,
    motion_threshold: float = 0.0375,
    frame_sample_rate: int = 40,
) -> List[CVClip]:
    """
    Splits a video into variable-length chunks based on pixel-level motion.

    Args:
        video_path (str): Path to the input video file.
        min_chunk (int): Minimum chunk length in seconds.
        max_chunk (int): Maximum chunk length in seconds.
        motion_threshold (float): Normalized threshold for motion activity (0–1).
        frame_sample_rate (int): Process every Nth frame for efficiency.

    Returns:
        list[CVClip]: List of subclips representing the chunks.
    """
    # Load full clip for later subclipping
    full_clip = CVClip(video_path)
    fps = float(full_clip.fps)
    total_duration = float(full_clip.duration)

    # OpenCV video reader for frame analysis
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read video file.")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_idx = 0
    current_start = 0.0
    chunks: List[CVClip] = []
    activity_window: list[float] = []

    # Safe smoothing window to avoid zero-length slicing
    win = max(1, int((fps * min_chunk) / max(1, frame_sample_rate)))

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
        motion_score = float(np.mean(diff) / 255.0)
        activity_window.append(motion_score)

        # Convert current frame to seconds
        elapsed_time = frame_idx / fps
        chunk_duration = elapsed_time - current_start

        # Smooth average motion over recent frames
        avg_motion = float(np.mean(activity_window[-win:])) if activity_window else 0.0

        # Decide if we should start a new chunk
        should_cut = (
            (chunk_duration >= min_chunk and avg_motion > motion_threshold)
            or (chunk_duration >= max_chunk)
        )
        if should_cut:
            end_time = min(elapsed_time, total_duration)
            chunk_clip = full_clip.subclip(current_start, end_time)
            chunks.append(chunk_clip)

            # Reset
            current_start = end_time
            activity_window = []

    # Handle remaining tail
    if current_start < total_duration:
        chunk_clip = full_clip.subclip(current_start, total_duration)
        chunks.append(chunk_clip)

    cap.release()
    # NOTE: Do NOT close `full_clip` here; subclips are independent views.
    return chunks


def close_video_chunks(chunks: List[CVClip]) -> None:
    """Close all chunk clips to free file handles."""
    for c in chunks:
        try:
            c.close()
        except Exception:
            pass


__all__ = ["chunk_video", "close_video_chunks"]

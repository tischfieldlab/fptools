import os
import cv2
import numpy as np


def get_frame_image(video: str, frame_idx: int) -> np.ndarray:
    """Get a frame from a video using openCV.

    Args:
        video: string path to the video file
        frame_idx: integer index of the frame to grab

    Returns:
        numpy array of the image at `frame_idx`
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        return

    cap.release()
    return frame

import os
from typing import Union
import cv2
import numpy as np


def get_frame_image(video: str, frame_idx: int) -> Union[np.ndarray, None]:
    """Get a frame from a video using openCV.

    Args:
        video: string path to the video file
        frame_idx: integer index of the frame to grab

    Returns:
        numpy array of the image at `frame_idx`, Or None if the frame could not be retrieved
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        return None

    cap.release()
    return frame

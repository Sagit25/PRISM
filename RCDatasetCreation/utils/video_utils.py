"""Video generation utilities for creating MP4 videos from image sequences."""

import os
import os.path as osp
from typing import Sequence, Union

import imageio.v2 as imageio
import numpy as np


def _load_frame(frame):
    """Load a single video frame from an ndarray or path."""
    if isinstance(frame, (str, os.PathLike)):
        return imageio.imread(frame)
    return np.asarray(frame)


def _prepare_frame(frame: np.ndarray) -> np.ndarray:
    """
    Ensure video frames are uint8 RGB as expected by imageio writers.

    Handles grayscale, RGBA, and floating-point conversions.
    """
    arr = np.asarray(frame)

    # Convert grayscale to RGB
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    # Convert RGBA to RGB
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Convert floating-point to uint8
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def gen_video_from_folder(frame_folder, video_path, fps=60):
    """
    Generate video from all images in a folder.

    Args:
        frame_folder: Directory containing image files
        video_path: Output video file path (.mp4)
        fps: Frames per second (default: 60)
    """
    paths = sorted(os.listdir(frame_folder))
    frames = []
    for path in paths:
        img_path = osp.join(frame_folder, path)
        if not osp.isfile(img_path):
            continue
        try:
            frames.append(imageio.imread(img_path))
        except Exception:
            continue
    gen_video(frames, video_path, fps)


def gen_video(frame_list: Sequence[Union[str, np.ndarray]], video_path, fps=60.0):
    """
    Generate video from a list of frames.

    Args:
        frame_list: List of image paths or numpy arrays
        video_path: Output video file path (.mp4)
        fps: Frames per second (default: 60.0)
    """
    writer_kwargs = {
        'fps': fps,
        'codec': 'libx264',
        'format': 'FFMPEG',
        'macro_block_size': None
    }
    with imageio.get_writer(video_path, **writer_kwargs) as writer:
        for frame in frame_list:
            arr = _prepare_frame(_load_frame(frame))
            writer.append_data(arr)

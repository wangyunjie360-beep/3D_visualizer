import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np


def default_screenshot_path(directory: str, timestamp: Optional[str] = None) -> str:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(directory, f"render_{stamp}.png")


def screenshot_render_size(width: int, height: int, scale: float = 1.0) -> Tuple[int, int]:
    scale = max(1.0, float(scale))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def render_scene_image(app, scene, width: int, height: int, scale: float = 1.0):
    render_width, render_height = screenshot_render_size(width, height, scale)
    return app.render_to_image(scene, render_width, render_height)


def image_to_uint8_array(image) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)
    if image_array.shape[2] > 3:
        image_array = image_array[:, :, :3]
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return image_array


def stitch_screenshot_arrays(images, rows: int, cols: int) -> np.ndarray:
    if not images:
        raise ValueError("no screenshot images to stitch")
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    arrays = [image_to_uint8_array(image) for image in images]
    cell_height = max(array.shape[0] for array in arrays)
    cell_width = max(array.shape[1] for array in arrays)
    channels = arrays[0].shape[2]
    stitched = np.full(
        (rows * cell_height, cols * cell_width, channels), 255, dtype=np.uint8
    )
    for idx, array in enumerate(arrays[: rows * cols]):
        row = idx // cols
        col = idx % cols
        y = row * cell_height
        x = col * cell_width
        stitched[y : y + array.shape[0], x : x + array.shape[1], : array.shape[2]] = array
    return stitched

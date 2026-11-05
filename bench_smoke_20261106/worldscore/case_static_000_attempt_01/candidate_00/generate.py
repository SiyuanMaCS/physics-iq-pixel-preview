"""Generate a smooth photographic push-in for the supplied cafe still."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import artifact_io


WIDTH = 256
HEIGHT = 175
FRAME_COUNT = 50


def smootherstep(value: float) -> float:
    """Ease the camera gently into and out of its five-second move."""
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def make_frames(source: np.ndarray) -> np.ndarray:
    frames = np.empty((FRAME_COUNT, HEIGHT, WIDTH, 3), dtype=np.uint8)
    frames[0] = source

    # Move the crop center a little upward as we advance toward the coffee bar.
    # The restrained 18% push retains the chairs and counter as recognizable
    # anchors while making the camera move unambiguous.
    xs, ys = np.meshgrid(
        np.arange(WIDTH, dtype=np.float32),
        np.arange(HEIGHT, dtype=np.float32),
    )
    output_cx = (WIDTH - 1) / 2.0
    output_cy = (HEIGHT - 1) / 2.0

    for index in range(1, FRAME_COUNT):
        progress = smootherstep(index / (FRAME_COUNT - 1))
        zoom = 1.0 + 0.18 * progress
        source_cx = output_cx + 1.5 * progress
        source_cy = output_cy - 6.0 * progress

        map_x = source_cx + (xs - output_cx) / zoom
        map_y = source_cy + (ys - output_cy) / zoom
        frames[index] = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return frames


def main() -> None:
    case_dir = Path(__file__).resolve().parent
    source = np.asarray(Image.open(case_dir / "first_frame.png").convert("RGB"))
    if source.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"Unexpected first-frame shape: {source.shape}")
    frames = make_frames(source)
    artifact_io.save(frames, case_dir)


if __name__ == "__main__":
    main()

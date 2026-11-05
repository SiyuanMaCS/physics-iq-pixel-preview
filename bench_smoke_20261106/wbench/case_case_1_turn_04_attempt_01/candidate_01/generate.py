"""Generate a short, deterministic 2.5-D counterclockwise camera orbit.

The source is deliberately treated as pixels rather than as a scene to be
re-rendered.  The road contact point of the dog is the orbit pivot.  Distant
pixels travel furthest across the frame while the dog is reprojected about its
feet, which supplies both parallax and a small, plausible change of viewing
angle without making the animal walk.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import artifact_io


HERE = Path(__file__).resolve().parent
W, H = 256, 143
N = 20


def smoothstep(t: float) -> float:
    """A camera move with gentle starts and stops."""
    return t * t * (3.0 - 2.0 * t)


def dog_matte() -> np.ndarray:
    """A compact hand-authored matte for the central dog."""
    im = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(im)
    outline = [
        (128, 72), (133, 73), (137, 77), (137, 82), (133, 88),
        (133, 98), (135, 105), (133, 113), (134, 126), (132, 131),
        (128, 131), (126, 115), (124, 115), (123, 128), (121, 131),
        (117, 129), (118, 119), (119, 108), (120, 98), (121, 89),
        (123, 84), (122, 78), (125, 74),
    ]
    draw.polygon(outline, fill=255)
    # A tiny feather retains the soft, low-resolution edge in the source.
    matte = np.asarray(im, dtype=np.float32) / 255.0
    matte = cv2.GaussianBlur(matte, (0, 0), 0.65)
    return np.clip(matte, 0.0, 1.0)


def clean_plate(source: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """Remove the dog so the background can move behind its silhouette."""
    hard = (matte > 0.025).astype(np.uint8) * 255
    hard = cv2.dilate(hard, np.ones((5, 5), np.uint8), iterations=1)
    # Telea interpolation is only used on this small road-colored region.
    return cv2.inpaint(source, hard, 5, cv2.INPAINT_TELEA)


def orbit_background(plate: np.ndarray, amount: float) -> np.ndarray:
    """Warp the scene about the dog's road contact using depth-like parallax."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    pivot_x, pivot_y = 126.0, 129.0
    horizon = 61.0

    # At the horizon the old street axis travels right as the camera travels
    # counterclockwise around the subject.  On the ground that displacement
    # falls to zero at the dog and reverses slightly in the near foreground.
    below = (pivot_y - yy) / (pivot_y - horizon)
    depth = np.where(yy <= horizon,
                     1.0 + 0.10 * (horizon - yy) / horizon,
                     below)
    depth = np.clip(depth, -0.22, 1.12)
    shift = 30.0 * amount * depth

    # Mild horizontal perspective contraction makes the motion an orbit/yaw,
    # rather than a simple sliding backdrop.  The sinusoidal term supplies a
    # subtle ground-plane turn while leaving vertical structures stable.
    scale = 1.0 + 0.025 * amount * np.clip(depth, 0.0, 1.0)
    bend = 1.1 * amount * np.sin((xx - pivot_x) * np.pi / W) * np.clip(below, 0.0, 1.0)
    map_x = pivot_x + (xx - shift - pivot_x) / scale
    map_y = yy - bend

    pad = 40
    extended = cv2.copyMakeBorder(plate, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    return cv2.remap(
        extended,
        map_x + pad,
        map_y + pad,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def orbit_dog(source: np.ndarray, matte: np.ndarray, amount: float):
    """Reproject the stationary dog around its planted feet.

    Moving to the dog's right exposes more of its side: its head and shoulders
    therefore project progressively to screen-right, while the paws remain at
    exactly the same scene point.
    """
    pivot_x, pivot_y = 126.0, 130.0
    side_shear = 0.155 * amount
    width_scale = 1.0 + 0.075 * amount
    matrix = np.array(
        [
            [
                width_scale,
                -side_shear,
                pivot_x - width_scale * pivot_x + side_shear * pivot_y,
            ],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    dog = cv2.warpAffine(
        source,
        matrix,
        (W, H),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    alpha = cv2.warpAffine(
        matte,
        matrix,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # The newly side-facing coat catches a fraction more skylight.  This is
    # intentionally modest to preserve all of the original animal texture.
    shade = np.ones((H, W, 1), dtype=np.float32)
    side_light = np.clip((np.arange(W, dtype=np.float32) - pivot_x) / 14.0, 0.0, 1.0)
    shade += (0.035 * amount * side_light)[None, :, None]
    dog = np.clip(dog.astype(np.float32) * shade, 0, 255)
    return dog, np.clip(alpha, 0.0, 1.0)


def make_frames() -> np.ndarray:
    source = np.asarray(Image.open(HERE / "first_frame.png").convert("RGB"), dtype=np.uint8)
    if source.shape != (H, W, 3):
        raise ValueError(f"Unexpected first-frame shape: {source.shape}")

    matte = dog_matte()
    plate = clean_plate(source, matte)
    frames = [source.copy()]

    for index in range(1, N):
        amount = smoothstep(index / (N - 1))
        background = orbit_background(plate, amount).astype(np.float32)
        dog, alpha = orbit_dog(source, matte, amount)
        frame = background * (1.0 - alpha[..., None]) + dog * alpha[..., None]
        frames.append(np.clip(np.rint(frame), 0, 255).astype(np.uint8))

    result = np.stack(frames)
    # The contract requires exact identity, not merely visual equivalence.
    result[0] = source
    return result


if __name__ == "__main__":
    generated = make_frames()
    artifact_io.save(generated, HERE)
    print(f"saved {generated.shape[0]} RGB frames: {generated.shape}, {generated.dtype}")


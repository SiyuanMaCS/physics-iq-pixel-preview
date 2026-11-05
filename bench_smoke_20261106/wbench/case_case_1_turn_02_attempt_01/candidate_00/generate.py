from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import artifact_io


WIDTH = 256
HEIGHT = 143
FRAME_COUNT = 20


def smoothstep(value: float) -> float:
    """A gentle camera move with no abrupt start or stop."""
    return value * value * (3.0 - 2.0 * value)


def make_subject_protection() -> tuple[np.ndarray, np.ndarray]:
    """Return hard and soft masks around the stationary dog.

    The soft mask brings optical flow smoothly to zero near the dog, while
    the hard mask restores the source pixels so the animal has no gait or
    body motion during the camera move.
    """
    hard_image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(hard_image)
    dog_outline = [
        (120, 73), (129, 73), (135, 78), (136, 86), (132, 93),
        (135, 102), (139, 105), (140, 113), (136, 117), (132, 115),
        (132, 129), (129, 133), (124, 133), (122, 116), (120, 130),
        (116, 131), (115, 126), (116, 112), (117, 101), (118, 94),
        (117, 87), (118, 79),
    ]
    draw.polygon(dog_outline, fill=255)
    hard = np.asarray(hard_image, dtype=np.float32) / 255.0

    # The pivot region includes the cast shadow. It belongs to the same patch
    # of road as the dog and therefore should not slide away during an orbit.
    pivot = np.zeros((HEIGHT, WIDTH), np.float32)
    cv2.ellipse(pivot, (128, 111), (18, 43), 0, 0, 360, 1.0, -1)
    cv2.ellipse(pivot, (143, 132), (31, 13), -8, 0, 360, 0.92, -1)
    pivot = np.maximum(pivot, hard)
    soft = cv2.GaussianBlur(pivot, (0, 0), sigmaX=8.0, sigmaY=8.0)
    soft = np.clip(soft, 0.0, 1.0)
    return hard, soft


def orbit_frame(source: np.ndarray, amount: float,
                hard_subject: np.ndarray,
                soft_protection: np.ndarray) -> np.ndarray:
    """Make one clockwise/rightward orbital view with a depth-aware warp."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)

    # A clockwise orbit moves the camera to the subject's right. Distant
    # scenery consequently drifts left, points at the target stay fixed, and
    # the closest road acquires a small counter-parallax to the right.
    depth_flow = np.interp(
        yy,
        [0.0, 54.0, 82.0, 111.0, 127.0, 142.0],
        [-12.0, -13.5, -9.0, -3.0, 0.0, 3.2],
    ).astype(np.float32)

    # A mild yaw/perspective term makes the two street walls separate around
    # the fixed target instead of translating like a flat photograph.
    yaw_flow = (xx - 127.0) * 0.025
    side_weight = np.clip(np.abs(xx - 127.0) / 126.0, 0.0, 1.0)
    facade_flow = side_weight * np.interp(
        yy, [0.0, 70.0, 142.0], [-1.2, 0.0, 2.0]
    ).astype(np.float32)
    visual_dx = (depth_flow + yaw_flow + facade_flow) * amount
    visual_dx *= (1.0 - soft_protection)

    # Tiny vertical projective lift away from the pivot reinforces an arc
    # while remaining far below the magnitude of the horizontal orbit.
    visual_dy = (
        0.0055 * (xx - 127.0) * np.clip((112.0 - yy) / 112.0, 0.0, 1.0)
    ) * amount
    visual_dy *= (1.0 - soft_protection)

    # remap asks where each output pixel comes from, hence subtract visual flow.
    map_x = xx - visual_dx
    map_y = yy - visual_dy
    warped = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Preserve the animal itself as an absolutely stationary photographic
    # subject. A sub-pixel feather avoids a cutout edge at this resolution.
    alpha = cv2.GaussianBlur(hard_subject, (0, 0), 0.55)[..., None]
    composed = warped.astype(np.float32) * (1.0 - alpha)
    composed += source.astype(np.float32) * alpha
    return np.clip(np.rint(composed), 0, 255).astype(np.uint8)


def save_contact_sheet(frames: np.ndarray, destination: Path) -> None:
    """Save an intermediate candidate showing the orbit at five moments."""
    selected = [0, 4, 9, 14, 19]
    sheet = np.concatenate([frames[index] for index in selected], axis=1)
    Image.fromarray(sheet, "RGB").save(destination)


def main() -> None:
    case_dir = Path(__file__).resolve().parent
    source = np.asarray(Image.open(case_dir / "first_frame.png").convert("RGB"))
    if source.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"unexpected first-frame shape: {source.shape}")

    hard_subject, soft_protection = make_subject_protection()
    generated = []
    for index in range(FRAME_COUNT):
        if index == 0:
            # Do not resample the contractual first frame.
            generated.append(source.copy())
            continue
        progress = index / (FRAME_COUNT - 1)
        generated.append(
            orbit_frame(source, smoothstep(progress), hard_subject, soft_protection)
        )

    frames = np.stack(generated, axis=0).astype(np.uint8)
    assert frames.shape == (FRAME_COUNT, HEIGHT, WIDTH, 3)
    assert np.array_equal(frames[0], source)

    save_contact_sheet(frames, case_dir / "candidate_orbit_contact_sheet.png")
    artifact_io.save(frames, case_dir)


if __name__ == "__main__":
    main()

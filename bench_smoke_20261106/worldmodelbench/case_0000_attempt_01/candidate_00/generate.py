"""Generate the requested bridge stop as a deterministic RGB pixel sequence.

The source is already a convincing dash-camera view, so the animation uses a
small perspective flow field rather than redrawing it.  Ground-plane features
move more than the skyline, the motion eases to zero, and the last second is a
stationary hold at the light.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from artifact_io import save


CASE_DIR = Path(__file__).resolve().parent
WIDTH, HEIGHT = 256, 144
FRAME_COUNT = 20


def smoothstep(edge0, edge1, value):
    value = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def travel_progress(frame_index):
    """Distance travelled while braking, followed by a complete stop."""
    # The car comes to rest at frame 15.  A cubic ease-out gives continuously
    # diminishing displacement and therefore legible deceleration.
    stop_frame = 15.0
    u = min(frame_index / stop_frame, 1.0)
    return 1.0 - (1.0 - u) ** 3


def inverse_camera_map(progress):
    """Construct an inverse map for modest forward movement on the bridge."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    vanishing_x, horizon_y = 129.0, 70.5

    # A slight global enlargement covers distant, approximately vertical
    # structures.  Its inverse map samples inward around the vanishing point.
    distant_scale = 1.0 + 0.012 * progress
    far_x = vanishing_x + (xx - vanishing_x) / distant_scale
    far_y = horizon_y + (yy - horizon_y) / distant_scale

    # Forward translation over a flat road creates increasingly strong image
    # flow below the horizon.  This rational form stays stable at the bottom
    # edge and is inverted here for cv2.remap.
    out_depth = np.maximum(yy - horizon_y, 0.0)
    road_rate = 0.00122 * progress
    source_depth = out_depth / (1.0 + road_rate * out_depth)
    depth_ratio = np.divide(
        source_depth,
        out_depth,
        out=np.ones_like(source_depth),
        where=out_depth > 1e-5,
    )
    road_x = vanishing_x + (xx - vanishing_x) * depth_ratio
    road_y = horizon_y + source_depth

    # Transition gradually so railings and low buildings do not tear at a
    # hard horizon mask.  Foreground road/bridge pixels receive full parallax.
    near_weight = smoothstep(58.0, 100.0, yy)
    map_x = far_x * (1.0 - near_weight) + road_x * near_weight
    map_y = far_y * (1.0 - near_weight) + road_y * near_weight

    # Braking gently compresses the front suspension.  The sub-pixel pitch is
    # damped back to neutral before the stopped hold begins.
    u = min(progress, 1.0)
    pitch = 0.55 * np.sin(np.pi * u) * (u ** 1.7)
    map_y += pitch
    return map_x, map_y


def render_frame(source, frame_index):
    if frame_index == 0:
        return source.copy()

    progress = travel_progress(frame_index)
    map_x, map_y = inverse_camera_map(progress)
    frame = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Keep interpolation ringing within displayable RGB bounds and retain the
    # source's natural low-resolution dash-camera texture.
    return np.clip(frame, 0, 255).astype(np.uint8)


def write_motion_preview(frames):
    """Save an intermediate strip that exposes approach and stopped states."""
    picks = (0, 4, 8, 12, 15, 19)
    scale = 2
    margin = 18
    canvas = Image.new(
        "RGB", (WIDTH * scale * 3, (HEIGHT * scale + margin) * 2), (22, 25, 30)
    )
    draw = ImageDraw.Draw(canvas)
    for cell, frame_index in enumerate(picks):
        x = (cell % 3) * WIDTH * scale
        y = (cell // 3) * (HEIGHT * scale + margin)
        draw.text((x + 4, y + 3), f"candidate 01 / frame {frame_index:02d}", fill=(235, 239, 243))
        image = Image.fromarray(frames[frame_index]).resize(
            (WIDTH * scale, HEIGHT * scale), Image.Resampling.NEAREST
        )
        canvas.paste(image, (x, y + margin))
    canvas.save(CASE_DIR / "candidate_01.png")


def main():
    source = np.asarray(Image.open(CASE_DIR / "first_frame.png").convert("RGB"))
    if source.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"unexpected source shape: {source.shape}")

    frames = np.stack(
        [render_frame(source, index) for index in range(FRAME_COUNT)], axis=0
    ).astype(np.uint8)
    assert np.array_equal(frames[0], source)
    # Stopped frames are intentionally identical: no residual camera creep.
    assert np.array_equal(frames[15], frames[19])

    write_motion_preview(frames)
    save(frames, CASE_DIR)


if __name__ == "__main__":
    main()

"""Render the future using the historical RGB plate and measured ball motion.

Only numpy and PIL are used in the generation path. Times are relative to the
last supplied historical frame; the first prediction is at source time 3.0 s.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import artifact_io


ROOT = Path(__file__).resolve().parent
HEIGHT, WIDTH = 144, 256
YY, XX = np.mgrid[:HEIGHT, :WIDTH]


def read_history(index):
    return np.asarray(Image.open(ROOT / "conditioning_frames" / f"{index:02d}.png"), dtype=np.float64)


def measure_motion():
    measurements = {"blue": [], "yellow": []}
    for index in range(82, 90):
        rgb = read_history(index)
        region = (XX > 10) & (XX < 234) & (YY > 28) & (YY < 90)
        blue = region & (rgb[..., 2] > 1.4 * rgb[..., 0]) & (rgb[..., 2] > 1.13 * rgb[..., 1]) & (rgb[..., 2] > 42)
        yellow = region & (rgb[..., 1] > 1.05 * rgb[..., 0]) & (rgb[..., 1] > 1.6 * rgb[..., 2]) & (rgb[..., 0] > 100)
        for name, mask in (("blue", blue), ("yellow", yellow)):
            y, x = np.where(mask)
            measurements[name].append([index / 30.0 - 89 / 30.0, x.mean(), y.mean()])
    result = {}
    for name, values in measurements.items():
        values = np.asarray(values)
        # Estimate motion from recent positions, anchoring to the last frame.
        velocity = np.array([np.polyfit(values[:, 0], values[:, c], 1)[0] for c in (1, 2)])
        result[name] = (values[-1, 1:], velocity)
    return result


def sample(array, x, y):
    """Bilinear pixel sampling with zero beyond the sprite boundary."""
    h, w = array.shape[:2]
    valid = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    x = np.clip(x, 0, w - 1.00001)
    y = np.clip(y, 0, h - 1.00001)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - ix, y - iy
    if array.ndim == 3:
        dx, dy, valid = dx[..., None], dy[..., None], valid[..., None]
    sampled = ((1 - dx) * (1 - dy) * array[iy, ix]
               + dx * (1 - dy) * array[iy, ix + 1]
               + (1 - dx) * dy * array[iy + 1, ix]
               + dx * dy * array[iy + 1, ix + 1])
    return sampled * valid


def make_sprite(source, clean, center, radius):
    cx, cy = center
    x0, y0 = int(cx) - 27, int(cy) - 20
    color = source[y0:y0 + 43, x0:x0 + 57].copy()
    plate = clean[y0:y0 + 43, x0:x0 + 57]
    sy, sx = np.mgrid[:43, :57]
    u, v = sx + x0 - cx, sy + y0 - cy
    distance = np.sqrt((u / radius) ** 2 + (v / (radius * .97)) ** 2)
    # A soft subpixel matte includes the photographed fuzzy ball edges.
    alpha = np.clip((1.08 - distance) / .16, 0, 1)
    ratio = np.mean(color / np.maximum(plate, 1), axis=2)
    darkness = np.clip(1 - ratio - .012, 0, .66)
    # Preserve both the contact shadow and the two broad directional shadows.
    support = np.exp(-((u / 24) ** 6 + (v / 15) ** 6))
    darkness *= support
    darkness *= 1 - np.clip((1.04 - distance) / .10, 0, 1)
    return dict(rgb=color, alpha=alpha, darkness=darkness,
                origin=(cx - x0, cy - y0), radius=radius)


def background_plate(last):
    clean = np.median(np.stack([read_history(i) for i in range(68, 78)]), axis=0)
    # Use the last frame everywhere except the ball and shadow footprints.
    mask = np.zeros((HEIGHT, WIDTH), dtype=float)
    for cx, cy in ((82, 51), (148, 57)):
        d = ((XX - cx) / 34) ** 6 + ((YY - cy) / 24) ** 6
        mask = np.maximum(mask, np.clip(2 - d, 0, 1))
    plate = last * (1 - mask[..., None]) + clean * mask[..., None]
    return plate, clean


def contact_time(motions):
    p0, v0 = motions["blue"]
    p1, v1 = motions["yellow"]
    dp, dv = p1 - p0, v1 - v0
    # Contact is at the measured sum of the ball radii, on the common path.
    radius_sum = 15.9
    coefficients = [dv @ dv, 2 * dp @ dv, dp @ dp - radius_sum ** 2]
    roots = np.roots(coefficients)
    return float(min(root.real for root in roots if abs(root.imag) < 1e-6 and root.real > 0))


def ball_position(name, time, motions, collision):
    position, velocity = motions[name]
    if time <= collision:
        return position + velocity * time
    position = position + velocity * collision
    elapsed = time - collision
    # Comparable masses rebound in opposite directions; rolling losses reduce
    # their speed. The paths use the perspective directions seen in history.
    rebound = -.80 * velocity
    speed = np.linalg.norm(rebound)
    deceleration = 26.0
    moving_time = min(elapsed, speed / deceleration)
    distance = speed * moving_time - .5 * deceleration * moving_time ** 2
    return position + rebound / speed * distance


def render_ball(canvas, sprite, center):
    cx, cy = center
    if cx < -26 or cx > WIDTH + 26 or cy < -24 or cy > HEIGHT + 24:
        return
    # Local apparent diameter changes smoothly with tabletop perspective.
    scale = np.clip((7.90 + .0105 * (cx - 110)) / sprite["radius"], .85, 1.18)
    ox, oy = sprite["origin"]
    sx, sy = (XX - cx) / scale + ox, (YY - cy) / scale + oy
    shade = sample(sprite["darkness"], sx, sy)
    canvas *= 1 - shade[..., None]
    alpha = sample(sprite["alpha"], sx, sy)
    foreground = sample(sprite["rgb"], sx, sy)
    canvas[:] = canvas * (1 - alpha[..., None]) + foreground * alpha[..., None]


def main():
    last = np.asarray(Image.open(ROOT / "first_frame.png"), dtype=float)
    plate, clean = background_plate(last)
    motions = measure_motion()
    collision = contact_time(motions)
    sprites = {
        "blue": make_sprite(last, clean, motions["blue"][0] + [.10, .20], 7.55),
        "yellow": make_sprite(last, clean, motions["yellow"][0] + [.25, .12], 8.32),
    }
    # The existing pipe mouths remain in the foreground as balls reach them.
    pipe_shape = Image.new("L", (WIDTH, HEIGHT), 0)
    pipe_draw = ImageDraw.Draw(pipe_shape)
    pipe_draw.ellipse((-6, 30, 12, 52), fill=255)
    pipe_draw.ellipse((231, 57, 246, 83), fill=255)
    pipe_draw.polygon([(240, 57), (255, 60), (255, 83), (239, 82)], fill=255)
    pipe = np.asarray(pipe_shape) > 0
    frames = []
    for index in range(20):
        time = 1 / 30.0 + index / 4.0
        canvas = plate.copy()
        for name in ("blue", "yellow"):
            center = ball_position(name, time, motions, collision)
            render_ball(canvas, sprites[name], center)
        canvas[pipe] = plate[pipe]
        frames.append(np.clip(np.rint(canvas), 0, 255).astype(np.uint8))
    frames = np.stack(frames)
    # Retain this complete candidate before producing final deliverables.
    candidate_dir = ROOT / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    np.save(candidate_dir / "candidate_02.npy", frames)
    preview = Image.new("RGB", (WIDTH * 4, (HEIGHT + 20) * 2), (24, 26, 30))
    draw = ImageDraw.Draw(preview)
    for index in range(8):
        x, y = index % 4 * WIDTH, index // 4 * (HEIGHT + 20)
        draw.text((x + 4, y + 3), f"future {index / 4:.2f} s", fill="white")
        preview.paste(Image.fromarray(frames[index]), (x, y + 20))
    preview.save(candidate_dir / "candidate_02.png")
    artifact_io.save(frames, ROOT)
    print("Generated", frames.shape, frames.dtype)
    print("Estimated collision source time:", 89 / 30 + collision)
    for name, (position, velocity) in motions.items():
        print(name, "last center", position, "incoming pixels/second", velocity)


if __name__ == "__main__":
    main()

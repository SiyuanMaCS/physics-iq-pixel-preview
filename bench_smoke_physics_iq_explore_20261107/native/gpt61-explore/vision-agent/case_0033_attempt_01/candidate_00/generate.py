"""A locked camera, photographed ball textures, and analytic rolling motion."""
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

import artifact_io


ROOT = Path(__file__).resolve().parent
SOURCE = np.asarray(Image.open(ROOT / "first_frame.png").convert("RGB"))
H, W = SOURCE.shape[:2]
Y, X = np.mgrid[:H, :W].astype(float)


def smoothstep(a, b, values):
    z = np.clip((values - a) / (b - a), 0, 1)
    return z * z * (3 - 2 * z)


def ellipse(cx, cy, rx, ry, soft=0.25):
    d = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)
    return 1 - smoothstep(1 - soft, 1, d)


def restore_tabletop():
    """Interpolate the exposed wood grain across the two small moving objects."""
    original = SOURCE.astype(float)
    cover_blue = np.maximum(ellipse(77, 50, 12, 12), ellipse(84, 48, 22, 10))
    cover_yellow = np.maximum(ellipse(149, 58, 12, 13), ellipse(140, 57, 13, 12))
    cover = np.maximum(cover_blue, cover_yellow)
    filled = original.copy()
    low = np.asarray(Image.fromarray(SOURCE).filter(ImageFilter.GaussianBlur(1.1))).astype(float)
    grain = original - low
    for y in range(H):
        for left, right in [(56, 109), (126, 166)]:
            span = np.flatnonzero(cover[y, left:right] > 0.001)
            if not len(span):
                continue
            lo, hi = left + int(span[0]), left + int(span[-1])
            # Average just beyond the object, retaining the board's horizontal bands.
            lv = original[y, lo - 4:lo - 1].mean(axis=0)
            rv = original[y, hi + 2:hi + 5].mean(axis=0)
            xx = np.arange(lo, hi + 1)
            t = ((xx - lo + 2) / (hi - lo + 4))[:, None]
            base = lv[None] * (1 - t) + rv[None] * t
            donor_x = xx + (108 if left == 56 else 45)
            donor_y = min(H - 1, y + (6 if left == 56 else 4))
            donor_x = np.clip(donor_x, 0, W - 1)
            base += grain[donor_y, donor_x] * 0.65
            filled[y, xx] = base
    return original * (1 - cover[..., None]) + filled * cover[..., None]


def sample(array, xx, yy):
    ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
    fx, fy = xx - ix, yy - iy
    ix0, iy0 = np.clip(ix, 0, W - 1), np.clip(iy, 0, H - 1)
    ix1, iy1 = np.clip(ix + 1, 0, W - 1), np.clip(iy + 1, 0, H - 1)
    if array.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return ((array[iy0, ix0] * (1 - fx) + array[iy0, ix1] * fx) * (1 - fy)
            + (array[iy1, ix0] * (1 - fx) + array[iy1, ix1] * fx) * fy)


def make_ball(background, cx, cy, radius, name):
    # Premultiplied photographic pixels preserve camera blur and felt texture.
    alpha = 1 - smoothstep(radius - 0.85, radius + 1.15,
                          np.sqrt((X - cx) ** 2 + (Y - cy) ** 2))
    delta = SOURCE.astype(float) - background
    window = (ellipse(77, 50, 13, 13) if name == "blue"
              else ellipse(149, 58, 13, 14))
    shadow_window = (ellipse(84, 48, 23, 11) if name == "blue"
                     else ellipse(140, 57, 14, 13))
    window = np.maximum(window, shadow_window)
    delta *= window[..., None]
    return {"cx": cx, "cy": cy, "r": radius, "alpha": alpha,
            "delta": delta, "background": background, "name": name}


def place_ball(canvas, ball, cx, cy, distance, speed):
    scale = 1 + (cy - ball["cy"]) * 0.012
    # Subframe exposure gives a small amount of camera motion blur during rolling.
    exposure = 0.014
    deposits = []
    for shutter in [-0.5, 0.0, 0.5]:
        mx = cx + shutter * speed * exposure
        my = cy + shutter * speed * exposure * 0.113
        sx = (X - mx) / scale + ball["cx"]
        sy = (Y - my) / scale + ball["cy"]
        delta = sample(ball["delta"], sx, sy)
        alpha = sample(ball["alpha"], sx, sy)
        old_ground = sample(ball["background"], sx, sy)
        # Correct the opaque ball for the brightness of its new ground location.
        change = delta - alpha[..., None] * (canvas - old_ground)
        deposits.append(change)
    return canvas + np.mean(deposits, axis=0)


def motion(t):
    blue0, yellow0 = 76.5, 149.1
    axis = np.array([1.0, 0.113])
    axis /= np.linalg.norm(axis)
    speed = 28.5
    separation = (yellow0 - blue0) / axis[0]
    contact = 15.5
    hit = (separation - contact) / (2 * speed)
    if t <= hit:
        travel = speed * t
        velocity = speed
    else:
        elapsed = t - hit
        rebound = speed * 0.87
        friction = 1.55
        elapsed = min(elapsed, rebound / friction)
        retreat = rebound * elapsed - 0.5 * friction * elapsed ** 2
        travel = speed * hit - retreat
        velocity = -(rebound - friction * elapsed)
    blue = np.array([blue0, 50.0]) + axis * travel
    yellow = np.array([yellow0, 58.2]) - axis * travel
    return blue, yellow, velocity * axis[0]


def generate():
    background = restore_tabletop()
    Image.fromarray(np.clip(background, 0, 255).astype(np.uint8)).save(ROOT / ".tmp" / "background.png")
    blue = make_ball(background, 76.5, 50.0, 7.05, "blue")
    yellow = make_ball(background, 149.1, 58.2, 7.75, "yellow")
    frames = [SOURCE.copy()]
    for index in range(1, 20):
        bp, yp, speed = motion(index / 4)
        canvas = background.copy()
        canvas = place_ball(canvas, blue, *bp, bp[0] - blue["cx"], speed)
        canvas = place_ball(canvas, yellow, *yp, yp[0] - yellow["cx"], -speed)
        frames.append(np.clip(np.rint(canvas), 0, 255).astype(np.uint8))
    frames = np.stack(frames)
    candidate = ROOT / "candidates" / "v1"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "prompt.txt").write_text((ROOT / "prompt.txt").read_text())
    (candidate / "task_spec.json").write_text((ROOT / "task_spec.json").read_text())
    artifact_io.save(frames, candidate)
    artifact_io.save(frames, ROOT)
    assert frames.shape == (20, 144, 256, 3)
    assert frames.dtype == np.uint8
    assert np.array_equal(frames[0], SOURCE)
    print("Saved 20 RGB frames, 256x144, 4 fps. Frame 00 matches the input exactly.")


if __name__ == "__main__":
    generate()

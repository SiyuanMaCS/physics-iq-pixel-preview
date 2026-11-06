"""Deterministic RGB animation, using the supplied photograph and pixel geometry."""
from pathlib import Path
import math
import shutil
import subprocess

import numpy as np
from PIL import Image

import artifact_io


ROOT = Path(__file__).resolve().parent
SOURCE = np.asarray(Image.open(ROOT / "first_frame.png").convert("RGB"))
H, W = SOURCE.shape[:2]
Y, X = np.mgrid[:H, :W].astype(np.float64)
PIVOT = np.array([112.2, 43.5])
ANGLE = math.atan2(1.0, 1.59)
U0 = np.array([math.cos(ANGLE), math.sin(ANGLE)])
N0 = np.array([-U0[1], U0[0]])


def sample(array, x, y):
    """Bilinear RGB sampling, at coordinates measured in source pixels."""
    x = np.clip(x, 0, W - 1.000001)
    y = np.clip(y, 0, H - 1.000001)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    ax, ay = (x - ix)[..., None], (y - iy)[..., None]
    return ((1 - ay) * ((1 - ax) * array[iy, ix] + ax * array[iy, ix + 1])
            + ay * ((1 - ax) * array[iy + 1, ix] + ax * array[iy + 1, ix + 1]))


def clean_plate():
    """Fill only the narrow footprint of the original stick and burning end."""
    dx, dy = X - PIVOT[0], Y - PIVOT[1]
    along = dx * U0[0] + dy * U0[1]
    normal = dx * N0[0] + dy * N0[1]
    body = (np.abs(normal) < 2.85) & (along < 31.8)
    end = (np.abs(normal + 0.30) < 3.35) & (along > 29) & (along < 46.4)
    flame = ((X - 143.1) / 4.1) ** 2 + ((Y - 59.2) / 3.6) ** 2 < 1
    mask = body | end | flame
    a = SOURCE.astype(float)
    plate = a.copy()
    # An initial horizontal estimate retains the wall/table boundary exactly.
    for row in range(H):
        xx = np.flatnonzero(mask[row])
        if not len(xx):
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(xx) > 1) + 1]
        stops = np.r_[starts[1:], len(xx)]
        for start, stop in zip(starts, stops):
            left, right = int(xx[start]), int(xx[stop - 1])
            l = max(0, left - 1)
            r = min(W - 1, right + 1)
            f = (np.arange(left, right + 1) - l) / max(1, r - l)
            plate[row, left:right + 1] = (a[row, l] * (1 - f[:, None])
                                         + a[row, r] * f[:, None])
    # Locally harmonic RGB interpolation continues the glass and water edges.
    for _ in range(480):
        pad = np.pad(plate, ((1, 1), (1, 1), (0, 0)), mode="edge")
        smooth = (pad[1:-1, :-2] + pad[1:-1, 2:]
                  + pad[:-2, 1:-1] + pad[2:, 1:-1]) * 0.25
        plate[mask] = smooth[mask]
    # Preserve the exact photographed horizontal background edge above the glass.
    for row in range(35):
        xx = np.flatnonzero(mask[row])
        if len(xx):
            l, r = max(0, xx[0] - 1), min(W - 1, xx[-1] + 1)
            f = (xx - l) / max(1, r - l)
            plate[row, xx] = a[row, l] * (1 - f[:, None]) + a[row, r] * f[:, None]
    return plate, mask


PLATE, REMOVED = clean_plate()


def body_alpha(q):
    # This width comes from the numerical cross section of the original stick.
    return np.clip((2.30 - np.abs(q)) / 0.91, 0, 1)


def material(s, q):
    """Source RGB texture with its former background removed algebraically."""
    sx = PIVOT[0] + s * U0[0] + q * N0[0]
    sy = PIVOT[1] + s * U0[1] + q * N0[1]
    # The unseen upper length uses additional samples of the same wooden shaft.
    outside = sy < 2
    wrapped_y = 5 + np.mod(sy, 23)
    sx = np.where(outside, 43.0 + 1.59 * wrapped_y - q / U0[1], sx)
    sy = np.where(outside, wrapped_y, sy)
    alpha = np.maximum(body_alpha(q), 0.16)[..., None]
    raw = sample(SOURCE.astype(float), sx, sy)
    bg = sample(PLATE, sx, sy)
    fg = np.clip((raw - (1 - alpha) * bg) / alpha, 0, 255)
    # Use unobstructed wood for the last few millimeters; the water reflection
    # remains attached to the glass instead of being carried by the match.
    upper_y = 16 + np.mod(s * 0.54, 15)
    upper_x = 43 + 1.59 * upper_y - q / U0[1]
    wood = sample(SOURCE.astype(float), upper_x, upper_y)
    wood_bg = sample(PLATE, upper_x, upper_y)
    wood = np.clip((wood - (1 - alpha) * wood_bg) / alpha, 0, 255)
    mix = np.clip((s - 23) / 5.5, 0, 1)[..., None]
    return fg * (1 - mix) + wood * mix


def composite(rgb, color, opacity):
    opacity = np.clip(opacity, 0, 1)[..., None]
    return rgb * (1 - opacity) + color * opacity


def flame_layer(rgb, head, time, strength):
    """A small upright warm flame; its geometry is independent of stick tilt."""
    if strength <= 0:
        return rgb
    hx, hy = head
    base_x, base_y = hx - 3.5, hy - 4.7
    flicker = 0.3 * math.sin(time * 25) + 0.18 * math.sin(time * 39)
    height = (5.7 + 0.6 * math.sin(time * 31)) * (0.6 + 0.4 * strength)
    v = (base_y - Y) / height
    center = base_x - 0.55 * v + flicker * v * v
    width = 1.65 * np.sqrt(np.clip(1 - v, 0, 1)) * (0.55 + 0.6 * np.sin(np.clip(v, 0, 1) * math.pi))
    alpha = np.clip((width - np.abs(X - center) + 0.45) / 0.8, 0, 1)
    alpha *= np.clip((v + 0.15) * 6, 0, 1) * np.clip((1.07 - v) * 8, 0, 1) * strength
    core = np.exp(-((X - center) / 0.7) ** 2) * np.clip(1 - v, 0, 1)
    colors = np.zeros_like(rgb)
    colors[..., 0] = 237 + core * 18
    colors[..., 1] = 147 + core * 74
    colors[..., 2] = 64 + core * 103
    glow = np.exp(-((X - base_x) / 3.6) ** 2 - ((Y - base_y + 2) / 4.6) ** 2)
    rgb = np.clip(rgb + glow[..., None] * np.array([9, 3, 0]) * strength, 0, 255)
    return composite(rgb, colors, alpha * 0.85)


def water_response(rgb, time):
    """Weak elliptical rings and a short, diffuse wisp after extinction."""
    since = time - 0.60
    if since <= 0:
        return rgb
    cx, cy = 142.0, 65.7
    water_window = np.clip(1 - ((X - 135.5) / 27.2) ** 8, 0, 1)
    water_window *= np.clip(1 - ((Y - 69) / 9.5) ** 8, 0, 1)
    for delay in (0, 0.26):
        age = since - delay
        if 0 < age < 2.4:
            radius = 2.0 + 6.6 * age
            distance = np.sqrt((X - cx) ** 2 + ((Y - cy) / 0.22) ** 2)
            wave = np.exp(-((distance - radius) / 0.8) ** 2)
            trough = np.exp(-((distance - radius - 1.0) / 0.9) ** 2)
            amplitude = 6.0 * math.exp(-age * 1.4)
            rgb += (wave - trough)[..., None] * amplitude * water_window[..., None]
    if since < 1.55:
        age = since
        lift = 9.0 * age
        top = cy - 3 - lift
        length = 5 + 6 * age
        v = np.clip((cy - 2 - Y) / length, 0, 1)
        bend = cx - 1.6 - 2.5 * age + 0.65 * np.sin(v * 6.8 + age * 3.0)
        density = np.exp(-((X - bend) / (0.8 + 0.8 * v)) ** 2)
        density *= np.exp(-((Y - top) / (2.8 + 2.2 * age)) ** 2)
        density *= (Y < cy - 2) * 0.095 * math.sin(math.pi * min(age / 1.55, 1))
        rgb = composite(rgb, np.array([175, 171, 165]), density)
    return np.clip(rgb, 0, 255)


def make_frame(index):
    if index == 0:
        return SOURCE.copy()
    time = index / 4
    progress = 0.5 - 0.5 * math.cos(math.pi * min(time / 3.1, 1.0))
    theta = ANGLE + math.radians(22.0) * progress
    u = np.array([math.cos(theta), math.sin(theta)])
    n = np.array([-u[1], u[0]])
    insertion = 16.0 * progress
    anchor = PIVOT + insertion * u
    dx, dy = X - anchor[0], Y - anchor[1]
    s = dx * u[0] + dy * u[1]
    q = dx * n[0] + dy * n[1]
    rgb = PLATE.copy()
    wood_alpha = body_alpha(q) * np.clip((34.3 - s) / 2.4, 0, 1)
    wood = material(s, q)
    # The narrow glass-wall highlights lie in front of the stick. This also
    # makes the submersed portion slightly softer and less contrasty.
    glass = np.clip((Y - 43) / 20, 0, 1)
    wood = wood * (1 - 0.06 * glass[..., None]) + PLATE * (0.06 * glass[..., None])
    under = np.clip((Y - 65.5) / 2.5, 0, 1)
    wood = wood * (1 - 0.15 * under[..., None]) + PLATE * (0.15 * under[..., None])
    rgb = composite(rgb, wood, wood_alpha)
    # The end is already charred in the input. Copy its local texture, avoiding
    # the adjacent glare spot on the glass.
    head_end = 44.6
    head_start = 31.0
    cap_s = np.clip(s, head_start + 1.5, head_end - 1.7)
    cap_dist = np.sqrt((s - cap_s) ** 2 + (q + 0.40) ** 2)
    head_alpha = np.clip((2.35 - cap_dist) / 0.85, 0, 1)
    sx = PIVOT[0] + s * U0[0] + q * N0[0]
    sy = PIVOT[1] + s * U0[1] + q * N0[1]
    raw_head = sample(SOURCE.astype(float), sx, sy)
    old_bg = sample(PLATE, sx, sy)
    a = np.maximum(head_alpha, 0.2)[..., None]
    char = np.clip((raw_head - (1 - a) * old_bg) / a, 0, 255)
    # Limit any surface reflection accidentally included at the edge of the
    # original head while retaining its brown upper end and black lower end.
    dark_factor = np.clip((s - 31.5) / 7.0, 0, 1)[..., None]
    char = np.minimum(char, np.array([155, 114, 74]) * (1 - dark_factor)
                      + np.array([61, 54, 51]) * dark_factor)
    char = char * (1 - under[..., None] * 0.15) + PLATE * (under[..., None] * 0.15)
    rgb = composite(rgb, char, head_alpha)
    tip = anchor + u * 44.0 - n * 0.4
    flame_strength = max(0.0, 1 - (time / 0.8) ** 1.4)
    rgb = flame_layer(rgb, tip, time, flame_strength)
    # Keep strong photographed glints anchored to the glass, including the rim.
    original_brightness = SOURCE.astype(float).mean(axis=-1)
    front = np.clip((original_brightness - 200) / 55, 0, 1)
    front *= ((X > 104) & (X < 164) & (Y > 42) & (Y < 130))
    front *= (~REMOVED)
    rgb = composite(rgb, SOURCE.astype(float), front * 0.73)
    rgb = water_response(rgb, time)
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def main():
    frames = np.stack([make_frame(i) for i in range(20)])
    assert frames.shape == (20, 144, 256, 3)
    assert frames.dtype == np.uint8
    assert np.array_equal(frames[0], SOURCE)
    candidates = ROOT / "candidates"
    candidates.mkdir(exist_ok=True)
    np.savez_compressed(candidates / "candidate_01.npz", frames=frames)
    shutil.copyfile(__file__, candidates / "generate_01.py")
    artifact_io.save(frames, ROOT)
    if shutil.which("ffmpeg"):
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", "4",
            "-i", str(ROOT / "frames" / "%02d.png"),
            "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(ROOT / "video.mp4"),
        ], check=True)
    print("Saved 20 RGB frames at 256x144, 4 fps, 5 seconds.")
    print("Frame 00 is pixel-identical to first_frame.png.")


if __name__ == "__main__":
    main()

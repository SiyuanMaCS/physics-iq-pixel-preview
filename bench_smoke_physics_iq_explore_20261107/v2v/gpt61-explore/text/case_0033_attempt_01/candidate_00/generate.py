"""Pixel-based continuation of the supplied rolling-ball demonstration."""
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image

import artifact_io


ROOT = Path(__file__).resolve().parent
W, H = 256, 144
history = np.stack([
    np.asarray(Image.open(p).convert("RGB"))
    for p in sorted((ROOT / "conditioning_frames").glob("*.png"))
])
last = np.asarray(Image.open(ROOT / "first_frame.png").convert("RGB"))
# This historical exposure precedes the release and contains the unobstructed
# table underneath both balls. Keep its photographic grain and all apparatus.
background = history[-2].astype(np.float32)

BALLS = [
    {"name": "blue", "center": (36.995, 46.83), "axes": (8.45, 7.85), "sign": 1},
    {"name": "yellow", "center": (201.02, 65.60), "axes": (10.20, 8.85), "sign": -1},
]

# A one-dimensional perspective coordinate along the table: equal distances
# on the tabletop have larger pixel displacements near the right-hand pipe.
radius_slope = (BALLS[1]["axes"][0] - BALLS[0]["axes"][0]) / (201.02 - 36.995)
radius_intercept = BALLS[0]["axes"][0] - radius_slope * 36.995
radius_reference = radius_intercept + radius_slope * 119.0
path_slope = (65.60 - 46.83) / (201.02 - 36.995)
path_intercept = 46.83 - path_slope * 36.995


def radius(x):
    return radius_intercept + radius_slope * x


def to_ground(x):
    return -radius_reference ** 2 / (radius_slope * radius(x))


def to_screen(s):
    return (-radius_reference ** 2 / (radius_slope * s) - radius_intercept) / radius_slope


initial_ground = np.array([to_ground(b["center"][0]) for b in BALLS])
initial_speed = 106.0
rolling_drag = 23.5
restitution = 0.67


def precontact(t):
    distance = initial_speed * t - 0.5 * rolling_drag * t * t
    return initial_ground + np.array([distance, -distance])


def center_from_ground(s):
    x = to_screen(s)
    return np.array([x, path_intercept + path_slope * x])


def contact_gap(t):
    p = [center_from_ground(s) for s in precontact(t)]
    return np.linalg.norm(p[1] - p[0]) - 0.98 * (radius(p[0][0]) + radius(p[1][0]))


lo, hi = 0.0, 1.5
for _ in range(60):
    middle = (lo + hi) / 2
    if contact_gap(middle) > 0:
        lo = middle
    else:
        hi = middle
contact_time = (lo + hi) / 2
contact_ground = precontact(contact_time)
rebound_speed = restitution * (initial_speed - rolling_drag * contact_time)
stop_after_contact = rebound_speed / rolling_drag


def positions(t):
    if t <= contact_time:
        ground = precontact(t)
    else:
        u = min(t - contact_time, stop_after_contact)
        d = rebound_speed * u - 0.5 * rolling_drag * u * u
        ground = contact_ground + np.array([-d, d])
    return [center_from_ground(s) for s in ground], ground


# Recover a small photographed sphere texture, including its original shading,
# from the last historical image. Split off fine detail so its rotation does
# not rotate the illumination with the ball.
texture_size = 81
uv = np.linspace(-1.22, 1.22, texture_size, dtype=np.float32)
uu, vv = np.meshgrid(uv, uv)
rr = np.sqrt(uu ** 2 + vv ** 2)
for ball in BALLS:
    cx, cy = ball["center"]
    rx, ry = ball["axes"]
    mx = (cx + uu * rx).astype(np.float32)
    my = (cy + vv * ry).astype(np.float32)
    photographed = cv2.remap(last.astype(np.float32), mx, my, cv2.INTER_LINEAR)
    beneath = cv2.remap(background, mx, my, cv2.INTER_LINEAR)
    alpha = np.clip((1.08 - rr) / 0.25, 0, 1)[..., None]
    recovered = (photographed - beneath * (1 - alpha)) / np.maximum(alpha, 0.22)
    # Fill the unobserved fringe with the nearest interior color rather than
    # introducing bits of the original tabletop into the translated sphere.
    shrink = np.minimum(1, 0.80 / np.maximum(rr, 1e-5))
    fallback = cv2.remap(
        last.astype(np.float32),
        (cx + uu * shrink * rx).astype(np.float32),
        (cy + vv * shrink * ry).astype(np.float32), cv2.INTER_LINEAR,
    )
    blend = np.clip((alpha - 0.24) / 0.45, 0, 1)
    texture = np.clip(blend * recovered + (1 - blend) * fallback, 0, 255)
    shade = cv2.GaussianBlur(texture, (0, 0), 2.15)
    detail = texture - shade
    ball["shade"] = shade
    ball["detail"] = detail
    ball["aspect"] = ry / rx


SCALE = 3
Y, X = np.mgrid[0:H * SCALE, 0:W * SCALE].astype(np.float32)
X = (X + 0.5) / SCALE - 0.5
Y = (Y + 0.5) / SCALE - 0.5
base_high = cv2.resize(background, (W * SCALE, H * SCALE), interpolation=cv2.INTER_LINEAR)


def texture_sample(texture, x, y):
    return cv2.remap(
        texture,
        ((x + 1.22) * (texture_size - 1) / 2.44).astype(np.float32),
        ((y + 1.22) * (texture_size - 1) / 2.44).astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def paint(t):
    centers, ground = positions(t)
    canvas = base_high.copy()
    for i, (ball, p) in enumerate(zip(BALLS, centers)):
        cx, cy = p
        rx = radius(cx)
        ry = rx * ball["aspect"]
        # The original two balls have soft shadows directed toward the center
        # of the table. Preserve that photographic lighting arrangement.
        inward = ball["sign"]
        sx, sy = cx + inward * rx * 0.93, cy + 0.08 * ry
        cast = np.exp(-2.2 * (((X - sx) / (1.35 * rx)) ** 2 + ((Y - sy) / (0.64 * ry)) ** 2))
        contact = np.exp(-2.0 * (((X - cx) / (0.84 * rx)) ** 2 + ((Y - (cy + 0.75 * ry)) / (0.30 * ry)) ** 2))
        shadow = np.clip(0.27 * cast + 0.17 * contact, 0, 0.40)
        canvas *= (1 - shadow[..., None])

    for i in np.argsort([p[1] for p in centers]):
        ball, (cx, cy) = BALLS[i], centers[i]
        rx, ry = radius(cx), radius(cx) * ball["aspect"]
        ux, uy = (X - cx) / rx, (Y - cy) / ry
        radial = np.sqrt(ux * ux + uy * uy)
        alpha = np.clip((1.035 - radial) / 0.085, 0, 1)
        z = np.sqrt(np.maximum(0, 1 - ux * ux - uy * uy))
        angle = (ground[i] - initial_ground[i]) / radius_reference
        # Rodrigues rotation of the surface around the rolling axis.
        direction = np.array([1.0, path_slope / ball["aspect"]])
        direction /= np.linalg.norm(direction)
        ax, ay = -direction[1], direction[0]
        dot = ax * ux + ay * uy
        co, si = np.cos(angle), np.sin(angle)
        tx = ux * co - ay * z * si + ax * dot * (1 - co)
        ty = uy * co + ax * z * si + ay * dot * (1 - co)
        shade = texture_sample(ball["shade"], ux, uy)
        detail = texture_sample(ball["detail"], tx, ty)
        color = np.clip(shade + 0.62 * detail, 0, 255)
        canvas = canvas * (1 - alpha[..., None]) + color * alpha[..., None]

    small = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    # Preserve exactly the original RGB values wherever there is no ball or
    # measurable shadow, avoiding a full-frame resampling of the apparatus.
    active = np.zeros((H, W), dtype=bool)
    y, x = np.mgrid[0:H, 0:W]
    for ball, (cx, cy) in zip(BALLS, centers):
        r = radius(cx)
        active |= ((x - cx) / (3.4 * r)) ** 2 + ((y - cy) / (2.0 * r)) ** 2 < 1
    small[~active] = background[~active]
    return np.clip(np.rint(small), 0, 255).astype(np.uint8)


def main():
    # Last historical frame is at 2.75 s; future frame 00 is at 3.00 s.
    times = 0.25 + np.arange(20) / 4
    frames = np.stack([paint(float(t)) for t in times])
    np.savez_compressed(ROOT / "candidate_01.npz", frames=frames)
    sheet = np.concatenate([np.concatenate(list(frames[j:j + 4]), axis=1) for j in range(0, 20, 4)], axis=0)
    Image.fromarray(sheet).save(ROOT / "candidate_01.png")
    record = {
        "contact_seconds_after_last_history": contact_time,
        "rest_seconds_after_last_history": contact_time + stop_after_contact,
        "positions": [
            {"frame": int(i), "time_after_history": float(t),
             "blue": positions(float(t))[0][0].tolist(),
             "yellow": positions(float(t))[0][1].tolist()}
            for i, t in enumerate(times)
        ],
        "shape": list(frames.shape), "dtype": str(frames.dtype),
    }
    (ROOT / "generation_check.json").write_text(json.dumps(record, indent=2))
    artifact_io.save(frames, ROOT)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

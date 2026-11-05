from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import artifact_io


W, H = 256, 143
CX, HORIZON = 127.5, 61.0
FOCAL = 205.0


def bilinear(image, x, y):
    """Sample an RGB image at floating point pixel coordinates."""
    h, w = image.shape[:2]
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    a = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    b = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return a * (1.0 - fy) + b * fy


def scalar_bilinear(field, x, y):
    h, w = field.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    return ((field[y0, x0] * (1.0 - fx) + field[y0, x1] * fx) * (1.0 - fy)
            + (field[y1, x0] * (1.0 - fx) + field[y1, x1] * fx) * fy)


def make_subject_mask():
    """A hand-traced soft matte for the dog and its ground shadow."""
    dog = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(dog)
    outline = [
        (124, 72), (130, 73), (134, 77), (135, 81), (131, 85),
        (133, 91), (131, 98), (134, 104), (132, 112), (135, 119),
        (134, 128), (131, 133), (127, 133), (126, 118), (124, 115),
        (123, 124), (122, 134), (118, 133), (116, 127), (117, 118),
        (119, 109), (117, 103), (118, 95), (119, 88), (119, 84),
        (117, 81), (120, 78), (120, 75),
    ]
    draw.polygon(outline, fill=255)

    # The original contact shadow belongs with the stationary dog.  Keeping it
    # on this layer prevents it from sliding with the road during the orbit.
    shadow = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shadow)
    sd.ellipse((123, 124, 153, 143), fill=170)
    shadow = shadow.filter(ImageFilter.GaussianBlur(3.0))
    dog = dog.filter(ImageFilter.GaussianBlur(0.65))
    return np.maximum(np.asarray(dog), np.asarray(shadow)).astype(np.float32) / 255.0


def make_clean_background(source, cut):
    """Fill the small subject footprint from nearby road samples."""
    bg = source.astype(np.float32).copy()
    # Horizontal road interpolation is stable here because all affected rows
    # are below the horizon and contain nearly horizontal asphalt texture.
    for yy in range(69, H):
        xs = np.flatnonzero(cut[yy] > 0.025)
        if not len(xs):
            continue
        lo = max(0, int(xs.min()) - 3)
        hi = min(W - 1, int(xs.max()) + 3)
        left = bg[yy, max(0, lo - 5):lo]
        right = bg[yy, hi + 1:min(W, hi + 6)]
        lv = left.mean(axis=0) if len(left) else bg[yy, lo]
        rv = right.mean(axis=0) if len(right) else bg[yy, hi]
        ramp = np.linspace(0.0, 1.0, hi - lo + 1)[:, None]
        fill = lv[None, :] * (1.0 - ramp) + rv[None, :] * ramp
        # Preserve low-amplitude asphalt grain so the repair is not a flat bar.
        grain = ((np.arange(hi - lo + 1) * 37 + yy * 17) % 13 - 6)[:, None] * 0.34
        bg[yy, lo:hi + 1] = np.clip(fill + grain, 0, 255)
    # Blend only where the removed layer was visible.
    blur = Image.fromarray(np.uint8(np.clip(cut * 255, 0, 255))).filter(
        ImageFilter.GaussianBlur(2.2))
    a = np.asarray(blur).astype(np.float32)[..., None] / 255.0
    return source.astype(np.float32) * (1.0 - a) + bg * a


def make_depth():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # The road is a ground plane whose depth tends to infinity at the horizon.
    road_z = 8.0 * (130.0 - HORIZON) / np.maximum(yy - HORIZON, 1.0)
    road_z = np.clip(road_z, 7.6, 92.0)

    # Facades at the edges are close, while the central city canyon is distant.
    side_z = 8.5 + 78.0 * np.power(1.0 - np.minimum(np.abs(xx - CX) / 132.0, 1.0), 1.55)
    vertical = side_z + np.maximum(HORIZON - yy, 0.0) * 0.10
    depth = np.where(yy > HORIZON, road_z, vertical)

    # Sky and the bright opening down the street form the farthest layer.
    sky = (yy < 52) & (xx > 74) & (xx < 166)
    distance_opening = (np.abs(xx - CX) < (18.0 + np.maximum(68.0 - yy, 0.0) * 0.45))
    depth[sky] = 115.0
    depth[distance_opening & (yy < 84)] = np.maximum(depth[distance_opening & (yy < 84)], 92.0)

    # Light separable smoothing prevents depth stair-steps without erasing
    # broad parallax (and works on float depth in every Pillow version).
    kernel = np.array([1, 4, 7, 10, 7, 4, 1], dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(depth, ((0, 0), (3, 3)), mode="edge")
    smooth = sum(kernel[k] * padded[:, k:k + W] for k in range(7))
    padded = np.pad(smooth, ((3, 3), (0, 0)), mode="edge")
    smooth = sum(kernel[k] * padded[k:k + H, :] for k in range(7))
    return smooth.astype(np.float32)


def project(source, depth, angle):
    """Inverse-map a yawing camera that orbits a target eight units away."""
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx, sy = gx.copy(), gy.copy()
    s, c = np.sin(angle), np.cos(angle)
    radius = 8.0
    cam_x = radius * s
    cam_z = radius * (1.0 - c)

    # Fixed-point inversion of the depth-dependent forward projection.
    for _ in range(8):
        z = scalar_bilinear(depth, sx, sy)
        wx = (sx - CX) * z / FOCAL
        wy = (sy - HORIZON) * z / FOCAL
        rx = c * (wx - cam_x) + s * (z - cam_z)
        rz = -s * (wx - cam_x) + c * (z - cam_z)
        rz = np.maximum(rz, 1.0)
        px = CX + FOCAL * rx / rz
        py = HORIZON + FOCAL * wy / rz
        sx += np.clip(gx - px, -6.0, 6.0) * 0.72
        sy += np.clip(gy - py, -4.0, 4.0) * 0.72
    return bilinear(source, sx, sy)


def warp_subject(source, mask, amount):
    """Give the fixed dog a subtle viewpoint change without moving its feet."""
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    anchor_x, anchor_y = 126.5, 132.0
    # Head moves a few pixels relative to planted paws as the view swings from
    # directly behind toward the dog's rear quarter.
    vertical = np.clip((anchor_y - gy) / 60.0, 0.0, 1.0)
    out_x = anchor_x + (gx - anchor_x) * (1.0 + 0.055 * amount) + 4.2 * amount * vertical
    out_y = gy - 0.5 * amount * vertical
    # Invert the simple displacement to sample the original layer.
    src_x = anchor_x + (gx - anchor_x - 4.2 * amount * vertical) / (1.0 + 0.055 * amount)
    src_y = gy + 0.5 * amount * vertical
    rgb = bilinear(source.astype(np.float32), src_x, src_y)
    alpha = scalar_bilinear(mask, src_x, src_y)[..., None]
    # Discard samples whose forward location is well outside the canvas.
    valid = ((out_x >= -1) & (out_x <= W) & (out_y >= -1) & (out_y <= H))[..., None]
    return rgb, alpha * valid


def generate():
    root = Path(__file__).resolve().parent
    source = np.asarray(Image.open(root / "first_frame.png").convert("RGB"), dtype=np.uint8)
    subject = make_subject_mask()
    background = make_clean_background(source, subject)
    depth = make_depth()

    frames = []
    for i in range(20):
        if i == 0:
            frames.append(source.copy())
            continue
        u = i / 19.0
        # Smooth start and finish reads as a deliberate camera move rather than
        # an abrupt image effect. Total orbit is about 32 degrees.
        eased = u * u * (3.0 - 2.0 * u)
        angle = np.deg2rad(32.0) * eased
        frame = project(background, depth, angle)
        dog_rgb, dog_a = warp_subject(source, subject, eased)
        frame = frame * (1.0 - dog_a) + dog_rgb * dog_a
        frames.append(np.uint8(np.clip(frame + 0.5, 0, 255)))

    result = np.stack(frames).astype(np.uint8)
    assert result.shape == (20, H, W, 3)
    assert np.array_equal(result[0], source)
    artifact_io.save(result, root)


if __name__ == "__main__":
    generate()

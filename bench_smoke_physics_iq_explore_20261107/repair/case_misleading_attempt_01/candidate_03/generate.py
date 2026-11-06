"""Continue the supplied still using RGB operations and analytic cuboid geometry.

The rightmost block is purple in the source image. Face textures come only
from that image. Blocks tilt about their fixed lower left edges, contact
limits keep adjacent blocks from crossing, and the thin wooden arm turns
clockwise. No learned renderer, external references, or physics engine is used.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
import cv2
import imageio_ffmpeg
import artifact_io

ROOT = Path(__file__).resolve().parent
SS = 4
SOURCE = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
H, W = SOURCE.shape[:2]
ORDER = ['Y', 'R', 'K', 'G', 'B', 'P']
# Each block: front lower left point, front width, depth, upright height.
# The floor is seen obliquely, so its two projected axes are not orthogonal.
GEOMETRY = {
    'Y': ((56, 92), (7, 1), (8, -5), (-7, -40)),
    'R': ((72, 95), (8, 1), (8, -5), (-6, -40)),
    'K': ((89, 100), (9, 0), (7, -5), (-4, -42)),
    'G': ((108, 103), (9, 1), (7, -5), (-5, -43)),
    'B': ((126, 107), (8, 1), (6, -5), (-3, -44)),
    'P': ((139, 110), (10, 0), (6, -6), (-3, -44)),
}
GEOMETRY = {k: tuple(np.array(v, dtype=float) for v in vv)
            for k, vv in GEOMETRY.items()}
PIVOT = np.array([208., 57.])
DIRECTION = np.array([-56., 24.])
LENGTH = float(np.linalg.norm(DIRECTION))
DIRECTION /= LENGTH
START = dict(zip(ORDER, [12.2, 10.15, 8.1, 6.05, 4.0, 1.95]))
FALL_DURATION = 4.9


def polygon_mask(points):
    im = Image.new('L', (W * SS, H * SS))
    ImageDraw.Draw(im).polygon([tuple(p * SS) for p in points], fill=255)
    return np.asarray(im.resize((W, H), Image.Resampling.LANCZOS)) / 255.


# Full object silhouettes retain the original face colors and soft edges.
MASKS = {}
for name, (p, vx, vd, vz) in GEOMETRY.items():
    outline = [p, p + vx, p + vx + vd, p + vx + vd + vz,
               p + vd + vz, p + vz]
    MASKS[name] = polygon_mask(outline)

# Color support fills small differences between the manually measured corners
# and the rounded wooden edges. It is confined to the block's own neighborhood.
r, g, b = SOURCE.astype(float).transpose(2, 0, 1)
color = {
    'Y': (r > 150) & (g > 100) & (b < 95) & (r - b > 80),
    'R': (r > g + 38) & (g < 85) & (b < 65),
    'K': (r > 135) & (g > 75) & (b > 75) & (r > g + 35),
    'G': (g > r + 10) & (g > b + 10),
    'B': (b > r + 40) & (b > 65),
    'P': (r > g + 12) & (b > g + 18) & (b > 50),
}
for name in ORDER:
    vicinity = ndimage.binary_dilation(MASKS[name] > .1, iterations=2)
    core = color[name] & vicinity
    MASKS[name] = np.maximum(MASKS[name], core.astype(float))

# Some narrow side faces are partly occluded by the next block or by the arm
# in the still. Fill those texture samples from the nearest visible pixel of
# this same block, rather than carrying a neighbor's color into a moving face.
CLEAN_SOURCE = {}
for name in ORDER:
    valid = color[name] & ndimage.binary_dilation(MASKS[name] > .1, iterations=2)
    indices = ndimage.distance_transform_edt(~valid, return_distances=False,
                                            return_indices=True)
    CLEAN_SOURCE[name] = SOURCE[indices[0], indices[1]]

# Erase stationary silhouettes and their table shadows only where an object
# moves. The rest of the locked camera image is copied unchanged each frame.
ERASE = {}
for name, (p, vx, vd, vz) in GEOMETRY.items():
    mask = ndimage.binary_dilation(MASKS[name] > .05, iterations=2)
    yy, xx = np.mgrid[0:H, 0:W]
    ground = ((xx - (p[0] + 6)) / 14)**2 + ((yy - (p[1] + 1)) / 5.5)**2 < 1
    ERASE[name] = mask | ground


def arm_mask():
    yy, xx = np.mgrid[0:H, 0:W]
    dx, dy = xx - PIVOT[0], yy - PIVOT[1]
    along = dx * DIRECTION[0] + dy * DIRECTION[1]
    across = -dx * DIRECTION[1] + dy * DIRECTION[0]
    wood = (along >= 14) & (along < LENGTH + 1.5) & (abs(across) < 2.1)
    neck = (along > 4) & (along < 18) & (abs(across) < 4)
    return ndimage.binary_dilation(wood | neck, iterations=1)


ARM_ERASE = arm_mask()
ALL_ERASE = ARM_ERASE.copy()
for mask in ERASE.values():
    ALL_ERASE |= mask
PLATE = cv2.inpaint(SOURCE, ALL_ERASE.astype(np.uint8) * 255, 4, cv2.INPAINT_TELEA)
# Fit the table's broad illumination and sample its fixed grain only from
# unobstructed wood. Blending inside the hole retains the original boundaries.
blur = cv2.GaussianBlur(SOURCE.astype(float), (0, 0), 1.0)
residual = SOURCE.astype(float) - blur
yy, xx = np.mgrid[0:H, 0:W]
sx = xx % 45
sy = 96 + yy % 47
grain = residual[sy, sx]
wood_repair = ALL_ERASE & (xx < 172)
xf, yf = xx / W, yy / H
basis = np.stack([np.ones_like(xf), xf, yf, xf*xf, xf*yf, yf*yf], axis=-1)
samples = ((xx < 168) & (yy > 43) &
           ~ndimage.binary_dilation(ALL_ERASE, iterations=5))
coef = np.linalg.lstsq(basis[samples], blur[samples], rcond=None)[0]
tone = basis @ coef
weight = np.clip(ndimage.distance_transform_edt(ALL_ERASE)/6., 0, 1)
weight *= wood_repair
PLATE = np.clip(PLATE.astype(float)*(1-weight[...,None]) +
                (tone + grain*.95)*weight[...,None], 0, 255).astype(np.uint8)


def planar_rectangle(name, angle):
    """A mathematical contact cross section in the row/height plane."""
    p, vx, vd, vz = GEOMETRY[name]
    width, height = vx[0], -vz[1]
    a = np.deg2rad(angle)
    v = np.array([width * np.cos(a), width * np.sin(a)])
    z = np.array([-height * np.sin(a), height * np.cos(a)])
    base = np.array([p[0], 0.])
    return np.array([base, base + v, base + v + z, base + z])


def intersects(a, b):
    # Strict interior overlap of two convex rectangles, excluding contact.
    for points in (a, b):
        for edge in np.roll(points, -1, axis=0) - points:
            normal = np.array([-edge[1], edge[0]])
            ap, bp = a @ normal, b @ normal
            if ap.max() <= bp.min() + 1e-7 or bp.max() <= ap.min() + 1e-7:
                return False
    return True


def contact_limit(left, left_angle, right):
    obstacle = planar_rectangle(left, left_angle)
    # Find the first contact continuously from the upright pose.
    lo = 0.
    for hi in np.linspace(0., 90., 91)[1:]:
        if intersects(obstacle, planar_rectangle(right, hi)):
            for _ in range(20):
                mid = (lo + hi) / 2
                if intersects(obstacle, planar_rectangle(right, mid)):
                    hi = mid
                else:
                    lo = mid
            return max(0., lo - .025)
        lo = hi
    return 90.


REST = {'Y': 90.}
for left, right in zip(ORDER[:-1], ORDER[1:]):
    REST[right] = contact_limit(left, REST[left], right)


def smooth(q):
    q = np.clip(q, 0., 1.)
    return q * q * (3. - 2. * q)


def angles_at(t):
    # Staggered acceleration represents the right-to-left cascade. Contact
    # limits prevent a leading block from passing through its neighbor.
    result = {}
    for i, name in enumerate(ORDER):
        desired = REST[name] * smooth((t - START[name]) / FALL_DURATION)
        if i:
            left = ORDER[i - 1]
            desired = min(desired, contact_limit(left, result[left], name))
        result[name] = desired
    return result


# Directly rasterize a parallelogram, sampling only the corresponding face of
# the initial RGB image. The analytic transform exposes the correct surfaces
# and keeps the lower ground edge at its measured image coordinates.
def paint_face(canvas, point, axis1, axis2, src_point, src_axis1, src_axis2,
               texture_source, light=1.):
    corners = np.array([point, point + axis1, point + axis1 + axis2, point + axis2])
    low = np.maximum(np.floor(corners.min(0) - 1).astype(int), 0)
    high = np.minimum(np.ceil(corners.max(0) + 1).astype(int), [W, H])
    x0, y0 = low
    x1, y1 = high
    if x1 <= x0 or y1 <= y0:
        return
    matrix = np.column_stack((axis1, axis2))
    if abs(np.linalg.det(matrix)) < .1:
        return
    gy, gx = np.mgrid[y0*SS:y1*SS, x0*SS:x1*SS]
    xy = np.stack(((gx + .5)/SS - point[0], (gy + .5)/SS - point[1]))
    uv = np.einsum('ij,jhw->ihw', np.linalg.inv(matrix), xy)
    u, v = uv
    coverage = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    # Slightly inset texture coordinates exclude background antialias pixels.
    us, vs = np.clip(u, .055, .945), np.clip(v, .025, .975)
    src = src_point[:, None, None] + src_axis1[:, None, None]*us + src_axis2[:, None, None]*vs
    rgb = np.stack([ndimage.map_coordinates(texture_source[..., c].astype(float),
                     [src[1], src[0]], order=1, mode='nearest') for c in range(3)], -1)
    rgb *= light
    alpha = coverage.astype(float)
    # Area reduction provides stable subpixel edges at the target resolution.
    sh = (y1-y0, SS, x1-x0, SS)
    alpha_small = alpha.reshape(sh).mean((1, 3))
    premul = (rgb * alpha[..., None]).reshape(*sh, 3).mean((1, 3))
    region = canvas[y0:y1, x0:x1]
    region[:] = region * (1-alpha_small[..., None]) + premul


def draw_block(canvas, name, angle):
    p, vx, vd, vz = GEOMETRY[name]
    a = np.deg2rad(angle)
    width, height = vx[0], -vz[1]
    tilted_x = vx*np.cos(a) + vz*(width/height)*np.sin(a)
    tilted_z = vz*np.cos(a) - vx*(height/width)*np.sin(a)
    # Front face remains visible throughout the fall.
    paint_face(canvas, p, tilted_x, tilted_z, p, vx, vz,
               CLEAN_SOURCE[name],
               light=1. - .045*np.sin(a))
    # The upper end turns away, while the bottom end can become visible.
    determinant = np.linalg.det(np.column_stack((tilted_x, vd)))
    if determinant < 0:
        paint_face(canvas, p+tilted_z, tilted_x, vd, p+vz, vx, vd,
                   CLEAN_SOURCE[name],
                   light=1. - .2*np.sin(a))
    elif determinant > 0:
        paint_face(canvas, p, tilted_x, vd, p+vz, vx, vd,
                   CLEAN_SOURCE[name], light=.62)
    # The original right side becomes the upward facing long wooden surface.
    paint_face(canvas, p+tilted_x, vd, tilted_z, p+vx, vd, vz,
               CLEAN_SOURCE[name],
               light=1. + .40*np.sin(a))


def shadow_layer(angles):
    layer = Image.new('L', (W*SS, H*SS))
    dr = ImageDraw.Draw(layer)
    for name in ORDER:
        angle = angles[name]
        if angle < .05:
            continue
        p, vx, vd, vz = GEOMETRY[name]
        a = np.deg2rad(angle)
        tip = p - vx * ((-vz[1])/vx[0]) * np.sin(a)
        # Floor footprint, gently offset away from the diffuse overhead light.
        pts = [p+[0,2], p+vx+vd+[2,3], tip+vx+vd+[3,4], tip+[1,3]]
        opacity = int(48 + 18*np.sin(a))
        dr.polygon([tuple(q*SS) for q in pts], fill=opacity)
        dr.ellipse(tuple((p + [-5, -1])*SS) + tuple((p + [13, 4])*SS), fill=76)
    layer = layer.filter(ImageFilter.GaussianBlur(1.6*SS))
    return np.asarray(layer.resize((W,H), Image.Resampling.LANCZOS))/255.


def arm_angle(t):
    return 26. * smooth(t / 14.)


def draw_arm(canvas, angle):
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    direction = np.array([DIRECTION[0]*c-DIRECTION[1]*s,
                          DIRECTION[0]*s+DIRECTION[1]*c])
    begin = PIVOT + direction*15.5
    end = PIVOT + direction*LENGTH
    layer = Image.new('RGBA', (W*SS,H*SS))
    dr = ImageDraw.Draw(layer)
    # Thin rounded black connector and the pale wood, matching the still.
    dr.line([tuple((PIVOT+direction*3)*SS), tuple(begin*SS)],
            fill=(20,19,18,255), width=round(3.0*SS))
    dr.line([tuple(begin*SS), tuple(end*SS)],
            fill=(130,108,76,255), width=round(2.0*SS))
    normal = np.array([-direction[1], direction[0]])
    dr.line([tuple((begin-normal*.25)*SS), tuple((end-normal*.25)*SS)],
            fill=(205,182,144,255), width=round(1.3*SS))
    dr.ellipse(tuple((end-.7)*SS) + tuple((end+.7)*SS), fill=(193,165,124,255))
    layer = layer.resize((W,H), Image.Resampling.LANCZOS)
    rgba = np.asarray(layer).astype(float)
    alpha = rgba[...,3:4]/255.
    canvas[:] = canvas*(1-alpha) + rgba[...,:3]*alpha
    # Keep the small stationary mounting hub from the initial image.
    yy, xx = np.mgrid[0:H, 0:W]
    hub = (xx-PIVOT[0])**2 + (yy-PIVOT[1])**2 < 4.2**2
    canvas[hub] = SOURCE[hub]


def render(t):
    if t == 0:
        return SOURCE.copy()
    angles = angles_at(t)
    erase = ARM_ERASE.copy()
    for name in ORDER:
        if angles[name] > .03:
            erase |= ERASE[name]
    canvas = SOURCE.astype(float).copy()
    canvas[erase] = PLATE[erase]
    shadow = shadow_layer(angles)
    canvas *= (1 - shadow[...,None]*.62)
    draw_arm(canvas, arm_angle(t))
    for name in ORDER:
        if angles[name] <= .03:
            alpha = MASKS[name][...,None]
            canvas[:] = canvas*(1-alpha) + SOURCE*alpha
        else:
            draw_block(canvas, name, angles[name])
    return np.clip(np.rint(canvas),0,255).astype(np.uint8)


if __name__ == '__main__':
    frames = np.stack([render(t) for t in range(20)])
    assert frames.shape == (20,144,256,3) and frames.dtype == np.uint8
    assert np.array_equal(frames[0], SOURCE)
    artifact_io.save(frames, ROOT)
    # Lossless H.264 RGB keeps even the decoded first video frame exact.
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-framerate', '4',
               '-i', str(ROOT/'frames'/'%02d.png'), '-c:v', 'libx264rgb',
               '-crf', '0', '-preset', 'medium', '-pix_fmt', 'rgb24',
               '-movflags', '+faststart', str(ROOT/'video.mp4')]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    (ROOT/'.tmp'/'video_encode.log').write_text(result.stderr)
    report = {'frames':20, 'size':[256,144], 'fps':4,
              'frame_00_exact':bool(np.array_equal(frames[0], SOURCE)),
              'rightmost_source_median_rgb':np.median(SOURCE[73:95,139:148],axis=(0,1)).tolist(),
              'rest_angles':REST,
              'timeline':[{name:round(a,3) for name,a in angles_at(t).items()} for t in range(20)]}
    (ROOT/'.tmp'/'generation_check.json').write_text(json.dumps(report,indent=2))
    print('Saved 20 RGB frames, pixels.npz, index.html, and lossless video.mp4; '
          'frame 00 matches exactly.')

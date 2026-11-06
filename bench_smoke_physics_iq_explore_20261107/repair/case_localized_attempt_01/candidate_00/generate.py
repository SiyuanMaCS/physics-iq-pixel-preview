"""
PixelWorld-Text case_0036 generation.

Scene: six colored wooden cuboids lined up diagonally on a wooden table
beside a black platform. A wooden stick attached to a black hub on the
platform rotates clockwise (screen sense), strikes the purple (nearest)
block, and the row topples as a domino cascade.

Rendering approach:
  * numeric RGB analysis -> exact masks for 6 blocks + stick + hub;
  * a clean background plate is built by 2-D membrane inpainting of the
    erased object pixels (plus matched grain);
  * block sprites cut from the original are rotated about their
    left-bottom ground-contact pivot;
  * the wood stick + black neck are rendered analytically (3x
    supersampled segment shading) behind the block layer; the static
    hub disc is never touched;
  * frame 0 is the untouched first_frame.png.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, uniform_filter
import artifact_io

ROOT = Path(__file__).parent
SS = 3  # stick supersampling

# ---------------------------------------------------------------- geometry
PIVOT = np.array([208.0, 57.0])
ANG0 = np.array([-0.925, 0.390])          # frame-0 stick unit dir (x,y)
STICK_L = 60.0
BLOCK_ORDER = ['Y', 'R', 'K', 'G', 'B', 'P']          # far -> near
PIVOTS = {'Y': (55, 92), 'R': (68, 95), 'K': (88, 95),
          'G': (107, 102), 'B': (125, 106), 'P': (137, 109)}
T0 = {'P': 4, 'B': 6, 'G': 8, 'K': 10, 'R': 12, 'Y': 14}
PHI_MAX = {'P': 78, 'B': 75, 'G': 70, 'K': 66, 'R': 62, 'Y': 58}
FALL_DUR = 5.2

# ---------------------------------------------------------------- load
frame0 = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
H, W, _ = frame0.shape
imf = frame0.astype(int)
r, g, b = imf[..., 0], imf[..., 1], imf[..., 2]
MX = imf.max(axis=2)
yy, xx = np.mgrid[0:H, 0:W]


def biggest(mask, minsz=25):
    lab, n = ndimage.label(mask)
    out = np.zeros_like(mask)
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() >= minsz:
            out |= c
    return out


# ------------------------------------------------------- block color cores
mY = (r > 160) & (g > 105) & (b < 95) & (r - b > 80)
mR = (((r > 95) & (g < 80) & (b < 70) & (r - g > 45)) |
      ((r > 60) & (r < 110) & (g < 55) & (b < 50) & (r > g + 25)))
mK = ((r > 150) & (g > 85) & (g < 150) & (b > 80) & (b < 140) &
      (r - g > 40) & (abs(g - b) < 38))
mG = (g > 40) & (g > r + 12) & (g > b + 12)
mB = (b > 80) & (b > r + 45)
mP = (r > 50) & (b > 55) & (g < r - 12) & (g < b - 15)
cores = {'Y': biggest(mY), 'R': biggest(mR), 'K': biggest(mK),
         'G': biggest(mG), 'B': biggest(mB), 'P': biggest(mP)}

# ground-front line across the block zone
gx0, gx1 = 45, 160
ground_y = np.zeros(W, float)
col_bottom = {}
for j in BLOCK_ORDER:
    ys, xs = np.where(cores[j])
    for x in range(xs.min(), xs.max() + 1):
        yycol = np.where(cores[j][:, x])[0]
        if len(yycol):
            col_bottom[x] = max(col_bottom.get(x, 0), yycol.max())
cols = np.array(sorted(col_bottom))
vals = np.array([col_bottom[x] for x in cols])
ground_y[:] = np.interp(np.arange(W), cols, ndimage.uniform_filter1d(vals, 5))

# assign very-dark rim pixels to nearest block core (distance <= 3)
union_core = np.zeros((H, W), bool)
for mk in cores.values():
    union_core |= mk
above_ground = yy <= ground_y[None, :] + 3
candidate = (ndimage.binary_dilation(union_core, iterations=2) &
            (MX < 100) & above_ground & (xx < 172) & ~union_core)
dists = {j: distance_transform_edt(~cores[j]) for j in BLOCK_ORDER}
dstack = np.stack([dists[j] for j in BLOCK_ORDER])
nearest = dstack.argmin(axis=0)
nearest_d = dstack.min(axis=0)
sprite_mask = {j: cores[j].copy() for j in BLOCK_ORDER}
rim = candidate & (nearest_d <= 3.0)
for k, j in enumerate(BLOCK_ORDER):
    sprite_mask[j] |= rim & (nearest == k)
    sprite_mask[j] = biggest(sprite_mask[j], minsz=20)

# ---------------------------------------------------------------- stick mask
rel = np.dstack([xx - PIVOT[0], yy - PIVOT[1]])
along0 = rel[..., 0] * ANG0[0] + rel[..., 1] * ANG0[1]
perp0 = rel[..., 0] * (-ANG0[1]) + rel[..., 1] * ANG0[0]
hub = ((xx - PIVOT[0]) ** 2 + (yy - PIVOT[1]) ** 2) <= 8.5 ** 2
stick_band = (along0 >= 6) & (along0 <= STICK_L) & (np.abs(perp0) <= 4.6)
neck_wedge = (MX < 55) & (xx >= 186) & (xx <= 220) & (yy >= 49) & (yy <= 72) & \
             ~hub & (perp0 > 1) & (along0 > 2) & (along0 < 22)
neck_wedge = biggest(neck_wedge, minsz=8)

# ---------------------------------------------------------------- erase/plate
gap_shadow = (ndimage.binary_dilation(union_core, iterations=4) & (MX < 95) &
              above_ground & (xx < 175))
erase_blocks = ndimage.binary_closing(
    union_core | rim | gap_shadow, iterations=2)
# Color thresholds miss very dark side faces and antialiased corners.
# Fill the row silhouette down to its contact line so those pixels cannot
# survive as stationary fragments after the colored sprites move away.
for x in np.where(erase_blocks.any(axis=0))[0]:
    ys = np.where(erase_blocks[:, x])[0]
    bottom = min(H - 1, max(int(ys.max()), int(np.ceil(ground_y[x] + 3))))
    erase_blocks[ys.min():bottom + 1, x] = True
erase_blocks = ndimage.binary_dilation(erase_blocks, iterations=1)
erase = erase_blocks | ((stick_band | neck_wedge) & ~union_core)

plate = frame0.copy().astype(float)
good = ~erase
# per-row linear initialisation across erased runs
for y in range(H):
    bx = np.where(erase[y])[0]
    if len(bx):
        gx = np.where(good[y])[0]
        for c in range(3):
            plate[y, bx, c] = np.interp(bx, gx, frame0[y, gx, c].astype(float))
# 2-D membrane smoothing of erased pixels from good neighbours
for _ in range(45):
    flt = uniform_filter(plate, size=(9, 9, 1), mode='nearest')
    plate[erase] = flt[erase]

# grain matched to local table/platform residuals
blur = uniform_filter(frame0.astype(float), size=(1, 5, 1), mode='nearest')
resid = frame0.astype(float) - blur
table_zone = (xx < 172)
sig_t = resid[table_zone & good & (yy > 44) & (yy < 130)].std()
sig_p = resid[~table_zone & good & (yy > 40) & (yy < 100)].std()
rng = np.random.default_rng(36036)
noise = rng.normal(0, 1.0, (H, W, 3))
noise = uniform_filter(noise, size=(1, 3, 1))
noise_t = rng.normal(0, 1.0, (H, W))
sig = np.where(table_zone, sig_t, sig_p)
plate += (erase & ~hub)[..., None] * noise * sig[..., None]
plate = np.clip(plate, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- sprites
def cut_sprite(name):
    mk = sprite_mask[name]
    ys, xs = np.where(mk)
    x0, x1, y0, y1 = xs.min() - 2, xs.max() + 3, ys.min() - 2, ys.max() + 3
    crop = frame0[y0:y1, x0:x1].copy()
    a = mk[y0:y1, x0:x1].astype(np.uint8) * 255
    # darken ground-contact edge of the sprite (underside line)
    bottom = np.where(mk.any(axis=1))[0].max()
    rgba = np.dstack([crop, a])
    return rgba, (x0, y0), bottom


SPRITES = {j: cut_sprite(j) for j in BLOCK_ORDER}


def paste_sprite(canvas, name, phi_deg):
    rgba, (x0, y0), bottom = SPRITES[name]
    px, py = PIVOTS[name]
    if phi_deg < 0.05:
        layer = Image.fromarray(rgba, 'RGBA')
        ox, oy = x0, y0
    else:
        img = Image.fromarray(rgba, 'RGBA')
        rot = img.rotate(phi_deg, resample=Image.BILINEAR, expand=True)
        # original pivot in crop coords
        lpx, lpy = px - x0, py - y0
        w0, h0 = img.size
        # pivot maps to center of expanded canvas
        cx0, cy0 = (w0 - 1) / 2.0, (h0 - 1) / 2.0
        w1, h1 = rot.size
        cx1, cy1 = (w1 - 1) / 2.0, (h1 - 1) / 2.0
        ox = px - lpx + (cx0 - cx1)
        oy = py - lpy + (cy0 - cy1)
        layer = rot
        ox, oy = int(round(ox)), int(round(oy))
    base = Image.fromarray(canvas, 'RGB').convert('RGBA')
    base.alpha_composite(layer, (ox, oy))
    return np.array(base.convert('RGB'))


# ---------------------------------------------------------------- stick draw
def cw(theta_deg):
    """Rotate frame-0 direction clockwise (screen sense)."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([ANG0[0] * c - ANG0[1] * s,
                     ANG0[0] * s + ANG0[1] * c])


def stick_layer(theta_deg):
    u = cw(theta_deg)
    nrm = np.array([-u[1], u[0]])
    Hs, Ws = H * SS, W * SS
    P = PIVOT * SS
    us = u[SS] if False else u
    yyi, xxi = np.mgrid[0:Hs, 0:Ws]
    d = np.dstack([xxi - P[0], yyi - P[1]]).astype(float)
    along = d[..., 0] * u[0] + d[..., 1] * u[1]
    perp = d[..., 0] * nrm[0] + d[..., 1] * nrm[1]

    def band(r0, r1, hw):
        return (along >= r0 * SS) & (along <= r1 * SS) & (np.abs(perp) <= hw * SS)

    rgba = np.zeros((Hs, Ws, 4), np.float64)
    def paint(mask, color, alpha, soft=0.0):
        m = mask.astype(float)
        if soft:
            edge = np.clip((soft * SS - np.abs(np.abs(perp) - 0)) / (soft * SS), 0, 1)
            m *= np.maximum(np.clip((soft * SS - np.abs(np.abs(perp) -
                  (np.clip(np.abs(perp)/(hw_s*SS),0,1)*0)))/(soft*SS),0,1), 0)
        cover = m * alpha
        for k, v in enumerate(color):
            rgba[..., k] = cover * v + (1 - cover) * rgba[..., k]
        rgba[..., 3] = np.maximum(rgba[..., 3], cover * 255)

    # soft neck halo
    hw_s = 9.0
    m = band(6, 20, 9.0)
    cover = m * 0.5
    for k, v in enumerate((10, 8, 6)):
        rgba[..., k] = cover * v + (1 - cover) * rgba[..., k]
    rgba[..., 3] = np.maximum(rgba[..., 3], cover * 255)
    # black neck
    m = band(7, 18, 3.6)
    cover = m * 0.95
    for k, v in enumerate((14, 11, 8)):
        rgba[..., k] = cover * v + (1 - cover) * rgba[..., k]
    rgba[..., 3] = np.maximum(rgba[..., 3], cover * 255)
    # wood stick: shade body
    m = band(14, STICK_L, 3.4)
    cover = m * 0.97
    for k, v in enumerate((116, 84, 52)):
        rgba[..., k] = cover * v + (1 - cover) * rgba[..., k]
    rgba[..., 3] = np.maximum(rgba[..., 3], cover * 255)
    # bright core, offset toward lower-left (-normal) side
    mc = (along >= 15 * SS) & (along <= (STICK_L - 1) * SS) & \
         (perp >= -3.2 * SS) & (perp <= 1.6 * SS)
    cover = mc * 0.9
    for k, v in enumerate((184, 156, 122)):
        rgba[..., k] = cover * v + (1 - cover) * rgba[..., k]
    rgba[..., 3] = np.maximum(rgba[..., 3], cover * 255)
    # lit rim
    ml = (along >= 16 * SS) & (along <= (STICK_L - 2) * SS) & \
         (perp >= -3.0 * SS) & (perp <= -1.9 * SS)
    cover = ml * 0.8
    for k, v in enumerate((206, 184, 152)):
        rgba[..., k] = cover * v + (1 - cover) * rgba[..., k]
    rgba[..., 3] = np.maximum(rgba[..., 3], cover * 255)
    # rounded tip cap
    tip = P + u * STICK_L * SS
    mcap = ((xxi - tip[0]) ** 2 + (yyi - tip[1]) ** 2) <= (3.4 * SS) ** 2
    cover = mcap * 0.95
    for k, v in enumerate((120, 88, 55)):
        rgba[..., k] = cover * v + (1 - cover) * rgba[..., k]
    rgba[..., 3] = np.maximum(rgba[..., 3], cover * 255)
    mcap2 = ((xxi - tip[0]) ** 2 + (yyi - tip[1]) ** 2) <= (1.8 * SS) ** 2
    cover = mcap2 * 0.9
    for k, v in enumerate((184, 156, 122)):
        rgba[..., k] = cover * v + (1 - cover) * rgba[..., k]
    rgba[..., 3] = np.maximum(rgba[..., 3], cover * 255)

    small = Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), 'RGBA')
    return small.resize((W, H), Image.LANCZOS)


def ground_shadow(canvas, phi_by_name):
    # Draw coverage separately. Drawing RGBA directly onto the background
    # and then converting to RGB discards alpha instead of blending it.
    mask = Image.new('L', (W * SS, H * SS), 0)
    dr = ImageDraw.Draw(mask)
    for j in BLOCK_ORDER:
        phi = np.deg2rad(phi_by_name[j])
        px, py = PIVOTS[j]
        ext = 13 * abs(np.sin(phi))
        rx = 8 + ext
        x0 = px - rx - ext * 0.9
        x1 = px + 5
        dy = 2.2
        if phi_by_name[j] <= 0:
            continue
        opacity = 60 * smooth(phi_by_name[j] / 12.0)
        dr.ellipse([x0 * SS, (py - dy) * SS,
                    x1 * SS, (py + dy + 1) * SS], fill=round(opacity))
    alpha = np.asarray(mask.resize((W, H), Image.Resampling.LANCZOS),
                       dtype=float) / 255.0
    alpha = ndimage.gaussian_filter(alpha, sigma=1.4)[..., None]
    # Retain the table's grain and lighting under a single soft shadow.
    # Overlapping block footprints share coverage, avoiding dark buildup.
    shadow_color = np.array([18.0, 13.0, 9.0])
    return np.clip(canvas.astype(float) * (1 - alpha) + shadow_color * alpha,
                   0, 255).round().astype(np.uint8)


# ---------------------------------------------------------------- timeline
def smooth(q):
    q = np.clip(q, 0, 1)
    return q * q * (3 - 2 * q)


def stick_angle(t):
    return 26.0 * np.clip(t / 13.0, 0, 1) ** 1.7


def lean(t, name):
    q = (t - T0[name]) / FALL_DUR
    return PHI_MAX[name] * smooth(q)


frames = [frame0]
for t in range(1, 20):
    canvas = plate.copy()
    phi = {j: lean(t, j) for j in BLOCK_ORDER}
    canvas = ground_shadow(canvas, phi)
    base = Image.fromarray(canvas, 'RGB').convert('RGBA')
    base.alpha_composite(stick_layer(stick_angle(t)))
    canvas = np.array(base.convert('RGB'))
    for j in BLOCK_ORDER:                       # far -> near
        canvas = paste_sprite(canvas, j, phi[j])
    frames.append(canvas)

frames = np.stack(frames).astype(np.uint8)
assert frames.shape == (20, 144, 256, 3)
assert np.array_equal(frames[0], frame0)
artifact_io.save(frames, ROOT)

# transition check sheet: selected frames + amplified predecessor diffs
sel = [0, 3, 6, 9, 12, 15, 18, 19]
sheet = Image.new('RGB', (4 * 256, 2 * (144 + 18)), (25, 28, 35))
dr = ImageDraw.Draw(sheet)
for i, t in enumerate(sel):
    x, y = (i % 4) * 256, (i // 4) * (144 + 18)
    dr.text((x + 5, y + 3), f'frame {t+1}/20  {(t+1)/4:.2f}s', fill='white')
    sheet.paste(Image.fromarray(frames[t]), (x, y + 18))
    if t > 0:
        d = np.abs(frames[t].astype(int) - frames[t - 1].astype(int)).sum(2)
        heat = np.clip(d * 3, 0, 255).astype(np.uint8)
        diff = np.zeros_like(frames[t])
        diff[..., 0] = heat
        diff[..., 1] = (heat // 3)
        sheet.paste(Image.fromarray(diff), (x, y + 18 + (144 + 18)))
sheet.save(ROOT / 'transition_check.png')
print('done')

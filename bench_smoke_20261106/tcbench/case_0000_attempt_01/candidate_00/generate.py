"""Generate a stylized close-up of a chameleon changing colour.

The scene is drawn directly with Pillow and numpy.  A spatially staggered
blend makes the green travel from the animal's face across its body and into
the curled tail, while the background and branch remain visually stable.
"""

from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

import artifact_io


W, H, N = 256, 144, 20
S = 3
SW, SH = W * S, H * S
ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(7123)


def sc(v):
    """Scale a coordinate or coordinate tuple for antialiased drawing."""
    if isinstance(v, (tuple, list)):
        return tuple(int(round(x * S)) for x in v)
    return int(round(v * S))


def mix(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(x * (1 - t) + y * t)) for x, y in zip(a, b))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# Fixed background decorations keep the video coherent from frame to frame.
LEAVES = []
for _ in range(25):
    LEAVES.append(
        (
            float(RNG.uniform(-10, W + 10)),
            float(RNG.uniform(-8, H + 5)),
            float(RNG.uniform(7, 23)),
            float(RNG.uniform(4, 13)),
            float(RNG.uniform(-0.8, 0.8)),
            float(RNG.uniform(0.0, 1.0)),
        )
    )

SCALE_POINTS = []
for yy in range(45, 105, 5):
    for xx in range(30, 225, 6):
        SCALE_POINTS.append((xx + ((yy // 5) % 2) * 2.5, yy, RNG.uniform(-1, 1)))


def background():
    y = np.linspace(0, 1, SH)[:, None, None]
    top = np.array([17, 48, 43], dtype=float)[None, None, :]
    bottom = np.array([76, 91, 47], dtype=float)[None, None, :]
    arr = np.broadcast_to(top * (1 - y) + bottom * y, (SH, SW, 3)).copy()
    # A little center light suggests a humid forest clearing.
    yy, xx = np.mgrid[0:SH, 0:SW]
    glow = np.exp(-(((xx / S - 139) / 104) ** 2 + ((yy / S - 55) / 57) ** 2))
    arr += glow[..., None] * np.array([21, 30, 7])[None, None, :]
    img = Image.fromarray(np.uint8(np.clip(arr, 0, 255)), "RGB")

    blur = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    d = ImageDraw.Draw(blur, "RGBA")
    for x, y0, rx, ry, ang, shade in LEAVES:
        # Soft overlapping oval foliage, deliberately out of focus.
        col = (
            int(20 + shade * 35),
            int(66 + shade * 55),
            int(39 + shade * 24),
            int(55 + shade * 60),
        )
        d.ellipse(sc((x - rx, y0 - ry, x + rx, y0 + ry)), fill=col)
    blur = blur.filter(ImageFilter.GaussianBlur(sc(5.0)))
    img = Image.alpha_composite(img.convert("RGBA"), blur)

    # Warm, defocused highlights make the subject's silhouette read cleanly.
    bokeh = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bokeh, "RGBA")
    for x, y0, r in [(31, 29, 8), (223, 22, 11), (239, 89, 7), (89, 17, 5)]:
        bd.ellipse(sc((x-r, y0-r, x+r, y0+r)), fill=(206, 211, 113, 35), outline=(234, 226, 143, 25), width=sc(1))
    return Image.alpha_composite(img, bokeh.filter(ImageFilter.GaussianBlur(sc(2.2))))


def draw_branch(img):
    d = ImageDraw.Draw(img, "RGBA")
    pts = [(-8, 122), (42, 117), (91, 115), (141, 108), (198, 106), (266, 97)]
    d.line([sc(p) for p in pts], fill=(34, 26, 18, 125), width=sc(19), joint="curve")
    d.line([sc(p) for p in pts], fill=(87, 59, 35, 255), width=sc(15), joint="curve")
    d.line([sc((p[0], p[1]-2)) for p in pts], fill=(139, 96, 52, 220), width=sc(3), joint="curve")
    d.line([sc((p[0], p[1]+4)) for p in pts], fill=(48, 38, 25, 190), width=sc(3), joint="curve")
    # Bark scars.
    for x, y, dx in [(18,119,10),(61,115,14),(111,111,9),(158,106,15),(215,101,12)]:
        d.line([sc((x,y)), sc((x+dx,y-2))], fill=(48,35,23,150), width=sc(1))
        d.line([sc((x+2,y-4)), sc((x+dx-1,y-5))], fill=(176,124,66,115), width=sc(1))
    # A small twig behind the animal.
    d.line([sc((169,108)), sc((189,88)), sc((196,76))], fill=(72,48,29,255), width=sc(4), joint="curve")
    d.line([sc((169,106)), sc((190,86))], fill=(150,99,52,170), width=sc(1))


def skin_progress(x, y, progress, phase=0.0):
    """Local blend: transformation advances from the head toward the tail."""
    if progress <= 0:
        return 0.0
    if progress >= 1:
        return 1.0
    # Higher x changes sooner. Low-amplitude ripples make the boundary organic.
    xn = np.clip((x - 25.0) / 205.0, 0, 1)
    threshold = 0.82 - 0.69 * xn + 0.035 * math.sin(y * 0.24 + x * 0.09 + phase)
    return float(smoothstep((progress - threshold + 0.14) / 0.28))


def make_chameleon_mask(breathe):
    mask = Image.new("L", (SW, SH), 0)
    d = ImageDraw.Draw(mask)
    by = breathe
    # Curled tail, torso, neck and angular chameleon head.
    tail = [(76,83+by), (59,83+by), (42,91+by), (26,91+by), (22,82+by), (29,74+by), (41,75+by), (45,82+by), (39,87+by), (33,85+by)]
    d.line([sc(p) for p in tail], fill=255, width=sc(13), joint="curve")
    d.ellipse(sc((56,43+by,177,106+by)), fill=255)
    d.polygon([sc(p) for p in [(147,49+by),(180,44+by),(198,58+by),(191,84+by),(160,92+by),(146,78+by)]], fill=255)
    d.polygon([sc(p) for p in [(178,45+by),(210,47+by),(230,59+by),(226,75+by),(203,81+by),(181,72+by)]], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(sc(0.35)))


def render_frame(frame_index):
    t = frame_index / (N - 1)
    progress = float(smoothstep(t))
    breathe = math.sin(t * math.pi * 2) * 0.7
    img = background()
    draw_branch(img)

    mask = make_chameleon_mask(breathe)
    yy, xx = np.mgrid[0:SH, 0:SW]
    x = xx / S
    y = yy / S
    xn = np.clip((x - 25.0) / 205.0, 0, 1)
    threshold = 0.82 - 0.69 * xn + 0.035 * np.sin(y * 0.24 + x * 0.09 + t * 2.0)
    if frame_index == 0:
        local = np.zeros_like(x)
    elif frame_index == N - 1:
        local = np.ones_like(x)
    else:
        local = smoothstep((progress - threshold + 0.14) / 0.28)

    brown = np.array([111, 70, 35], dtype=float)
    green = np.array([66, 205, 46], dtype=float)
    skin = brown[None,None,:] * (1-local[...,None]) + green[None,None,:] * local[...,None]
    # Rounded volume, warm dorsal ridge and cool shaded belly.
    volume = 0.72 + 0.30 * np.clip(1 - ((y - (66+breathe))/35) ** 2, 0, 1)
    dorsal = 17 * np.exp(-((y - (49+breathe))/10) ** 2)
    mottling = 5.0 * np.sin(x * .31 + y * .19) + 3.0 * np.sin(x * .11 - y * .37)
    skin = skin * volume[...,None] + dorsal[...,None] + mottling[...,None]
    skin = np.uint8(np.clip(skin, 0, 255))
    skin_img = Image.fromarray(skin, "RGB").convert("RGBA")
    img = Image.composite(skin_img, img, mask)

    d = ImageDraw.Draw(img, "RGBA")
    outline = mix((56,37,22), (24,92,25), progress)
    # Silhouette accents and back crest.
    d.line([sc(p) for p in [(45,82+breathe),(56,59+breathe),(78,45+breathe),(112,41+breathe),(148,46+breathe),(178,45+breathe),(210,47+breathe),(230,59+breathe)]], fill=outline+(220,), width=sc(1.4), joint="curve")
    for i, (cx, cy, ht) in enumerate([(82,44,5),(94,42,6),(107,41,6),(120,42,6),(133,44,5),(146,47,4)]):
        lp = skin_progress(cx, cy, progress, t*2)
        crest = mix((103,64,31),(53,175,38),lp)
        d.polygon([sc((cx-4,cy+2+breathe)),sc((cx,cy-ht+breathe)),sc((cx+4,cy+2+breathe))], fill=crest+(255,), outline=outline+(170,))

    # Honeycomb-like scales, clipped precisely to the animal silhouette.
    scale_layer = Image.new("RGBA", (SW,SH), (0,0,0,0))
    sd = ImageDraw.Draw(scale_layer, "RGBA")
    mask_arr = np.asarray(mask)
    for px, py, jitter in SCALE_POINTS:
        sy = py + breathe
        ix, iy = int(px*S), int(sy*S)
        if 0 <= ix < SW and 0 <= iy < SH and mask_arr[iy,ix] > 180:
            lp = skin_progress(px, sy, progress, t*2)
            base = mix((80,49,27),(36,145,31),lp)
            hi = mix((148,100,53),(116,235,76),lp)
            alpha = int(48 + 18*jitter)
            r = 1.45 + .2*jitter
            sd.ellipse(sc((px-r,sy-r,px+r,sy+r)), outline=base+(alpha+38,), width=sc(.55))
            sd.ellipse(sc((px-.45,sy-.55,px+.45,sy+.35)), fill=hi+(alpha,))
    clipped_alpha = ImageChops.multiply(scale_layer.getchannel("A"), mask)
    scale_layer.putalpha(clipped_alpha)
    img = Image.alpha_composite(img, scale_layer)
    d = ImageDraw.Draw(img, "RGBA")

    # Two gripping legs and small toes. Their colour follows the nearest skin.
    for pts, anchor_x in [([(99,84+breathe),(91,102),(84,110)],99), ([(165,81+breathe),(174,97),(171,107)],165)]:
        lp = skin_progress(anchor_x, 86, progress, t*2)
        legc = mix((88,54,29),(43,168,38),lp)
        dark = mix((51,35,23),(24,99,28),lp)
        d.line([sc(p) for p in pts], fill=dark+(255,), width=sc(8), joint="curve")
        d.line([sc(p) for p in pts], fill=legc+(255,), width=sc(5), joint="curve")
        ex, ey = pts[-1]
        for off in (-4,0,4):
            d.line([sc((ex,ey)),sc((ex+off,ey+2))],fill=dark+(240,),width=sc(1.5))

    # Facial fold, mouth and nostril strongly identify the animal.
    face_lp = skin_progress(199, 62, progress, t*2)
    face_dark = mix((62,39,23),(20,104,27),face_lp)
    d.line([sc((184,70+breathe)),sc((203,73+breathe)),sc((225,69+breathe))], fill=face_dark+(210,), width=sc(1.2))
    d.line([sc((196,78+breathe)),sc((213,77+breathe)),sc((226,71+breathe))], fill=(30,25,18,180), width=sc(1))
    d.ellipse(sc((218,58+breathe,221,61+breathe)), fill=(27,28,19,210))

    # Raised eye turret, golden iris, and a tiny highlight. Pupil tracks slightly.
    eye_skin = mix((132,88,42),(83,220,57),face_lp)
    d.ellipse(sc((180,49+breathe,198,67+breathe)), fill=face_dark+(235,))
    d.ellipse(sc((182,50+breathe,197,65+breathe)), fill=eye_skin+(255,))
    d.ellipse(sc((185,53+breathe,195,63+breathe)), fill=(224,191,74,255))
    pupil_x = 190.5 + 1.0*math.sin(t*math.pi)
    d.ellipse(sc((pupil_x-2.6,55+breathe,pupil_x+2.2,61+breathe)), fill=(16,18,12,255))
    d.ellipse(sc((pupil_x-1.1,55.4+breathe,pupil_x+.1,56.6+breathe)), fill=(255,255,225,235))

    # A small moving sheen makes the vivid final green feel alive, not flat.
    shine_alpha = int(35 + 18*math.sin(t*math.pi))
    d.arc(sc((73,48+breathe,164,93+breathe)), 198, 325, fill=(240,255,176,shine_alpha), width=sc(1))

    out = img.convert("RGB").resize((W,H), Image.Resampling.LANCZOS)
    return np.asarray(out, dtype=np.uint8)


def main():
    frames = np.stack([render_frame(i) for i in range(N)], axis=0)
    # Keep a first-pass contact sheet as the required intermediate candidate.
    thumbs = Image.new("RGB", (W*5, H), (20,20,20))
    for j, idx in enumerate((0, 5, 10, 14, 19)):
        thumbs.paste(Image.fromarray(frames[idx]), (j*W, 0))
    thumbs.save(ROOT / "candidate_01.png")
    artifact_io.save(frames, ROOT)


if __name__ == "__main__":
    main()

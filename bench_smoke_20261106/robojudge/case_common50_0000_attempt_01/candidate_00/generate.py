"""Generate a short, deterministic RGB manipulation sequence from first_frame.png.

The animation is deliberately made from image pixels and vector geometry only.  The
photographic frame is retained as the workspace plate; a masked, rigid two-link arm
and small fruit sprites supply the visible action.
"""

from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import artifact_io


ROOT = Path(__file__).resolve().parent
W, H = 256, 192
SCALE = 3


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    t = ease(t)
    return (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t)


def poly_mask(size, polygons):
    im = Image.new("L", size, 0)
    d = ImageDraw.Draw(im)
    for p in polygons:
        d.polygon(p, fill=255)
    return np.asarray(im)


def make_clean_plate(source):
    """Remove the original moving right arm, leaving the fixed right base alone."""
    mask = poly_mask(
        (W, H),
        [
            # long silver forearm and the black wrist/claw
            [(134, 58), (224, 24), (241, 35), (233, 68), (166, 109), (139, 106)],
            [(128, 55), (163, 53), (181, 97), (165, 116), (137, 108), (145, 83)],
            # cable and the small part of the link immediately at the right base
            [(221, 38), (253, 31), (255, 80), (225, 86), (215, 66)],
        ],
    )
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    # TELEA works well here because nearby crate slats and fruit colours continue
    # naturally under the relatively thin diagonal arm.
    return cv2.inpaint(source, mask, 5, cv2.INPAINT_TELEA)


def remove_fruit(plate, kind):
    out = plate.copy()
    specs = {
        "apple": ((78, 112), 10),
        "orange": ((207, 111), 11),
        "star": ((121, 99), 12),
    }
    (x, y), r = specs[kind]
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (x, y), r, 255, -1)
    return cv2.inpaint(out, m, 4, cv2.INPAINT_TELEA)


def hi_layer():
    return Image.new("RGBA", (W * SCALE, H * SCALE), (0, 0, 0, 0))


def P(point):
    return tuple(int(round(v * SCALE)) for v in point)


def capsule(draw, a, b, width, fill, outline=None, outline_width=0):
    a, b, width = P(a), P(b), int(width * SCALE)
    if outline and outline_width:
        draw.line((a, b), fill=outline, width=width + 2 * outline_width * SCALE)
        rr = (width + 2 * outline_width * SCALE) // 2
        for q in (a, b):
            draw.ellipse((q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr), fill=outline)
    draw.line((a, b), fill=fill, width=width)
    rr = width // 2
    for q in (a, b):
        draw.ellipse((q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr), fill=fill)


def arm_geometry(grip):
    """Stable articulated geometry with a fixed shoulder and a bent elbow."""
    sx, sy = 244.0, 60.0
    gx, gy = grip
    dx, dy = gx - sx, gy - sy
    dist = max(1.0, math.hypot(dx, dy))
    # A shallow bend makes joint motion legible and keeps all links rigid-looking.
    nx, ny = -dy / dist, dx / dist
    bend = 15.0 if gy < 132 else 11.0
    ex = sx + dx * 0.50 + nx * bend
    ey = sy + dy * 0.50 + ny * bend
    return (sx, sy), (ex, ey), (gx, gy)


def draw_robot(base, grip, closed, item_kind=None):
    layer = hi_layer()
    d = ImageDraw.Draw(layer)
    shoulder, elbow, wrist = arm_geometry(grip)

    # Moving arm: dark seams, aluminium bodies, then fixed-size joint bearings.
    capsule(d, shoulder, elbow, 22, (126, 155, 161, 255), (18, 29, 31, 255), 3)
    capsule(d, elbow, wrist, 17, (151, 182, 188, 255), (15, 27, 29, 255), 3)
    capsule(d, (shoulder[0] - 2, shoulder[1] - 3), (elbow[0] - 1, elbow[1] - 3), 4,
            (214, 230, 229, 210))
    for x, y, r in ((*shoulder, 11), (*elbow, 10), (*wrist, 8)):
        q = P((x, y)); rr = int(r * SCALE)
        d.ellipse((q[0]-rr, q[1]-rr, q[0]+rr, q[1]+rr), fill=(18, 30, 32, 255))
        ri = int((r-3) * SCALE)
        d.ellipse((q[0]-ri, q[1]-ri, q[0]+ri, q[1]+ri), fill=(111, 143, 149, 255))
        rc = max(2, int((r-7) * SCALE))
        d.ellipse((q[0]-rc, q[1]-rc, q[0]+rc, q[1]+rc), fill=(31, 46, 47, 255))

    # Wrist points toward the lower-left; two black jaws maintain their dimensions.
    gx, gy = wrist
    jaw_gap = 5.0 if closed else 11.0
    for sign in (-1, 1):
        y0 = gy + sign * jaw_gap
        capsule(d, (gx - 1, y0), (gx - 13, y0 + sign * 2), 4.5,
                (12, 22, 22, 255), (2, 7, 7, 255), 1)
        capsule(d, (gx - 13, y0 + sign * 2), (gx - 17, gy + sign * (jaw_gap - 1)), 3.5,
                (9, 18, 18, 255))

    # A short red pneumatic lead follows the arm without changing thickness.
    pts = [P(shoulder), P((elbow[0]+2, elbow[1]+8)), P((wrist[0]+3, wrist[1]+8))]
    d.line(pts, fill=(151, 26, 30, 230), width=2 * SCALE, joint="curve")

    layer = layer.resize((W, H), Image.Resampling.LANCZOS)
    return Image.alpha_composite(base.convert("RGBA"), layer)


def fruit_sprite(kind, center, scale=1.0, alpha=255):
    layer = hi_layer()
    d = ImageDraw.Draw(layer)
    cx, cy = P(center)
    sc = SCALE * scale
    if kind == "apple":
        r = int(8 * sc)
        d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(198, 24, 32, alpha),
                  outline=(111, 15, 20, alpha), width=max(1, int(sc)))
        d.ellipse((cx-int(4*sc), cy-int(5*sc), cx, cy-int(1*sc)), fill=(245, 71, 68, 190))
        d.line((cx, cy-r, cx+int(sc), cy-int(11*sc)), fill=(65, 53, 25, alpha), width=max(1, int(sc)))
        d.ellipse((cx+int(sc), cy-int(11*sc), cx+int(5*sc), cy-int(8*sc)), fill=(50, 99, 36, alpha))
    elif kind == "orange":
        r = int(8 * sc)
        d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(241, 101, 20, alpha),
                  outline=(169, 58, 13, alpha), width=max(1, int(sc)))
        d.ellipse((cx-int(4*sc), cy-int(5*sc), cx, cy-int(sc)), fill=(255, 173, 61, 210))
        d.ellipse((cx-int(sc), cy-r, cx+int(2*sc), cy-r+int(2*sc)), fill=(58, 91, 32, alpha))
    else:
        # A golden, five-ridged side view of a ripe star fruit.
        pts = [(-12,0), (-8,-6), (-3,-4), (2,-7), (6,-3), (12,0),
               (6,3), (2,7), (-3,4), (-8,6)]
        pp = [(cx + int(x*sc), cy + int(y*sc)) for x, y in pts]
        d.polygon(pp, fill=(225, 190, 38, alpha), outline=(133, 112, 18, alpha))
        d.line((cx-int(9*sc), cy, cx+int(9*sc), cy), fill=(255, 225, 73, alpha), width=max(1, int(sc)))
        d.line((cx-int(7*sc), cy-int(3*sc), cx+int(6*sc), cy+int(3*sc)),
               fill=(176, 144, 24, 210), width=max(1, int(sc)))
    return layer.resize((W, H), Image.Resampling.LANCZOS)


def add_bag_contents(im, deposited):
    """Render accumulated produce under a translucent bag film."""
    out = im.convert("RGBA")
    slots = {"apple": (111, 161), "orange": (128, 164), "star": (120, 149)}
    for kind in deposited:
        out = Image.alpha_composite(out, fruit_sprite(kind, slots[kind], 0.86, 235))

    film = hi_layer()
    d = ImageDraw.Draw(film)
    bag = [P(p) for p in [(99,138), (108,130), (136,133), (148,151), (143,188), (100,188), (92,154)]]
    d.polygon(bag, fill=(220, 231, 229, 23), outline=(230, 239, 236, 145))
    d.line([P((98,144)), P((109,166)), P((106,185))], fill=(242,247,244,125), width=1*SCALE)
    d.line([P((140,139)), P((132,158)), P((140,184))], fill=(232,240,238,110), width=1*SCALE)
    d.arc((*P((101,128)), *P((144,146))), 185, 350, fill=(240,246,241,180), width=2*SCALE)
    film = film.filter(ImageFilter.GaussianBlur(0.25*SCALE)).resize((W,H), Image.Resampling.LANCZOS)
    return Image.alpha_composite(out, film)


def main():
    source = np.asarray(Image.open(ROOT / "first_frame.png").convert("RGB"))
    clean = make_clean_plate(source)

    # One pose per frame.  "held" means the fruit is rigidly carried just left of
    # the wrist; deposits persist visibly in the transparent cart bag.
    poses = [
        ((154, 92), False, None, ()),                   # frame 00 replaced verbatim below
        ((116, 102), False, None, ()),
        ((83, 112), True, "apple", ()),
        ((96, 127), True, "apple", ()),
        ((117, 148), True, "apple", ()),
        ((150, 118), False, None, ("apple",)),
        ((181, 110), False, None, ("apple",)),
        ((205, 111), True, "orange", ("apple",)),
        ((183, 126), True, "orange", ("apple",)),
        ((151, 140), True, "orange", ("apple",)),
        ((125, 151), True, "orange", ("apple",)),
        ((159, 117), False, None, ("apple", "orange")),
        ((142, 105), False, None, ("apple", "orange")),
        ((123, 99), True, "star", ("apple", "orange")),
        ((124, 116), True, "star", ("apple", "orange")),
        ((124, 134), True, "star", ("apple", "orange")),
        ((121, 149), True, "star", ("apple", "orange")),
        ((151, 118), False, None, ("apple", "orange", "star")),
        ((170, 102), False, None, ("apple", "orange", "star")),
        ((154, 92), False, None, ("apple", "orange", "star")),
    ]

    frames = []
    removed = set()
    for index, (grip, closed, held, deposited) in enumerate(poses):
        if index == 0:
            frames.append(source.copy())
            continue
        if held:
            removed.add(held)
        plate = clean.copy()
        for kind in removed:
            plate = remove_fruit(plate, kind)
        im = Image.fromarray(plate)
        im = draw_robot(im, grip, closed, held)
        if held:
            # Fruit sits centered between the two rigid jaws.
            im = Image.alpha_composite(im.convert("RGBA"), fruit_sprite(held, (grip[0]-17, grip[1]), 0.78))
        im = add_bag_contents(im, deposited)
        frames.append(np.asarray(im.convert("RGB"), dtype=np.uint8))

    frames = np.stack(frames).astype(np.uint8)
    assert frames.shape == (20, H, W, 3)
    assert np.array_equal(frames[0], source)
    artifact_io.save(frames, ROOT)


if __name__ == "__main__":
    main()

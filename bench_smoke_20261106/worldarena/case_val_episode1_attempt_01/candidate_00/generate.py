from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import artifact_io


ROOT = Path(__file__).resolve().parent
W, H = 256, 192
SS = 4


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def scaled_point(p):
    return (int(round(p[0] * SS)), int(round(p[1] * SS)))


def scaled_points(points):
    return [scaled_point(p) for p in points]


def line(draw, points, fill, width, joint="curve"):
    draw.line(scaled_points(points), fill=fill, width=max(1, int(width * SS)), joint=joint)


def ellipse(draw, box, fill, outline=None, width=1):
    b = tuple(int(round(v * SS)) for v in box)
    draw.ellipse(b, fill=fill, outline=outline, width=max(1, int(width * SS)))


def polygon(draw, points, fill, outline=None, width=1):
    ps = scaled_points(points)
    draw.polygon(ps, fill=fill)
    if outline is not None:
        draw.line(ps + [ps[0]], fill=outline, width=max(1, int(width * SS)), joint="curve")


def unit(vx, vy):
    d = math.hypot(vx, vy) or 1.0
    return vx / d, vy / d


def segment_poly(a, b, r0, r1):
    ux, uy = unit(b[0] - a[0], b[1] - a[1])
    nx, ny = -uy, ux
    return [
        (a[0] + nx * r0, a[1] + ny * r0),
        (b[0] + nx * r1, b[1] + ny * r1),
        (b[0] - nx * r1, b[1] - ny * r1),
        (a[0] - nx * r0, a[1] - ny * r0),
    ]


def ik_elbow(shoulder, target, l1=96.0, l2=96.0):
    vx, vy = target[0] - shoulder[0], target[1] - shoulder[1]
    d = min(l1 + l2 - 0.05, max(abs(l1 - l2) + 0.05, math.hypot(vx, vy)))
    ux, uy = vx / d, vy / d
    along = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    height = math.sqrt(max(0.0, l1 * l1 - along * along))
    # The positive normal keeps the elbow low in the workspace, like the
    # folded left arm visible at the edge of the supplied first frame.
    nx, ny = -uy, ux
    return (shoulder[0] + ux * along + nx * height,
            shoulder[1] + uy * along + ny * height)


def make_clean_background(base):
    """Remove only the tiny initial arm sliver for later articulated frames."""
    arr = np.asarray(base).copy()
    # Reconstruct the pale wall row-by-row from nearby undisturbed wall.
    for y in range(94, 160):
        sample = np.median(arr[y, 18:34].astype(np.float32), axis=0)
        for x in range(0, 18):
            # Preserve the gentle left/right illumination falloff.
            shade = (x - 17) * 0.08
            arr[y, x] = np.clip(sample + shade, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def door_geometry(open_amount):
    # Perspective projection of a rigid door rotating about its left hinge.
    theta = math.radians(76.0) * open_amount
    perspective = 1.0 / (1.0 - 0.32 * math.sin(theta))
    hinge_x, center_y = 53.5, 35.5
    outer_x = hinge_x + 62.5 * math.cos(theta) * perspective
    outer_top = center_y + (16.0 - center_y) * perspective
    outer_bottom = center_y + (55.5 - center_y) * perspective
    quad = [(52.5, 13.5), (outer_x, outer_top),
            (outer_x, outer_bottom), (59.5, 58.0)]
    handle = (outer_x - 2.0, lerp(35.5, center_y + 1.0, open_amount))
    return quad, handle, perspective


def draw_microwave_opening(layer, amount):
    d = ImageDraw.Draw(layer, "RGBA")

    # The inner frame and dark cavity become visible behind the swinging door.
    polygon(d, [(52.5, 13.5), (117.5, 15.5), (117.0, 56.5), (59.5, 58.0)],
            (108, 109, 108, 255), (205, 205, 202, 255), 1.2)
    polygon(d, [(56.5, 17.2), (113.5, 18.5), (113.0, 52.5), (62.0, 53.5)],
            (26, 28, 29, 255), (63, 65, 65, 255), 1.0)
    # Rear wall and glass turntable highlights give the cavity readable depth.
    polygon(d, [(61.0, 21.0), (108.5, 21.7), (106.0, 46.8), (64.0, 47.5)],
            (38, 40, 40, 255))
    line(d, [(66, 46), (103, 45.5)], (105, 108, 106, 145), 1.2)
    line(d, [(64, 23), (107, 23)], (89, 92, 91, 90), 0.8)

    quad, handle, persp = door_geometry(amount)

    # A fixed rigid quadrilateral is shaded as a glossy dark-gray glass door.
    polygon(d, quad, (49, 52, 53, 255), (122, 124, 122, 255), 1.3)
    q0, q1, q2, q3 = quad
    inset = 0.12
    inner = [
        (lerp(q0[0], q1[0], inset), lerp(q0[1], q3[1], inset)),
        (lerp(q1[0], q0[0], inset), lerp(q1[1], q2[1], inset)),
        (lerp(q2[0], q3[0], inset), lerp(q2[1], q1[1], inset)),
        (lerp(q3[0], q2[0], inset), lerp(q3[1], q0[1], inset)),
    ]
    polygon(d, inner, (56, 59, 60, 235), (34, 36, 37, 220), 0.8)

    # Soft reflection streak moves with the planar door.
    top_a = (lerp(q0[0], q1[0], .22), lerp(q0[1], q1[1], .22))
    top_b = (lerp(q0[0], q1[0], .72), lerp(q0[1], q1[1], .72))
    bot_a = (lerp(q3[0], q2[0], .22), lerp(q3[1], q2[1], .22))
    bot_b = (lerp(q3[0], q2[0], .72), lerp(q3[1], q2[1], .72))
    polygon(d, [top_a, top_b, bot_b, bot_a], (165, 168, 166, 27))

    # The original vertical outer-edge handle remains rigidly on the door.
    hx = handle[0]
    span = 9.0 * min(1.28, persp)
    line(d, [(hx, handle[1] - span), (hx, handle[1] + span)],
         (25, 27, 28, 255), 3.2)
    line(d, [(hx - .65, handle[1] - span + .8),
             (hx - .65, handle[1] + span - .8)], (145, 148, 146, 150), .65)
    return handle


def draw_arm(layer, shoulder, elbow, wrist, grip_angle=0.0):
    # Diffuse contact shadow on the pale wall.
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, "RGBA")
    line(sd, [(shoulder[0] + 3, shoulder[1] + 4),
              (elbow[0] + 3, elbow[1] + 4)], (35, 35, 35, 65), 15)
    line(sd, [(elbow[0] + 3, elbow[1] + 4),
              (wrist[0] + 3, wrist[1] + 4)], (35, 35, 35, 55), 12)
    shadow = shadow.filter(ImageFilter.GaussianBlur(2.4 * SS))
    layer.alpha_composite(shadow)

    d = ImageDraw.Draw(layer, "RGBA")
    upper = segment_poly(shoulder, elbow, 9.0, 8.0)
    fore = segment_poly(elbow, wrist, 7.2, 5.5)

    polygon(d, upper, (31, 34, 36, 255), (10, 12, 13, 255), 1.2)
    polygon(d, fore, (55, 59, 61, 255), (14, 16, 17, 255), 1.1)

    # Narrow specular strips reinforce that the links are rigid metal shells.
    line(d, [
        (lerp(shoulder[0], elbow[0], .12), lerp(shoulder[1], elbow[1], .12) - 2),
        (lerp(shoulder[0], elbow[0], .82), lerp(shoulder[1], elbow[1], .82) - 2),
    ], (142, 148, 149, 105), 1.2)
    line(d, [
        (lerp(elbow[0], wrist[0], .13), lerp(elbow[1], wrist[1], .13) - 1.2),
        (lerp(elbow[0], wrist[0], .78), lerp(elbow[1], wrist[1], .78) - 1.2),
    ], (179, 183, 182, 110), 1.0)

    ellipse(d, (elbow[0] - 10, elbow[1] - 10, elbow[0] + 10, elbow[1] + 10),
            (23, 26, 28, 255), (9, 10, 11, 255), 1.1)
    ellipse(d, (elbow[0] - 5.2, elbow[1] - 5.2, elbow[0] + 5.2, elbow[1] + 5.2),
            (94, 99, 101, 255), (174, 178, 177, 190), .7)
    ellipse(d, (shoulder[0] - 11, shoulder[1] - 11, shoulder[0] + 11, shoulder[1] + 11),
            (22, 24, 25, 255), (7, 8, 9, 255), 1.0)

    # Wrist collar aligns with the final link.
    ux, uy = unit(wrist[0] - elbow[0], wrist[1] - elbow[1])
    nx, ny = -uy, ux
    back = (wrist[0] - ux * 8, wrist[1] - uy * 8)
    collar = [(back[0] + nx * 6.7, back[1] + ny * 6.7),
              (wrist[0] + nx * 5.2, wrist[1] + ny * 5.2),
              (wrist[0] - nx * 5.2, wrist[1] - ny * 5.2),
              (back[0] - nx * 6.7, back[1] - ny * 6.7)]
    polygon(d, collar, (18, 20, 21, 255), (112, 117, 118, 220), .8)

    # Two compact fingers visibly wrap around the vertical microwave handle.
    palm = (wrist[0] - 1.2, wrist[1])
    ellipse(d, (palm[0] - 5.8, palm[1] - 5.8, palm[0] + 5.8, palm[1] + 5.8),
            (21, 23, 24, 255), (8, 9, 10, 255), .8)
    line(d, [(palm[0] - 1, palm[1] - 3), (palm[0] + 4.2, palm[1] - 6.8)],
         (16, 18, 19, 255), 3.5)
    line(d, [(palm[0] - 1, palm[1] + 3), (palm[0] + 4.2, palm[1] + 6.8)],
         (16, 18, 19, 255), 3.5)
    ellipse(d, (palm[0] + 2.2, palm[1] - 8.2, palm[0] + 5.7, palm[1] - 4.7),
            (92, 97, 98, 255))
    ellipse(d, (palm[0] + 2.2, palm[1] + 4.7, palm[0] + 5.7, palm[1] + 8.2),
            (92, 97, 98, 255))


def composite_overlay(background, overlay):
    small = overlay.resize((W, H), Image.Resampling.LANCZOS)
    return Image.alpha_composite(background.convert("RGBA"), small).convert("RGB")


def render_frame(index, original, clean):
    if index == 0:
        return np.asarray(original).copy()

    # Remove the supplied edge sliver gently over the first few frames.
    bg_mix = smoothstep(index / 7.0)
    background = Image.blend(original, clean, bg_mix)
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))

    shoulder = (-12.0, 146.0)
    reach = smoothstep((index - 4) / 43.0)
    closed_handle = (113.0, 35.8)
    initial_wrist = (8.0, 120.0)
    wrist = (lerp(initial_wrist[0], closed_handle[0], reach),
             lerp(initial_wrist[1], closed_handle[1], reach))

    opening = smoothstep((index - 58) / 45.0)
    if opening > 0:
        handle = draw_microwave_opening(layer, opening)
        wrist = handle

    elbow = ik_elbow(shoulder, wrist)
    draw_arm(layer, shoulder, elbow, wrist)

    # A short stable hold after opening makes the completed action unambiguous.
    return np.asarray(composite_overlay(background, layer), dtype=np.uint8)


def main():
    original = Image.open(ROOT / "first_frame.png").convert("RGB")
    clean = make_clean_background(original)
    frames = np.stack([render_frame(i, original, clean) for i in range(121)])
    assert np.array_equal(frames[0], np.asarray(original)), "frame zero changed"
    artifact_io.save(frames, ROOT)


if __name__ == "__main__":
    main()

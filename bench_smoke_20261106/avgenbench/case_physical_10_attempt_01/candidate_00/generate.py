from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import artifact_io


ROOT = Path(__file__).resolve().parent
W, H, N, FPS = 256, 144, 40, 4
S = 3


def C(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def pts(values):
    return [(round(x * S), round(y * S)) for x, y in values]


def ellipse(draw, box, fill, outline=None, width=1):
    draw.ellipse(tuple(round(v * S) for v in box), fill=fill, outline=outline, width=width * S)


def rect(draw, box, fill, outline=None, width=1, radius=0):
    box = tuple(round(v * S) for v in box)
    if radius:
        draw.rounded_rectangle(box, radius=radius * S, fill=fill, outline=outline, width=width * S)
    else:
        draw.rectangle(box, fill=fill, outline=outline, width=width * S)


def line(draw, values, fill, width=1):
    draw.line(pts(values), fill=fill, width=width * S, joint="curve")


def polygon(draw, values, fill, outline=None):
    draw.polygon(pts(values), fill=fill, outline=outline)


def static_scene():
    im = Image.new("RGB", (W * S, H * S), C("#d8c8b6"))
    d = ImageDraw.Draw(im)

    # Warm wall with soft vertical illumination.
    for y in range(0, 108 * S):
        yy = y / S
        shade = int(232 - 0.19 * yy)
        d.line([(0, y), (W * S, y)], fill=(shade, shade - 9, shade - 22))
    rect(d, (0, 106, 256, 111), C("#b99b7d"))
    rect(d, (0, 111, 256, 144), C("#9c7859"))
    # Rug and floorboards.
    polygon(d, [(48, 119), (221, 119), (252, 144), (20, 144)], C("#638b87"))
    for x in (22, 79, 185, 238):
        line(d, [(x, 111), (x - 10, 144)], C("#8b684e"), 1)
    for y in (128, 136):
        line(d, [(0, y), (256, y + 1)], C("#8b684e"), 1)

    # Framed prints.
    rect(d, (87, 14, 114, 45), C("#6c5140"), radius=1)
    rect(d, (90, 17, 111, 42), C("#f3ede3"))
    polygon(d, [(91, 39), (98, 28), (103, 34), (108, 24), (111, 39)], C("#87a590"))
    ellipse(d, (94, 20, 99, 25), C("#dcad55"))
    rect(d, (119, 18, 145, 43), C("#6c5140"), radius=1)
    rect(d, (122, 21, 142, 40), C("#f0e4d4"))
    ellipse(d, (128, 25, 136, 33), C("#d28d69"))
    line(d, [(123, 37), (130, 31), (136, 37), (142, 29)], C("#77959a"), 2)

    # Window and curtains on the right.
    rect(d, (183, 8, 235, 75), C("#725744"), radius=1)
    rect(d, (186, 11, 232, 72), C("#a9d3db"))
    for yy in range(12, 72):
        col = (174 + yy // 8, 215 + yy // 18, 222 + yy // 20)
        d.line([(187 * S, yy * S), (231 * S, yy * S)], fill=col)
    line(d, [(209, 11), (209, 72)], C("#f0e4cf"), 2)
    line(d, [(186, 43), (232, 43)], C("#f0e4cf"), 2)
    polygon(d, [(178, 6), (190, 7), (188, 75), (176, 83)], C("#b86756"))
    polygon(d, [(229, 6), (241, 8), (243, 83), (232, 74)], C("#b86756"))
    line(d, [(179, 7), (178, 77)], C("#d48670"), 2)
    line(d, [(238, 8), (240, 78)], C("#914a43"), 2)

    # Sofa behind the experiment.
    rect(d, (9, 75, 105, 119), C("#637d73"), outline=C("#425c56"), width=2, radius=9)
    rect(d, (5, 90, 111, 121), C("#6f8c80"), outline=C("#425c56"), width=2, radius=7)
    line(d, [(54, 91), (54, 119)], C("#526e65"), 1)
    rect(d, (15, 80, 50, 104), C("#ca8e59"), radius=5)
    line(d, [(18, 84), (45, 100)], C("#e3ad70"), 1)
    line(d, [(45, 83), (18, 101)], C("#a76c47"), 1)
    rect(d, (61, 81, 94, 104), C("#d2c5a6"), radius=5)
    line(d, [(65, 94), (90, 94)], C("#b4a582"), 1)

    # Side table and lamp.
    rect(d, (224, 90, 253, 95), C("#5f4435"), radius=2)
    rect(d, (228, 95, 231, 121), C("#5f4435"))
    rect(d, (247, 95, 250, 121), C("#5f4435"))
    line(d, [(239, 91), (239, 67)], C("#5e5047"), 2)
    polygon(d, [(227, 68), (251, 68), (247, 53), (231, 53)], C("#edc274"), outline=C("#a77c45"))
    ellipse(d, (236, 49, 242, 54), C("#f7dfa4"))
    return im


def draw_air(d, frame):
    # Semi-transparent-looking blue-white streaks converge on the nozzle.
    phase = frame * 0.83
    for k in range(8):
        y0 = 92 - ((frame * 7 + k * 13) % 58)
        xoff = math.sin(phase + k * 1.7) * (2.3 + 0.15 * k)
        x = 137 + xoff
        length = 7 + (k % 3) * 3
        alpha_col = (190 + (k % 2) * 25, 224 + (k % 2) * 14, 236 + (k % 2) * 11)
        line(d, [(x, y0 + length), (x + math.sin(phase + k) * 1.2, y0)], alpha_col, 1)
    # Curving outer boundaries communicate a narrow upward jet.
    for side in (-1, 1):
        wig = math.sin(frame * 0.6 + side) * 1.2
        line(d, [(137 + side * 7, 96), (137 + side * (9 + wig), 78),
                 (137 + side * (11 - wig), 61), (137 + side * (13 + wig), 43)],
             C("#c7e3e9"), 1)


def draw_dryer_and_holder(d, frame):
    # Cast shadow.
    ellipse(d, (106, 134, 194, 143), C("#725746"))

    # Right hand continuously holding the dryer's handle.
    polygon(d, [(256, 120), (236, 112), (216, 113), (194, 125), (197, 144), (256, 144)], C("#c47c58"))
    line(d, [(235, 116), (215, 120), (203, 130)], C("#995a42"), 1)
    polygon(d, [(221, 112), (199, 106), (181, 109), (183, 116), (205, 120)], C("#d99168"))
    line(d, [(202, 111), (183, 112)], C("#a96348"), 1)

    # Vertical hair dryer: nozzle, motor housing, intake, and angled handle.
    polygon(d, [(127, 105), (146, 105), (151, 116), (145, 127), (130, 127), (122, 117)], C("#176f91"), outline=C("#123f57"))
    rect(d, (125, 94, 148, 108), C("#278fb1"), outline=C("#123f57"), width=2, radius=4)
    polygon(d, [(129, 95), (132, 82), (144, 82), (147, 95)], C("#237f9f"), outline=C("#123f57"))
    rect(d, (131, 78, 145, 84), C("#142f43"), outline=C("#0b2332"), width=1, radius=2)
    rect(d, (133, 79, 143, 81.5), C("#8ec5d3"), radius=1)
    ellipse(d, (128, 99, 146, 117), C("#236e8a"), outline=C("#0e3d54"), width=1)
    ellipse(d, (131, 102, 143, 114), C("#173f56"), outline=C("#62b7cb"), width=1)
    for a in range(0, 360, 45):
        cx, cy = 137, 108
        line(d, [(cx, cy), (cx + 5 * math.cos(math.radians(a)), cy + 5 * math.sin(math.radians(a)))], C("#72bacc"), 1)
    # Handle projects down-right into holding hand.
    polygon(d, [(145, 113), (157, 112), (169, 137), (157, 140), (145, 121)], C("#176b8a"), outline=C("#123f57"))
    rect(d, (149, 116, 154, 122), C("#e47a47"), outline=C("#743b2b"), width=1, radius=1)
    # Fingers wrapped over handle.
    for j in range(3):
        y = 122 + j * 4
        line(d, [(178, y + 4), (161, y), (154, y + 1)], C("#dc926a"), 4)
        line(d, [(161, y + 2), (155, y + 2)], C("#9c5d45"), 1)
    # Power cord.
    line(d, [(162, 139), (169, 143), (180, 144)], C("#263139"), 2)

    # Sound marks pulse beside the running motor.
    pulse = 1 if frame % 2 == 0 else 0
    for j in range(3):
        x = 153 + j * 4 + pulse
        line(d, [(x, 88 - j * 2), (x + 2, 91), (x, 94 + j * 2)], C("#f0b34d"), 1)


def ball_position(frame):
    # The placing hand carries the ball into the jet, then the ball stays supported.
    if frame <= 10:
        u = frame / 10
        u = u * u * (3 - 2 * u)
        return 72 + 65 * u, 62 - 8 * u
    q = frame - 11
    # Coupled lateral wobble and smaller vertical bob, centered over the nozzle.
    return (137 + 4.7 * math.sin(q * 0.82) + 1.4 * math.sin(q * 1.91),
            53.5 + 2.7 * math.sin(q * 0.58 + 0.7) + 0.8 * math.sin(q * 1.47))


def draw_placing_hand(d, frame, bx, by):
    if frame > 20:
        return
    if frame <= 10:
        palm_x = bx - 30
        openness = 0
    else:
        u = min(1, (frame - 11) / 9)
        palm_x = (bx - 30) * (1 - u) + (-24) * u
        openness = math.sin(min(1, (frame - 11) / 4) * math.pi / 2)
    palm_y = 67 + (palm_x - 42) * 0.035

    # Forearm enters from left; sleeve ends before the palm.
    polygon(d, [(-8, palm_y - 13), (palm_x - 11, palm_y - 10),
                (palm_x - 9, palm_y + 9), (-8, palm_y + 14)], C("#784e7e"))
    line(d, [(-2, palm_y - 9), (palm_x - 12, palm_y - 7)], C("#a46f9e"), 2)
    ellipse(d, (palm_x - 14, palm_y - 10, palm_x + 13, palm_y + 11), C("#d8946c"), outline=C("#9d6048"), width=1)

    if frame <= 10:
        # Pinch grip: fingers visibly touch the ball at top and bottom.
        polygon(d, [(palm_x + 2, palm_y - 8), (bx - 4, by - 6), (bx - 2, by - 3),
                    (palm_x + 5, palm_y - 2)], C("#dd9c76"), outline=C("#9d6048"))
        polygon(d, [(palm_x + 6, palm_y + 2), (bx - 3, by + 3), (bx - 3, by + 7),
                    (palm_x + 2, palm_y + 9)], C("#d68d68"), outline=C("#9d6048"))
    else:
        # Fingers open immediately at release, then withdraw.
        spread = 5 + 9 * openness
        polygon(d, [(palm_x + 3, palm_y - 7), (palm_x + 23, palm_y - spread),
                    (palm_x + 25, palm_y - spread + 4), (palm_x + 7, palm_y - 1)], C("#dd9c76"), outline=C("#9d6048"))
        polygon(d, [(palm_x + 5, palm_y + 1), (palm_x + 24, palm_y + spread - 4),
                    (palm_x + 22, palm_y + spread), (palm_x + 2, palm_y + 7)], C("#d68d68"), outline=C("#9d6048"))


def draw_ball(d, bx, by, frame):
    # A faint displaced shadow gives the sphere volume against the bright wall.
    ellipse(d, (bx - 7.0, by - 6.2, bx + 7.6, by + 8.2), C("#907f72"))
    ellipse(d, (bx - 7, by - 7, bx + 7, by + 7), C("#f8f5e9"), outline=C("#6a6b68"), width=1)
    ellipse(d, (bx - 4.3, by - 4.5, bx - 0.8, by - 1.0), C("#ffffff"))
    # Tiny seam/brand-like dash rotates as it wobbles.
    ang = frame * 0.55
    line(d, [(bx + 1.3 * math.cos(ang), by + 1.3 * math.sin(ang)),
             (bx + 3.8 * math.cos(ang), by + 3.8 * math.sin(ang))], C("#c3c0b7"), 1)


def make_frame(base, frame):
    im = base.copy()
    d = ImageDraw.Draw(im)

    draw_air(d, frame)
    bx, by = ball_position(frame)
    draw_placing_hand(d, frame, bx, by)
    # Ball is in front of the fingers while being pinched and free thereafter.
    draw_ball(d, bx, by, frame)
    draw_dryer_and_holder(d, frame)

    # A tiny moving tuft on a nearby cushion reinforces the upward draft.
    sway = math.sin(frame * 0.85) * 2.0
    line(d, [(102, 84), (105 + sway, 78), (102 + sway, 73)], C("#d7c3a0"), 1)

    return np.asarray(im.resize((W, H), Image.Resampling.LANCZOS), dtype=np.uint8)


def make_audio():
    sr = 24000
    count = 240000
    t = np.arange(count, dtype=np.float64) / sr
    rng = np.random.default_rng(31010)
    white = rng.normal(0, 1, count + 8)
    # Several short differences create a strong, broad rushing-air spectrum.
    rush = (white[8:] - 0.48 * white[7:-1] - 0.22 * white[5:-3]
            + 0.16 * white[:-8])
    # Smoothed low component plus motor fundamentals make it read as a dryer.
    kernel = np.ones(17) / 17
    low = np.convolve(white[8:], kernel, mode="same")
    motor = (0.33 * np.sin(2 * np.pi * 178 * t + 0.06 * np.sin(2 * np.pi * 0.7 * t))
             + 0.16 * np.sin(2 * np.pi * 356 * t)
             + 0.08 * np.sin(2 * np.pi * 712 * t))
    signal = 0.58 * rush + 0.44 * low + motor
    # Constant roar with only a very short click-free start/end ramp.
    env = np.ones(count)
    fade = 360
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    signal *= env
    signal /= max(1e-9, np.max(np.abs(signal)))
    return np.round(signal * 28000).astype(np.int16)[:, None]


def save_candidate_sheet(frames):
    chosen = [0, 7, 11, 16, 23, 30, 35, 39]
    sheet = Image.new("RGB", (W * 4, H * 2), (20, 20, 20))
    for i, idx in enumerate(chosen):
        sheet.paste(Image.fromarray(frames[idx]), ((i % 4) * W, (i // 4) * H))
    sheet.save(ROOT / "candidate_01.png")


def main():
    base = static_scene()
    frames = np.stack([make_frame(base, i) for i in range(N)]).astype(np.uint8)
    save_candidate_sheet(frames)
    audio = make_audio()
    artifact_io.save(frames, ROOT, audio=audio)


if __name__ == "__main__":
    main()

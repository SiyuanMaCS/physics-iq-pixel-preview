"""Draw the future RGB frames from the supplied single historical image.

The table is repaired with nearby wood pixels. Two extracted, softly matted
balls follow explicit rolling trajectories with a symmetric collision.
Only numpy/PIL pixel operations are used.
"""
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw

import artifact_io


ROOT = Path(__file__).resolve().parent
H, W = 144, 256
YY, XX = np.mgrid[:H, :W].astype(float)


def sample(a, x, y):
    x = np.clip(x, 0, a.shape[1] - 1.001)
    y = np.clip(y, 0, a.shape[0] - 1.001)
    ix, iy = x.astype(int), y.astype(int)
    u, v = (x - ix)[..., None], (y - iy)[..., None]
    return ((1-u)*(1-v)*a[iy, ix] + u*(1-v)*a[iy, ix+1]
            + (1-u)*v*a[iy+1, ix] + u*v*a[iy+1, ix+1])


def ellipse(x, y, rx, ry, feather=0.18):
    distance = np.sqrt(((XX-x)/rx)**2 + ((YY-y)/ry)**2)
    z = np.clip((1+feather-distance)/(2*feather), 0, 1)
    return z*z*(3-2*z)


def repair(original, mask):
    """Continue the wood's rows across the occluded ball and shadow regions."""
    slope = 0.067
    aligned = sample(original, XX, YY+slope*(XX-128))
    aligned_mask = sample(mask[...,None], XX, YY+slope*(XX-128))[...,0]
    filled = aligned.copy()
    for y in range(H):
        valid = aligned_mask[y] < 0.001
        for channel in range(3):
            filled[y,:,channel] = np.interp(np.arange(W),
                                           np.flatnonzero(valid),
                                           aligned[y,valid,channel])
    restored = sample(filled, XX, YY-slope*(XX-128))
    return original*(1-mask[...,None]) + restored*mask[...,None]


def prepare():
    original = np.asarray(Image.open(ROOT/'first_frame.png').convert('RGB'), float)
    blue_mask = np.maximum(ellipse(76.5, 49.7, 10.2, 9.1),
                           ellipse(81.0, 49.8, 24.0, 10.5))
    yellow_mask = np.maximum(ellipse(148.8, 58.1, 10.2, 10.0),
                             ellipse(144.2, 58.0, 17.4, 10.0))
    bg = repair(original, np.maximum(blue_mask, yellow_mask))
    objects = []
    for center, radii, mask in [((76.5,49.7),(7.35,7.05),blue_mask),
                                ((148.8,58.1),(8.0,7.65),yellow_mask)]:
        cx, cy = center
        # The larger matte includes the optical softness of the reference.
        body = ellipse(cx, cy, radii[0], radii[1], feather=0.21)
        diff = original-bg
        foreground = np.clip(bg + diff/np.maximum(body[..., None], 0.12), 0, 255)
        # Cast shadows retain their observed shape and opposing light sources.
        ratio = np.clip(original/np.maximum(bg, 1), 0.45, 1.05)
        shadow = (ratio-1)*mask[..., None]*(1-body[..., None])
        objects.append((center, body, foreground, shadow))
    Image.fromarray(bg.clip(0,255).astype('uint8')).save(ROOT/'.tmp/table_repaired.png')
    return bg, objects


def trajectory(t):
    # Motion is specified directly, in pixels and seconds, without a simulator.
    left, right = 76.5, 148.8
    speed, drag = 28.2, 1.0
    contact_gap = 15.15
    travel = (right-left-contact_gap)/2
    hit = (speed-math.sqrt(speed*speed-2*drag*travel))/drag
    if t <= hit:
        d = speed*t-0.5*drag*t*t
        return left+d, right-d
    u = t-hit
    rebound = 0.84*(speed-drag*hit)
    rolling_drag = 3.2
    u = min(u, rebound/rolling_drag)
    d = rebound*u-0.5*rolling_drag*u*u
    return left+travel-d, right-travel+d


def draw_ball(frame, obj, position, blue=False):
    (cx, cy), matte, foreground, shadow = obj
    x, y = position
    sx, sy = XX-(x-cx), YY-(y-cy)
    moved_shadow = sample(shadow, sx, sy)
    frame = frame*(1+moved_shadow)
    alpha = sample(matte[...,None], sx, sy)
    if blue:
        # A small texture turn changes the visible seam during rolling while
        # retaining the photographic softness and color of the original ball.
        angle = 0.42*math.sin((x-cx)/7.35)
        ux, uy = sx-cx, sy-cy
        tx = cx+math.cos(angle)*ux-math.sin(angle)*uy
        ty = cy+math.sin(angle)*ux+math.cos(angle)*uy
        color = sample(foreground, tx, ty)
    else:
        color = sample(foreground, sx, sy)
    return frame*(1-alpha)+color*alpha


def generate():
    bg, objects = prepare()
    frames = []
    for i in range(20):
        # The historical sample is at 2.9667 seconds; prediction starts at 3.
        t = 1/30+i/4
        xb, xy = trajectory(t)
        yb = 49.7+(xb-76.5)*0.1162
        yy = 58.1+(xy-148.8)*0.1162
        frame = bg.copy()
        frame = draw_ball(frame, objects[0], (xb,yb), blue=True)
        frame = draw_ball(frame, objects[1], (xy,yy))
        frames.append(np.rint(frame).clip(0,255).astype(np.uint8))
    frames = np.stack(frames)
    candidate = ROOT/'candidate_03'
    candidate.mkdir(exist_ok=True)
    np.save(candidate/'rgb.npy', frames)
    for i in [0,2,3,4,8,12,16,19]:
        Image.fromarray(frames[i]).save(candidate/f'{i:02d}.png')
    sheet = Image.new('RGB',(W*4,(H+20)*2),(25,28,35))
    draw = ImageDraw.Draw(sheet)
    for j,i in enumerate([0,2,3,4,8,12,16,19]):
        x,y = j%4*W,j//4*(H+20)
        draw.text((x+4,y+3),f'{i/4:.2f}s',fill='white')
        sheet.paste(Image.fromarray(frames[i]),(x,y+20))
    sheet.save(candidate/'inspection.png')
    artifact_io.save(frames, ROOT)


if __name__ == '__main__':
    generate()

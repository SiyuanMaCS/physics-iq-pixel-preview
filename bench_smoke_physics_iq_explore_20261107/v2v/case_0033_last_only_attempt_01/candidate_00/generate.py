"""Photographic pixel compositing and analytic motion for two rolling balls."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import artifact_io

ROOT = Path(__file__).resolve().parent
SOURCE = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'), dtype=float)
H, W = SOURCE.shape[:2]
Y, X = np.mgrid[:H, :W].astype(float)


def sample(a, x, y):
    x = np.clip(x, 0, a.shape[1] - 1.001)
    y = np.clip(y, 0, a.shape[0] - 1.001)
    ix, iy = x.astype(int), y.astype(int)
    fx, fy = (x - ix)[..., None], (y - iy)[..., None]
    return ((1-fx)*(1-fy)*a[iy, ix] + fx*(1-fy)*a[iy, ix+1]
            + (1-fx)*fy*a[iy+1, ix] + fx*fy*a[iy+1, ix+1])


def blur(a, radius):
    return np.asarray(Image.fromarray(np.uint8(np.clip(a, 0, 255))).filter(
        ImageFilter.GaussianBlur(radius)), dtype=float)


def clean_table():
    """Fill the two small occupied regions from nearby aligned wood texture."""
    clean = SOURCE.copy()
    for cx, cy, rx, ry, dx in [(42, 46, 20, 12, 37), (198, 65, 22, 16, -42)]:
        r = np.sqrt(((X-cx)/rx)**2 + ((Y-cy)/ry)**2)
        donor = sample(SOURCE, X+dx, Y+dx*0.095)
        ring = (r > 0.92) & (r < 1.4)
        # A smooth color correction preserves the illumination gradient.
        basis = np.stack([np.ones_like(X), (X-cx)/rx, (Y-cy)/ry], -1)
        difference = blur(SOURCE, 2) - blur(donor, 2)
        fit = np.linalg.lstsq(basis[ring], difference[ring], rcond=None)[0]
        donor += basis @ fit
        weight = np.clip((1.18-r)/0.25, 0, 1)[..., None]
        clean = clean*(1-weight) + donor*weight
    return np.clip(clean, 0, 255)


BACKGROUND = clean_table()
BALLS = [dict(center=np.array([36.9, 46.6]), rx=7.3, ry=6.9, shadow=1),
         dict(center=np.array([201.8, 66.0]), rx=8.8, ry=8.1, shadow=-1)]


def ball_texture(ball):
    # Recover the observed color behind a soft silhouette so compositing does
    # not carry a rectangular patch of the original tabletop into the future.
    cx, cy = ball['center']
    u, v = (X-cx)/ball['rx'], (Y-cy)/ball['ry']
    r = np.sqrt(u*u + v*v)
    alpha = np.clip((1.15-r)/0.24, 0, 1)
    fg = BACKGROUND + (SOURCE-BACKGROUND)/np.maximum(alpha[..., None], 0.18)
    fg = np.clip(fg, 0, 255)
    # Fill beyond the silhouette with the nearest interior sphere color.
    q = np.maximum(r, 0.85)/0.85
    edge = sample(SOURCE, cx+(X-cx)/q, cy+(Y-cy)/q)
    fg = np.where((r > 1.12)[..., None], edge, fg)
    return fg


for ball in BALLS:
    ball['texture'] = ball_texture(ball)

start_b, start_y = [b['center'] for b in BALLS]
axis = start_y-start_b
distance = np.linalg.norm(axis)
axis /= distance
radius_sum = 16.1
contact_time = 1.75
drag_in = 4.0
travel_to_contact = (distance-radius_sum)/2
speed_in = travel_to_contact/contact_time + 0.5*drag_in*contact_time
speed_out = (speed_in-drag_in*contact_time)*0.86
drag_out = 8.0


def motion(t):
    if t <= contact_time:
        s = speed_in*t - 0.5*drag_in*t*t
        v = speed_in-drag_in*t
        return [start_b+axis*s, start_y-axis*s], [v, -v], s
    elapsed = min(t-contact_time, speed_out/drag_out)
    rebound = speed_out*elapsed - 0.5*drag_out*elapsed*elapsed
    v = max(0, speed_out-drag_out*elapsed)
    s = travel_to_contact-rebound
    return [start_b+axis*s, start_y-axis*s], [-v, v], s


def render_at(t):
    centers, velocities, travel = motion(t)
    frame = BACKGROUND.copy()
    # Soft cast shadows move with the balls and retain the two-sided lighting
    # that is visible in the photograph.
    for i, (ball, center) in enumerate(zip(BALLS, centers)):
        cx, cy = center
        scale = 1 + (cy-ball['center'][1])*0.0035
        shadow_dx = np.clip((119-cx)*0.118, -11, 11)
        sx, sy = cx+shadow_dx, cy-2.7
        rx, ry = ball['rx']*scale, ball['ry']*scale
        cast = np.exp(-0.5*(((X-sx)/(rx*0.9))**2 + ((Y-sy)/(ry*0.55))**2))
        contact = np.exp(-0.5*(((X-cx)/(rx*0.7))**2 + ((Y-(cy+ry*0.45))/(ry*0.29))**2))
        shade = np.clip(cast*0.30+contact*0.22, 0, 0.52)
        frame *= 1-shade[..., None]
    for i, (ball, center) in enumerate(zip(BALLS, centers)):
        cx, cy = center
        scale = 1 + (cy-ball['center'][1])*0.0035
        rx, ry = ball['rx']*scale, ball['ry']*scale
        u, v = (X-cx)/rx, (Y-cy)/ry
        r = np.sqrt(u*u+v*v)
        alpha = np.clip((1.15-r)/0.24, 0, 1)
        tex = sample(ball['texture'], ball['center'][0]+u*ball['rx'],
                     ball['center'][1]+v*ball['ry'])
        # A subtle rolling seam on the blue sphere: retain its photographed
        # diffuse shading and move only the fine surface color variation.
        if i == 0:
            smooth = sample(blur(ball['texture'], 1.8),
                            ball['center'][0]+u*ball['rx'],
                            ball['center'][1]+v*ball['ry'])
            theta = -travel/ball['rx']
            ur, vr = np.cos(theta)*u-np.sin(theta)*v, np.sin(theta)*u+np.cos(theta)*v
            rotating = sample(ball['texture'], ball['center'][0]+ur*ball['rx'],
                              ball['center'][1]+vr*ball['ry'])
            rotating_smooth = sample(blur(ball['texture'], 1.8),
                              ball['center'][0]+ur*ball['rx'],
                              ball['center'][1]+vr*ball['ry'])
            interior = np.clip((0.90-r)/0.25, 0, 1)[..., None]
            tex = tex*(1-interior) + (smooth+rotating-rotating_smooth)*interior
        frame = frame*(1-alpha[..., None]) + tex*alpha[..., None]
    return np.clip(frame, 0, 255)


def generate():
    candidate = ROOT / 'candidates' / '01'
    candidate.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(BACKGROUND)).save(candidate / 'background.png')
    frames = []
    trajectories = []
    for index in range(20):
        t = (index+1)/4.0
        # Short camera exposure adds a small, directionally correct blur.
        frame = np.mean([render_at(t+dt) for dt in [-0.012, 0, 0.012]], axis=0)
        frames.append(np.uint8(np.rint(frame)))
        trajectories.append(np.concatenate(motion(t)[0]))
    frames = np.stack(frames)
    np.save(candidate / 'rgb.npy', frames)
    np.save(candidate / 'trajectories.npy', trajectories)
    Image.fromarray(frames[0]).save(candidate / 'first_prediction.png')
    artifact_io.save(frames, ROOT)
    # Retain the candidate and the code used to produce it for inspection.
    (candidate / 'generate.py').write_text(Path(__file__).read_text())
    print('Saved', frames.shape, frames.dtype, 'at 4 fps')
    print('Collision at future t=', contact_time, '; initial speed=', round(speed_in, 2))


if __name__ == '__main__':
    generate()

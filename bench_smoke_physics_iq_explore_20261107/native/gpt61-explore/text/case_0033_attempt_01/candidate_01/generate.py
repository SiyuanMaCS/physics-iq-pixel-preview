"""Draw a photographic two-ball collision using only the supplied RGB frame."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import artifact_io

ROOT = Path(__file__).resolve().parent
SOURCE = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
H, W = SOURCE.shape[:2]
Y, X = np.mgrid[:H, :W]
BALLS = [dict(name='blue', center=np.array([76.25, 50.10]), rx=8.05, ry=7.85),
         dict(name='yellow', center=np.array([149.05, 58.30]), rx=9.55, ry=8.85)]
START_A, START_B = [b['center'] for b in BALLS]
DIRECTION = (START_B - START_A) / np.linalg.norm(START_B - START_A)
DISTANCE = float(np.linalg.norm(START_B - START_A))
CONTACT_TIME = 1.5
SCALE_SLOPE = 0.020


def reconstruct_table():
    """Interpolate through the original balls and their local shadows."""
    mask = (((X - 76.0) / 19.0)**2 + ((Y - 50.0) / 12.0)**2 < 1)
    mask |= (((X - 149.0) / 21.0)**2 + ((Y - 58.0) / 13.0)**2 < 1)
    points = np.argwhere(mask)
    ids = np.full((H, W), -1, dtype=int)
    ids[mask] = np.arange(len(points))
    matrix = lil_matrix((len(points), len(points)))
    rhs = np.zeros((len(points), 3))
    for k, (y, x) in enumerate(points):
        matrix[k, k] = 4
        for yy, xx in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
            if mask[yy, xx]:
                matrix[k, ids[yy, xx]] = -1
            else:
                rhs[k] += SOURCE[yy, xx].astype(float)
    background = SOURCE.astype(float).copy()
    background[mask] = spsolve(matrix.tocsr(), rhs)
    # Carry nearby photographic grain into the otherwise smooth interpolation.
    residual = SOURCE.astype(float) - gaussian_filter(SOURCE.astype(float), (1.2, 1.2, 0))
    feather = np.minimum(1, np.minimum(
        np.maximum(0, 1 - ((X-76)/19)**2 - ((Y-50)/12)**2),
        np.ones((H, W))))
    feather = np.maximum(feather, np.maximum(0, 1 - ((X-149)/21)**2 - ((Y-58)/13)**2))
    yy, xx = points.T
    grain = residual[yy + 27, xx]
    background[mask] += 0.70 * grain * np.minimum(1, feather[mask, None]*4)
    return np.clip(background, 0, 255)


BACKGROUND = reconstruct_table()


def sample(array, xs, ys):
    return np.stack([map_coordinates(array[..., c], [ys, xs], order=1, mode='nearest')
                     for c in range(3)], axis=-1)


def make_ball_material(ball):
    # A disc sampled from the photograph separates fixed illumination from felt.
    size = 81
    gy, gx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    rad = np.sqrt(gx*gx + gy*gy)
    cx, cy = ball['center']
    raw = sample(SOURCE.astype(float), cx + gx*ball['rx'], cy + gy*ball['ry'])
    bg = sample(BACKGROUND, cx + gx*ball['rx'], cy + gy*ball['ry'])
    # Photograph edges already have antialiasing; remove the table contribution.
    alpha = np.clip((1.055-rad)*min(ball['rx'],ball['ry']), 0.08, 1)
    rgba = np.clip((raw-bg*(1-alpha[...,None]))/alpha[...,None], 0, 255)
    interior = rad < 0.89
    # Extend valid material to the rim using its nearest inner spherical point.
    safe = np.minimum(1, 0.89 / np.maximum(rad, 1e-6))
    extension = sample(rgba, (gx*safe+1)*(size-1)/2, (gy*safe+1)*(size-1)/2)
    rgba[~interior] = extension[~interior]
    shade = gaussian_filter(rgba, (4.3,4.3,0))
    texture = rgba - shade
    ball['shade'], ball['texture'] = shade, texture
    ball['material_size'] = size


for ball in BALLS:
    make_ball_material(ball)


def support_radius(ball):
    return 1 / np.sqrt((DIRECTION[0]/ball['rx'])**2 + (DIRECTION[1]/ball['ry'])**2)


R_A, R_B = [support_radius(b) for b in BALLS]
# Perspective changes apparent radius slightly as a ball moves down the table.
Q_CONTACT = (DISTANCE-R_A-R_B) / (2 + SCALE_SLOPE*DIRECTION[1]*(R_A-R_B))
SPEED = Q_CONTACT / CONTACT_TIME


def displacement(t):
    if t <= CONTACT_TIME:
        return SPEED*t
    elapsed = t-CONTACT_TIME
    return Q_CONTACT - 0.94*SPEED*elapsed + 0.5*0.40*elapsed**2


def pose(ball, t):
    q = displacement(t)
    sign = 1 if ball['name']=='blue' else -1
    center = ball['center'] + sign*q*DIRECTION
    scale = 1 + SCALE_SLOPE*(center[1]-ball['center'][1])
    return center, scale, sign*q


def draw_shadow(frame, ball, center, scale):
    cx, cy = center
    # The source lighting casts broad soft shadows towards the left.
    rx, ry = ball['rx']*scale, ball['ry']*scale
    broad = np.exp(-0.5*(((X-(cx-3.4*scale))/(rx*1.12))**2
                        + ((Y-(cy+1.2*scale))/(ry*0.61))**2))
    contact = np.exp(-0.5*(((X-(cx-0.8*scale))/(rx*0.64))**2
                          + ((Y-(cy+ry*0.74))/(1.35*scale))**2))
    attenuation = 1 - 0.22*broad - 0.14*contact
    frame *= attenuation[...,None]
    frame[...,2] -= 2.5*broad


def draw_ball(frame, ball, center, scale, travel):
    cx, cy = center
    rx, ry = ball['rx']*scale, ball['ry']*scale
    x0, x1 = max(0,int(np.floor(cx-rx-2))), min(W,int(np.ceil(cx+rx+3)))
    y0, y1 = max(0,int(np.floor(cy-ry-2))), min(H,int(np.ceil(cy+ry+3)))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    nx, ny = (xx-cx)/rx, (yy-cy)/ry
    radial = np.sqrt(nx*nx+ny*ny)
    alpha = np.clip((1.04-radial)*min(rx,ry)+0.12, 0, 1)
    size = ball['material_size']
    light = sample(ball['shade'], (nx+1)*(size-1)/2, (ny+1)*(size-1)/2)
    # Rotate the photographed felt on a sphere, leaving illumination fixed.
    nz = np.sqrt(np.maximum(0, 1-nx*nx-ny*ny))
    angle = travel / (ball['rx']*1.28)
    ca, sa = np.cos(angle), np.sin(angle)
    rotated_x = nx*ca-nz*sa
    rotated_z = nx*sa+nz*ca
    rotated_y = ny
    # The unseen hemisphere reuses supplied felt, with a half-turn about z.
    rotated_x = np.where(rotated_z < 0, -rotated_x, rotated_x)
    rotated_y = np.where(rotated_z < 0, -rotated_y, rotated_y)
    felt = sample(ball['texture'], (rotated_x+1)*(size-1)/2,
                  (rotated_y+1)*(size-1)/2)
    color = np.clip(light + felt, 0, 255)
    frame[y0:y1,x0:x1] = frame[y0:y1,x0:x1]*(1-alpha[...,None])+color*alpha[...,None]


def render(t):
    frame = BACKGROUND.copy()
    poses = [(b, *pose(b,t)) for b in BALLS]
    for ball, center, scale, travel in poses:
        draw_shadow(frame, ball, center, scale)
    for ball, center, scale, travel in poses:
        draw_ball(frame, ball, center, scale, travel)
    # Outside the tiny local animation area, original pixels remain exact.
    active = np.any(np.abs(frame-BACKGROUND)>0.06, axis=2)
    restored = np.any(np.abs(BACKGROUND-SOURCE.astype(float))>0.06, axis=2)
    frame[~(active|restored)] = SOURCE[~(active|restored)]
    return np.clip(frame,0,255)


def main():
    frames = [SOURCE.copy()]
    for index in range(1,20):
        t = index/4
        # Short camera exposure provides a small natural amount of motion blur.
        samples = [render(max(0,t+dt)) for dt in (-0.025,0,0.025)]
        frames.append(np.rint(np.mean(samples,axis=0)).astype(np.uint8))
    frames = np.stack(frames)
    candidate = ROOT/'candidates'/'candidate_01'
    candidate.mkdir(parents=True,exist_ok=True)
    np.save(candidate/'rgb.npy',frames)
    (candidate/'generate.py').write_text(Path(__file__).read_text())
    stats = {
        'shape':list(frames.shape), 'dtype':str(frames.dtype), 'fps':4,
        'first_frame_exact':bool(np.array_equal(frames[0],SOURCE)),
        'contact_time_seconds':CONTACT_TIME,
        'initial_equal_speed_pixels_per_second':SPEED,
        'coefficient_of_restitution':0.94,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'centers':[[pose(b,i/4)[0].round(4).tolist() for b in BALLS] for i in range(20)],
    }
    (candidate/'validation.json').write_text(json.dumps(stats,indent=2))
    artifact_io.save(frames,ROOT)
    import imageio.v2 as imageio
    imageio.mimwrite(ROOT/'video.mp4', frames, fps=4, codec='libx264',
                    quality=9, macro_block_size=1, ffmpeg_params=['-pix_fmt','yuv420p'])
    print(json.dumps({k:v for k,v in stats.items() if k!='centers'},indent=2))


if __name__=='__main__':
    main()

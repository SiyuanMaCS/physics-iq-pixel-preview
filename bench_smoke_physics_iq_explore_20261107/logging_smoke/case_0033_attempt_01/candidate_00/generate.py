from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import artifact_io

ROOT = Path(__file__).resolve().parent
SOURCE = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'), dtype=np.float32)
H, W = SOURCE.shape[:2]
Y, X = np.mgrid[:H, :W].astype(np.float32)


def sample(a, x, y):
    """Bilinear pixel resampling, including subpixel motion."""
    x = np.clip(x, 0, a.shape[1] - 1.001)
    y = np.clip(y, 0, a.shape[0] - 1.001)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - ix)[..., None], (y - iy)[..., None]
    if a.ndim == 2:
        a = a[..., None]
    return ((a[iy, ix] * (1-fx) + a[iy, ix+1] * fx) * (1-fy)
            + (a[iy+1, ix] * (1-fx) + a[iy+1, ix+1] * fx) * fy)


def blurred(a, radius):
    return np.asarray(Image.fromarray(np.uint8(np.clip(a, 0, 255))).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)


def removal_mask(cx, cy, color):
    # Include the original soft cast shadows as well as the colored spheres.
    if color == 'blue':
        ball = np.sqrt(((X-cx)/11.0)**2 + ((Y-cy)/10.0)**2)
        cast = np.sqrt(((X-(cx+3))/23.0)**2 + ((Y-(cy-2))/9.0)**2)
    else:
        ball = np.sqrt(((X-cx)/11.8)**2 + ((Y-cy)/11.2)**2)
        cast = np.sqrt(((X-(cx-7))/13.5)**2 + ((Y-(cy-1))/10.0)**2)
    return np.clip((1.0-np.minimum(ball,cast))/0.25, 0, 1)


objects = [(np.array([76.6, 49.8]), 7.1, 'blue'),
           (np.array([149.3, 58.3]), 7.9, 'yellow')]
background = SOURCE.copy()
for center, radius, color in objects:
    cx, cy = center
    mask = removal_mask(cx, cy, color)
    # Reconstruct the tabletop from adjacent pixels along its slanting grain.
    left = cx - (25 if color == 'blue' else 23)
    right = cx + (29 if color == 'blue' else 24)
    slope = 0.115
    lx = np.full_like(X, left)
    rx = np.full_like(X, right)
    lval = sample(SOURCE, lx, Y+slope*(left-X))
    rval = sample(SOURCE, rx, Y+slope*(right-X))
    frac = np.clip((X-left)/(right-left), 0, 1)[...,None]
    smooth = lval*(1-frac)+rval*frac
    # Carry fine native wood grain into the reconstructed region.
    texture = sample(SOURCE-blurred(SOURCE,1.4), X+35, Y+35*slope)
    fill = smooth + texture * 0.45
    background = background*(1-mask[...,None]) + fill*mask[...,None]

# The original sphere pixels retain the source illumination and defocus.
# Moving these differences also carries the photographed contact/cast shadows.
residuals = []
for center, radius, color in objects:
    cx,cy = center
    mask = removal_mask(cx,cy,color)
    delta = (SOURCE-background)*mask[...,None]
    residuals.append(delta)

p0, p1 = objects[0][0], objects[1][0]
axis = (p1-p0)/np.linalg.norm(p1-p0)
diameter = objects[0][1]+objects[1][1]
contact_time = 1.25
speed = (np.linalg.norm(p1-p0)-diameter)/(2*contact_time)
restitution = 0.94
rolling_deceleration = 0.48


def position(index, t):
    sign = 1 if index == 0 else -1
    initial = objects[index][0]
    if t <= contact_time:
        travel = speed*t
    else:
        dt = t-contact_time
        travel = speed*contact_time - restitution*speed*dt + .5*rolling_deceleration*dt*dt
    return initial + sign*axis*travel


frames=[]
for k in range(20):
    t=k/4
    canvas = background.copy()
    # The farther ball is composed first at the instant of contact.
    for j in range(2):
        center,radius,color = objects[j]
        pos=position(j,t)
        shift=pos-center
        moved=sample(residuals[j],X-shift[0],Y-shift[1])
        canvas+=moved
    frames.append(np.uint8(np.clip(np.rint(canvas),0,255)))
frames=np.stack(frames)
frames[0]=SOURCE.astype(np.uint8)

# Preserve a reviewable first candidate before exporting final artifacts.
candidate = ROOT / 'candidate_01'
candidate.mkdir(exist_ok=True)
np.save(candidate/'rgb.npy',frames)
Image.fromarray(np.uint8(np.clip(background,0,255))).save(candidate/'tabletop.png')
sheet=Image.new('RGB',(256*5,164*4),(25,28,35))
draw=ImageDraw.Draw(sheet)
for k in range(20):
    x,y=(k%5)*256,(k//5)*164
    sheet.paste(Image.fromarray(frames[k]),(x,y+20))
    draw.text((x+5,y+3),f'{k:02d}  {k/4:.2f}s',fill=(255,255,255))
sheet.save(candidate/'contact_sheet.png')
artifact_io.save(frames, ROOT)
print(f'Saved {frames.shape}, {frames.dtype}; initial speed {speed:.2f} px/s; contact at {contact_time:.2f}s')

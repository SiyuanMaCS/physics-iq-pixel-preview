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
    # Broad feathered support includes the camera's soft ball/shadow edges.
    if color == 'blue':
        distance = np.sqrt(((X-(cx+2))/25.0)**2 + ((Y-(cy-1))/12.0)**2)
    else:
        distance = np.sqrt(((X-(cx-3))/19.5)**2 + ((Y-cy)/12.0)**2)
    return np.clip((1.20-distance)/0.22,0,1)


objects = [(np.array([76.6, 49.8]), 7.1, 'blue'),
           (np.array([149.3, 58.3]), 7.9, 'yellow')]
background = SOURCE.copy()
for center, radius, color in objects:
    cx, cy = center
    mask = removal_mask(cx, cy, color)
    # Reconstruct each missing strip using clean pixels along its wood grain.
    left, right = cx-32, cx+35
    slope=0.115
    lval=sample(SOURCE,np.full_like(X,left),Y+slope*(left-X))
    rval=sample(SOURCE,np.full_like(X,right),Y+slope*(right-X))
    frac=np.clip((X-left)/(right-left),0,1)[...,None]
    fill=lval*(1-frac)+rval*frac
    texture=sample(SOURCE-blurred(SOURCE,1.4),X+40,Y+40*slope)
    fill+=texture*0.30
    background=background*(1-mask[...,None])+fill*mask[...,None]

# The original sphere pixels retain the source illumination and defocus.
# Moving these differences also carries the photographed contact/cast shadows.
residuals = []
for center, radius, color in objects:
    cx,cy = center
    mask = removal_mask(cx,cy,color)
    delta = (SOURCE-background) * (mask>0)[...,None]
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
        pipe_pixels = ((X < 14) & (Y < 55)) | ((X > 230) & (Y > 53) & (Y < 86))
        canvas+=moved*(~pipe_pixels)[...,None]
    frames.append(np.uint8(np.clip(np.rint(canvas),0,255)))
frames=np.stack(frames)
frames[0]=SOURCE.astype(np.uint8)

# Preserve a reviewable first candidate before exporting final artifacts.
candidate = ROOT / 'candidate_03'
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

from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter, distance_transform_edt
import artifact_io

ROOT = Path(__file__).resolve().parent
SOURCE = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB')).astype(np.float64)
H, W = SOURCE.shape[:2]
Y, X = np.mgrid[:H, :W].astype(float)
END0 = np.array([150.5, 67.5])
THETA0 = np.arctan2(1, 1.603)
U0 = np.array([np.cos(THETA0), np.sin(THETA0)])
V0 = np.array([-U0[1], U0[0]])
S0 = (X-END0[0])*U0[0] + (Y-END0[1])*U0[1]
N0 = (X-END0[0])*V0[0] + (Y-END0[1])*V0[1]

# Replace only the splint and its small flame with locally interpolated pixels.
mask = (np.abs(N0)<3.4) & (S0 < 1.5)
mask |= (((X-140.5)/7.5)**2 + ((Y-61)/5.8)**2 < 1)
mask |= (((X-147.5)/6.2)**2 + ((Y-66)/4.0)**2 < 1)
mask &= (S0 > -145)
_, nearest = distance_transform_edt(mask, return_indices=True)
background = SOURCE.copy()
background[mask] = SOURCE[nearest[0][mask], nearest[1][mask]]
# Discrete harmonic interpolation is useful across this narrow diagonal gap.
for _ in range(600):
    pad = np.pad(background, ((1,1),(1,1),(0,0)), mode='edge')
    mean = (pad[:-2,1:-1]+pad[2:,1:-1]+pad[1:-1,:-2]+pad[1:-1,2:])/4
    background[mask] = mean[mask]

# Horizontal texture interpolation preserves the straight wall/table boundary.
for row in range(40):
    selected = np.flatnonzero(mask[row] & (X[row] < 105))
    if not len(selected):
        continue
    left, right = int(selected[0])-1, int(selected[-1])+1
    left=max(left,0); right=min(right,W-1)
    fraction=(selected-left)/(right-left)
    background[row,selected] = SOURCE[row,left]*(1-fraction[:,None])+SOURCE[row,right]*fraction[:,None]

# Restore the two thin rim contours where the stationary match hid them.
for front in [False, True]:
    dx = (X-135.0)/28.4
    cy = 46.1 + (1 if front else -1)*11.6*np.sqrt(np.maximum(0,1-dx*dx))
    edge = np.exp(-0.5*((Y-cy)/0.43)**2)*(np.abs(dx)<0.985)
    strength = edge*mask*(0.34 if front else 0.60)
    tint = np.array([177.,177.,177.]) if front else np.array([186.,184.,181.])
    background = background*(1-strength[...,None])+tint*strength[...,None]

SCALE = 4
hh, ww = H*SCALE, W*SCALE
hy, hx = np.mgrid[:hh,:ww].astype(float)
hx = (hx+0.5)/SCALE-0.5
hy = (hy+0.5)/SCALE-0.5
base_hi = np.asarray(Image.fromarray(np.uint8(np.clip(background,0,255))).resize((ww,hh),Image.Resampling.BICUBIC)).astype(float)

base_down = np.asarray(Image.fromarray(base_hi.astype(np.uint8)).resize((W,H),Image.Resampling.LANCZOS)).astype(np.int16)
background_down = np.uint8(np.clip(background,0,255)).astype(np.int16)

def sample(arr, sx, sy):
    return np.stack([map_coordinates(arr[...,c], [sy,sx], order=1, mode='nearest') for c in range(3)],axis=-1)

# The shaft's texture and shading come from the supplied image alone.
def old_alpha(s,n):
    width = np.where(s>-12,1.75,1.58)
    a = np.clip((width+0.62-np.abs(n))/1.02,0,1)
    a *= np.clip((0.8-s)/1.7,0,1)
    return a

def render_splint(canvas, end, angle, time):
    u=np.array([np.cos(angle),np.sin(angle)])
    v=np.array([-u[1],u[0]])
    # A small refraction shift applies to the submerged section.
    wet=np.clip((hy-69.6)/1.4,0,1) if time >= .5 else np.zeros_like(hy)
    xx=hx-1.1*wet
    yy=hy-.30*wet
    s=(xx-end[0])*u[0]+(yy-end[1])*u[1]
    n=(xx-end[0])*v[0]+(yy-end[1])*v[1]
    a=old_alpha(s,n)
    ox=END0[0]+s*U0[0]+n*V0[0]
    oy=END0[1]+s*U0[1]+n*V0[1]
    # Unmix antialiased source edges from the old background.
    raw=sample(SOURCE,ox,oy)
    bg=sample(background,ox,oy)
    source_a=np.maximum(old_alpha(s,n),.16)
    color=np.clip((raw-bg*(1-source_a[...,None]))/source_a[...,None],0,255)
    # Above the frame, continue the same natural wood shading.
    off=oy<0
    wood=np.zeros_like(color)
    shade=8*np.cos(n*1.3)+2*np.sin(s*1.7)
    wood[:]=np.array([176,143,102])
    wood += shade[...,None]
    color[off]=wood[off]
    char = np.clip((s+12)/4,0,1)
    # Once extinguished, the ember becomes neutral charcoal.
    extinguish=np.clip((time-.45)/.22,0,1)
    hot_zone=np.clip((s+17)/5,0,1)*np.clip((-s-6)/5,0,1)
    neutral=np.stack([0.28*color[...,0]+0.35*color[...,1]+0.37*color[...,2]]*3,axis=-1)
    color=color*(1-(hot_zone*extinguish*.8)[...,None])+neutral*(hot_zone*extinguish*.8)[...,None]
    color=color*(1-.17*wet[...,None])+np.array([130,117,98])*(.17*wet[...,None])
    a*=1-.14*wet
    canvas=canvas*(1-a[...,None])+color*a[...,None]
    # The front lip remains in front of the inserted stick.
    dx=(hx-135)/28.4
    edge_y=46.1+11.6*np.sqrt(np.maximum(0,1-dx*dx))
    rim=np.exp(-.5*((hy-edge_y)/.44)**2)*(np.abs(dx)<.98)*a*.52
    canvas=canvas*(1-rim[...,None])+base_hi*rim[...,None]
    return canvas

def flame(canvas, end, angle, t):
    # This splint has only a tiny flame around the charred end.
    if t >= .65: return canvas
    u=np.array([np.cos(angle),np.sin(angle)])
    p=end-9*u
    fade=1-np.clip((t-.25)/.4,0,1)
    layer=Image.new('RGBA',(ww,hh))
    d=ImageDraw.Draw(layer)
    cx,cy=p[0]*SCALE,(p[1]-.8)*SCALE
    flick=.45*np.sin(t*24)
    d.polygon([(cx-2.3*SCALE,cy+.6*SCALE),(cx-1.6*SCALE,cy-1.5*SCALE),(cx+flick*SCALE,cy-4.2*SCALE),(cx+.2*SCALE,cy-1.5*SCALE),(cx+2.3*SCALE,cy+.5*SCALE)],fill=(238,143,47,int(120*fade)))
    d.polygon([(cx-1.1*SCALE,cy),(cx-.3*SCALE,cy-2.2*SCALE),(cx+.9*SCALE,cy)],fill=(252,207,99,int(150*fade)))
    layer=layer.filter(ImageFilter.GaussianBlur(.45*SCALE))
    arr=np.asarray(layer).astype(float)
    a=arr[...,3:]/255
    return canvas*(1-a)+arr[...,:3]*a

def ripples(canvas,t):
    if t<.5: return canvas
    surface=(((hx-135)/28)**2+((hy-67.2)/11.3)**2)<.93
    intensity=np.zeros_like(hx)
    for start in [.53,.90,1.28,1.68]:
        age=t-start
        if age<0 or age>1.75: continue
        cx=144-3.6*min(start,1.7)
        cy=69.5
        rr=np.sqrt((hx-cx)**2+((hy-cy)/.34)**2)
        radius=1+12*age
        wave=np.exp(-((rr-radius)/.7)**2)-.6*np.exp(-((rr-radius-1.0)/.85)**2)
        intensity += 4.5*np.exp(-age*1.8)*wave
    canvas += (intensity*surface)[...,None]*np.array([1.,1.02,1.05])
    return canvas

def smoke(canvas,t):
    # Very faint smoke disperses just after the tip contacts the water.
    if not (.45<t<1.9): return canvas
    age=t-.45
    opacity=.12*np.sin(np.pi*age/1.45)*np.exp(-age*.7)
    center=145-2.5*age+1.6*np.sin((69-hy)*.30+age*4)
    plume=np.exp(-((hx-center)/(1.0+age*1.6))**2)
    plume*=np.exp(-((hy-(67-12*age))/(5+4*age))**2)
    plume*=np.clip((69-hy)/5,0,1)
    a=opacity*plume
    return canvas*(1-a[...,None])+np.array([185,184,179])*a[...,None]

def generate():
    frames=[SOURCE.astype(np.uint8)]
    for i in range(1,20):
        t=i/4
        # A continuous hand-driven descent, slowing to a submerged hold.
        p=min(t/2.9,1.)
        ease=p*p*(3-2*p)
        end=np.array([150.5-9.5*ease,67.5+28.0*ease])
        angle=THETA0+(np.deg2rad(67)-THETA0)*ease
        end[0]+=.16*np.sin(t*3.4)*min(p*3,1)
        end[1]+=.12*np.sin(t*4.1)*min(p*3,1)
        canvas=base_hi.copy()
        canvas=ripples(canvas,t)
        canvas=render_splint(canvas,end,angle,t)
        canvas=flame(canvas,end,angle,t)
        canvas=smoke(canvas,t)
        im=Image.fromarray(np.uint8(np.clip(canvas,0,255))).resize((W,H),Image.Resampling.LANCZOS)
        # Preserve the original static pixels outside the animated regions.
        delta = np.asarray(im).astype(np.int16)-base_down
        frames.append(np.uint8(np.clip(background_down+delta,0,255)))
    return np.stack(frames)

if __name__=='__main__':
    frames=generate()
    candidate=ROOT/'.tmp'/'candidate_v2'
    candidate.mkdir(parents=True,exist_ok=True)
    for name in ['prompt.txt','task_spec.json']:
        shutil.copyfile(ROOT/name,candidate/name)
    artifact_io.save(frames,candidate)
    artifact_io.save(frames,ROOT)
    print('Saved',frames.shape,frames.dtype,'to pixels.npz and index.html')

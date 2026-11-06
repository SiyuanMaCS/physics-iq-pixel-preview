from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import artifact_io

ROOT = Path(__file__).resolve().parent
SOURCE = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
H, W = SOURCE.shape[:2]
S = 4
rng = np.random.default_rng(35)


def polygon_mask(vertices, scale=1):
    im = Image.new('L', (W*scale, H*scale))
    ImageDraw.Draw(im).polygon([tuple(np.asarray(p)*scale) for p in vertices], fill=255)
    return np.array(im, dtype=float)/255


def sample(arr, x, y):
    x = np.clip(x, 0, W-1.001)
    y = np.clip(y, 0, H-1.001)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x-ix, y-iy
    return ((1-fx)[...,None]*(1-fy)[...,None]*arr[iy,ix] +
            fx[...,None]*(1-fy)[...,None]*arr[iy,ix+1] +
            (1-fx)[...,None]*fy[...,None]*arr[iy+1,ix] +
            fx[...,None]*fy[...,None]*arr[iy+1,ix+1])


# Cuboid face coordinates measured from the RGB input. A is the lower
# leading corner, width points along the row, height points upward, and
# depth runs toward the rear of the tabletop.
blocks = [
    dict(name='yellow', a=(53.3,106.1), w=(10.8,0), v=(-6.0,-48.1), d=(5.0,-6.7), height=48.1, start=.91, final=90),
    dict(name='orange', a=(78.5,106.7), w=(10.3,0), v=(-3.8,-48.2), d=(3.6,-6.6), height=48.2, start=.75, final=79),
    dict(name='pink', a=(100.2,107.3), w=(10.2,0), v=(-.9,-47.0), d=(2.0,-6.6), height=47.0, start=.59, final=77),
    dict(name='green', a=(125.0,109.0), w=(10.1,0), v=(0,-50.1), d=(.1,-5.7), height=50.1, start=.43, final=76),
    dict(name='blue', a=(146.2,109.1), w=(10.2,0), v=(1.0,-48.8), d=(.2,-6.2), height=48.8, start=.27, final=75),
    dict(name='purple', a=(166.8,109.2), w=(9.8,0), v=(-3.6,-48.3), d=(-1.6,-6.7), height=48.3, start=.10, final=74),
]
for b in blocks:
    for k in ['a','w','v','d']:
        b[k] = np.array(b[k], dtype=float)
    a,w,v,d = [b[k] for k in ['a','w','v','d']]
    # Include the subpixel photographic borders in the removal patch.
    hull = [a-.8, a+w+[.8,.5], a+w+d+[1,0], a+w+d+v+[.8,-.7], a+d+v+[-.8,-.7], a+v+[-.8,0]]
    b['mask'] = polygon_mask(hull)


# Estimate the previously occluded wood from the visible tabletop. The
# polynomial accounts for the broad illumination; residual RGB samples
# carry the original small wood grain into newly revealed areas.
yy,xx = np.indices((H,W),dtype=float)
xn,yn = (xx-128)/128, (yy-94)/50
features = np.stack([np.ones_like(xx),xn,yn,xn*xn,xn*yn,yn*yn,
                     xn**3,xn*xn*yn,xn*yn*yn,yn**3],axis=-1)
valid = ((yy>=44)&(xx<43)) | ((yy>=88)&(xx>185)) | (yy>=120) | ((yy>=44)&(yy<=49)&(xx>44)&(xx<160))
coeff = np.linalg.lstsq(features[valid], SOURCE[valid].astype(float),rcond=None)[0]
lighting = features @ coeff
texture_y = 121 + ((yy.astype(int)-44) % 22)
residual = SOURCE[texture_y,xx.astype(int)].astype(float)-lighting[texture_y,xx.astype(int)]
wood = np.clip(lighting + .62*residual,0,255)
# Restore the small uncovered gaps using their actual source wood grain.
occupied = np.maximum.reduce([b['mask'] for b in blocks])
clean_gaps = (occupied==0)&(yy>44)&(yy<85)&(xx<162)
wood[clean_gaps] = SOURCE[clean_gaps]


pivot = np.array([197.1,55.4])
arm_tip = np.array([176.1,74.2])
arm_vec = arm_tip-pivot
arm_perp = np.array([-arm_vec[1],arm_vec[0]]) / np.linalg.norm(arm_vec) * 1.85
arm_vertices = [pivot-arm_perp, pivot+arm_perp, arm_tip+arm_perp, arm_tip-arm_perp]
arm_mask = polygon_mask(arm_vertices)
# Extra half-pixel antialiasing border around the photographed arm.
arm_mask = np.array(Image.fromarray((arm_mask*255).astype('uint8')).filter(ImageFilter.MaxFilter(3)),dtype=float)/255
arm_clean = SOURCE.astype(float).copy()
perp = arm_perp / np.linalg.norm(arm_perp)
for y,x in np.argwhere(arm_mask>.01):
    vals=[]
    for distance in [-7,-6,-5,5,6,7]:
        p=np.array([x,y])+perp*distance
        ix,iy=np.round(p).astype(int)
        if 0<=ix<W and 0<=iy<H and arm_mask[iy,ix]<.1:
            vals.append(SOURCE[iy,ix])
    if vals:
        arm_clean[y,x]=np.median(vals,axis=0)


def texture_face(canvas, dest_a, dest_u, dest_v, src_a, src_u, src_v, shade=1.0):
    basis = np.stack([dest_u,dest_v],axis=1)
    if abs(np.linalg.det(basis))<.05:
        return
    vertices = [dest_a,dest_a+dest_u,dest_a+dest_u+dest_v,dest_a+dest_v]
    xmin=max(0,int(math.floor(min(p[0] for p in vertices)*S))-1)
    xmax=min(W*S,int(math.ceil(max(p[0] for p in vertices)*S))+2)
    ymin=max(0,int(math.floor(min(p[1] for p in vertices)*S))-1)
    ymax=min(H*S,int(math.ceil(max(p[1] for p in vertices)*S))+2)
    if xmax<=xmin or ymax<=ymin:
        return
    gy,gx=np.indices((ymax-ymin,xmax-xmin),dtype=float)
    positions=np.stack([(gx+xmin+.5)/S,(gy+ymin+.5)/S],axis=-1)-dest_a
    uv = positions @ np.linalg.inv(basis).T
    u,v=uv[...,0],uv[...,1]
    inside=(u>=0)&(u<=1)&(v>=0)&(v<=1)
    coords=src_a+u[...,None]*src_u+v[...,None]*src_v
    colors=sample(SOURCE,coords[...,0],coords[...,1])*shade
    patch=canvas[ymin:ymax,xmin:xmax]
    patch[inside]=np.clip(colors[inside],0,255)


def angle_at(t,b):
    elapsed=t-b['start']
    if elapsed<=0:
        return 0.0
    # A push starts the fall; the angular speed grows under gravity and
    # then the cuboid rests on its neighbor after a very small rebound.
    duration=.48
    u=min(elapsed/duration,1)
    theta=b['final']*(.18*u+.82*u*u)
    if elapsed>duration:
        e=elapsed-duration
        theta=b['final']-2.1*math.exp(-e*10)*math.sin(e*24)
    return math.radians(theta)


def geometry(b,theta):
    a,w,v,d,h=[b[k] for k in ['a','w','v','d','height']]
    z=v/h
    xaxis=np.array([1.0,.018])
    return a, w[0]*(xaxis*math.cos(theta)+z*math.sin(theta)), v*math.cos(theta)-h*xaxis*math.sin(theta), d


def shadow_layer(shapes):
    layer=Image.new('L',(W*S,H*S))
    dr=ImageDraw.Draw(layer)
    for vertices,opacity in shapes:
        dr.polygon([tuple(p*S) for p in vertices],fill=int(opacity))
    return np.array(layer.filter(ImageFilter.GaussianBlur(1.05*S)),dtype=float)/255


def render(t):
    base=SOURCE.astype(float).copy()
    # The arm vacates its initial position as soon as the drive turns.
    base = base*(1-arm_mask[...,None])+arm_clean*arm_mask[...,None]
    angles=[angle_at(t,b) for b in blocks]
    for b,theta in zip(blocks,angles):
        if theta<=0:
            continue
        m=b['mask'].copy()
        # Release the original broad ground shadow as its caster lowers.
        x=b['a'][0]
        shadow_region=np.exp(-((xx-(x+8))/13)**4)*np.clip((yy-81)/14,0,1)*np.clip((119-yy)/6,0,1)
        shadow_region *= min(1,theta/.36)
        m=np.maximum(m,shadow_region)
        base=base*(1-m[...,None])+wood*m[...,None]
    # Supersampling keeps the small moving edges smooth in the RGB frames.
    canvas=np.asarray(Image.fromarray(np.uint8(np.clip(base,0,255))).resize((W*S,H*S),Image.Resampling.BICUBIC),dtype=float).copy()
    shapes=[]
    for b,theta in zip(blocks,angles):
        if theta<=0:
            continue
        a,w,v,d=geometry(b,theta)
        # Light comes from above and to camera left. The cast shadow gets
        # shorter and sharper as the block approaches the table.
        height=b['height']*math.cos(theta)
        top_ground=a-np.array([b['height']*math.sin(theta),b['height']*.018*math.sin(theta)])
        off=np.array([.29*height,.18*height])
        shapes.append(([a,a+d,a+d+w,top_ground+d+w+off,top_ground+off,a+w],80))
    shadows=shadow_layer(shapes)
    canvas*=1-.33*shadows[...,None]
    # Draw in contact order: each later falling block overlaps its neighbor.
    for b,theta in zip(blocks,angles):
        a,w0,v0,d0=[b[k] for k in ['a','w','v','d']]
        if theta<=0:
            m=np.asarray(Image.fromarray(np.uint8(b['mask']*255)).resize((W*S,H*S),Image.Resampling.BICUBIC),dtype=float)/255
            photo=np.asarray(Image.fromarray(SOURCE).resize((W*S,H*S),Image.Resampling.BICUBIC),dtype=float)
            canvas=canvas*(1-m[...,None])+photo*m[...,None]
            continue
        a,w,v,d=geometry(b,theta)
        # The two narrow depth faces and the original painted front face.
        if d[0]<0:
            texture_face(canvas,a,d,v,a,d0,v0,.96)
        else:
            texture_face(canvas,a+w,d,v,a+w0,d0,v0,.98)
        texture_face(canvas,a+v,w,d,a+v0,w0,d0,1.0+.06*math.sin(theta))
        texture_face(canvas,a,w,v,a,w0,v0,1.0-.025*math.sin(theta))
    # Clockwise rotation in the camera projection, anchored at the source
    # attachment point. The arm continues turning after the domino fall.
    rot=math.radians(70*t)
    c,s=math.cos(rot),math.sin(rot)
    mat=np.array([[c,-s],[s,c]])
    av=mat@arm_vec
    ap=mat@arm_perp
    a=pivot-ap
    texture_face(canvas,a,2*ap,av,pivot-arm_perp,2*arm_perp,arm_vec,1.0)
    # Expose the narrow darker wooden edge to retain the stick's thickness.
    edge=[pivot+ap,pivot+av+ap,pivot+av+ap+[.2,1.0],pivot+ap+[.2,1.0]]
    overlay=Image.new('RGBA',(W*S,H*S))
    ImageDraw.Draw(overlay).polygon([tuple(np.asarray(p)*S) for p in edge],fill=(101,76,53,130))
    canvas=np.asarray(Image.alpha_composite(Image.fromarray(np.uint8(np.clip(canvas,0,255))).convert('RGBA'),overlay).convert('RGB'))
    out=np.array(Image.fromarray(canvas).resize((W,H),Image.Resampling.LANCZOS))
    return out.astype(np.uint8)


def main():
    frames=np.stack([SOURCE.copy()]+[render(i/4) for i in range(1,20)])
    candidates=ROOT/'candidates'
    candidates.mkdir(exist_ok=True)
    np.savez_compressed(candidates/'candidate_01.npz',frames=frames)
    (candidates/'generate_01.py').write_text(Path(__file__).read_text())
    for i in [1,2,4,6,10,19]:
        Image.fromarray(frames[i]).save(candidates/f'candidate_01_{i:02}.png')
    artifact_io.save(frames,ROOT)
    try:
        import imageio.v2 as imageio
        writer=imageio.get_writer(str(ROOT/'video.mp4'),fps=4,codec='libx264',quality=9,macro_block_size=None,ffmpeg_params=['-pix_fmt','yuv420p'])
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except ImportError:
        pass
    print('Saved RGB frames:',frames.shape,frames.dtype)
    print('First frame identical:',np.array_equal(frames[0],SOURCE))
    print('Changed pixels:',[int(np.count_nonzero(np.any(f!=SOURCE,axis=-1))) for f in frames])


if __name__=='__main__':
    main()

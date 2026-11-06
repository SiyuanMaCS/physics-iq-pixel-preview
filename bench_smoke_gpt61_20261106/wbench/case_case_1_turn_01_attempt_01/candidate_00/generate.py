"""Render a following shot using source RGB pixels and hand-written warps."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import artifact_io

ROOT = Path(__file__).resolve().parent
S = 4
W, H = 256, 143
source = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))

def mask_polygon(points, holes=()):
    m = Image.new('L', (W*S,H*S))
    d = ImageDraw.Draw(m)
    d.polygon([(round(x*S),round(y*S)) for x,y in points],fill=255)
    for hole in holes:
        d.polygon([(round(x*S),round(y*S)) for x,y in hole],fill=0)
    return np.asarray(m,dtype=np.float32)/255

def sample(a,x,y):
    x=np.clip(x,0,a.shape[1]-1.001)
    y=np.clip(y,0,a.shape[0]-1.001)
    x0=x.astype(np.int32); y0=y.astype(np.int32)
    fx=(x-x0)[...,None]; fy=(y-y0)[...,None]
    if a.ndim==2:
        a=a[...,None]
    return ((1-fy)*((1-fx)*a[y0,x0]+fx*a[y0,x0+1])+
            fy*((1-fx)*a[y0+1,x0]+fx*a[y0+1,x0+1]))

# The contours are geometric annotations made from the supplied first image.
outline=[(120.6,73.2),(122.5,72.7),(124.6,76.0),(130.8,76.1),
         (133.3,73.1),(134.8,73.4),(135.1,77.8),(136.2,81.5),
         (135.5,85.1),(134.8,88.1),(137.0,92.5),(138.2,98.3),
         (138.7,105.1),(137.7,111.7),(136.4,116.0),(136.3,122.0),
         (137.0,126.8),(137.2,129.2),(133.7,130.2),(132.0,129.1),
         (132.5,126.9),(131.5,118.3),(131.3,114.2),(129.5,110.6),
         (127.9,108.5),(126.0,110.3),(124.4,115.5),(124.0,122.7),
         (123.0,128.5),(123.0,130.3),(119.4,130.8),(118.9,129.4),
         (120.0,124.9),(120.2,118.3),(118.4,112.2),(117.8,106.0),
         (118.1,99.0),(119.4,93.0),(120.9,88.7),(120.0,86.0),
         (119.0,82.3),(119.6,78.2)]
dog_alpha=mask_polygon(outline)
hi_rgb=np.asarray(Image.fromarray(source).resize((W*S,H*S),Image.Resampling.BICUBIC),dtype=np.float32)
dog_rgba=np.concatenate([hi_rgb*dog_alpha[...,None],dog_alpha[...,None]],axis=2)

# Reconstruct only the small exposed road area, using other road pixels from
# this same frame. Feathering keeps the grain and lighting continuous.
erase_points=[(118,71),(137,71),(141,94),(142,108),(155,110),
              (156,116),(152,128),(139,134),(117,134),(115,108),(116,87)]
erase=Image.fromarray((mask_polygon(erase_points)*255).astype(np.uint8))
erase=erase.filter(ImageFilter.GaussianBlur(1.4*S))
erase=np.asarray(erase,dtype=np.float32)/255
Y,X=np.mgrid[0:H*S,0:W*S].astype(np.float32)
# A slight slant avoids repeated vertical columns in the repaired road.
texture=sample(hi_rgb,X-36*S+(Y-103*S)*0.12,Y)
plate=hi_rgb*(1-erase[...,None])+texture*erase[...,None]
Image.fromarray(np.uint8(np.clip(plate,0,255))).resize((W,H),Image.Resampling.LANCZOS).save(ROOT/'.tmp/clean_plate.png')
Image.fromarray(np.uint8(dog_alpha*255)).save(ROOT/'.tmp/source_contour.png')

# Build a soft cast shadow with an irregular silhouette, on the street to
# the right of the feet, matching the sunlight in the supplied image.
shadow=mask_polygon([(120,129.0),(131,130),(142,126.2),(150,122.0),
                     (150,118),(152.5,113),(155,112),(151,110.5),
                     (143,110.1),(138,112.4),(130,123)])
shadow=np.asarray(Image.fromarray(np.uint8(shadow*255)).filter(ImageFilter.GaussianBlur(.65*S)),dtype=np.float32)/255

xp=X/S; yp=Y/S

def street(t):
    travel=0.30*t/4.75
    # The ground is a perspective plane; street facades use their bases to
    # estimate distance. This yields greater parallax on nearby asphalt.
    vx,vy=130.,58.
    camx=.25*np.sin(2*np.pi*.85*t)
    camy=.24*np.sin(4*np.pi*.85*t)
    dx=xp-vx-camx; dy=yp-vy-camy
    sx=vx+dx; sy=vy+dy
    for _ in range(10):
        side=np.where(sx<vx,.25,.32)*np.abs(sx-vx)
        road=np.maximum(sy-vy,0)
        depth=np.maximum(side,road)/85.
        shrink=1-travel*depth
        sx=vx+dx*shrink; sy=vy+dy*shrink
    return sample(plate,sx*S,sy*S)

def dog_displacement(x,y,t):
    phase=2*np.pi*.85*t
    sway=.65*np.sin(phase)
    bob=-.75*(1-np.cos(2*phase))
    low=np.clip((y-105)/24,0,1)
    left=1/(1+np.exp(np.clip((x-128)*1.1,-40,40)))
    a=np.sin(phase); b=-a
    # Hind feet alternate between ground contact and a lifted forward swing.
    lift_l=-5.3*max(0,a)-1.3*a
    lift_r=-5.3*max(0,b)-1.3*b
    dy=bob+low**1.2*(left*lift_l+(1-left)*lift_r)
    dx=sway+low**1.25*(left*1.8*a+(1-left)*1.8*b)
    # Slight tail wag and shoulder rotation, with continuous texture warps.
    tail=np.exp(-((x-128.1)/4.0)**2-((y-91)/8.5)**2)
    dx+=1.15*np.sin(phase)*tail
    dx+=.32*np.sin(phase)*(np.clip((100-y)/26,0,1))
    return dx,dy

def render(t):
    bg=street(t)
    phase=2*np.pi*.85*t
    # The shadow follows the dog's slight gait motion while the tracking
    # camera preserves the subject's distance and footprint.
    sh=sample(shadow,X-.55*np.sin(phase)*S,Y+.4*(1-np.cos(2*phase))*S)[...,0]
    bg=bg*(1-.46*sh[...,None])+np.array([16,29,36],np.float32)*(.46*sh[...,None])
    # Inverse deformation of the original textured silhouette.
    qx=xp.copy(); qy=yp.copy()
    for _ in range(7):
        dx,dy=dog_displacement(qx,qy,t)
        qx=xp-dx; qy=yp-dy
    rgba=sample(dog_rgba,qx*S,qy*S)
    out=bg*(1-rgba[...,3:])+rgba[...,:3]
    return np.asarray(Image.fromarray(np.uint8(np.clip(out,0,255))).resize((W,H),Image.Resampling.LANCZOS))

frames=[source.copy()]
for i in range(1,20):
    frames.append(render(i/4))
frames=np.stack(frames).astype(np.uint8)
assert frames.shape==(20,143,256,3)
assert np.array_equal(frames[0],source)
# Keep the candidate tensor as well as the output artifacts for inspection.
np.save(ROOT/'.tmp/candidate_01.npy',frames)
artifact_io.save(frames,ROOT)

# An ordinary codec packages the rendered RGB sequence for convenient viewing.
try:
    import imageio_ffmpeg
    writer=imageio_ffmpeg.write_frames(str(ROOT/'video.mp4'),(W,H),fps=4,
        codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv444p',
        macro_block_size=1,quality=9,ffmpeg_log_level='error')
    writer.send(None)
    for frame in frames:
        writer.send(frame)
    writer.close()
except ImportError:
    pass
print('Saved 20 RGB frames, pixels.npz, index.html, and video.mp4.')

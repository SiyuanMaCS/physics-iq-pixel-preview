from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageFilter
import artifact_io

ROOT = Path(__file__).resolve().parent
W, H, N = 256, 144, 20
Y, X = np.mgrid[:H, :W].astype(np.float32)


def smooth(a, b, v):
    t = np.clip((v-a)/(b-a), 0, 1)
    return t*t*(3-2*t)


def sample(im, sx, sy):
    sx = np.clip(sx, 0, W-1)
    sy = np.clip(sy, 0, H-1)
    x0 = sx.astype(int)
    y0 = sy.astype(int)
    x1 = np.minimum(x0+1, W-1)
    y1 = np.minimum(y0+1, H-1)
    fx = (sx-x0)[..., None]
    fy = (sy-y0)[..., None]
    return ((im[y0,x0]*(1-fx)+im[y0,x1]*fx)*(1-fy)
            +(im[y1,x0]*(1-fx)+im[y1,x1]*fx)*fy)


def blend(frame, color, alpha):
    return frame*(1-alpha[...,None])+np.asarray(color)*alpha[...,None]


def prepare(src):
    # Separate the stationary setup from the glass's observed transmission
    # and reflection, using only spatial RGB interpolation in the input.
    plain = src.copy()
    flame_window = (smooth(148,153,X)*(1-smooth(171,177,X))
                    *(1-smooth(15,22,Y)))
    for y in range(23):
        row = np.arange(W, dtype=np.float32)
        u = np.clip((row-149)/27,0,1)[:,None]
        bg = src[y,149][None,:]*(1-u)+src[y,176][None,:]*u
        plain[y] = src[y]*(1-flame_window[y,:,None])+bg*flame_window[y,:,None]
    fire = src-plain

    background = plain.copy()
    for y in range(111):
        if y < 41:
            left, right = 114, 196
        elif y < 77:
            left, right = 113, 196
        elif y < 96:
            left, right = 104, 195
        else:
            left, right = 98, 194
        u = np.clip((np.arange(W)-left)/(right-left),0,1)[:,None]
        line = plain[y,left][None,:]*(1-u)+plain[y,right][None,:]*u
        glass_width = smooth(116.5,121,X[y])*(1-smooth(189,194,X[y]))
        glass_width *= 1-smooth(104,111,float(y))
        background[y] = plain[y]*(1-glass_width[:,None])+line*glass_width[:,None]

    sticks = smooth(131,135,X)*(1-smooth(158,162,X))
    sticks *= smooth(22,27,Y)*(1-smooth(92,98,Y))
    stationary = background*(1-sticks[...,None])+plain*sticks[...,None]
    reflection = plain-stationary
    # Offscreen glass texture is continued numerically from the top row.
    return stationary, reflection, fire, sticks


def render(src, frame_index, layers, version=1):
    if frame_index == 0:
        return src.astype(np.uint8)
    t = frame_index/4
    stationary, reflection, fire, sticks = layers
    descent = smooth(0,2.35,t)
    dy = 13.5*descent
    dx = -3.8*descent
    # The inverted glass settles almost vertically, with a small centering
    # movement; the fixed supports and bowl remain in camera coordinates.
    sx = X-dx
    sy = Y-dy
    moving = sample(reflection,sx,sy)
    moving *= (1-smooth(108+dy,115+dy,Y))[...,None]
    moving *= (1-0.88*sticks)[...,None]
    out = stationary+moving

    # Flicker and shortening are affine deformations of input RGB flame
    # pixels. There is no generated image source or simulation engine.
    strength = 1-smooth(2.15,3.75,t)
    contraction = 1-0.80*smooth(2.25,3.7,t)
    flicker = 0.035*np.sin(frame_index*2.3)+0.022*np.sin(frame_index*4.9)
    scale = contraction*(1+flicker)
    anchor_x, anchor_y = 149,29
    fx = anchor_x+(X-anchor_x-0.9*np.sin(frame_index*2.1))/(0.96+0.06*np.sin(frame_index*1.7))
    fy = anchor_y+(Y-anchor_y)/scale
    hot = sample(fire,fx,fy)
    valid = ((fy >= 0)&(fy <= 23)).astype(float)
    # Extend an initially cropped tongue just beyond the frame boundary.
    valid = np.where((fy<0)&(Y<2)&(strength>0.5),1,valid)
    out += hot*valid[...,None]*strength

    # A thick glass mouth has two very close elliptical edges. Its contrast
    # is restrained to match the light and reflections of the photograph.
    cx = 154.5+dx
    lip_y = 96.0+dy
    rx, ry = 34.3,6.2
    ellipse = np.sqrt(((X-cx)/rx)**2+((Y-lip_y)/ry)**2)
    edge = np.exp(-((ellipse-1)/0.065)**2)
    front = smooth(lip_y-2,lip_y+3,Y)
    lip_amount = 0.25*smooth(0,0.9,t)
    out = blend(out,(65,61,48),edge*lip_amount*(0.5+0.5*front))
    ellipse2 = np.sqrt(((X-cx)/(rx-1.0))**2+((Y-lip_y-0.8)/(ry-0.65))**2)
    highlight = np.exp(-((ellipse2-1)/0.055)**2)
    out = blend(out,(160,151,130),highlight*0.20*smooth(0,0.9,t))

    # After the flame dies, the modest increase in the internal liquid level
    # is rendered as a red surface with the bowl's original sampled texture.
    rise = 8.5*smooth(2.8,4.75,t)
    level = 99.5-rise
    xx = (X-cx)/31.5
    inside = (1-smooth(0.94,1.03,np.abs(xx)))
    surface_curve = np.sqrt(np.maximum(0,1-xx*xx))
    top_back = level-4.5*surface_curve
    front_surface = level+4.5*surface_curve
    filled = smooth(top_back-0.6,top_back+0.6,Y)
    filled *= 1-smooth(lip_y+5,lip_y+7,Y)
    filled *= inside*smooth(2.6,4.2,t)
    source_water = sample(src, X, np.clip(Y+rise+7,104,119))
    water = source_water*np.array([1.08,0.95,0.90])+np.array([4.0,8.0,6.0])
    water += (1.4*np.sin(X*0.53+Y*0.28+t*2))[...,None]
    out = out*(1-0.46*filled[...,None])+water*(0.46*filled[...,None])
    meniscus = np.exp(-((Y-front_surface)/0.65)**2)*inside
    meniscus *= smooth(2.8,4.75,t)
    out = blend(out,(133,61,45),meniscus*0.22)

    # Soft local ripples damp rapidly after the mouth enters the water.
    ripple_time = max(t-1.9,0)
    if ripple_time > 0:
        dist = np.sqrt(((X-cx)/1.0)**2+((Y-104)*3.0)**2)
        ring = np.exp(-((dist-(28+15*ripple_time))/2.2)**2)
        ring *= np.exp(-ripple_time*1.2)*0.045
        ring *= smooth(89,96,Y)*(1-smooth(115,120,Y))
        ring *= smooth(83,89,X)*(1-smooth(196,202,X))
        out = blend(out,(133,57,35),ring)

    # Small traces of smoke stay inside the transparent enclosure.
    smoke_amount = smooth(3.1,3.8,t)*(1-smooth(4.4,5.3,t))*0.065
    plume_x = 146+2.2*np.sin(Y*0.17-t*1.2)
    smoke = np.exp(-((X-plume_x)/(2.5+0.08*np.maximum(30-Y,0)))**2)
    smoke *= smooth(2,7,Y)*(1-smooth(29,35,Y))*smoke_amount
    out = blend(out,(169,175,171),smoke)
    return np.clip(np.rint(out),0,255).astype(np.uint8)


def main():
    src = np.asarray(Image.open(ROOT/'first_frame.png').convert('RGB'),dtype=np.float32)
    assert src.shape == (H,W,3)
    layers = prepare(src)
    frames = np.stack([render(src,i,layers) for i in range(N)])
    candidate = ROOT/'candidates'/'candidate_01'
    candidate.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(candidate/'rgb_frames.npz',frames=frames)
    for i in (0,1,5,9,13,19):
        Image.fromarray(frames[i]).save(candidate/f'{i:02d}.png')
    (candidate/'generate.py').write_text(Path(__file__).read_text())
    artifact_io.save(frames,ROOT)
    checks = {
        'shape':list(frames.shape),'dtype':str(frames.dtype),'fps':4,
        'first_frame_identical':bool(np.array_equal(frames[0],src.astype(np.uint8))),
        'frame_change_mean':[round(float(np.abs(frames[i].astype(float)-frames[i-1]).mean()),4) for i in range(1,N)],
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'candidate':'candidates/candidate_01/rgb_frames.npz'
    }
    (ROOT/'generation_log.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()

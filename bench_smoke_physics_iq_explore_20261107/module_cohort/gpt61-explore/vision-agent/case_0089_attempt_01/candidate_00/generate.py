"""Direct RGB continuation of a full glass under a red beverage dispenser."""
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from artifact_io import save


ROOT = Path(__file__).resolve().parent
SCALE = 4
W, H = 256, 144
base = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
canvas = np.asarray(Image.fromarray(base).resize((W*SCALE, H*SCALE), Image.Resampling.BICUBIC)).astype(float)
y, x = np.mgrid[:H*SCALE, :W*SCALE].astype(float)
x = (x + .5)/SCALE
y = (y + .5)/SCALE


def blend(im, color, opacity):
    a = np.clip(opacity, 0, 1)[..., None]
    return im*(1-a) + np.asarray(color)*a


def line_mask(points, width):
    layer = Image.new('L', (W*SCALE, H*SCALE))
    ImageDraw.Draw(layer).line([(round(a*SCALE), round(b*SCALE)) for a,b in points], fill=255,
                              width=max(1, round(width*SCALE)), joint='curve')
    return np.asarray(layer).astype(float)/255


def glass_region():
    # Boundary follows the photographed glass, including its rounded heel.
    left = 135.0 + 3.3*np.exp(-((y-22)/7)**2) + .017*np.maximum(y-64, 0)
    right = 173.1 - 2.5*np.exp(-((y-22)/8)**2) - .066*np.maximum(y-65,0)
    heel = 87.4 + 4.7*np.sqrt(np.clip(1-((x-153.8)/18.4)**2, 0, 1))
    return (x>left)&(x<right)&(y>20)&(y<heel)


glass = glass_region()


def render(i):
    if i == 0:
        return base.copy()
    t = i/4
    im = canvas.copy()
    elapsed = max(t-.28, 0)
    amount = 1-np.exp(-elapsed/.95)

    # Thin, reflective liquid advances across the metal tray. The elliptical
    # footprint is projected onto the tray plane and has an irregular wet edge.
    growth = np.sqrt(elapsed/4.47)
    cx, cy = 155.1+2.0*growth, 91.1+9.0*growth
    rx, ry = 18.2+46.0*growth, 3.8+14.5*growth
    theta = np.arctan2((y-cy)/ry, (x-cx)/rx)
    irregular = 1 + .024*np.sin(5*theta+.6) + .012*np.sin(9*theta-1.2)
    distance = np.sqrt(((x-cx)/rx)**2+((y-cy)/ry)**2)/irregular
    pool = np.clip((1-distance)*min(rx,ry)*2.5,0,1)*min(elapsed*3.5,1)
    pool *= ~glass
    pool *= y>82.0
    # The shallow red liquid preserves the photographed metal texture.
    depth = np.clip((1-distance)*2.0, 0, 1)
    pool_color = canvas*np.array([1.12,.43,.35])+np.array([60,3,1])
    pool_color += depth[...,None]*np.array([42,4,1])
    pool_color += (2*np.sin(x*.23+y*.5))[...,None]*np.array([1,.35,.2])
    im = blend(im, pool_color, pool*(.65+.27*depth))
    wet_edge = np.exp(-((distance-.974)*min(rx,ry)/.30)**2)*pool
    im = blend(im, [126,34,28], wet_edge*.26)

    # Soft reflected glass and vertical lighting reflections on the pool.
    reflected_glass = np.exp(-((x-154)/15)**4)*np.exp(-((y-98)/11)**2)*pool
    im = blend(im, [186,48,34], reflected_glass*.28)
    glint = (np.exp(-((y-(cy+ry*.6+.006*(x-cx)**2))/ .40)**2)
             * np.exp(-((x-145)/26)**2)*pool)
    im = blend(im, [199,126,108], glint*.15)

    # Waves originate where runoff arrives beside the glass, becoming shallower
    # as they move across the tray. They are small surface brightness changes.
    for source_x in [135.5, 170.4]:
        radius = np.sqrt(((x-source_x)/1.7)**2+((y-92.3)/.64)**2)
        waves = np.sin(radius*2.3-t*14)*np.exp(-radius/10.0)
        wave_zone = pool*(y>91.0)*.035*amount
        im += (waves*wave_zone)[...,None]*np.array([75,57,49])

    # Overflow runs over the existing glass rather than moving its silhouette.
    # Narrow surface sheets retain the glass's white highlights and shading.
    for side, offset in [(0,0.),(1,1.3),(2,2.4)]:
        start = [.45,.68,1.65][side]
        run_time = max(0,t-start)
        end = min(91.0, 24.0 + 81*run_time)
        if end <= 24:
            continue
        yy = np.linspace(24,end,160)
        if side == 0:
            xx = 137.0 - 1.1*(1-np.exp(-(yy-24)/7)) + .35*np.sin(yy*.34-t*4)
            width = 1.6
        elif side == 1:
            xx = 170.6 + .95*np.sin((yy-24)/29) + .22*np.sin(yy*.38-t*3)
            width = 1.8
        else:
            xx = 161.6 + .42*np.sin(yy*.14+.8) + .15*np.sin(yy*.7-t*5)
            width = 1.0
        wet = line_mask(list(zip(xx,yy)), width)
        wet *= np.clip(run_time*5,0,1)
        tint = canvas*np.array([1.03,.72,.60])+np.array([3,0,0])
        im = blend(im,tint,wet*.52)
        edge = line_mask(list(zip(xx-.36,yy)), .3)
        im = blend(im, [253,133,100], edge*.27*np.clip(run_time*5,0,1))
        # Tiny falling beads bridge the rounded heel and the pool below it.
        if run_time>.9 and side<2:
            for j in range(2):
                phase = (t*3.1+j*.48+offset)%1
                dy = 88.9 + 5.8*phase*phase
                dx = (136.6 if side==0 else 170.4)+.22*np.sin(t*8+j)
                drop = np.clip(1-((x-dx)/.63)**2-((y-dy)/(.6+.55*phase))**2,0,1)
                im = blend(im,[217,50,32],drop*.9)

    # Meniscus swells at the full rim; waves and small entrained bubbles move
    # around the impact point while remaining inside the photographed mouth.
    surface_r = np.sqrt(((x-154.1)/16.2)**2+((y-23.3)/3.2)**2)
    surface = np.clip((1-surface_r)*7,0,1)
    movement = np.sin(surface_r*22-t*13)+.5*np.sin((x-153)*1.2-t*8)
    im += (surface*movement*.65)[...,None]*np.array([8,13,10])
    impact = np.exp(-(((x-153.8)/2.9)**2+((y-23.4)/1.2)**2))
    im = blend(im,[251,110,78],impact*(.16+.04*np.sin(t*16)))
    rimwet = line_mask([(138.0,23.0),(143.0,25.2),(151.0,26.0),(162.0,25.8),(169.5,23.2)],.5)
    im = blend(im,[255,129,103],rimwet*.14*amount)
    for j in range(7):
        phase = (t*.69+j*.153)%1
        bx = 154.1 + (j%2*2-1)*(1.2+10.3*phase)
        by = 23.0 + .50*np.sin(j*1.9+t*4)+.35*phase
        radius = .22+.17*np.sin(j*2.7)**2
        bubble = np.exp(-(((x-bx)/radius)**2+((y-by)/(radius*.55))**2))
        im = blend(im,[255,167,143],bubble*.48*np.sin(np.pi*phase))

    # The very short exposed jet starts at the underside of the blue nozzle.
    jet_y = np.linspace(16.8,23.9,90)
    jet_x = 153.7 + .12*np.sin(jet_y*1.9-t*11)*(jet_y-16.8)/7.1
    jet = line_mask(list(zip(jet_x,jet_y)),1.8)
    # Leave the nozzle exactly as photographed; only draw below its outlet.
    jet *= y>=17.2
    im = blend(im,[245,65,42],jet*.62)
    highlight = line_mask(list(zip(jet_x-.38,jet_y)),.45)*(y>=17.5)
    im = blend(im,[255,150,119],highlight*(.35+.07*np.sin(t*17)))

    image = Image.fromarray(np.uint8(np.clip(im,0,255)))
    return np.asarray(image.resize((W,H),Image.Resampling.LANCZOS))


def main():
    frames = np.stack([render(i) for i in range(20)])
    assert frames.dtype == np.uint8 and frames.shape == (20,144,256,3)
    assert np.array_equal(frames[0],base)
    candidate = ROOT/'candidates'/'candidate_01'
    candidate.mkdir(parents=True,exist_ok=True)
    for name in ['prompt.txt','task_spec.json','generate.py']:
        shutil.copyfile(ROOT/name,candidate/name)
    save(frames,candidate)
    save(frames,ROOT)
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(ROOT/'video.mp4',frames,fps=4,codec='libx264',quality=9,
                         pixelformat='yuv420p',macro_block_size=1)
    except ImportError:
        pass
    print('Saved 20 RGB frames, pixels.npz, inspection sheets, and HTML player.')


if __name__ == '__main__':
    main()

"""A fixed-camera RGB animation, made from the supplied photograph and geometry."""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import artifact_io

ROOT = Path(__file__).resolve().parent
W, H = 256, 144
Y, X = np.mgrid[:H, :W].astype(np.float32)
SOURCE = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))


def smooth(a, b, v):
    z = np.clip((v-a)/(b-a), 0, 1)
    return z*z*(3-2*z)


def raster(drawer, blur=0):
    im = Image.new('L', (W*4, H*4))
    drawer(ImageDraw.Draw(im), 4)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur*4))
    return np.asarray(im.resize((W, H), Image.Resampling.LANCZOS), dtype=np.float32)/255


def sample(a, xx, yy):
    xx = np.clip(xx, 0, W-1)
    yy = np.clip(yy, 0, H-1)
    xi, yi = np.floor(xx).astype(int), np.floor(yy).astype(int)
    xj, yj = np.minimum(xi+1, W-1), np.minimum(yi+1, H-1)
    dx, dy = (xx-xi)[..., None], (yy-yi)[..., None]
    return (a[yi,xi]*(1-dx)*(1-dy) + a[yi,xj]*dx*(1-dy)
            + a[yj,xi]*(1-dx)*dy + a[yj,xj]*dx*dy)


def overlay(frame, color, alpha):
    return frame*(1-alpha[..., None]) + np.array(color)*alpha[..., None]


def ellipse_mask(cx, cy, rx, ry, feather=.65):
    d = np.sqrt(((X-cx)/rx)**2 + ((Y-cy)/ry)**2)
    return np.clip((1-d)*min(rx, ry)/feather+.5, 0, 1)


def prepare():
    original = SOURCE.astype(np.float32)
    # Take the fire out of the photograph by interpolating the glass immediately
    # beside it. The resulting fire residual can be warped independently.
    unlit = original.copy()
    left = original[:, 150:153].mean(axis=1)
    right = original[:, 172:175].mean(axis=1)
    interp = left[:, None]*(1-np.clip((X-151)/22,0,1)[...,None])
    interp += right[:, None]*np.clip((X-151)/22,0,1)[...,None]
    fm = smooth(149,154,X)*(1-smooth(168,175,X))*(1-smooth(16,23,Y))
    unlit = unlit*(1-fm[...,None])+interp*fm[...,None]
    flame = np.maximum(original-unlit, 0)

    # Reconstruct the wall/table seen through the jar, using pixels on either
    # side at the same height. The bowl rim and match bundle remain stationary.
    back = unlit.copy()
    l = unlit[:,113:117].mean(axis=1)
    r = unlit[:,194:198].mean(axis=1)
    ratio = np.clip((X-115)/81, 0, 1)
    estimate = l[:,None]*(1-ratio[...,None])+r[:,None]*ratio[...,None]
    back = estimate
    glass = unlit-back
    objects = np.zeros((H,W), dtype=np.float32)
    objects[25:95,135:159] = 1
    # Interpolate only the optical contribution across the matchbox: its RGB
    # texture is part of the stationary background layer.
    for yy in range(25,95):
        a = glass[yy,130:134].mean(axis=0)
        b = glass[yy,162:166].mean(axis=0)
        for xx in range(134,160):
            u=(xx-133)/27
            glass[yy,xx]=a*(1-u)+b*u
    # The outer vessel's thin elliptical lip stays fixed in front of the jar.
    rim = np.exp(-((Y-(44+.00065*(X-145)**2))/1.4)**2)
    rim *= smooth(119,125,X)*(1-smooth(185,192,X))
    glass *= (1-.86*rim[...,None])
    # Sharp horizontal features belong to the fixed bowl and its contents.
    # Moving only the broad glass reflection prevents a second bowl rim from
    # appearing lower down as the jar descends.
    weights=np.exp(-np.arange(-12,13,dtype=np.float32)**2/(2*4.0**2))
    weights/=weights.sum()
    padded=np.pad(glass,((12,12),(0,0),(0,0)),mode='edge')
    glass=sum(weight*padded[k:k+H] for k,weight in enumerate(weights))

    leftedge=119+.055*Y
    rightedge=190-.025*Y
    side=smooth(-1.4,1.2,X-leftedge)*(1-smooth(-1.2,1.4,X-rightedge))
    bottom=94+5*np.sqrt(np.maximum(0,1-((X-155)/32)**2))
    jm=side*(1-smooth(bottom-1.3,bottom+1.3,Y))
    glass *= jm[...,None]
    # Start with this exact decomposition, so movement never changes the
    # static background or the original arrangement of matches.
    fixed = unlit-glass
    return fixed, glass, flame


def glass_cap(frame, shift):
    cy=-9+shift
    # The closed bottom of the inverted jar enters from the top of the shot.
    body=ellipse_mask(154.3,cy,35.3,5.1)
    frame=overlay(frame, (124,137,137), body*.42)
    def lines(d,s):
        box=(119*s,(cy-5.1)*s,189.6*s,(cy+5.1)*s)
        d.ellipse(box,outline=170,width=round(1.1*s))
        d.arc((121*s,(cy-3.8)*s,188*s,(cy+3.8)*s),0,180,fill=255,width=round(.7*s))
    rim=raster(lines,.25)
    frame=overlay(frame,(191,203,199),rim*.50)
    shade=ellipse_mask(154,cy-.5,31.8,3.1)*.12
    return overlay(frame,(76,85,80),shade)


def water(frame, t, shift):
    sealed=smooth(.8,1.4,t)
    rise=12.0*smooth(2.0,4.75,t)
    surface=96-rise
    rx=30.3
    # Surface ellipse and transmitted red beneath it. The object's lower end
    # can be seen through this layer with a small refractive displacement.
    edge=smooth(-1,.5,X-125.2)*(1-smooth(-.5,1,X-185.8))
    ey=surface+3.3*np.sqrt(np.maximum(0,1-((X-155.5)/rx)**2))
    depth=smooth(ey-.5,ey+1.1,Y)*(1-smooth(105,109,Y))*edge
    alpha=depth*.54*sealed*smooth(0,4,rise)
    refracted=sample(frame,X+.3*np.sin((Y-surface)/5),Y-1.2)
    red=refracted*np.array([1.13,.59,.52])+np.array([10,-3,-5])
    frame=frame*(1-alpha[...,None])+red*alpha[...,None]
    top=ellipse_mask(155.5,surface,rx,3.4)
    frame=overlay(frame,(123,47,40),top*.30*sealed*smooth(0,4,rise))
    line=np.exp(-((Y-ey)/.50)**2)*edge
    frame=overlay(frame,(169,108,95),line*.23*sealed*smooth(0,4,rise))
    # A damp glass lip, partially below the liquid surface.
    cy=93.5+shift
    ring = raster(lambda d,s: d.ellipse((124.2*s,(cy-5.2)*s,187*s,(cy+5.2)*s),outline=255,width=round(.75*s)),.25)
    frame=overlay(frame,(142,90,79),ring*.29*sealed)
    # Only a faint, quickly damped disturbance in the surrounding bowl.
    amp=.11*np.exp(-2.6*max(0,t-1.1))*smooth(.55,1.1,t)
    for j in range(2):
        rx2=33+8*j+5*max(0,t-1)
        ry2=5.5+1.8*j+.8*max(0,t-1)
        d=np.sqrt(((X-154)/rx2)**2+((Y-97)/ry2)**2)
        rip=np.exp(-((d-1)*8)**2)*(1-top)
        frame=overlay(frame,(171,109,99),rip*amp)
    return frame


def fire_and_smoke(frame, flame, t):
    burn=1-smooth(1.4,2.65,t)
    scale=burn*(1+.065*np.sin(19*t)+.028*np.sin(31*t))
    if scale>.02:
        # Shrink the flame toward the match heads as oxygen is consumed.
        sx=max(.12,.68+.32*scale)
        sy=max(.09,scale)
        xx=145+(X-145-1.0*np.sin(17*t))/sx
        yy=29+(Y-29)/sy
        fire=sample(flame,xx,yy)
        valid=(xx>=0)&(xx<=255)&(yy>=0)&(yy<=143)
        frame+=fire*valid[...,None]*(.85+.15*burn)
        # Soft lower flame, joining the bright tongue to the bundle.
        points=[(142,29),(143,23),(148,17-9*scale),(153,14-9*scale),
                (151,21),(149,25),(148,29)]
        tail=raster(lambda d,s:d.polygon([(x*s,y*s) for x,y in points],fill=255),.65)
        frame=overlay(frame,(214,132,64),tail*.13*burn)
        glow=np.exp(-((X-146)/7)**2-((Y-26)/8)**2)
        frame+=glow[...,None]*np.array([7,3,0])*burn
    ember=(smooth(1.8,2.6,t)*(1-smooth(2.65,3.4,t)))
    e=np.exp(-((X-145.5)/2)**2-((Y-28.7)/1.2)**2)
    frame+=e[...,None]*np.array([35,9,1])*ember
    smoke_strength=smooth(2.1,2.65,t)*(1-smooth(3.35,4.75,t))
    smoke=np.zeros((H,W),dtype=np.float32)
    for k in range(6):
        phase=max(0,t-2.1)*9+k*2.5
        cy=28-phase%23
        cx=147+2.3*np.sin(cy*.27+t*3)+.12*(28-cy)
        smoke+=np.exp(-((X-cx)/(1.7+(28-cy)*.09))**2-((Y-cy)/3.5)**2)
    smoke*=smoke_strength*.028*smooth(3,8,Y)*(1-smooth(27,31,Y))
    return overlay(frame,(171,175,170),np.clip(smoke,0,.075))


def generate():
    fixed, glass, flame=prepare()
    frames=[SOURCE.copy()]
    for i in range(1,20):
        t=i/4
        shift=15.3*smooth(0,1.35,t)
        # Pixels above the source frame are extended from its glass texture;
        # the curved cap supplies the cylinder's newly visible upper boundary.
        yy=Y-shift
        gy=np.where(yy<0,-yy*.7+17,yy)
        moved=sample(glass,X,gy)
        cap_y=-9+shift
        cut=smooth(cap_y-2.5,cap_y+1.2,Y)
        frame=fixed+moved*cut[...,None]
        frame=glass_cap(frame,shift)
        frame=water(frame,t,shift)
        frame=fire_and_smoke(frame,flame,t)
        frames.append(np.uint8(np.clip(np.rint(frame),0,255)))
    return np.stack(frames)


if __name__=='__main__':
    frames=generate()
    candidate=ROOT/'candidates'/'candidate_02'
    candidate.mkdir(parents=True,exist_ok=True)
    np.save(candidate/'frames.npy',frames)
    shutil.copyfile(__file__,candidate/'generate.py')
    artifact_io.save(frames,ROOT)
    shutil.copyfile(ROOT/'inspection.png',candidate/'inspection.png')
    # Lossless RGB encoding retains the exact first-frame pixels in the movie.
    import imageio.v2 as imageio
    with imageio.get_writer(str(ROOT/'video.mp4'),fps=4,codec='libx264rgb',
                            pixelformat='rgb24',macro_block_size=1,
                            ffmpeg_params=['-crf','0','-preset','veryslow']) as writer:
        for frame in frames:
            writer.append_data(frame)
    print('Saved 20 uint8 RGB frames, pixels.npz, index.html, and video.mp4')

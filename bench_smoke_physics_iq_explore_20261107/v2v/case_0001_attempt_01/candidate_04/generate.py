"""Future RGB frames from the historical photograph and explicit pixel motion."""
from pathlib import Path
import math
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import artifact_io

ROOT = Path(__file__).resolve().parent
W, H = 256, 144
PHOTO = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'), dtype=np.float64)
Y, X = np.indices((H, W), dtype=np.float64)


def polygon_mask(points, smooth=False):
    if smooth:
        p = np.asarray(points, float)
        expanded = []
        for i in range(len(p)):
            a, b, c, d = p[(i-1) % len(p)], p[i], p[(i+1) % len(p)], p[(i+2) % len(p)]
            for t in np.linspace(0, 1, 12, endpoint=False):
                expanded.append(.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
        points = expanded
    im = Image.new('L', (W*8, H*8))
    ImageDraw.Draw(im).polygon([(round(x*8), round(y*8)) for x,y in points], fill=255)
    return np.asarray(im.resize((W,H), Image.Resampling.LANCZOS), float)/255


def sample(array, sx, sy):
    sx = np.clip(sx, 0, W-1.001)
    sy = np.clip(sy, 0, H-1.001)
    ix, iy = sx.astype(int), sy.astype(int)
    fx, fy = sx-ix, sy-iy
    if array.ndim == 3:
        fx, fy = fx[...,None], fy[...,None]
    return ((1-fx)*(1-fy)*array[iy,ix] + fx*(1-fy)*array[iy,ix+1]
            +(1-fx)*fy*array[iy+1,ix]+fx*fy*array[iy+1,ix+1])


def wall_patch(left, right):
    # The concealed wall is interpolated from visible pixels on the same rows.
    l = np.median(PHOTO[:,left-4:left], axis=1)
    r = np.median(PHOTO[:,right:right+4], axis=1)
    f = np.clip((X-left)/(right-left),0,1)[...,None]
    return l[:,None,:]*(1-f)+r[:,None,:]*f


BALL_CENTER = (83.5,26.5)
BALL_MASK = polygon_mask([(81.0,14.8),(86,15.1),(90.5,17),(93.5,20.5),
 (95.4,25.5),(95.1,30.0),(92.9,34.1),(89.5,37.1),(84.7,38.5),
 (80.4,38.0),(76.6,36.0),(73.7,32.8),(72.0,28.6),(71.8,24.1),
 (73.3,19.8),(76.5,16.5)], smooth=True)
# Keep the blue jaws separate from the ball's texture.
BLUE = (PHOTO[...,2] > PHOTO[...,0]*1.12) & (PHOTO[...,2] > PHOTO[...,1]*1.10)
BALL_MASK[BLUE & (Y<29)] = 0
BALL_WALL = wall_patch(64,101)
BALL_CLEAR = np.asarray(Image.fromarray(np.uint8(BALL_MASK*255)).filter(ImageFilter.MaxFilter(3)),float)/255
BALL_CLEAR[BLUE & (Y<29)] = 0
BALL_CLEAR[(Y<15)] = 0
BALL_RGB = PHOTO.copy()
# Recover dark fuzzy edges rather than carrying the original wall fringe along.
edge = (BALL_MASK>0) & (BALL_MASK<.90)
u, v = X-BALL_CENTER[0],Y-BALL_CENTER[1]
r = np.maximum(np.hypot(u,v),1)
inner = sample(PHOTO, X-u/r*1.3, Y-v/r*1.3)
BALL_RGB[edge] = inner[edge]
BALL_RGB = BALL_RGB*BALL_MASK[...,None]

BLOCK_CENTER = (217.35,29.25)
BLOCK_MASK = polygon_mask([(210.3,16.1),(219.0,18.8),(224.65,40.9),
                            (223.0,41.8),(216.7,42.1),(211.9,23.4)])
BLOCK_WALL = wall_patch(188,232)
BLOCK_CLEAR = np.asarray(Image.fromarray(np.uint8(BLOCK_MASK*255)).filter(ImageFilter.MaxFilter(3)),float)/255
# Include the photographed antialias fringe below the stationary finger.
orange_fringe = ((X>=212)&(X<=227)&(Y>=25)&(Y<=43)
                 &(PHOTO[...,0]>PHOTO[...,1]*1.35)
                 &(PHOTO[...,0]>PHOTO[...,2]*1.35)&(PHOTO[...,0]>45))
fringe_clear = np.asarray(Image.fromarray(np.uint8(orange_fringe)*255)
                          .filter(ImageFilter.MaxFilter(3)),float)/255
BLOCK_CLEAR = np.maximum(BLOCK_CLEAR,fringe_clear)
# Fully remove pale reddish edge pixels that fall outside the solid silhouette.
edge_cleanup = Image.new('L',(W*8,H*8))
ImageDraw.Draw(edge_cleanup).rectangle((round(215.5*8),18*8,round(227.5*8),44*8),fill=255)
edge_cleanup = edge_cleanup.resize((W,H),Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(.45))
BLOCK_CLEAR = np.maximum(BLOCK_CLEAR,np.asarray(edge_cleanup,float)/255)
# The dark finger is in front of the upper part of the released block.
FINGER = (Y<25.5)&(Y>17)&(X<213)&(X>199)&(np.max(PHOTO,axis=2)<53)
BLOCK_CLEAR[FINGER] = 0
BLOCK_RGB = PHOTO.copy()
covered = (BLOCK_MASK>.10)&(Y<26)&(PHOTO[...,0]<65)
BLOCK_RGB[covered] = np.array([101,16,2])
edge = (BLOCK_MASK>0)&(BLOCK_MASK<.85)
u,v = X-BLOCK_CENTER[0],Y-BLOCK_CENTER[1]
inner = sample(BLOCK_RGB, X-u/np.maximum(np.hypot(u,v),1)*1.1,
                            Y-v/np.maximum(np.hypot(u,v),1)*1.1)
BLOCK_RGB[edge] = inner[edge]
BLOCK_RGB *= BLOCK_MASK[...,None]

# The two blue side arms open by about two pixels at their tips.
LEFT_ARM = polygon_mask([(66.6,14.5),(74.5,14.5),(74.0,20),(73.5,28.5),
                         (70.7,29),(68,24),(66.0,19)])
RIGHT_ARM = polygon_mask([(91.6,14.5),(97.7,14.5),(98,22.5),(96.8,28.8),
                          (94.7,28.9),(93.4,23)])
LEFT_ARM *= (1-BALL_MASK)
RIGHT_ARM *= (1-BALL_MASK)
# Only copy the arm; concealed brown areas must never remain on the gripper.
for m in (LEFT_ARM,RIGHT_ARM):
    m[(PHOTO[...,0]>PHOTO[...,2]*1.3)&(Y>16)] = 0
ARM_CLEAR = np.maximum(LEFT_ARM,RIGHT_ARM)

BASE = PHOTO*(1-BALL_CLEAR[...,None])+BALL_WALL*BALL_CLEAR[...,None]
BASE = BASE*(1-BLOCK_CLEAR[...,None])+BLOCK_WALL*BLOCK_CLEAR[...,None]
BASE = BASE*(1-ARM_CLEAR[...,None])+BALL_WALL*ARM_CLEAR[...,None]


def layer(rgb, mask, origin, center, angle=0, scale_x=1, scale_y=1):
    a = math.radians(angle)
    dx,dy = X-center[0],Y-center[1]
    sx = origin[0]+(math.cos(a)*dx+math.sin(a)*dy)/scale_x
    sy = origin[1]+(-math.sin(a)*dx+math.cos(a)*dy)/scale_y
    return sample(rgb,sx,sy), sample(mask,sx,sy)


def over(base, rgb, alpha):
    return base*(1-alpha[...,None])+rgb


RELEASE = .27
G = 1320.0
BALL_FLOOR = 104.3
BLOCK_FLOOR = 103.8
BALL_HIT = math.sqrt(2*(BALL_FLOOR-BALL_CENTER[1])/G)
BLOCK_HIT = math.sqrt(2*(BLOCK_FLOOR-BLOCK_CENTER[1])/G)


def ball_pose(t):
    q = t-RELEASE
    if q<0:
        return (*BALL_CENTER,0,1,1,0)
    if q < BALL_HIT:
        return (BALL_CENTER[0]+1.0*q, BALL_CENTER[1]+.5*G*q*q,
                14*q,1,1,0)
    s = q-BALL_HIT
    # A soft pillow absorbs most of the impact. A low, damped rebound follows.
    rebound_duration = .30
    if s < rebound_duration:
        height = 5.5*math.sin(math.pi*s/rebound_duration)
        cy = BALL_FLOOR-height+1.15*math.exp(-s/.05)
    else:
        cy = BALL_FLOOR+.38*math.exp(-(s-rebound_duration)/.18)
    roll = 3.4*(1-math.exp(-s/0.62))
    cx = BALL_CENTER[0]+.36+roll
    angle = 5+roll/12*180/math.pi
    squish = .075*math.exp(-s/.045)
    return (cx,cy,angle,1+squish,1-squish,.70+.3*math.exp(-s/.3))


def block_pose(t):
    q = t-(RELEASE+.035)
    if q<0:
        return (*BLOCK_CENTER,0,0)
    if q < BLOCK_HIT:
        return (BLOCK_CENTER[0]+.7*q, BLOCK_CENTER[1]+.5*G*q*q, 8*q,0)
    s = q-BLOCK_HIT
    # The long block tips on its first contact, then rests on the fabric.
    turn = 1-math.exp(-s/.14)
    angle = -71*turn + 3.5*math.sin(23*s)*math.exp(-s/.22)
    relative = math.radians(angle-13)
    halfheight = abs(12.2*math.cos(relative))+abs(4.5*math.sin(relative))
    bottom = 116.7+1.3*math.exp(-s/.11)
    return (BLOCK_CENTER[0]+1.7*turn,bottom-halfheight,angle,1)


def pillow_warp(base,t):
    bp=ball_pose(t)
    kp=block_pose(t)
    ball_age=t-RELEASE-BALL_HIT
    block_age=t-RELEASE-.035-BLOCK_HIT
    displace=np.zeros((H,W))
    if ball_age>=0:
        amount=.55+1.3*math.exp(-ball_age/.14)
        displace+=amount*np.exp(-((X-bp[0])/18)**2-((Y-116)/13)**2)
    if block_age>=0:
        amount=.35+.95*math.exp(-block_age/.16)
        displace+=amount*np.exp(-((X-kp[0])/20)**2-((Y-117)/12)**2)
    # Limit the indentation to cloth, preserving the wall and pillow silhouette.
    boundary=np.where(X<137,102+np.maximum(0,50-X)*.3,96+np.maximum(0,X-191)*.13)
    cloth=np.clip((Y-boundary-1)/6,0,1)
    return sample(base,X,Y-displace*cloth)


def render(t):
    if t<RELEASE:
        return PHOTO.copy()
    base=pillow_warp(BASE.copy(),t)
    opening=np.clip((t-RELEASE)/.12,0,1)
    for m, direction in [(LEFT_ARM,-1),(RIGHT_ARM,1)]:
        sx=X-direction*2.0*opening*np.clip((Y-14.5)/14,0,1)
        a=sample(m,sx,Y)
        rgb=sample(PHOTO*m[...,None],sx,Y)
        base=over(base,rgb,a)
    # Short shutter exposure makes the falling objects slightly blurred.
    accum=np.zeros_like(base)
    for shutter in [-.006,0,.006]:
        tm=t+shutter
        frame=base.copy()
        bx,by,ba,bsx,bsy,contact=ball_pose(tm)
        kx,ky,ka,kcontact=block_pose(tm)
        # Contact shadows sit on the pillow, darkening and narrowing at landing.
        if tm>=RELEASE:
            proximity=np.clip(1-max(BALL_FLOOR-by,0)/65,0,1)
            shadow=.19*proximity*np.exp(-((X-(bx+2))/12)**2-((Y-117.5)/3.9)**2)
            if by<90:
                shadow+=.034*np.exp(-((X-bx-4)/14)**2-((Y-by-5)/12)**2)
            frame *= (1-shadow[...,None])
        if tm>=RELEASE+.035:
            proximity=np.clip(1-max(BLOCK_FLOOR-ky,0)/65,0,1)
            shadow=.18*proximity*np.exp(-((X-kx-1.0)/15)**2-((Y-117.4)/3.0)**2)
            frame *= (1-shadow[...,None])
        rgb,a=layer(BALL_RGB,BALL_MASK,BALL_CENTER,(bx,by),ba,bsx,bsy)
        frame=over(frame,rgb,a)
        rgb,a=layer(BLOCK_RGB,BLOCK_MASK,BLOCK_CENTER,(kx,ky),ka)
        # The upper face is lit a little more after tipping onto the pillow.
        rgb*=1+.06*kcontact*(1-math.exp(-max(tm-RELEASE-.035-BLOCK_HIT,0)/.3))
        frame=over(frame,rgb,a)
        # Restore the stationary foreground finger while objects leave its grip.
        frame[FINGER]=PHOTO[FINGER]
        accum+=frame/3
    return accum


def generate():
    return np.stack([np.clip(np.rint(render(i/4)),0,255).astype(np.uint8) for i in range(20)])


def main():
    frames=generate()
    candidate=ROOT/'candidate_v3'
    candidate.mkdir(exist_ok=True)
    for filename in ['prompt.txt','task_spec.json']:
        shutil.copyfile(ROOT/filename,candidate/filename)
    shutil.copyfile(ROOT/'generate.py',candidate/'generate.py')
    artifact_io.save(frames,candidate)
    artifact_io.save(frames,ROOT)
    # A denser temporal sheet makes the release and soft contact easy to inspect.
    sheet=Image.new('RGB',(W*5,(H+20)*4),(23,25,29))
    d=ImageDraw.Draw(sheet)
    for i,f in enumerate(frames):
        x,y=(i%5)*W,(i//5)*(H+20)
        d.text((x+4,y+3),f'{i:02d}  future {i/4:.2f}s',fill='white')
        sheet.paste(Image.fromarray(f),(x,y+20))
    sheet.save(candidate/'all_frames.png')
    Image.fromarray(np.uint8(np.clip(BASE,0,255))).save(ROOT/'.tmp'/'cleared_background.png')
    print(f'Saved {frames.shape}, {frames.dtype}; range {frames.min()}..{frames.max()}')

if __name__=='__main__':
    main()

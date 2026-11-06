"""Render a clockwise striking arm and a domino cascade from supplied RGB.

Only local image operations and analytic cuboid geometry are used. The camera
is fixed; the original first frame is retained verbatim. Source face textures
are projected onto rotating cuboids whose ground contact corners stay fixed.
"""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve
from PIL import Image, ImageDraw
import artifact_io

ROOT = Path(__file__).resolve().parent
SS = 4
FRAME0 = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
H, W = FRAME0.shape[:2]
ORDER = ['Y', 'R', 'K', 'G', 'B', 'P']
# Front-left, front-right, back-right, back-left top corners, plus the
# front-left and front-right ground corners. Coordinates are from frame 0.
CORNERS = {
 'Y': [[49.5,52.5],[56.5,54],[65,49],[58,47.5],[56,91],[63,92.5]],
 'R': [[67,55],[75,56],[82,51.5],[74,50.5],[73,95],[80.5,97]],
 'K': [[84,59],[92.5,60],[101,54.5],[92.5,53],[88.5,99],[97,101]],
 'G': [[104,60],[112.5,62],[120,56],[111.5,54.5],[110,104],[118.5,106]],
 'B': [[124,63],[132.5,64.5],[139.5,59],[131,57.5],[128,106],[136.5,108]],
 'P': [[136,65],[144,65.5],[150,60],[142.5,59],[143,109.5],[150.5,111]],
}
CORNERS = {k: np.asarray(v, dtype=np.float64) for k,v in CORNERS.items()}
ROW = np.array([0.98, 0.20])
# Strike begins at 0.5 sec, each collision launches its immediate neighbour.
START = {'P': .52, 'B': .69, 'G': .85, 'K': 1.01, 'R': 1.17, 'Y': 1.33}
REST = {'Y': 89.8, 'R': 83.0, 'K': 78.0, 'G': 74.0, 'B': 70.5, 'P': 67.5}
DURATION = .62


def poly_mask(points, scale=1):
    mask = np.zeros((H*scale, W*scale), np.uint8)
    cv2.fillConvexPoly(mask, np.round(points*scale).astype(np.int32), 255)
    return mask


def vertices(name, angle):
    p = CORNERS[name]
    origin, right = p[4], p[5]
    up = p[0] - origin
    across = right - origin
    depth = .5 * ((p[3]-p[0]) + (p[2]-p[1]))
    height = abs(up[1])
    width = np.linalg.norm(across)
    a = np.deg2rad(angle)
    # Rotate in the vertical plane of the row, about the lower left edge.
    rise = up*np.cos(a) - ROW*height*np.sin(a)
    edge = across*np.cos(a) + (up/height)*width*np.sin(a)
    bottom = np.array([origin, origin+edge, origin+edge+depth, origin+depth])
    top = bottom + rise
    if angle == 0:
        top = p[:4].copy()
        bottom[1] = right
    return bottom, top


# Erase the photographed objects while retaining all static scene pixels.
yy, xx = np.mgrid[:H, :W]
block_masks = {}
object_union = np.zeros((H,W), np.uint8)
rr,gg,bb=FRAME0.astype(np.int16).transpose(2,0,1)
color_masks={
 'Y': (rr>160)&(gg>105)&(bb<95)&(rr-bb>80),
 'R': (rr>70)&(gg<80)&(bb<70)&(rr-gg>35),
 'K': (rr>150)&(gg>85)&(gg<150)&(bb>80)&(bb<140)&(rr-gg>40),
 'G': (gg>40)&(gg>rr+12)&(gg>bb+12),
 'B': (bb>80)&(bb>rr+45),
 'P': (rr>50)&(bb>55)&(gg<rr-12)&(gg<bb-15),
}
for name in ORDER:
    bottom, top = vertices(name,0)
    hull = cv2.convexHull(np.vstack([bottom,top]).astype(np.float32))[:,0]
    m = poly_mask(hull)
    zone=(xx>45)&(xx<162)&(yy>44)&(yy<115)
    m[color_masks[name]&zone]=255
    m = cv2.dilate(m,np.ones((3,3),np.uint8))
    block_masks[name] = m
    object_union = np.maximum(object_union,m)

# Original block shadows occupy the narrow strip below the bases.
shadow_erase = np.zeros((H,W),np.uint8)
cv2.fillConvexPoly(shadow_erase,np.array([[53,85],[153,102],[165,116],[64,101]],np.int32),255)
table_erase = np.maximum(object_union,shadow_erase)
table_erase=cv2.dilate(table_erase,np.ones((5,5),np.uint8))
# Fit the broad tabletop colour gradient using the surrounding valid table.
valid = (xx<169)&(yy>44)&(table_erase==0)
xn,yn=xx/W,yy/H
basis=np.stack([np.ones_like(xn),xn,yn,xn*yn,xn*xn,yn*yn],axis=-1)
coef=np.linalg.lstsq(basis[valid],FRAME0[valid].astype(float),rcond=None)[0]
wood_base=basis@coef
# Reuse grain from the unobstructed lower table at the same grain angle.
low=cv2.GaussianBlur(FRAME0.astype(np.float32),(0,0),8)
residual=FRAME0.astype(np.float32)-low
source_y=118+(yy-47)*.35
source_x=xx-.65*(source_y-yy)
source_x=np.clip(source_x,2,168).astype(np.float32)
source_y=np.clip(source_y,118,142).astype(np.float32)
grain=cv2.remap(residual,source_x,source_y,cv2.INTER_LINEAR)
wood=np.clip(wood_base+grain,0,255)
# Match the surrounding illumination by solving the classical Poisson image
# blending problem, using only the local table grain and photographed border.
inside=table_erase>0
ys,xs=np.where(inside)
count=len(ys)
indices=np.full((H,W),-1,dtype=np.int32)
indices[inside]=np.arange(count)
row_ids=[np.arange(count)]
col_ids=[np.arange(count)]
coeffs=[np.full(count,4.)]
rhs=4.*wood[ys,xs].astype(np.float64)
for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
    ny,nx=ys+dy,xs+dx
    rhs-=wood[ny,nx]
    neighbour=indices[ny,nx]
    connected=neighbour>=0
    row_ids.append(np.flatnonzero(connected))
    col_ids.append(neighbour[connected])
    coeffs.append(np.full(connected.sum(),-1.))
    rhs[~connected]+=FRAME0[ny[~connected],nx[~connected]]
matrix=sparse.coo_matrix((np.concatenate(coeffs),
 (np.concatenate(row_ids),np.concatenate(col_ids))),shape=(count,count)).tocsc()
plate=FRAME0.astype(np.float32).copy()
plate[inside]=np.clip(spsolve(matrix,rhs),0,255)

PIVOT=np.array([208.,57.])
TIP0=np.array([152.,80.5])
# A circle in the tabletop plane is an ellipse in this fixed camera view.
# These projection scales match the arm's initial endpoint exactly.
AX=75.
AY=35.
PHASE=np.arctan2((TIP0-PIVOT)[1]/AY,(TIP0-PIVOT)[0]/AX)
AX=(TIP0-PIVOT)[0]/np.cos(PHASE)
AY=(TIP0-PIVOT)[1]/np.sin(PHASE)

arm_erase=np.zeros((H,W),np.uint8)
cv2.line(arm_erase,tuple(TIP0.astype(int)),(201,60),255,5)
cv2.fillConvexPoly(arm_erase,np.array([[185,61],[198,54],[209,55],[207,60],[193,68]],np.int32),255)
# Do not erase the fixed little hub at the right-hand end of the black linkage.
arm_erase[((xx-208)**2+(yy-57)**2)<3.5**2]=0
arm_clean=cv2.inpaint(FRAME0,arm_erase,4,cv2.INPAINT_TELEA).astype(np.float32)
# Outside the platform, the arm reveals the same tabletop texture.
arm_table=(arm_erase>0)&(xx<172)
arm_clean[arm_table]=plate[arm_table]
arm_clean[arm_table&(table_erase==0)]=wood[arm_table&(table_erase==0)]
plate[arm_erase>0]=arm_clean[arm_erase>0]
plate=np.clip(plate,0,255).astype(np.uint8)

SOURCE_BIG=cv2.resize(FRAME0,(W*SS,H*SS),interpolation=cv2.INTER_CUBIC)
# Each source face's original texture is warped onto its changing quadrilateral.
FACE_SOURCE={}
for name in ORDER:
    bottom,top=vertices(name,0)
    FACE_SOURCE[name]={
      'front':np.array([bottom[0],bottom[1],top[1],top[0]]),
      'side':np.array([bottom[1],bottom[2],top[2],top[1]]),
      'top':top,
    }


def render_face(canvas,src,dst,light=1.):
    if abs(cv2.contourArea(dst.astype(np.float32)))<.1:
        return
    transform=cv2.getPerspectiveTransform((src*SS).astype(np.float32),(dst*SS).astype(np.float32))
    rgb=cv2.warpPerspective(SOURCE_BIG,transform,(W*SS,H*SS),flags=cv2.INTER_LINEAR)
    m=poly_mask(dst,SS).astype(np.float32)/255
    canvas[:]=canvas*(1-m[...,None])+np.clip(rgb.astype(np.float32)*light,0,255)*m[...,None]


def render_block(canvas,name,angle):
    bottom,top=vertices(name,angle)
    if angle < .01:
        # Restore photographed stationary blocks, including their natural edges.
        m=cv2.resize(block_masks[name],(W*SS,H*SS),interpolation=cv2.INTER_NEAREST)>0
        canvas[m]=SOURCE_BIG[m]
        return
    faces={
      'front':np.array([bottom[0],bottom[1],top[1],top[0]]),
      'side':np.array([bottom[1],bottom[2],top[2],top[1]]),
      'top':top,
    }
    # The broad front remains visible as it tips; keep its photographed grain.
    for face in ('side','top','front'):
        light=1.+(.055 if face=='front' else -.025)*np.sin(np.deg2rad(angle))
        render_face(canvas,FACE_SOURCE[name][face],faces[face],light)


def fall_angle(t,name):
    q=np.clip((t-START[name])/DURATION,0,1)
    # Accelerating fall, followed by a small damped settling motion.
    progress=(1-np.cos(q*np.pi*.5))**.8
    a=REST[name]*progress
    after=t-(START[name]+DURATION)
    if after>0:
        a-=1.3*np.exp(-after*7)*abs(np.sin(after*19))
    return float(a)


def arm_angle(t):
    # Constant clockwise motion through contact, then a gentle motor stop.
    if t<=2.5:
        return 22*t
    q=np.clip((t-2.5)/.6,0,1)
    return 55+13.2*(q-.5*q*q)


def draw_arm(canvas,angle):
    phase=PHASE+np.deg2rad(angle)
    tip=PIVOT+np.array([AX*np.cos(phase),AY*np.sin(phase)])
    v=tip-PIVOT
    norm=v/np.linalg.norm(v)
    n=np.array([-norm[1],norm[0]])
    # The black chuck occupies the first quarter of the radius.
    wood_start=PIVOT+v*.26
    im=Image.fromarray(np.clip(canvas,0,255).astype(np.uint8))
    dr=ImageDraw.Draw(im)
    def line(a,b,color,width):
        dr.line([tuple(a*SS),tuple(b*SS)],fill=color,width=max(1,round(width*SS)))
    line(PIVOT+v*.05,PIVOT+v*.27,(23,22,20),4.0)
    line(PIVOT+v*.05-n*.55,PIVOT+v*.27-n*.55,(43,41,37),1.)
    # A thin wooden dowel, with a lit edge and a small rounded end.
    line(wood_start,tip,(115,88,54),2.35)
    line(wood_start-n*.38,tip-n*.38,(189,161,115),1.6)
    line(wood_start-n*.7,tip-n*.7,(215,188,142),.45)
    r=.8*SS
    tx,ty=tip*SS
    dr.ellipse((tx-r,ty-r,tx+r,ty+r),fill=(183,153,104))
    return np.asarray(im).astype(np.float32)


def add_shadows(canvas,angles):
    shadow=np.zeros((H*SS,W*SS),np.float32)
    contact=np.zeros_like(shadow)
    for name in ORDER:
        bottom,top=vertices(name,angles[name])
        # Project each vertex onto the table away from the broad light source.
        p=CORNERS[name]
        origin=p[4]
        height=abs((p[0]-origin)[1])
        width=np.linalg.norm(p[5]-origin)
        up=(p[0]-origin)/height
        a=np.deg2rad(angles[name])
        top_h=height*np.cos(a)+np.array([0,width,width,0])*np.sin(a)
        bot_h=np.array([0,width,width,0])*np.sin(a)
        shift=np.array([.28,.13])-up
        projected=np.vstack([bottom+bot_h[:,None]*shift,top+top_h[:,None]*shift])
        hull=cv2.convexHull(projected.astype(np.float32))[:,0]
        temp=poly_mask(hull,SS).astype(np.float32)/255
        shadow=np.maximum(shadow,temp*.17)
        # Soft contact darkness only near the actual footprint and resting faces.
        ground=cv2.convexHull(np.vstack([bottom,top if angles[name]>60 else bottom]).astype(np.float32))[:,0]
        temp=poly_mask(ground,SS).astype(np.float32)/255
        contact=np.maximum(contact,temp*(.12 if angles[name]>60 else .05))
    shadow=cv2.GaussianBlur(shadow,(0,0),1.9*SS)
    contact=cv2.GaussianBlur(contact,(0,0),.6*SS)
    strength=np.clip(shadow+contact,0,.28)
    return canvas*(1-strength[...,None])


def render(t):
    angles={name:fall_angle(t,name) for name in ORDER}
    canvas=cv2.resize(plate,(W*SS,H*SS),interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # Keep the photographed contact shadows until their associated block moves.
    unmoved=np.zeros((H,W),np.uint8)
    for name in ORDER:
        if angles[name]<.01:
            bottom,_=vertices(name,0)
            m=np.zeros((H,W),np.uint8)
            p=bottom.mean(axis=0)
            cv2.ellipse(m,(round(p[0]+6),round(p[1]+2)),(10,4),8,0,360,255,-1)
            unmoved=np.maximum(unmoved,m)
    m=cv2.resize(unmoved,(W*SS,H*SS),interpolation=cv2.INTER_LINEAR)/255.
    canvas=canvas*(1-m[...,None])+SOURCE_BIG*m[...,None]
    canvas=add_shadows(canvas,angles)
    canvas=draw_arm(canvas,arm_angle(t))
    for name in ORDER:
        render_block(canvas,name,angles[name])
    frame=cv2.resize(np.clip(canvas,0,255).astype(np.uint8),(W,H),interpolation=cv2.INTER_AREA)
    # Preserve every background pixel outside the region occupied by any moving
    # object, its original location, and its soft shadow.
    active=arm_erase.copy()
    for name in ORDER:
        if angles[name]<.01:
            continue
        active=np.maximum(active,cv2.dilate(block_masks[name],np.ones((5,5),np.uint8)))
        bottom,top=vertices(name,angles[name])
        hull=cv2.convexHull(np.vstack([bottom,top]).astype(np.float32))[:,0]
        active=np.maximum(active,poly_mask(hull))
        p=CORNERS[name]
        origin=p[4]
        height=abs((p[0]-origin)[1])
        width=np.linalg.norm(p[5]-origin)
        a=np.deg2rad(angles[name])
        top_h=height*np.cos(a)+np.array([0,width,width,0])*np.sin(a)
        bot_h=np.array([0,width,width,0])*np.sin(a)
        shift=np.array([.28,.13])-(p[0]-origin)/height
        projected=np.vstack([bottom+bot_h[:,None]*shift,top+top_h[:,None]*shift])
        hull=cv2.convexHull(projected.astype(np.float32))[:,0]
        active=np.maximum(active,poly_mask(hull))
        center=vertices(name,0)[0].mean(axis=0)+[8,3]
        cv2.ellipse(active,tuple(np.round(center).astype(int)),(16,5),8,0,360,255,-1)
    phase=PHASE+np.deg2rad(arm_angle(t))
    tip=PIVOT+np.array([AX*np.cos(phase),AY*np.sin(phase)])
    cv2.line(active,tuple(PIVOT.astype(int)),tuple(tip.astype(int)),255,7)
    # Include the soft cast-shadow region; restore all other source RGB values.
    active=cv2.dilate(active,np.ones((13,13),np.uint8))
    frame[active==0]=FRAME0[active==0]
    return frame


def main():
    frames=np.stack([FRAME0]+[render(i/4) for i in range(1,20)]).astype(np.uint8)
    assert frames.shape==(20,144,256,3)
    assert np.array_equal(frames[0],FRAME0)
    revision=ROOT/'.tmp/revisions/04'
    revision.mkdir(parents=True,exist_ok=True)
    for filename in ('prompt.txt','task_spec.json','generate.py'):
        shutil.copy2(ROOT/filename,revision/filename)
    artifact_io.save(frames,revision)
    artifact_io.save(frames,ROOT)
    np.save(ROOT/'.tmp/rendered_frames.npy',frames)
    # Lossless RGB H.264 keeps even the encoded video's first frame exact.
    import imageio_ffmpeg
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo',
      '-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate','4',
      '-i','pipe:0','-an','-c:v','libx264rgb','-crf','0','-preset','medium',
      '-pix_fmt','rgb24','-movflags','+faststart',str(ROOT/'video.mp4')]
    subprocess.run(command,input=frames.tobytes(),stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,check=True)
    print('Saved 20 RGB frames at 256x144, 4 fps; frame 00 matches exactly.')

if __name__=='__main__':
    main()

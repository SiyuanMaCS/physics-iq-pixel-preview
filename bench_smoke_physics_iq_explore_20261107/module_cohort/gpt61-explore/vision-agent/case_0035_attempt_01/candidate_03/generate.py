from pathlib import Path
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw
from artifact_io import save

ROOT = Path(__file__).resolve().parent
SOURCE = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'))
S = 4
W,H=256,144
# Coordinates of the actual painted faces in the supplied still.
BLOCKS=[
 {'front':[(54,106),(64,106),(59,58),(48,59)], 'depth':(4,-7), 'start':1.50},
 {'front':[(79,108),(89,108),(87,59),(76,59)], 'depth':(2,-6), 'start':1.30},
 {'front':[(101,109),(111,109),(111,61),(100,61)], 'depth':(2,-7), 'start':1.10},
 {'front':[(124,109),(134,109),(134,60),(124,60)], 'depth':(1,-7), 'start':.90},
 {'front':[(146,109),(156,109),(157,60),(147,60)], 'depth':(0,-6), 'start':.70},
 {'front':[(166,110),(177,109),(175,61),(164,61)], 'depth':(-2,-6), 'start':.50},
]

def polygon_mask(points, shape=(H*S,W*S)):
    m=np.zeros(shape,np.uint8)
    cv2.fillPoly(m,[np.rint(np.array(points)*S).astype(np.int32)],255,lineType=cv2.LINE_AA)
    return m

# Recover the small occluded portions of the stationary table and platform.
mask=np.zeros((H,W),np.uint8)
for b in BLOCKS:
    f=np.array(b['front'],np.float32); d=np.array(b['depth'])
    hull=cv2.convexHull(np.vstack([f,f+d]).astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask,hull,255)
mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
cv2.line(mask,(175,77),(205,49),255,5)
clean_source=cv2.inpaint(SOURCE,mask,7,cv2.INPAINT_TELEA)
background=clean_source.copy()
# Source-derived wood texture under the removed upright blocks; an affine
# table-plane mapping continues the grain from a clear part of the same still.
yy,xx=np.mgrid[:H,:W].astype(np.float32)
texture=cv2.remap(SOURCE, np.clip(xx+(yy-122)*.18,0,255),np.clip(121+(yy-60)*.28,115,142),cv2.INTER_LINEAR)
wood_mask=mask.copy()
# The circular platform is fixed. Restore its silhouette where the purple
# block previously hid it, using colors sampled on its unobstructed surface.
platform=np.zeros((H,W),np.uint8)
cv2.ellipse(platform,(209,49),(44,15),3,0,360,255,-1)
cv2.ellipse(platform,(209,69),(42,17),0,0,180,255,-1)
cv2.rectangle(platform,(168,50),(249,69),255,-1)
# Better approximation of the left cylindrical wall.
cv2.fillPoly(platform,[np.array([(166,51),(169,67),(175,77),(187,82),(209,86),(235,82),(248,73),(251,55)],np.int32)],255)
platform=cv2.GaussianBlur(platform,(3,3),.45)
for y in range(48,115):
    for x in range(40,182):
        if mask[y,x]:
            if platform[y,x]>128 and x>=165 and y<85:
                background[y,x]=clean_source[y,min(255,x+24)]
            else:
                background[y,x]=texture[y,x]
# Continue one contiguous patch of table grain, with a feathered boundary.
# Using a common patch avoids silhouettes at the vacated block locations.
replacement=np.zeros((H,W),np.uint8)
cv2.rectangle(replacement,(43,50),(179,112),255,-1)
replacement=cv2.GaussianBlur(replacement,(25,25),7).astype(np.float32)/255
replacement=np.maximum(replacement,mask.astype(np.float32)/255)
# The fixed black fixture sits in front of this table patch.
platform=np.zeros((H,W),np.uint8)
cv2.ellipse(platform,(209,49),(44,15),3,0,360,255,-1)
cv2.rectangle(platform,(166,49),(251,68),255,-1)
cv2.ellipse(platform,(209,68),(42,17),0,0,180,255,-1)
platform=cv2.GaussianBlur(platform,(3,3),.45)
replacement*=1-platform.astype(np.float32)/255
background=(background*(1-replacement[...,None])+texture*replacement[...,None]).astype(np.uint8)
# Reconstruct only the occluded sliver of the platform from the same still.
for y in range(48,86):
    for x in range(163,182):
        if mask[y,x] and platform[y,x]>128:
            background[y,x]=clean_source[y,min(255,x+24)]
BG=cv2.resize(background,(W*S,H*S),interpolation=cv2.INTER_CUBIC).astype(np.float32)
SRC=cv2.resize(SOURCE,(W*S,H*S),interpolation=cv2.INTER_CUBIC).astype(np.float32)

# Texture patches are all measured from the six original painted blocks.
def face(canvas, src_points, dst_points, tone=1.0):
    src=np.array(src_points,np.float32)*S; dst=np.array(dst_points,np.float32)*S
    if abs(cv2.contourArea(dst)) < .4: return
    M=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(SRC,M,(W*S,H*S),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    alpha=polygon_mask(dst_points).astype(np.float32)/255
    # Supersampling preserves a soft photographic edge at the final size.
    canvas[:]=canvas*(1-alpha[...,None])+np.clip(warped*tone,0,255)*alpha[...,None]


def transformed(b, angle):
    f=np.array(b['front'],np.float32)
    p=f[0]; c,s=np.cos(angle),np.sin(angle)
    out=f.copy()
    u=np.array([0,1,1,0],np.float32)
    width=f[1,0]-f[0,0]
    out[:,0]=p[0]+(f[:,0]-p[0])*c+(f[:,1]-p[1])*s
    out[:,1]=p[1]+(f[:,1]-p[1])*c-u*width*s
    return out


def angle_at(t,b,index):
    u=(t-b['start'])/.72
    if u<=0: return 0.
    # A short accelerating fall, then contact with the table or the preceding
    # fallen block. Overlap determines the slightly raised resting angles.
    final=math.radians(89 if index==0 else 77.0)
    if u<1:
        return final*(u*u*(3-2*u))
    return final + math.radians(1.8)*math.exp(-(u-1)*6)*math.sin((u-1)*13)


def draw_stick(canvas,t):
    pivot=np.array([204.,49.])
    theta=2.215 + t*.50
    tip=pivot+np.array([47.5*math.cos(theta),35.0*math.sin(theta)])
    start=pivot+.24*(tip-pivot)
    # Thin cast shadow, offset in the same direction as the source lighting.
    layer=np.zeros((H*S,W*S),np.float32)
    cv2.line(layer,tuple(np.rint((start+[2,3])*S).astype(int)),tuple(np.rint((tip+[2,3])*S).astype(int)),.21,3*S,cv2.LINE_AA)
    layer=cv2.GaussianBlur(layer,(11,11),2.3)
    canvas*=1-layer[...,None]
    p0=tuple(np.rint(start*S).astype(int)); p1=tuple(np.rint(tip*S).astype(int))
    cv2.line(canvas,p0,p1,(116,87,48),int(2.3*S),cv2.LINE_AA)
    cv2.line(canvas,p0,p1,(192,165,109),int(1.6*S),cv2.LINE_AA)
    off=np.array([-.4,-.5])
    cv2.line(canvas,tuple(np.rint((start+off)*S).astype(int)),tuple(np.rint((tip+off)*S).astype(int)),(212,190,141),int(.55*S),cv2.LINE_AA)
    cv2.line(canvas,tuple(np.rint(pivot*S).astype(int)),p0,(35,34,34),int(2.2*S),cv2.LINE_AA)
    cv2.ellipse(canvas,tuple(np.rint(pivot*S).astype(int)),(4*S,int(1.6*S)),0,0,360,(46,45,46),-1,cv2.LINE_AA)


def render(t):
    canvas=BG.copy()
    angles=[angle_at(t,b,i) for i,b in enumerate(BLOCKS)]
    # Directly drawn soft shadows follow the bottom and descending upper end.
    shadow=np.zeros((H*S,W*S),np.float32)
    for b,a in zip(BLOCKS,angles):
        f=transformed(b,a)
        foot=np.array(b['front'][:2],np.float32)
        top=f[2:]
        s=np.array([foot[0]+[3,0],foot[1]+[12,-1],top[0]+[11,7+20*np.cos(a)],top[1]+[3,7+20*np.cos(a)]])
        cv2.fillPoly(shadow,[np.rint(s*S).astype(np.int32)],.21,cv2.LINE_AA)
        q=np.vstack([f,f+np.array(b['depth'])])
        # Table projection of the body, with a compact contact shadow.
        q[:,1]=np.maximum(102,q[:,1])+2
        hull=cv2.convexHull(q.astype(np.float32))
        cv2.fillConvexPoly(shadow,np.rint(hull*S).astype(np.int32),.22,cv2.LINE_AA)
    shadow=cv2.GaussianBlur(shadow,(25,25),5)
    canvas*=1-shadow[...,None]
    draw_stick(canvas,t)
    # The rear object is painted first; the descending blocks overlap like
    # solid wooden pieces instead of transparent sprites.
    for i in range(5,-1,-1):
        b=BLOCKS[i];a=angles[i]
        f=np.array(b['front'],np.float32);d=np.array(b['depth'],np.float32)
        q=transformed(b,a)
        # Top end grain.
        face(canvas,[f[3],f[2],f[2]+d,f[3]+d],[q[3],q[2],q[2]+d,q[3]+d],1-.12*np.sin(a))
        # The right narrow face becomes the upper broad face during a fall.
        right=d[0]*np.cos(a)-d[1]*np.sin(a)>0
        if right:
            srcside=[f[1],f[1]+d,f[2]+d,f[2]]
            if d[0]<=.6:
                # This face was hidden at time zero, so use the same block's
                # front wood texture for the newly exposed surface.
                srcside=[f[0],f[1],f[2],f[3]]
                shade=1.05+.17*np.sin(a)
            else:
                shade=1+.32*np.sin(a)
            face(canvas,srcside,[q[1],q[1]+d,q[2]+d,q[2]],shade)
        else:
            face(canvas,[f[0],f[3],f[3]+d,f[0]+d],[q[0],q[3],q[3]+d,q[0]+d],1)
        face(canvas,f,q,1-.07*np.sin(a))
    return np.clip(cv2.resize(canvas,(W,H),interpolation=cv2.INTER_AREA),0,255).astype(np.uint8)

frames=np.stack([SOURCE.copy() if k==0 else render(k/4) for k in range(20)])
# Keep the first candidate so subsequent inspection never destroys its history.
np.savez_compressed(ROOT/'candidate_03.npz',frames=frames)
save(frames,ROOT)
Image.fromarray(background).save(ROOT/'.tmp/background.png')
print('Saved',frames.shape,frames.dtype,'first frame exact:',np.array_equal(frames[0],SOURCE))

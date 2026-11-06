from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from artifact_io import save

ROOT = Path(__file__).resolve().parent
ORIGINAL = Image.open(ROOT / 'first_frame.png').convert('RGB')
W, H = ORIGINAL.size

def polygon(points):
    m = Image.new('L', (W,H)); ImageDraw.Draw(m).polygon(points,fill=255)
    return m

def rgba(im, mask):
    out=im.convert('RGBA'); out.putalpha(mask); return out

main = polygon([(256,18),(242,26),(221,39),(211,38),(202,43),(138,80),(144,109),(148,113),(169,107),(215,81),(225,81),(245,75),(256,72)])
upper = polygon([(122,77),(126,72),(127,65),(131,62),(134,56),(155,55),(158,58),(157,69),(153,69),(152,61),(139,61),(136,65),(140,69),(149,74),(143,78),(135,72),(132,71),(127,76)])
lower = polygon([(133,118),(140,115),(145,111),(149,110),(152,115),(157,114),(162,109),(169,108),(166,116),(158,123),(151,127),(146,125),(144,122),(140,121),(134,120)])
cable=Image.new('L',(W,H)); cd=ImageDraw.Draw(cable)
# The two flexible service lines are copied along with the rigid wrist assembly.
cd.line([(149,109),(149,99),(152,89),(158,82),(169,78),(181,77),(193,72),(204,70)],fill=255,width=5)
cd.line([(142,119),(150,120),(164,109),(177,99),(187,91),(199,86),(211,83),(226,83),(243,87),(255,94)],fill=255,width=5)
whole=np.maximum.reduce([np.asarray(m) for m in [main,upper,lower,cable]])
whole=Image.fromarray(whole).filter(ImageFilter.MaxFilter(3))

# Rebuild only the area hidden by the initial right hand from visible wooden
# grain, banana, and fruit pixels in the one supplied photograph.
clean=ORIGINAL.copy()
wood=ORIGINAL.crop((156,28,190,54))
wood=wood.resize((64,75),Image.Resampling.BICUBIC)
clean.paste(wood,(144,30))
# Central banana column: use the adjacent complete banana silhouettes.
ban=ORIGINAL.crop((99,57,128,89))
clean.paste(ban,(121,57))
ban2=ORIGINAL.crop((99,90,133,123))
clean.paste(ban2,(123,90))
# Recreate the right wall and the fruit underneath the initial metal fingers.
woodright=ORIGINAL.crop((163,29,190,54)).resize((55,69),Image.Resampling.BICUBIC)
clean.paste(woodright,(157,28))
d=ImageDraw.Draw(clean)
d.polygon([(196,22),(201,24),(224,122),(219,122)],fill=(141,113,77))
d.line([(199,25),(220,118)],fill=(177,143,98),width=2)
# Fill the back of the right fruit row with patches from visible apples/oranges.
clean.paste(ORIGINAL.crop((187,98,210,122)),(163,98))
clean.paste(ORIGINAL.crop((193,105,214,122)),(185,96))
# Preserve all pixels outside the exact robot silhouette.
base=Image.composite(clean,ORIGINAL,whole)

fruit_specs=[
    {'name':'apple','center':(82,104),'box':(75,95,89,115),'poly':[(80,95),(85,96),(88,100),(89,108),(86,112),(82,115),(78,112),(75,107),(76,100)]},
    {'name':'orange','center':(205,113),'box':(196,104,213,122),'poly':[(204,104),(209,106),(212,110),(213,115),(210,119),(204,122),(199,120),(196,115),(197,109),(200,106)]},
    {'name':'star','center':(215,111),'box':(208,100,222,122),'poly':[(214,100),(218,102),(221,107),(222,115),(219,121),(215,122),(211,117),(208,111),(209,104)]},
]
for spec in fruit_specs:
    mask=polygon(spec['poly'])
    spec['mask']=mask
    spec['sprite']=rgba(ORIGINAL,mask).crop(spec['box'])
    # Empty space is a shaded wooden pocket where the selected fruit rested.
    arr=np.asarray(ORIGINAL).copy()
    xx,yy=np.meshgrid(np.arange(W),np.arange(H))
    src=np.asarray(ORIGINAL)[30+(yy%18),164+(xx%20)].astype(float)
    src=src*.53
    hole=Image.fromarray(np.uint8(np.clip(src,0,255)))
    spec['hole']=hole

anchor=np.array([137.,98.]); wrist=np.array([235.,53.])
orig_angle=math.atan2(-45,98)
main_sprite=rgba(ORIGINAL,main)
cable_sprite=rgba(ORIGINAL,cable)
upper_sprite=rgba(ORIGINAL,upper)
lower_sprite=rgba(ORIGINAL,lower)

def transform(im,target,theta,offset=(0,0)):
    # A single rotation and translation preserve all tool lengths and details.
    c,s=math.cos(theta),math.sin(theta)
    r=np.array([[c,-s],[s,c]])
    trans=np.asarray(target)-r@anchor+r@np.asarray(offset)
    inv=r.T; b=-inv@trans
    coeff=(inv[0,0],inv[0,1],b[0],inv[1,0],inv[1,1],b[1])
    return im.transform((W,H),Image.Transform.AFFINE,coeff,Image.Resampling.BICUBIC)

def paste_fruit(canvas,spec,center,angle=0,scale=1,inside=False):
    fruit=spec['sprite']
    bw,bh=fruit.size
    fruit=fruit.resize((max(1,round(bw*scale)),max(1,round(bh*scale))),Image.Resampling.BICUBIC)
    fruit=fruit.rotate(angle,Image.Resampling.BICUBIC,expand=True)
    pos=(round(center[0]-fruit.width/2),round(center[1]-fruit.height/2))
    if inside:
        # Transparent front of the shopping bag mutes the contents.
        a=np.asarray(fruit).copy()
        a[:,:,:3]=(a[:,:,:3].astype(float)*.69+np.array([133,137,126])*.31).astype(np.uint8)
        a[:,:,3]=(a[:,:,3]*.88).astype(np.uint8)
        fruit=Image.fromarray(a)
    canvas.alpha_composite(fruit,pos)

# Positions specify approach, closure, lift, transport, insertion, and release.
poses=[
 ((137,98),0,None,None),
 ((107,101),.35,None,None),
 ((82,104),1,0,(82,104)),
 ((91,112),1,0,(91,112)),
 ((115,141),1,0,(115,141)),
 ((115,149),0,None,None),
 ((178,113),0,None,None),
 ((205,113),1,1,(205,113)),
 ((193,98),1,1,(193,98)),
 ((154,118),1,1,(154,118)),
 ((132,146),1,1,(132,146)),
 ((132,150),0,None,None),
 ((181,107),0,None,None),
 ((215,111),1,2,(215,111)),
 ((206,96),1,2,(206,96)),
 ((163,120),1,2,(163,120)),
 ((125,143),1,2,(125,143)),
 ((125,148),0,None,None),
 ((143,116),0,None,None),
 ((150,97),0,None,None),
]
picked=[2,7,13]; released=[5,11,17]
settled=[(113,159),(133,160),(124,153)]
frames=[]
for i,(target,closure,held,fruitpos) in enumerate(poses):
    if i==0:
        frames.append(np.asarray(ORIGINAL).copy()); continue
    bg=base.copy()
    for j,spec in enumerate(fruit_specs):
        if i>picked[j]:
            bg=Image.composite(spec['hole'],bg,spec['mask'])
    canvas=bg.convert('RGBA')
    # Mild bag folds remain anchored to the cart while fruit settles inside.
    for j,spec in enumerate(fruit_specs):
        if i>=released[j]:
            age=i-released[j]
            drop=min(age/1.5,1.)
            center=(settled[j][0],settled[j][1]-6*(1-drop))
            paste_fruit(canvas,spec,center,angle=7-j*9,scale=.92,inside=True)
    # Direction to the arm mount determines wrist rotation, with fixed length.
    desired=math.atan2(35-target[1],280-target[0])
    theta=desired-orig_angle
    if i==1: theta*=.3
    c,s=math.cos(theta),math.sin(theta)
    r=np.array([[c,-s],[s,c]])
    joint=np.asarray(target)+r@(wrist-anchor)
    # The cropped proximal link continues to the mounting point outside view.
    # Parallel metal faces, pins, and cylinders are explicit RGB geometry.
    link=Image.new('RGBA',(W*3,H*3)); ld=ImageDraw.Draw(link)
    aa=np.array([292.,24.]); bb=joint
    direction=(aa-bb)/np.linalg.norm(aa-bb); normal=np.array([-direction[1],direction[0]])
    def poly_link(vv,cc): ld.polygon([tuple(v*3) for v in vv],fill=cc)
    poly_link([bb+normal*12,aa+normal*12,aa-normal*12,bb-normal*12],(71,91,96,255))
    poly_link([bb+normal*8,aa+normal*8,aa-normal*6,bb-normal*6],(156,177,181,255))
    ld.line([tuple((bb+normal*8)*3),tuple((aa+normal*8)*3)],fill=(205,220,218,255),width=5)
    ld.line([tuple((bb-normal*8)*3),tuple((aa-normal*8)*3)],fill=(39,55,57,255),width=8)
    for rad,color in [(13,(76,92,94,255)),(10,(163,178,177,255)),(6,(102,126,129,255)),(3,(180,194,193,255))]:
        ld.ellipse(((bb[0]-rad)*3,(bb[1]-rad)*3,(bb[0]+rad)*3,(bb[1]+rad)*3),fill=color)
    link=link.resize((W,H),Image.Resampling.LANCZOS)
    canvas.alpha_composite(link)
    if held is not None:
        paste_fruit(canvas,fruit_specs[held],fruitpos,angle=math.degrees(theta)*.3)
    canvas.alpha_composite(transform(main_sprite,target,theta))
    canvas.alpha_composite(transform(upper_sprite,target,theta,(0,10*closure)))
    canvas.alpha_composite(transform(lower_sprite,target,theta,(0,-9*closure)))
    canvas.alpha_composite(transform(cable_sprite,target,theta))
    # Bag's rim and glossy plastic folds sit in front of inserted fruit.
    if i>=5:
        folds=Image.new('RGBA',(W,H)); fd=ImageDraw.Draw(folds)
        fd.arc((95,135,147,169),190,345,fill=(195,200,181,100),width=1)
        fd.line([(104,149),(108,158),(106,174)],fill=(208,211,189,95),width=1)
        fd.line([(138,151),(139,163),(144,177)],fill=(204,208,191,115),width=1)
        fd.line([(117,157),(120,163),(117,177)],fill=(197,201,184,65),width=1)
        canvas.alpha_composite(folds)
    frames.append(np.asarray(canvas.convert('RGB')))
frames=np.asarray(frames,dtype=np.uint8)
# Save the first candidate separately before visual inspection and refinement.
np.save(ROOT / '.tmp/candidate_v1.npy',frames)
save(frames,ROOT)

"""Pixel continuation using historical backplate and extracted RGB objects.

No renderer, model, reference continuation, or simulation package is used.
The pair follows a hand-written rolling trajectory and a single elastic impact.
"""
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import cv2
import artifact_io

ROOT = Path(__file__).resolve().parent
H, W = 144, 256


def get_backplate():
    history = [np.asarray(Image.open(ROOT / 'conditioning_frames' / f'{i:02d}.png').convert('RGB'), dtype=np.float32) for i in range(11)]
    plate = np.median(np.stack(history[-5:]), axis=0)
    last = np.asarray(Image.open(ROOT / 'first_frame.png').convert('RGB'), dtype=np.float32)
    yy, xx = np.mgrid[:H, :W]
    clear = ((xx-37)**2+(yy-47)**2 > 32**2) & ((xx-201)**2+(yy-66)**2 > 32**2)
    correction = np.median((last-plate)[clear], axis=0)
    return np.clip(plate+correction, 0, 255), last


def extract_ball(last, plate, center, radii):
    """Recover edge coverage, RGB texture, and the nearby real cast shadow."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cx, cy = center
    rx, ry = radii
    rr = np.sqrt(((xx-cx)/rx)**2 + ((yy-cy)/ry)**2)
    alpha = np.clip((1.10-rr)/0.21, 0, 1)
    # Premultiplied object colors undo the original tabletop at the soft edge.
    texture = last-(1-alpha[..., None])*plate
    texture *= (alpha > 0)[..., None]
    luminance = last.mean(axis=2)
    base_luminance = plate.mean(axis=2)
    dark = (base_luminance-luminance)/np.maximum(base_luminance, 20)
    support = np.sqrt(((xx-cx)/(rx*2.8))**2 + ((yy-cy)/(ry*1.9))**2)
    window = np.clip((1.0-support)/0.20, 0, 1)
    shadow = np.clip(dark-0.025, 0, 0.48)*window*(1-alpha)
    shadow = cv2.GaussianBlur(shadow, (0,0), 0.65)
    # Black contact immediately below the ball; retained texture covers its center.
    return {'center': np.asarray(center), 'radii': radii,
            'alpha': alpha, 'texture': texture, 'shadow': shadow}


def warp(field, sprite, center, scale):
    sx, sy = sprite['center']
    cx, cy = center
    matrix = np.array([[scale, 0, cx-scale*sx], [0, scale, cy-scale*sy]], dtype=np.float32)
    return cv2.warpAffine(field, matrix, (W,H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def positions(t):
    """t is time since the last historical frame, in seconds."""
    p_blue = np.array([37.0,46.85])
    p_yellow = np.array([201.25,65.95])
    difference = p_yellow-p_blue
    distance = np.linalg.norm(difference)
    direction = difference/distance
    # The larger near-side ball has a slightly larger image speed. Both have
    # the same tabletop speed. Visibility at 2.50 s bounds the approach speed.
    speed_blue, speed_yellow = 148.0, 164.0
    drag = 8.0
    contact_distance = 16.6
    total_speed = speed_blue+speed_yellow
    contact_t = (total_speed-math.sqrt(total_speed**2-4*drag*(distance-contact_distance)))/(2*drag)
    if t <= contact_t:
        blue = p_blue+direction*(speed_blue*t-0.5*drag*t*t)
        yellow = p_yellow-direction*(speed_yellow*t-0.5*drag*t*t)
    else:
        blue_contact = p_blue+direction*(speed_blue*contact_t-0.5*drag*contact_t**2)
        yellow_contact = p_yellow-direction*(speed_yellow*contact_t-0.5*drag*contact_t**2)
        dt = t-contact_t
        # A tennis-ball rebound retains most, but not all, of the approach speed.
        restitution = 0.82
        ub = restitution*(speed_blue-drag*contact_t)
        uy = restitution*(speed_yellow-drag*contact_t)
        post_drag = 12.0
        db = ub*min(dt, ub/post_drag)-0.5*post_drag*min(dt, ub/post_drag)**2
        dy = uy*min(dt, uy/post_drag)-0.5*post_drag*min(dt, uy/post_drag)**2
        blue = blue_contact-direction*db
        yellow = yellow_contact+direction*dy
    return blue, yellow


def render(plate, sprites, t):
    canvas = plate.copy()
    points = positions(t)
    yy, xx = np.mgrid[:H,:W]
    # Preserve the foreground lip of each pipe when the returning ball reaches it.
    pipe_left = ((xx-3.0)/9.0)**2+((yy-41.7)/10.5)**2 < 1
    pipe_right = (((xx-239.0)/7.4)**2+((yy-70.4)/13.0)**2 < 1) | ((xx>=240)&(yy>=58)&(yy<=81))
    occlusion = pipe_left|pipe_right
    layers = []
    for sprite, point in zip(sprites, points):
        # Continue the motion beyond the camera boundary, instead of fading it.
        if point[0] < -25 or point[0] > W+25:
            continue
        scale = 1+0.005*(point[1]-sprite['center'][1])
        alpha = warp(sprite['alpha'],sprite,point,scale)
        texture = warp(sprite['texture'],sprite,point,scale)
        shadow = warp(sprite['shadow'],sprite,point,scale)
        alpha[occlusion] = 0
        texture[occlusion] = 0
        shadow[occlusion] = 0
        canvas *= 1-shadow[..., None]
        layers.append((alpha,texture))
    for alpha, texture in layers:
        canvas = canvas*(1-alpha[..., None])+texture
    return np.clip(np.rint(canvas),0,255).astype(np.uint8)


def main():
    plate,last = get_backplate()
    sprites = [extract_ball(last,plate,(37.0,46.85),(7.15,6.75)),
               extract_ball(last,plate,(201.25,65.95),(9.0,7.9))]
    # Output begins at the next sample, 3.00 s of the original locked-off shot.
    frames = np.stack([render(plate,sprites,0.25+i/4) for i in range(20)])
    candidate = ROOT/'candidates'/'01'
    candidate.mkdir(parents=True,exist_ok=True)
    np.save(candidate/'rgb.npy',frames)
    for i,frame in enumerate(frames):
        Image.fromarray(frame).save(candidate/f'{i:02d}.png')
    Image.fromarray(render(plate,sprites,0)).save(ROOT/'reconstruction_check.png')
    sheet = Image.new('RGB',(W*4,(H+20)*2),(23,24,26))
    draw = ImageDraw.Draw(sheet)
    for i in range(8):
        x,y=(i%4)*W,(i//4)*(H+20)
        draw.text((x+5,y+3),f'future {i/4:.2f}s',fill='white')
        sheet.paste(Image.fromarray(frames[i]),(x,y+20))
    sheet.save(ROOT/'motion_check.png')
    artifact_io.save(frames,ROOT)
    print('Saved',frames.shape,frames.dtype,'to pixels.npz and frames/')
    print('Trajectory sample centers:',[tuple(p.round(2).tolist() for p in positions(0.25+i/4)) for i in range(6)])


if __name__ == '__main__':
    main()

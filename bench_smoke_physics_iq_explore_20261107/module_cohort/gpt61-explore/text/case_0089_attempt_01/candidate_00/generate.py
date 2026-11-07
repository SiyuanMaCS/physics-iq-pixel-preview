"""Render the pour directly in RGB, using the supplied still as the scene.

The camera, glass walls, tap and tray stay fixed. Only the liquid acquires
advected refraction, a gravity-fed jet, a rising meniscus and impact ripples.
No renderer, simulation package, or additional image source is used.
"""
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

import artifact_io

CASE_DIR = Path(__file__).resolve().parent
BASE = np.asarray(Image.open(CASE_DIR / 'first_frame.png').convert('RGB'))
H, W = BASE.shape[:2]
Y, X = np.mgrid[:H, :W].astype(np.float64)


def blend(canvas, color, alpha):
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    canvas[:] = canvas * (1.0 - alpha) + np.asarray(color) * alpha


def sample(image, xx, yy):
    xx = np.clip(xx, 0, W - 1.001)
    yy = np.clip(yy, 0, H - 1.001)
    ix, iy = xx.astype(int), yy.astype(int)
    dx, dy = (xx - ix)[..., None], (yy - iy)[..., None]
    return (image[iy, ix] * (1-dx) * (1-dy)
            + image[iy, ix+1] * dx * (1-dy)
            + image[iy+1, ix] * (1-dx) * dy
            + image[iy+1, ix+1] * dx * dy)


def soft_interval(a, lo, hi, feather=1.0):
    return np.minimum(np.clip((a-lo)/feather, 0, 1),
                      np.clip((hi-a)/feather, 0, 1))


def render(time):
    if time == 0:
        return BASE.copy()
    canvas = BASE.astype(np.float64).copy()
    # The original glass already contains red liquid. Its upper surface moves
    # only a few pixels as more liquid enters; the original filled body and
    # all the immobile optical reflections are retained.
    onset = min(1.0, time / 0.48)
    level = 25.9 - 2.1 * (1 - np.exp(-time / 2.9))
    surface = level + 0.19*np.sin(6.3*time)
    inside = soft_interval(X, 135.8, 171.5, 2.1)
    inside *= soft_interval(Y, 26.0, 88.5, 3.0)
    saturated = np.clip((BASE[...,0].astype(float)-BASE[...,1]-75)/50, 0, 1)
    inside *= saturated
    # Flow in the liquid bends its existing RGB texture. The high intensity
    # white reflections belong to the glass and are held in place.
    fixed_glints = np.clip((BASE[...,1].astype(float)-135)/75, 0, 1)
    motion_mask = inside * (1-fixed_glints) * onset
    depth = np.exp(-(Y-28).clip(0)/37)
    dx = depth*(0.24*np.sin((Y-25)*0.42-9.1*time)
                + 0.15*np.cos((Y-25)*0.21-4.2*time))
    dy = depth*(0.35*np.sin((X-152)*0.26+7.3*time))
    refracted = sample(BASE.astype(float), X+dx, Y+dy)
    blend(canvas, refracted, motion_mask*0.68)
    eddy = depth*(np.sin(0.46*Y-8.4*time+0.055*(X-153)**2)
                  + 0.38*np.sin(0.18*Y+0.61*X-5.9*time))
    canvas += motion_mask[...,None]*eddy[...,None]*np.array([1.7, 2.4, 1.9])

    # Surface tint, curved meniscus, and perspective-compressed wave rings.
    xx = (X-154.2)/17.0
    yy = (Y-surface)/2.05
    ellipse = xx*xx+yy*yy
    surface_alpha = np.clip((1.12-ellipse)*3.3, 0, 1)*onset*0.42
    surface_rgb = np.stack((244+4*np.cos(X*.28),
                            86+11*np.cos(X*.24),
                            61+8*np.cos(X*.24)), axis=-1)
    blend(canvas, surface_rgb, surface_alpha)
    meniscus = np.exp(-((np.sqrt(ellipse+1e-8)-1)/0.19)**2)
    meniscus *= soft_interval(X, 137.0, 171.1, 1.1)*onset
    # The nearer edge is dark red; the far edge catches the room light.
    blend(canvas, [176, 50, 38], meniscus*(Y>=surface)*0.26)
    blend(canvas, [255, 157, 129], meniscus*(Y<surface)*0.25)
    for j in range(3):
        phase = (time*1.65+j/3) % 1
        radius = 1.2+phase*14.8
        ring_dist = np.sqrt((X-154.6)**2+((Y-surface)/0.19)**2)
        ring = np.exp(-((ring_dist-radius)/0.75)**2)
        ring *= np.exp(-phase*2.3)*surface_alpha*0.59
        blend(canvas, [255, 165, 131], ring)

    # A thin continuous jet emerges exactly under the blue outlet. Its width
    # tapers with distance under gravity and its highlights travel downward.
    fall = np.maximum(0, Y-16.2)
    axis = 154.7 + 0.14*np.sin(time*8.1-fall*0.53)
    axis += 0.10*np.sin(time*13.8-fall*0.9)
    halfwidth = 1.47/np.sqrt(1+fall*0.055)
    halfwidth += 0.06*np.sin(time*11.2-fall*1.2)
    jet = np.clip(halfwidth+0.55-np.abs(X-axis), 0, 1)
    jet *= soft_interval(Y, 16.1, surface+0.8, 0.60)*onset
    transverse = np.clip((X-axis)/1.7, -1, 1)
    wave = np.sin(time*15.4-fall*1.8)
    jet_rgb = np.stack((233+8*wave,
                       61+7*wave-15*transverse,
                       41+5*wave-10*transverse), axis=-1)
    blend(canvas, jet_rgb, jet*0.88)
    gleam = np.exp(-((X-axis+0.56)/0.32)**2)*jet
    gleam *= 0.36+0.15*np.sin(time*17-fall*1.5)
    blend(canvas, [255, 195, 158], gleam)
    impact = np.exp(-((X-154.6)/2.1)**2-((Y-surface)/0.8)**2)*onset
    blend(canvas, [233, 59, 43], impact*0.65)
    impact_glint = np.exp(-((X-153.6)/0.6)**2-((Y-surface+0.45)/0.4)**2)
    blend(canvas, [255, 188, 151], impact_glint*onset*0.6)

    # Small entrained air bubbles move downward a short way and then rise.
    # They remain inside the liquid rather than affecting the glass edges.
    for j in range(6):
        age = (time+j*0.29) % 1.76
        by = surface+2+14*np.sin(age/1.76*np.pi)
        bx = 154.8 + (j%2*2-1)*(1.0+j*0.41)+0.35*np.sin(age*5+j)
        rad = 0.35 + (j%3)*0.12
        d = np.sqrt((X-bx)**2+((Y-by)/1.18)**2)
        bubble = np.exp(-((d-rad)/0.24)**2)
        bubble *= onset*min(1, age/0.18, (1.76-age)/0.20)*0.28
        blend(canvas, [255, 153, 124], bubble)
        center = np.exp(-(d/0.27)**2)*onset*0.15
        blend(canvas, [194, 64, 47], center)

    # Tiny red glints on the tray are reflections of the moving liquid only.
    reflected = soft_interval(X, 136, 172, 6)*soft_interval(Y, 94, 119, 5)
    shimmer = 0.65*np.sin(0.37*Y+5.2*time)*np.exp(-(Y-94).clip(0)/11)
    canvas += reflected[...,None]*shimmer[...,None]*np.array([1.0,0.20,0.12])
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def main():
    frames = np.stack([render(i/4) for i in range(20)])
    assert frames.shape == (20, 144, 256, 3)
    assert frames.dtype == np.uint8
    assert np.array_equal(frames[0], BASE)
    # Retain the generated candidate before writing the deliverable artifacts.
    candidates = CASE_DIR / 'candidates'
    candidates.mkdir(exist_ok=True)
    np.savez_compressed(candidates / 'candidate_01.npz', frames=frames)
    Image.fromarray(frames[10]).save(candidates / 'candidate_01_frame_10.png')
    artifact_io.save(frames, CASE_DIR)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', '256x144', '-r', '4', '-i', '-',
                        '-an', '-c:v', 'libx264', '-preset', 'slow',
                        '-crf', '16', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(CASE_DIR/'video.mp4')],
                       input=frames.tobytes(), check=True)
    delta = np.abs(frames.astype(np.int16)-BASE.astype(np.int16))
    print('RGB frames:', frames.shape, frames.dtype)
    print('Frame 00 pixel-identical:', np.array_equal(frames[0], BASE))
    print('Changed pixels per frame:', np.any(delta, axis=-1).sum((1,2)).tolist())
    print('Saved pixels.npz, frames/, index.html, and intermediate candidate.')


if __name__ == '__main__':
    main()

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import artifact_io


ROOT = Path(__file__).resolve().parent


def feather(mask, radius=1.15):
    """Turn a binary mask into a small, photographic antialiased matte."""
    layer = Image.fromarray(mask.astype(np.uint8) * 255)
    layer = layer.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(layer, dtype=np.float32) / 255.0


def extract_objects(source):
    hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)

    # The ball is nearly circular; the slight oversize matte retains its naturally
    # blurred boundary while excluding the long, soft cast shadow.
    yy, xx = np.mgrid[:144, :256]
    ball_shape = ((xx - 66.0) / 9.3) ** 2 + ((yy - 53.0) / 9.0) ** 2 <= 1.0
    ball_dark = hsv[..., 2] < 139
    ball_mask = ball_shape & ball_dark
    ball_mask = cv2.morphologyEx(ball_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8)) > 0
    ball_alpha = feather(ball_mask, 1.05)

    # Saturation cleanly separates the yellow/orange duck from the neutral table.
    duck_roi = np.zeros((144, 256), np.uint8)
    duck_roi[43:72, 174:216] = 1
    duck_mask = (duck_roi > 0) & (hsv[..., 1] > 72) & (hsv[..., 2] > 72)
    duck_mask = cv2.morphologyEx(duck_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8), iterations=2)
    # Fill the small black eye enclosed by the yellow head.
    cv2.circle(duck_mask, (190, 55), 2, 1, -1)
    duck_alpha = feather(duck_mask > 0, 0.85)

    ball_box = (53, 40, 80, 67)
    duck_box = (173, 41, 218, 74)

    def rgba_crop(box, alpha):
        x0, y0, x1, y1 = box
        rgb = source[y0:y1, x0:x1]
        a = np.clip(alpha[y0:y1, x0:x1] * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(np.dstack((rgb, a)))

    return rgba_crop(ball_box, ball_alpha), rgba_crop(duck_box, duck_alpha), ball_alpha, duck_alpha


def make_clean_plate(source, ball_alpha, duck_alpha):
    # Remove both subjects plus their cast shadows. Telea inpainting draws from
    # the adjacent horizontal wood grain and leaves all fixed scenery untouched.
    remove = np.zeros((144, 256), np.uint8)
    remove[ball_alpha > 0.025] = 255
    remove[duck_alpha > 0.025] = 255
    cv2.ellipse(remove, (72, 57), (15, 7), 0, 0, 360, 255, -1)
    cv2.ellipse(remove, (193, 67), (25, 9), 0, 0, 360, 255, -1)
    remove = cv2.dilate(remove, np.ones((3, 5), np.uint8), iterations=1)
    return cv2.inpaint(source, remove, 4.0, cv2.INPAINT_TELEA)


def shadow_layer(size, center, radii, opacity):
    # Shadows are rendered separately so they remain grounded as the objects move.
    scale = 4
    layer = Image.new("RGBA", (size[0] * scale, size[1] * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = center
    rx, ry = radii
    draw.ellipse(((cx - rx) * scale, (cy - ry) * scale,
                  (cx + rx) * scale, (cy + ry) * scale),
                 fill=(28, 25, 18, opacity))
    layer = layer.filter(ImageFilter.GaussianBlur(2.2 * scale))
    return layer.resize(size, Image.Resampling.LANCZOS)


def rotated_about_anchor(sprite, angle, anchor, canvas_size=(92, 92)):
    """Rotate clockwise around a sprite-local contact point without cropping."""
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    pivot = (canvas_size[0] // 2, canvas_size[1] // 2)
    canvas.alpha_composite(sprite, (pivot[0] - anchor[0], pivot[1] - anchor[1]))
    # PIL's positive angle is counter-clockwise, hence the minus sign.
    return canvas.rotate(-angle, resample=Image.Resampling.BICUBIC,
                         center=pivot, expand=False), pivot


def compose_frame(clean, ball_sprite, duck_sprite, index):
    frame = Image.fromarray(clean).convert("RGBA")

    ball_x = [66, 83, 100, 117, 134, 151, 166, 176, 181, 183,
              184, 184, 183, 182, 182, 182, 182, 182, 182, 182][index]
    # Contact begins between frames 6 and 7. A short damped wobble keeps the
    # collision readable without introducing implausible perpetual motion.
    duck_angle = [0, 0, 0, 0, 0, 0, 0, 3, 14, 30,
                  49, 67, 82, 91, 86, 89, 88, 88, 88, 88][index]
    duck_pivot_x = [195, 195, 195, 195, 195, 195, 195, 196, 198, 200,
                    202, 204, 205, 206, 207, 207, 207, 207, 207, 207][index]

    # Moving ball shadow. It becomes a little darker at rest, as in the source.
    frame = Image.alpha_composite(
        frame, shadow_layer(frame.size, (ball_x + 5, 59), (12, 4), 54)
    )

    # Duck shadow shrinks vertically while the duck tips over and widens along
    # its new horizontal footprint.
    tip = min(duck_angle / 88.0, 1.0)
    duck_shadow_center = (duck_pivot_x + 7 * tip, 68)
    frame = Image.alpha_composite(
        frame,
        shadow_layer(frame.size, duck_shadow_center,
                     (21 + 5 * tip, 5.5 - 1.5 * tip), int(54 + 12 * tip)),
    )

    # Rotate the mottled ball texture according to distance travelled. This is
    # what makes the translation read as rolling rather than sliding.
    roll_degrees = -((ball_x - 66) / 8.7) * 57.2958
    moving_ball = ball_sprite.rotate(roll_degrees, Image.Resampling.BICUBIC,
                                     expand=False)
    frame.alpha_composite(moving_ball,
                          (int(round(ball_x - ball_sprite.width / 2)),
                           int(round(53 - ball_sprite.height / 2))))

    # Original duck crop is 45x33. Its table contact is at local (22, 28).
    turned_duck, local_pivot = rotated_about_anchor(
        duck_sprite, duck_angle, anchor=(22, 28)
    )
    frame.alpha_composite(turned_duck,
                          (int(duck_pivot_x - local_pivot[0]),
                           int(69 - local_pivot[1])))
    return np.asarray(frame.convert("RGB"), dtype=np.uint8)


def main():
    source = np.asarray(Image.open(ROOT / "first_frame.png").convert("RGB"),
                        dtype=np.uint8)
    ball, duck, ball_alpha, duck_alpha = extract_objects(source)
    clean = make_clean_plate(source, ball_alpha, duck_alpha)

    frames = np.stack([
        source.copy() if i == 0 else compose_frame(clean, ball, duck, i)
        for i in range(20)
    ]).astype(np.uint8)

    assert np.array_equal(frames[0], source), "frame 00 must be pixel-identical"
    artifact_io.save(frames, ROOT)


if __name__ == "__main__":
    main()

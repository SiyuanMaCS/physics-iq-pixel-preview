from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import artifact_io


CASE_DIR = Path(__file__).resolve().parent
WIDTH, HEIGHT = 256, 143
FRAME_COUNT = 20


def dog_mask(image: np.ndarray) -> np.ndarray:
    """Extract the dog with a tightly constrained, classical GrabCut mask."""
    guide = np.zeros((HEIGHT, WIDTH), np.uint8)
    silhouette = np.array(
        [
            (124, 69), (128, 73), (133, 71), (132, 78), (135, 82),
            (135, 89), (132, 94), (133, 101), (136, 106), (137, 114),
            (136, 122), (136, 128), (133, 130), (130, 128), (130, 115),
            (127, 110), (126, 119), (126, 127), (123, 131), (120, 130),
            (119, 126), (120, 116), (120, 109), (118, 103), (119, 94),
            (121, 89), (119, 85), (120, 77), (123, 75),
        ],
        np.int32,
    )
    cv2.fillPoly(guide, [silhouette], cv2.GC_PR_FGD)

    # Known fur samples prevent similarly colored street pixels from leaking in.
    certain = np.array(
        [
            (124, 75), (129, 77), (132, 82), (128, 87), (124, 91),
            (127, 96), (130, 101), (125, 105), (132, 109), (123, 113),
            (133, 117), (122, 123), (133, 125),
        ],
        np.int32,
    )
    for x, y in certain:
        cv2.circle(guide, (int(x), int(y)), 2, cv2.GC_FGD, -1)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, guide, None, bg_model, fg_model, 7, cv2.GC_INIT_WITH_MASK)
    mask = np.where((guide == cv2.GC_FGD) | (guide == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Keep only the component around the dog's center and gently feather its edge.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count > 1:
        center_label = labels[95, 127]
        if center_label == 0:
            center_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == center_label, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(mask, (3, 3), 0.55)


def clean_background(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Replace the dog with nearby street texture for the moving background plate."""
    hole = cv2.dilate((alpha > 18).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    # Telea is a deterministic local pixel propagation, suitable for this small road patch.
    clean = cv2.inpaint(image, hole, 5, cv2.INPAINT_TELEA)

    # Restore a little asphalt grain so the filled area does not look airbrushed.
    rng = np.random.default_rng(7319)
    grain = rng.normal(0.0, 2.0, image.shape[:2]).astype(np.float32)
    area = cv2.GaussianBlur((hole > 0).astype(np.float32), (9, 9), 2.0)[..., None]
    clean = np.clip(clean.astype(np.float32) + grain[..., None] * area, 0, 255).astype(np.uint8)
    return clean


def warp_background(background: np.ndarray, index: int) -> np.ndarray:
    """Move the camera down the street with expansion about the vanishing point."""
    if index == 0:
        return background.copy()
    progress = index / (FRAME_COUNT - 1)
    # About 13% forward expansion over five seconds, with a subtle walking sway.
    scale = 1.0 + 0.132 * progress
    phase = 2.0 * np.pi * index / 7.5
    vp_x = 130.0 + 0.38 * np.sin(phase)
    vp_y = 48.0 + 0.22 * np.sin(phase * 2.0)
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    src_x = vp_x + (xx - vp_x) / scale
    src_y = vp_y + (yy - vp_y) / scale

    # The closest road moves a touch faster than the distant skyline.
    ground = np.clip((yy - vp_y) / (HEIGHT - vp_y), 0.0, 1.0)
    src_y -= progress * 1.15 * ground * ground
    src_x += 0.18 * np.sin(phase) * ground
    src_x = src_x.astype(np.float32)
    src_y = src_y.astype(np.float32)
    return cv2.remap(
        background,
        src_x,
        src_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def affine_layer(rgb: np.ndarray, alpha: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moved_rgb = cv2.warpAffine(
        rgb, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    moved_alpha = cv2.warpAffine(
        alpha, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return moved_rgb, moved_alpha


def composite(dst: np.ndarray, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    weight = alpha.astype(np.float32)[..., None] / 255.0
    return np.clip(dst.astype(np.float32) * (1.0 - weight) + rgb.astype(np.float32) * weight, 0, 255).astype(np.uint8)


def render_dog(frame: np.ndarray, source: np.ndarray, alpha: np.ndarray, index: int) -> np.ndarray:
    """Keep the dog framed while giving its hind legs a restrained walk cycle."""
    phase = 2.0 * np.pi * index / 7.5
    bob = -0.65 * (0.5 - 0.5 * np.cos(phase * 2.0))
    sway = 0.34 * np.sin(phase)

    yy, xx = np.indices((HEIGHT, WIDTH))
    left_region = np.array([(118, 101), (127, 102), (128, 112), (126, 131), (118, 132)], np.int32)
    right_region = np.array([(127, 101), (137, 103), (138, 131), (129, 131), (128, 112)], np.int32)
    left = np.zeros_like(alpha)
    right = np.zeros_like(alpha)
    cv2.fillPoly(left, [left_region], 255)
    cv2.fillPoly(right, [right_region], 255)
    left = cv2.min(left, alpha)
    right = cv2.min(right, alpha)
    leg_union = cv2.max(left, right)

    # Body overlaps the thighs, which hides the leg-layer seams.
    body_keep = np.where(yy <= 110, 255, 0).astype(np.uint8)
    body_keep = cv2.GaussianBlur(body_keep, (1, 5), 0.7)
    body_alpha = cv2.max(cv2.min(alpha, body_keep), cv2.subtract(alpha, leg_union))

    global_matrix = np.float32([[1, 0, sway], [0, 1, bob]])
    body_rgb, body_a = affine_layer(source, body_alpha, global_matrix)

    # Opposing shears pivot visually at the hips, alternating the paws by ~2 px.
    stride = 1.75 * np.sin(phase)
    lift_left = -0.8 * max(0.0, np.sin(phase))
    lift_right = -0.8 * max(0.0, -np.sin(phase))
    left_matrix = np.float32([[1, stride / 27.0, sway - stride * 108.0 / 27.0], [0, 1, bob + lift_left]])
    right_matrix = np.float32([[1, -stride / 27.0, sway + stride * 108.0 / 27.0], [0, 1, bob + lift_right]])
    left_rgb, left_a = affine_layer(source, left, left_matrix)
    right_rgb, right_a = affine_layer(source, right, right_matrix)

    # A soft contact shadow moves with the animal and sells its contact with the street.
    shadow = np.zeros((HEIGHT, WIDTH), np.uint8)
    center = (int(round(128 + sway)), int(round(129 + bob)))
    cv2.ellipse(shadow, center, (11, 3), 0, 0, 360, 62, -1)
    shadow = cv2.GaussianBlur(shadow, (9, 5), 2.0)
    dark = np.clip(frame.astype(np.float32) * 0.68, 0, 255).astype(np.uint8)
    frame = composite(frame, dark, shadow)
    frame = composite(frame, left_rgb, left_a)
    frame = composite(frame, right_rgb, right_a)
    frame = composite(frame, body_rgb, body_a)
    return frame


def save_candidate_strip(frames: np.ndarray) -> None:
    picks = [0, 4, 8, 12, 16, 19]
    canvas = Image.new("RGB", (WIDTH * 3, HEIGHT * 2 + 34), (22, 24, 29))
    draw = ImageDraw.Draw(canvas)
    for slot, frame_index in enumerate(picks):
        x = (slot % 3) * WIDTH
        y = (slot // 3) * (HEIGHT + 17)
        draw.text((x + 4, y + 2), f"candidate v1 - frame {frame_index:02d}", fill=(245, 245, 245))
        canvas.paste(Image.fromarray(frames[frame_index]), (x, y + 17))
    canvas.save(CASE_DIR / "candidate_v1.png")


def generate() -> np.ndarray:
    source = np.asarray(Image.open(CASE_DIR / "first_frame.png").convert("RGB"), dtype=np.uint8)
    alpha = dog_mask(source)
    background = clean_background(source, alpha)
    frames = [source.copy()]
    for index in range(1, FRAME_COUNT):
        frame = warp_background(background, index)
        frame = render_dog(frame, source, alpha, index)
        frames.append(frame)
    result = np.stack(frames).astype(np.uint8)
    assert result.shape == (20, 143, 256, 3)
    assert np.array_equal(result[0], source)
    return result


if __name__ == "__main__":
    generated = generate()
    save_candidate_strip(generated)
    artifact_io.save(generated, CASE_DIR)

# Case 0074 — 2D pixel continuation

The only visual input is `first_frame.png`, with motion given by `prompt.txt`. No old output, other case, reference continuation, model, network access, or 3D construction is used.

## Observation and event schedule

The camera and spotlight are fixed. A narrow blue upright rectangular block occupies approximately x=145–155, y=98–124; its base touches the black platform near (149,124). Its top face is dark, the principal face is cyan, and the right rim is lighter. The platform top is an ellipse near x=123–176, y=118–135. The large upright wall shadow occupies x=104–118, y=61–101, connected to the existing horizontal platform shadow. The beam comes from the right/front and sends the shadow toward the left/up. The illuminated wall, table, cable, and platform remain stable.

The 20 frames are at 4 fps: an initially narrow view broadens through about 1.5 s, turns to a narrow edge near 2.5 s, broadens with the other face visible around 3.75 s, and returns to another narrow view by 4.75 s. The visible face division advances from right toward left during the first part of the clockwise turn, then the following pair of faces repeats the cycle. No translation or rocking is added.

## Pixel method

An explicitly specified 2D outline and face-boundary schedule is interpolated at each frame. Each face, the top strip, rim light, and contact shade is rasterized anew at 4× working resolution, with subtle vertical color gradients and low-amplitude fixed grain. No old block or shadow image is transformed or copied. The wall shadow is a separate freshly drawn 2D polygon, whose width follows the current block silhouette, with a softly blurred penumbra and dark colored umbra. Its foot remains merged with the original platform shadow. A softly tapered blue platform reflection changes with the visible block width, fading into the platform rather than ending as a solid rectangle.

The old upright shadow is removed using a shaped mask and horizontal interpolation of adjacent illuminated wall colors, including small local texture variations. The original blue block and reflection are removed through a shaped mask and deterministic, non-learning local inpainting. The first output frame is the untouched supplied RGB input. All following RGB frames are computed independently from the cleaned static background and the current newly drawn geometry.

## Assumptions and limitations

The speed and exact number of turns are unspecified; this sequence assumes a steady modest rotation with a repeating contour cycle. The small source image does not reveal the hidden faces or block cross-section, so their widths and face transitions are hand designed. Shadow changes are a qualitative 2D construction, not a calibrated optical solution; subtle perspective differences are approximate. The reconstructed wall and tabletop areas may be smoother than the original grain, and the tiny contact/reflection detail is approximate. The platform has no clearly trackable marking, so its clockwise rotation is visible through the block's face sequence rather than artificial texture motion.

## Inspection and revision

The supplied still and frames 0, 2, 5, 8, 11, 14, 17, and 19 were opened at 4× enlargement, as well as the 8-time contact sheet. The first inspection found the rebuilt base approximately 3 pixels too low and a bright metallic streak pulled into the reflection by inpainting. The base was restored to y≈124 and the platform beneath it was rebuilt using neighboring same-row platform colors. The next face now enters at the opposite silhouette edge after the narrow transition, preserving the direction of the face-boundary sweep. Width interpolation was smoothed at its broad and narrow extrema. Revised contact sheets and key frames were inspected again.

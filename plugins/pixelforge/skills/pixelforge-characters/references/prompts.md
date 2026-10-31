# Prompt templates and reference limits for character series

Fill the brackets and keep the spec block byte-identical across every call.
Parameter values shown after each template are the ones to pass to the
PixelForge tool named in the heading.

## Contents

- Spec block
- Master (`generate_image`)
- Reference sheet (`generate_image`)
- Scene (`generate_image`)
- Storyboard strip (`generate_image`)
- Drift check (`compare_images`)
- Single-trait fix (`edit_image`)
- Reference image limits per model

## Spec block (reuse verbatim)

```
[NAME] is a [species or build], [age or scale], with [face shape and eyes],
[hair or fur, colour #RRGGBB], [skin or body colour #RRGGBB], wearing
[outfit, colours #RRGGBB], always carrying [signature prop]. Art style:
[style, line weight, shading], identical proportions in every image.
```

## Master (`generate_image`)

```
Create an image of [SPEC BLOCK] Full body, neutral standing pose, facing
the camera, arms relaxed, even soft studio lighting, plain flat light-grey
background, no text, no watermark.
```

`aspect_ratio="1:1"`, `image_size="1K"`, `output_filename="[name]_master.png"`.

## Reference sheet (`generate_image`)

Four-view turnaround (Google's character 360 pattern):

```
A character reference sheet of the same character as the reference image.
[SPEC BLOCK] Four full-body views side by side on one plain white
background, in this order: front view, three-quarter view, side profile
facing right, back view. Identical outfit, colours and proportions in every
view. Clean line-up, no text, no labels.
```

Expression and pose grid (Google's sprite-sheet pattern):

```
A 3x3 grid sprite sheet of the same character as the reference image.
[SPEC BLOCK] Each cell shows the full body on a plain white background.
Row 1: neutral, happy, surprised. Row 2: angry, sad, thinking. Row 3:
walking, waving, sitting. Same scale in every cell, no text.
```

`reference_images=["[master path]"]`, `aspect_ratio="1:1"`, `image_size="2K"`,
`output_filename="[name]_sheet.png"`.

## Scene (`generate_image`)

```
Create an image of [NAME], the same character as in the reference images.
[SPEC BLOCK] [NAME] is [action] in [setting], [time of day and lighting].
[Composition, for example medium shot, slight low angle, 35mm lens].
[Style note matching the master]. Keep [NAME]'s face, proportions, outfit
and colours exactly as in the references.
```

Two characters: "[NAME1], [spec block 1], stands on the left. [NAME2],
[spec block 2], stands on the right. They are [interaction]." Pass the
masters in `reference_images` in the same order as the names.

`reference_images=["[master path]", "[sheet path]"]`, `number_of_images=1`,
`output_filename="[name]_scene_[NN].png"`.

## Storyboard strip (`generate_image`)

```
Create a 4-panel horizontal comic strip of [NAME], the same character as in
the reference images. [SPEC BLOCK] Panel 1: [beat]. Panel 2: [beat].
Panel 3: [beat]. Panel 4: [beat]. Thin black gutters between panels, no
speech bubbles, no captions, identical character design in every panel.
```

`aspect_ratio="4:1"` (or `"8:1"` for eight panels), `image_size="2K"`,
`model="gemini-nano-banana-2.1"`, `reference_images=["[master path]", "[sheet path]"]`.

## Drift check (`compare_images`)

```
Image 1 is the approved master. For every other image answer in this
order: is it the same character, yes or no; then list each difference in
face shape, eyes, hair or fur colour, skin or body colour, outfit,
signature prop, proportions and art style. Ignore pose, expression,
background and lighting. End with one verdict per image, pass or fail.
```

`image_paths=["[master path]", "[scene path]", ...]` (2 to 10 images),
`model="gemini-3.8-flash"` when the verdict matters.

## Single-trait fix (`edit_image`)

```
Using the provided image, change only the [trait] to [exact value, hex code
if a colour]. Keep everything else in the image exactly the same,
preserving the original style, lighting and composition.
```

`input_image_path="[scene path]"`, same `model` as the series.

## Reference image limits per model

| Model | `reference_images` (server docs) | Character refs (Google docs) | Style refs | Panoramic ratios |
|---|---|---|---|---|
| `gemini-nano-banana-2.1` | 14 | 4 | no | yes |
| `gemini-3.1-flash-image` | 14 | 4 | no | yes |
| `gemini-3-pro-image` | 6 | 5 | up to 3 | no |
| `gemini-3.1-flash-lite-image` | not optimized for reference inputs | - | no | no |

The server validates only the 14-image total; the per-model columns are the
tool description and Google's documented guidance. Past a cap, collage several
views into one image and pass the collage as a single reference.

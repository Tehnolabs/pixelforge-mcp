# Edit prompts, server wrappers and verification snippets

Companion to the pixelforge-edit skill. Section 1 quotes Google's Gemini image
generation guide (ai.google.dev/gemini-api/docs/image-generation, page text CC BY 4.0).
Section 2 is the exact text the PixelForge server wraps around your prompt, copied from
its source, so you can write a prompt that reads naturally inside it.

Contents: 1. Google's editing templates, 2. Server wrappers per advanced_edit mode,
3. Verification prompts, 4. Mask and key-colour snippets.

## 1. Google's editing templates (verbatim)

Add or remove an element (`edit_image`):

> Using the provided image of [subject], please [add/remove/modify] [element] to/from the scene. Ensure the change is [description of how the change should integrate].

Change one thing and nothing else (`edit_image`; there is no mask parameter here):

> Using the provided image, change only the [specific element] to [new element/description]. Keep everything else in the image exactly the same, preserving the original style, lighting, and composition.

Style transfer (`edit_image`, or `advanced_edit` with `style_transfer`):

> Transform the provided photograph of [subject] into the artistic style of [artist/art style]. Preserve the original composition but render it with [description of stylistic elements].

Combine elements from several images (`generate_image` with `reference_images`):

> Create a new image by combining the elements from the provided images. Take the [element from image 1] and place it with/on the [element from image 2]. The final image should be a [description of the final scene].

High-fidelity detail preservation (`generate_image` with `reference_images`):

> Using the provided images, place [element from image 2] onto [element from image 1]. Ensure that the features of [element from image 1] remain completely unchanged. The added element should [description of how the element should integrate].

Sketch to finished image (`edit_image`):

> Turn this rough [medium] sketch of a [subject] into a [style description] photo. Keep the [specific features] from the sketch but add [new details/materials].

Rules Google states alongside them: focus the prompt on what changes and what stays;
start with a strong verb; there is no negative prompt, so write semantic negatives ("an
empty street with no traffic") instead of "no cars"; for text, put the exact words in
quotes and describe the font; iterate with follow-ups such as "Keep everything the same,
but change the character's expression".

## 2. What the server sends for each advanced_edit mode (verbatim)

Your `prompt` replaces `{prompt}`. Write it as the fragment the sentence expects.

| `edit_mode` | Instruction sent to the model |
|---|---|
| `inpaint_removal` | Remove the following from the image: {prompt}. Fill the vacated area so it blends seamlessly with the surroundings. Keep everything else exactly as it is. |
| `inpaint_insertion` | Add the following to the image: {prompt}. Integrate it naturally with matching lighting, perspective, scale, and shadows. Keep everything else exactly as it is. |
| `outpaint` | Extend the image beyond its current borders, continuing the scene naturally and consistently. {prompt}. Keep the original content unchanged in the center. |
| `background_swap` | Replace the background with: {prompt}. Keep the main subject exactly as it is, with clean edges and lighting that matches the new background. |
| `style_transfer` | Re-render the image in this style: {prompt}. Preserve the composition, subjects, and layout. |
| `product_image` | Create a professional product photograph of the main subject: {prompt}. Keep the product's shape, colors, labels, and proportions exactly as shown. |

With `mask_path`, the mask goes in as a second image followed by: "The second image is a
mask for the first image: white areas mark the region to change, black areas must stay
exactly as they are."

`upscale_image` sends the source plus: "Upscale this image to a higher resolution.
Reproduce it exactly: the same composition, subjects, colors, lighting, text, and framing.
Only add fine detail and sharpness. Do not add, remove, move, or restyle anything."

`remove_background` sends: "Remove the background completely. Keep only the main subject.
Make the background transparent or pure white."

## 3. Verification prompts

- Drift check with `compare_images`: "Image 1 is the original, image 2 the edit. The
  intended change was X. List every other difference, however small, in subject, pose,
  colours, lighting, text, or framing. Reply none if there is none."
- Text check: run `extract_text` on both files and diff the `text` fields, or ask
  `compare_images` "Compare all visible text in image 1 and image 2 word by word and list
  any difference."
- Product fidelity: "Compare the product in image 1 and image 2. Report any change in
  shape, proportions, colours, logos, or label text."
- Did the change land: `analyze_image` with `prompt="Is there a <object> at <place>? Answer yes or no, then describe it."`

## 4. Mask and key-colour snippets

Both need Pillow in the agent's Python (`pip install pillow`); ImageMagick is the
alternative for keying.

Rectangle mask for `advanced_edit` (white = change, black = keep):

```python
from PIL import Image, ImageDraw
src = Image.open("source.png")
mask = Image.new("RGB", src.size, "black")
ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill="white")
mask.save("mask.png")
```

Key out a flat backdrop after a `background_swap` to pure green:

```python
from PIL import Image
img = Image.open("on-green.png").convert("RGBA")
px = img.load()
key, tol = (0, 255, 0), 90
for y in range(img.height):
    for x in range(img.width):
        r, g, b, _ = px[x, y]
        if abs(r - key[0]) + abs(g - key[1]) + abs(b - key[2]) < tol:
            px[x, y] = (r, g, b, 0)
img.save("cutout.png")
```

ImageMagick one-liner: `magick on-green.png -fuzz 15% -transparent "#00FF00" cutout.png`.
Raise `tol` or the fuzz percentage if a green fringe remains; lower it if the subject loses pixels.

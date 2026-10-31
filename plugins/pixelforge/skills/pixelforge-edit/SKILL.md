---
name: pixelforge-edit
description: Edits an image file the user already has with the PixelForge MCP tools and chooses the right tool for each change. Use it whenever the request is about changing an existing picture, such as "remove the background", "cut out the product", "make the background transparent", "crop this to 16:9", "resize to 512 px", "rotate", "flip", "blur", "sharpen", "grayscale", "watermark it", "remove the person from the photo", "add a lamp on the desk", "extend the image", "outpaint", "swap the background", "inpaint this area", "restyle it as watercolor", "turn this into a product shot", "upscale to 4K", "make it higher resolution", "enlarge for print", or "change only X and keep everything else". Also use it to check that an edit changed nothing else.
allowed-tools: mcp__plugin_pixelforge_pixelforge__transform_image mcp__plugin_pixelforge_pixelforge__edit_image mcp__plugin_pixelforge_pixelforge__advanced_edit mcp__plugin_pixelforge_pixelforge__remove_background mcp__plugin_pixelforge_pixelforge__upscale_image mcp__plugin_pixelforge_pixelforge__compare_images mcp__plugin_pixelforge_pixelforge__generate_image mcp__plugin_pixelforge_pixelforge__analyze_image mcp__plugin_pixelforge_pixelforge__extract_text mcp__plugin_pixelforge_pixelforge__estimate_cost mcp__plugin_pixelforge_pixelforge__list_available_models mcp__plugin_pixelforge_pixelforge__get_server_info
---

# Editing an existing image with PixelForge

Every PixelForge editing tool takes a local png, jpg, jpeg or webp path and writes its
result into the server's output directory (`get_server_info` reports it as
`output_directory`). `output_filename` is a bare file name, never a path, and the same
name silently overwrites, so version the names and copy results into the project after.

## Pick the tool

Work down the list and stop at the first match.

1. Geometry or a pixel filter (crop, resize, rotate, flip, blur, sharpen, grayscale,
   text watermark): call the PixelForge `transform_image` tool. It is deterministic
   Pillow code, costs nothing, and touches only what it is told to.
2. The change is one of the six `advanced_edit` jobs (remove an object, add an object,
   extend the canvas, replace the background, restyle, product shot): call
   `advanced_edit` with that `edit_mode`. It wraps the prompt in a tuned instruction,
   re-renders at the smallest 1K/2K/4K tier that covers the source's longest edge (a
   16:9 1K source, 1376 px wide, comes back at 2K), pins the ratio, and accepts a mask.
3. Any other change described in words (colour, expression, lighting, materials, text,
   a small composition tweak): call `edit_image` with a prompt that names the one thing
   to change and says to keep everything else exactly the same.
4. More resolution: `upscale_image` re-renders and invents detail; `transform_image`
   with `operation="resize"` keeps the pixels exactly (pixel art, screenshots, UI,
   anything whose text must not be redrawn).
5. A cutout: `remove_background` gives the subject on flat white, never transparency.
   For a real alpha channel follow "Real cutouts" below.

## Workflow for a model-driven edit

1. Note the source's pixel size (`file image.png` prints it) and keep the original
   untouched. Every retry starts from the original, not from a previous output.
2. Pick the model. Default `gemini-nano-banana-2.1`. Pass `model="gemini-3-pro-image"`
   when the image carries text that must survive, when faces or product details drift,
   or when a chain of edits keeps degrading. Do not edit with
   `gemini-3.1-flash-lite-image` (1K only, not built for multi-turn editing;
   `upscale_image` rejects it outright).
3. Cost gate: for more than a few edits, or any source over 1024 px on its longest edge
   going through `advanced_edit` or `upscale_image`, call `estimate_cost` with
   `operation="edit"` (1K price) and read the per-tier prices in `list_available_models`.
4. Write the prompt: one attribute per pass, say what changes and what stays. Read
   `references/edit-prompts.md` for Google's edit templates and the exact wrapper each
   `advanced_edit` mode adds around your text.
5. Run the edit with a descriptive `output_filename` (`hero-v2-sky.png`) and read the
   `warnings` field of the result.
6. Verify. Call `compare_images` with `image_paths=[original, result]` and a prompt such
   as "Image 1 is the original, image 2 the edit. The intended change was X. List every
   other difference, however small. Reply none if there is none." Run `extract_text` on
   both when the image contains text, and `analyze_image` to confirm the change landed.
7. If anything else moved, retry from the original with a stronger keep clause, a mask
   through `advanced_edit`, a lower `temperature` (edit_image only, try 0.3), or
   `gemini-3-pro-image`. Never stack a corrective edit on a drifted output.

## transform_image operations

| `operation` | Required parameters | Behaviour |
|---|---|---|
| `crop` | `x`, `y`, `width`, `height` | Box from the top-left corner; a box past the edge is padded, not rejected |
| `resize` | `width`, `height` | `maintain_aspect=True` (default) fits inside the box and only shrinks; `maintain_aspect=False` enlarges or stretches to the exact size |
| `rotate` | `degrees` | Counter-clockwise; the canvas grows to fit, corners are filled |
| `flip` | `direction` | `"horizontal"` or `"vertical"` |
| `blur` | none | `radius`, default 2.0 |
| `sharpen` | none | `factor`, default 2.0; above 1 sharpens |
| `grayscale` | none | Returns RGB |
| `watermark` | `text` | `position` is `top-left`, `top-right`, `bottom-left`, `bottom-right` or `center`; `opacity` 0.0 to 1.0; fixed 24 px white text |

For an exact target canvas, `crop` to the ratio first, then `resize` with
`maintain_aspect=False` and both dimensions computed from that ratio.

## advanced_edit modes

| `edit_mode` | The wrapper asks the model to | Put in `prompt` |
|---|---|---|
| `inpaint_removal` | remove it, fill the gap seamlessly, keep the rest | the object to remove, as a noun phrase ("the red car on the left") |
| `inpaint_insertion` | add it with matching light, perspective, scale and shadows | the object and where it goes |
| `outpaint` | extend the scene past the borders, keep the centre unchanged | what the new area shows, and the target direction and shape |
| `background_swap` | replace the background, keep the subject, match lighting | the new background |
| `style_transfer` | re-render in that style, preserve composition and layout | the style in words |
| `product_image` | shoot the subject as a product photo, keep shape, colours, labels | surface, lighting, angle |

Facts the server enforces:

- Output size is the smallest of 1K/2K/4K that covers the source's longest edge, clamped
  to the model's largest tier. A source under 1024 px comes back at 1K, larger than it
  was; one over 1024 px bills 2K or 4K (a 16:9 1K source, 1376 px wide, comes back at 2K).
- Aspect ratio is pinned to the nearest ratio the model supports, except `outpaint`,
  which leaves the ratio to the model. State the target shape in the outpaint prompt
  ("extend left and right into a wide 16:9 landscape"), then crop or resize if the
  canvas must be exact. Panoramic sources (1:4, 4:1, 1:8, 8:1) keep their shape only on
  `gemini-nano-banana-2.1` or `gemini-3.1-flash-image`; other models get a standard
  ratio and a warning.
- `mask_path` is a png/jpg/jpeg/webp file that exists; draw it at the source's
  dimensions. White marks the region to change, black the region to keep. The server
  sends it as a second image with that instruction, and the model reads it semantically,
  so the boundary is not pixel-exact and unmasked areas are not guaranteed untouched. Use
  a mask when the prompt alone keeps picking the wrong object; draw it with Pillow (black
  canvas, white rectangle; snippet in the reference file).

## Real cutouts

`remove_background` is `edit_image` with a fixed prompt that asks for a transparent or
pure white background. The model paints pixels, so the result is the subject on flat
white, or a drawn checkerboard, and saving as png does not add alpha. Use it when a white
backdrop is the deliverable (catalogue shots), then stage with `advanced_edit`
`product_image` or `background_swap`. For a transparent PNG:

1. Call `advanced_edit` with `edit_mode="background_swap"` and
   `prompt="a flat, uniform, pure green (#00FF00) backdrop with no gradient, texture or shadow"`.
   Pick a key colour the subject does not contain (magenta for plants, green for people).
2. Key it out in code outside the server (Pillow or ImageMagick, snippets in the
   reference file).
3. Check the edges with `analyze_image`; widen the tolerance if a coloured fringe remains.

## upscale_image

A generative re-render: the source goes in as a reference with an instruction to
reproduce it exactly and only add detail. The server's tier math is longest edge times
the factor (`upscale_factor` is `"x2"` or `"x4"`), then the smallest of 1K (1024),
2K (2048) or 4K (4096) whose nominal long edge covers it, capped at 4K. A source already
4096 px or more on its longest edge is an error. The nominal edge is used even for wide
images, so a 16:9 1K source (1376 px wide) at `x2` lands in the 4K tier; read
`image_size`, `output_size` and `warnings` in the result, and price that tier with
`list_available_models` before upscaling many files.

- Models: default `gemini-nano-banana-2.1`; `gemini-3-pro-image` and
  `gemini-3.1-flash-image` also work; `gemini-3.1-flash-lite-image` is rejected.
- Aspect ratio is pinned the same way as `advanced_edit`, with the same panoramic rule.
- Text, logos and labels can be redrawn with mistakes. Run `extract_text` on source and
  result and diff them; if anything changed, use `transform_image` `resize` with
  `maintain_aspect=False` instead and accept the softer pixels.

## Style references

`edit_image` accepts one image and no references. To restyle an image after example
images, call `generate_image` with `model="gemini-3-pro-image"`,
`reference_images=[source, style1, ...]` and a prompt that names the roles ("Re-render
image 1 in the style of images 2 and 3. Keep the composition of image 1 exactly"). Use up
to 3 style references on `gemini-3-pro-image` (Google's documented limit); the server
accepts up to 14 reference paths in total. Pass `aspect_ratio` matching the source (the
default is 1:1) and an `image_size` matching its tier. For a style you can describe in
words, `advanced_edit` with `style_transfer` is cheaper and sizes the output from the
source automatically.

## Gotchas

- No model-backed tool can output an alpha channel. "Transparent" in a prompt yields
  painted white or a checkerboard. Key colour plus external keying is the only route.
- Two changes in one prompt often move a third thing. One attribute per pass, and end
  every prompt with "Keep everything else exactly the same, preserving the original
  style, lighting, and composition."
- `edit_image` and `remove_background` send no size or ratio settings; the model usually
  matches the source, but check the output's dimensions and fix any drift with
  `transform_image`. `advanced_edit` and `upscale_image` pin both from the source.
- `edit_image` always sends `temperature` (default 0.7); `advanced_edit`, `upscale_image`
  and `remove_background` never do. If the Vertex AI backend rejects an `edit_image`
  call over a generation parameter, rerun the edit through `advanced_edit`.
- Every model call bills one image at the output tier, including the ones you discard
  after verification. Iterate on a copy resized to 1024 px on its longest edge
  (`transform_image` `resize`) and run the accepted prompt once on the full-size original.
- Apply `watermark` last, at delivery size: its 24 px text is unreadable on 2K/4K.
- `jpeg` output flattens any alpha onto white; keep `png` until the final export.
- Identity drifts over chained edits. After two or three passes, compare the latest
  result with the original, not just with the previous step.
- Every image a model returns carries a SynthID watermark, edited ones included. Say so
  when the image is headed for stock licensing or provenance-sensitive use.
- Never use `-preview` model IDs or `gemini-2.5-flash-image`; `list_available_models`
  has the current list, and the server rewrites known old IDs with a warning.

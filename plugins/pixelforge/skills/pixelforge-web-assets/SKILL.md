---
name: pixelforge-web-assets
description: Produces the image assets a website or repository needs with the PixelForge MCP tools, namely a README or landing-page hero, an Open Graph or social share card at 1200x630, and an app icon, favicon and PWA icon family. Use it whenever the user asks for a hero image, header banner, blog cover, og image, social preview, Twitter, LinkedIn or GitHub card, favicon, app icon, apple-touch-icon, manifest icons, or any image that must land at exact pixel dimensions, because the Gemini image models only output fixed aspect ratios and this skill holds the generate, crop and resize sequences that hit the target sizes.
allowed-tools: mcp__plugin_pixelforge_pixelforge__generate_image mcp__plugin_pixelforge_pixelforge__batch_generate mcp__plugin_pixelforge_pixelforge__transform_image mcp__plugin_pixelforge_pixelforge__edit_image mcp__plugin_pixelforge_pixelforge__extract_text mcp__plugin_pixelforge_pixelforge__analyze_image mcp__plugin_pixelforge_pixelforge__detect_objects mcp__plugin_pixelforge_pixelforge__compare_images mcp__plugin_pixelforge_pixelforge__optimize_prompt mcp__plugin_pixelforge_pixelforge__estimate_cost mcp__plugin_pixelforge_pixelforge__list_available_models
---

# PixelForge web assets

Each job is one fixed sequence on the PixelForge MCP server: `generate_image` makes a
master at a native Gemini ratio, `transform_image` crops and resizes it to the exact
pixels the destination wants, and any headline or title is rendered in code, not by the
model. Read `references/dimensions.md` when the target is anything other than a
1200x630 card, a 16:9 or 21:9 hero, or a square icon.

## Rules shared by every job

1. Files land in the server's output directory (default `generated_images/` relative
   to where the server process started), not at the path the project serves from.
   Every result returns the absolute path; copy the final file into `public/`,
   `static/`, `assets/` or `app/`.
2. `output_filename` is a bare name with no path separators. Give it an extension that
   matches `output_format`; a name without one gets `.png` appended even when the bytes
   are JPEG (`output_filename="og-image.jpg", output_format="jpeg"`).
3. Neither `generate_image` nor `transform_image` reports pixel dimensions, and the
   model does not always return the size in Google's table. Measure before computing
   a crop: `file PATH` prints width x height (or `python3 -c "from PIL import Image;
   print(Image.open('PATH').size)"`).
4. Use the default model (`gemini-nano-banana-2.1`). `quality="fast"` is for throwaway
   placeholders only (1K, no panoramic ratios). `gemini-3-pro-image` is only for text
   that must be baked into the image.
5. Before more than four images, or any 2K or 4K job, call
   `estimate_cost(operation="generate", number_of_images=N)`. It prices at 1K; for 2K
   and 4K read `pricing` in `list_available_models`.
6. Same prompt, several candidates: `number_of_images` (max 4). Different prompts:
   `batch_generate`, which shares one model and ratio and has no `image_size`.
7. Keep PNG through intermediate steps. Set `output_format="jpeg"` or `"webp"` only on
   the last call, and only for photographic content that would otherwise exceed 1 MB.

## Job 1. README or landing-page hero

1. Decide where the headline goes. A landing page overlays it in HTML/CSS, so the
   image must be text-free with empty space on the headline side. A GitHub README
   cannot overlay text (styles are stripped), so keep the hero text-free and put the
   title in Markdown below it; bake text in only when the user insists (step 5).
2. Write the prompt with Google's negative-space template, filling the brackets:
   "A minimalist composition featuring a single [subject] positioned in the
   [right third] of the frame. The background is a vast, empty [colour] canvas,
   creating significant negative space. Soft, subtle lighting." Add the brand palette
   as hex codes and the mood. Phrase exclusions as a description ("a clean background
   with no text, letters, logos or watermarks"); there is no negative-prompt
   parameter. When the brief is thin, run `optimize_prompt(prompt=...,
   style="cinematic")` first (`"illustration"`, `"3d_render"` and `"abstract"` also
   fit heroes).
3. Generate. README: `generate_image(prompt=..., aspect_ratio="16:9",
   image_size="1K", output_filename="hero.png")` (about 1376x768, enough for a README
   column). Landing page: `aspect_ratio="21:9", image_size="2K"` (about 3168x1344),
   then serve at or below 1920 wide with `transform_image(image_path=...,
   operation="resize", width=1920, height=1080, maintain_aspect=true,
   output_filename="hero.webp", output_format="webp")`.
4. Check the composition: `analyze_image(image_path=..., prompt="Is the right third
   free of detail and suitable for white headline text? Is any text, lettering or
   watermark present?")`. Fix with one `edit_image` pass per change ("Keep everything
   exactly the same, but clear the right third to a plain gradient"), or regenerate.
5. Text inside the image, only when required: `generate_image(model=
   "gemini-3-pro-image", ...)` with the exact words in quotes plus a font description,
   then `extract_text(image_path=...)` and diff against the intended copy. Fix one
   wrong word per `edit_image` call and run `extract_text` again on the result.
6. Overlay in code, for example `.hero{background:url(/hero.webp) center/cover}` with
   the `h1` padded into the empty third. For a dark-mode README variant run
   `edit_image(prompt="Keep everything exactly the same but make the background dark
   charcoal (#0d1117) and lift the subject's lighting", input_image_path=...)` and
   serve both through `<picture>` with `prefers-color-scheme`. `edit_image` sends no
   size config, so measure the variant and `resize` it to match the original.

## Job 2. Open Graph or social card (1200x630)

1. Generate the background only, subject on one side, no text:
   `generate_image(prompt=..., aspect_ratio="16:9", image_size="1K",
   output_filename="og-source.png")`.
2. Measure the file (rule 3) and crop centred to 1.905 to 1 (1200/630): crop the width
   if width/height is above 1.905, otherwise the height. For a 1376x768 source the
   height becomes round(1376/1.905) = 722 and y = (768-722)/2 = 23:
   `transform_image(image_path=..., operation="crop", x=0, y=23, width=1376,
   height=722, output_filename="og-crop.png")`. Shift `x` or `y` off centre when the
   subject sits near an edge; `detect_objects` returns `box_2d` as
   `[y_min, x_min, y_max, x_max]` on a 0-1000 scale to locate it.
3. Resize to the exact target with stretching allowed (the crop already matches the
   ratio, so the stretch is under 1 percent): `transform_image(image_path=...,
   operation="resize", width=1200, height=630, maintain_aspect=false,
   output_filename="og-image.png")`.
4. Render the title, description and logo in code on top of the resized image (satori
   or `@vercel/og` in Next.js, Pillow, ImageMagick, or a screenshot of an HTML
   template), keeping everything inside a 60 px margin. The model misspells and
   misplaces text, and every platform crops the edges differently.
5. Verify with `file` (1200x630) and check the size stays under 1 MB; for a heavy
   photo repeat step 3 with `output_filename="og-image.jpg", output_format="jpeg"`.
   Add `og:image` (absolute URL), `og:image:width` 1200, `og:image:height` 630 and
   `twitter:card` `summary_large_image`. The same file serves Twitter, LinkedIn and
   Slack previews.

## Job 3. App icon, favicon and PWA icon family

1. One square master: `generate_image(prompt=..., aspect_ratio="1:1", image_size="1K",
   output_filename="icon-master.png")`. Prompt rules: one bold glyph centred, filling
   about 60 percent of the canvas, a flat solid background in the brand colour, "square
   corners, no text, no drop shadow, no device frame or mockup, legible at 16 pixels".
   Do not use `apply_template("icon_set", ...)` for the master; its wording bakes in
   rounded corners, and iOS and maskable manifest icons apply their own masks.
2. Test legibility before making the family: `transform_image(image_path=...,
   operation="resize", width=32, height=32, output_filename="probe-32.png")` and look
   at it. If the glyph muddies, simplify with one `edit_image` pass ("Keep everything
   exactly the same, but remove the inner detail lines of the glyph").
3. Derive every size from the master, never from a smaller output:
   `transform_image(image_path=MASTER, operation="resize", width=N, height=N,
   maintain_aspect=true, output_filename=...)` for N in 512, 192, 180, 48, 32, 16,
   named `icon-512.png`, `icon-192.png`, `apple-touch-icon.png`, `favicon-48.png`,
   `favicon-32.png`, `favicon-16.png`.
4. Package outside the server. `favicon.ico` (16, 32, 48 in one file):
   `python3 -c "from PIL import Image; Image.open('icon-master.png').save(
   'favicon.ico', sizes=[(16,16),(32,32),(48,48)])"` or `magick icon-master.png
   -define icon:auto-resize=48,32,16 favicon.ico`. Modern browsers accept a PNG
   `<link rel="icon">`, so the ICO is optional. SVG and monochrome mask icons are
   vector work the model cannot do.
5. Wire the `<link>` tags and manifest from section 4 of `references/dimensions.md`.
   Mark the 512 icon `"purpose": "any maskable"` only if the glyph stays inside the
   central 80 percent.
6. A glyph that must float on any tab colour needs real transparency, which no
   PixelForge generation or editing tool produces (`remove_background` returns a
   flat, usually white, background). Generate the master on a flat key colour ("solid
   pure magenta #FF00FF background, no gradient"), key it out with Pillow or
   ImageMagick at about 10 percent tolerance (the model returns a near-key colour,
   not the exact one), then run step 3 on the keyed-out PNG: `transform_image` keeps
   the alpha channel as long as `output_format` stays `"png"`.
7. Family variants (dark, monochrome): `edit_image` on the master with "Keep the
   glyph, proportions and margins exactly the same, change only ..." one attribute per
   call, then `compare_images(image_paths=[MASTER, VARIANT], prompt="List any
   difference in glyph shape, position or margins")`.

## Gotchas

- No PixelForge generation or editing tool outputs an alpha channel. "Transparent
  background" in a prompt yields a painted checkerboard or a flat colour.
  `transform_image` keeps an alpha channel the source already has when
  `output_format` is `"png"` or `"webp"`; `"jpeg"` flattens it onto white and
  `operation="grayscale"` drops it.
- Google's "16:9" output is 1376x768 (1.79 to 1), not exact 16:9. For an exact 16:9
  target such as 1280x720, resize with `maintain_aspect=false` (under 1 percent
  stretch) or crop to 1365x768 first.
- `maintain_aspect=true` fits within `width` x `height` and never enlarges; a resize
  larger than the source returns the same pixels. Needing to upscale a web asset means
  the master tier was too small: regenerate at `"2K"`.
- `crop` takes `x`, `y`, `width`, `height` of the kept region in pixels, and the region
  must fit inside the source; `resize` takes `width` and `height`.
- 1.91:1, 2:1, 3:1 and 5.9:1 are not native ratios. Crop from 16:9 (cards), 21:9 or
  4:1 (banners) and 8:1 (very wide banners). 4:1 and 8:1 render only on
  `gemini-nano-banana-2.1` and `gemini-3.1-flash-image`; `quality="fast"` and
  `gemini-3-pro-image` silently return a standard shape, so crop from 21:9 there.
- `0.5K` exists only on `gemini-3.1-flash-image`, and `gemini-3.1-flash-lite-image`
  outputs 1K only, so `quality="fast"` cannot make a 2K hero.
- Do not combine `quality` with `image_size`. The preset's size wins silently
  (`"balanced"` stays 1K, `"quality"` stays 2K) and `"fast"` rejects anything but 1K.
  For a 2K hero on the default model pass `image_size="2K"` and no `quality`.
- Each `transform_image` call re-encodes the file; crop then resize is two saves, so
  keep both as PNG until the final step.
- Any hero or card with model-rendered text needs `extract_text` on the final file,
  every time, including after an `edit_image` fix.
- Every output carries an invisible SynthID watermark; mention it when the user asks
  about provenance or stock licensing.

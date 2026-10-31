---
name: pixelforge-characters
description: Keeps one character, mascot or brand figure visually identical across a series of PixelForge images. Use it whenever a user asks for "the same character in different scenes", a mascot set, an expression or pose sheet, a sprite sheet, a turnaround, a storyboard or comic strip, "keep the face the same", matching brand illustrations, or any batch of images that must share a subject. It builds a reference sheet first, generates every scene one call at a time with reference_images, and checks drift with compare_images.
allowed-tools: mcp__plugin_pixelforge_pixelforge__generate_image mcp__plugin_pixelforge_pixelforge__edit_image mcp__plugin_pixelforge_pixelforge__advanced_edit mcp__plugin_pixelforge_pixelforge__compare_images mcp__plugin_pixelforge_pixelforge__transform_image mcp__plugin_pixelforge_pixelforge__estimate_cost mcp__plugin_pixelforge_pixelforge__list_available_models
---

# Consistent characters and mascots with PixelForge

Identity survives only when every call sees the same reference images and the
same written spec. Build the references once, then spend the budget on scenes.

## Defaults

- Model `gemini-nano-banana-2.1` (up to 14 `reference_images`, panoramic
  strips, 1K/2K/4K). Switch to `gemini-3-pro-image` for human faces, for
  style references (Pro only, at most 3), or when drift persists. Never use
  `quality="fast"` or `gemini-3.1-flash-lite-image` for a series; that model
  is not optimized for reference inputs and outputs 1K only.
- One image per call. Keep `number_of_images=1` and do not use
  `batch_generate`; both run images in parallel, and parallel runs drift.
- Master and sheet at `aspect_ratio="1:1"`; master at `image_size="1K"`,
  sheet at `"2K"` so each cell stays legible. Scenes at the ratio the slot
  needs.
- Keep `model` and `temperature` identical across the whole series.
- Pass absolute paths. The server resolves `reference_images`,
  `input_image_path` and `image_paths` in its own process, so use
  `images[0].path` from `generate_image` and `edit_image` results,
  `output_path` from `transform_image` and `advanced_edit` results, and
  expand user-supplied files before passing them.
- Name files with `output_filename` so the series is traceable, for example
  `pip_master.png`, `pip_sheet.png`, `pip_scene_01.png`. The value cannot
  contain path separators; letters, digits, `_`, `-` and `.` only.

## Workflow

1. Write the character spec. Give the character a short name and 5 to 8 fixed
   traits in one block (species or build, face, hair or fur colour with hex
   codes, outfit, signature prop, art style, line weight). Reuse this block
   verbatim in every prompt; the model follows text as much as pixels.
   Prompts are capped at 2000 characters, so keep the block under 400.
2. Generate the master. Call the PixelForge `generate_image` tool with the
   master prompt from `references/prompts.md` (spec block, full body, neutral
   pose, plain flat background, no text), `aspect_ratio="1:1"`,
   `output_filename="<name>_master.png"`. Show the user the result and get
   approval before spending more; every later image inherits its flaws.
3. Generate the reference sheet. Call `generate_image` with
   `reference_images=[<master path>]`, `image_size="2K"`, and the sheet
   prompt from `references/prompts.md` (4-view turnaround, or a 3x3 grid of
   poses and expressions). Say "the same character as the reference image"
   and repeat the spec block. Confirm `details.reference_images_count` in
   the result equals the number of paths you passed.
4. Optional: crop the sheet into single cells with the PixelForge
   `transform_image` tool, `operation="crop"` with `x`, `y`, `width` and
   `height` in pixels, so you can pass only the matching view (for example
   the back view) as an extra reference later.
5. Gate the cost. For more than three scenes call the PixelForge
   `estimate_cost` tool with `operation="generate"` and the chosen `model`
   (it prices 1K; read `list_available_models` for the 2K and 4K tiers) and
   tell the user the total, counting one redo pass per scene.
6. Generate scenes sequentially. For each scene call `generate_image` with
   `reference_images=[<master>, <sheet>]` (plus the cropped view that
   matches the pose when you have one) and the scene prompt from
   `references/prompts.md`, which names the character, repeats the spec
   block, then describes action, setting, composition and lighting. Wait
   for each result before sending the next call.
7. Check drift. After each scene, or after every batch of up to 9, call the
   PixelForge `compare_images` tool with `image_paths=[<master>, <scene>...]`
   and the drift prompt from `references/prompts.md`. Pass
   `model="gemini-3.8-flash"` when the verdict matters. Treat any flagged
   change to face, proportions, palette or signature items as a failure.
8. Fix a failed scene from the master, not from the failed image.
   Regenerate with the same references and a prompt that calls out the
   drifted trait ("the scarf is red, hex #C8102E, not orange"). If only one
   trait is off, call the PixelForge `edit_image` tool on the scene with the
   single-trait fix prompt from `references/prompts.md`. One trait per pass.

## More than one character

- Give every character its own name, spec block and master. In the scene
  prompt name each one and bind traits to names ("Pip, the blue robot, is on
  the left; Mara, the red fox, is on the right").
- Pass all masters in `reference_images` in the order you name them. The
  server accepts up to 14 images in total; `gemini-3-pro-image` takes 6.
- Keep it to three named characters per scene. Google's reference guidance
  caps character references at 4 per prompt on `gemini-nano-banana-2.1` and
  `gemini-3.1-flash-image` and 5 on `gemini-3-pro-image`, and traits start
  blending past three. For a crowd, generate the leads with references and
  describe everyone else generically.
- Set `thinking_budget` (for example 4096) on `gemini-nano-banana-2.1` when
  a multi-character scene comes back with swapped traits.

## Storyboards and comic strips

Pick one of two routes before generating.

- Strip route (cheapest, panels share one render): a single
  `generate_image` call with `aspect_ratio="4:1"`, `image_size="2K"`,
  `model="gemini-nano-banana-2.1"` (or `gemini-3.1-flash-image`), references
  attached, and the strip prompt from `references/prompts.md`, which numbers
  the panels. Use `"8:1"` for eight panels. Split the result with
  `transform_image` crops. Read the width from the file (`transform_image`
  does not report dimensions), divide by the panel count, and crop each
  panel at that `x` offset with the full `height`.
- Per-panel route (best faces, any model): one `generate_image` call per
  panel at the final ratio, same references, prompt prefixed with "Panel N
  of M of a storyboard". Drift-check every panel against the master.

## Re-using an existing character image

- New setting, same character: the PixelForge `advanced_edit` tool with
  `edit_mode="background_swap"`; its instruction keeps the main subject as
  it is.
- Campaign restyle: `advanced_edit` with `edit_mode="style_transfer"`, then
  drift-check, because a style pass can reshape faces.
- Any other change: `edit_image`, one attribute per pass, always ending
  with "Keep everything else exactly the same".

## Gotchas

- `edit_image` has no `reference_images` parameter; only `generate_image`
  does. The image you edit is the only reference the model sees. To place
  the character in a new scene, generate with references instead of editing.
- `batch_generate` has no `reference_images` and runs prompts in parallel, so
  a series through it loses the character entirely.
- Edits degrade regions you did not mention. Chains of edits drift; go back
  to the master after two passes on the same image.
- Reference images steer, they do not lock. Say "the same character as in
  the reference images" and still repeat the spec block every time.
- No transparent output exists. Generate mascots on a flat solid colour and
  key it out in code; `remove_background` returns a flat white background.
- Panoramic ratios (`1:4`, `4:1`, `1:8`, `8:1`) work only on
  `gemini-nano-banana-2.1` and `gemini-3.1-flash-image`; other models render
  a standard shape without an error.
- Photoreal identifiable real people are restricted by the API;
  `person_generation` is accepted but not enforced. Use an invented
  character or an illustrated style.
- Reference images cost input tokens on every call; `estimate_cost` does not
  include them.
- The same `output_filename` overwrites the earlier file. Number scenes.
- Read `references/prompts.md` for the master, sheet, scene, strip,
  drift-check and single-trait fix templates and the per-model reference
  limits table.

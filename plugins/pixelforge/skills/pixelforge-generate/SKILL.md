---
name: pixelforge-generate
description: Generates a new image from a text brief with the PixelForge MCP server (Gemini image models), from preset and aspect ratio choice through prompt writing, cost check, generation, verification and one-change-per-pass refinement. Use whenever the user wants a picture made from words, such as "generate an image", "create a logo", "make a hero image", "draw an illustration", "render a product shot", "give me three variations", "concept art", "banner", "poster", "sticker", or hands over a prompt and asks for the image. Also use before any batch of images so cost and model limits are checked first.
allowed-tools: mcp__plugin_pixelforge_pixelforge__generate_image mcp__plugin_pixelforge_pixelforge__batch_generate mcp__plugin_pixelforge_pixelforge__optimize_prompt mcp__plugin_pixelforge_pixelforge__list_templates mcp__plugin_pixelforge_pixelforge__apply_template mcp__plugin_pixelforge_pixelforge__estimate_cost mcp__plugin_pixelforge_pixelforge__list_available_models mcp__plugin_pixelforge_pixelforge__analyze_image mcp__plugin_pixelforge_pixelforge__extract_text mcp__plugin_pixelforge_pixelforge__edit_image
---

# Generate an image from text

Text-to-image with the PixelForge MCP server. For changing an existing image use the
`pixelforge-edit` skill; for hero images, OG cards and icon families `pixelforge-web-assets`;
for diagrams and slide visuals `pixelforge-diagrams`; for a recurring character or mascot
`pixelforge-characters`.

## Workflow

1. **Pin the brief.** Write one line each for subject, action, setting, composition and
   style, plus where the image will be used and whether any words must appear in it.
   Ask only when the destination or the text is unknown; otherwise choose defaults and
   say which you chose.
2. **Pick the preset.** Default to `quality="balanced"`. Use `quality="fast"` for drafts,
   placeholders and high-volume runs; `quality="quality"` when the image carries text,
   faces or brand assets. The decision table below lists the cases that need `model`
   instead. `quality` and `model` are mutually exclusive; passing both fails validation.
3. **Choose `aspect_ratio` and `image_size`.** Ratios the server accepts: `1:1` (default),
   `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`, and the panoramic
   `1:4`, `4:1`, `1:8`, `8:1`. Pick the one nearest the destination. When the destination
   needs an exact pixel size, generate here, then switch to the `pixelforge-web-assets`
   skill for the crop and resize step. Iterate at `1K`; re-render the chosen composition
   once at `2K` or `4K`.
4. **Gate the cost.** Before more than four images, any `4K` output, or any
   `gemini-3-pro-image` run, call the PixelForge `estimate_cost` tool with
   `operation="generate"`, the model and `number_of_images`, and tell the user the total.
   It prices at 1K only; for 2K/4K read the `pricing` map in the `list_available_models`
   result. Never quote prices from memory.
5. **Write the prompt** (under 2000 characters), in this order of preference:
   - A template, when one matches the job. Call `list_templates` (optionally with a
     `category`), then `apply_template(template_name, subject)` and use the returned
     `rendered_prompt`, `recommended_aspect_ratio` and `recommended_model`. A recommended
     model goes in `model`; drop `quality` for that call.
   - Your own narrative prompt. Read `references/prompting.md` when writing a prompt from
     scratch; it has the structure, the vocabulary and Google's own templates.
   - `optimize_prompt(prompt, style=...)` when the user's brief is only a few words. `style`
     is one of `photorealistic`, `illustration`, `3d_render`, `pixel_art`, `watercolor`,
     `oil_painting`, `sketch`, `anime`, `cinematic`, `product_photo`, `architecture`,
     `food`, `fashion`, `abstract`. Read the `enhanced_prompt` before using it: it is capped
     at about 500 characters and can add elements the brief did not ask for. Remove those.
6. **Generate.** Call `generate_image` with `prompt`, the preset or model, `aspect_ratio`
   and an `output_filename` that names the asset (`hero-dark.png`; no directories). The
   result's `images[].path` is absolute and lands under the server's output directory
   (default `./generated_images` relative to where the server was started); move the
   file to where the asset belongs. Read `warnings` in the result.
   Use `number_of_images` (1-4) for variations of one prompt.
7. **Verify before showing.** Call `analyze_image` with
   `prompt="Does this image match the brief: <brief>? List every mismatch."`. If the image
   must contain words, call `extract_text` and compare each string with the intended copy
   character by character. Look at the image yourself when the client renders it; the
   analysis is the fallback, not the authority.
8. **Refine one change per pass.** Call `edit_image(prompt, input_image_path)` with a single
   change and the phrase "keep everything else exactly the same". The edit inherits the
   source's size and framing, so settle ratio and tier before this step. When the
   composition itself is wrong, regenerate with a rewritten prompt instead of editing. Use
   `model="gemini-3-pro-image"` for a chain of several edits.

## Decision table

| Need | Call |
|---|---|
| Everyday image, no text | `quality="balanced"` (gemini-nano-banana-2.1 at 1K) |
| Draft, placeholder, many cheap images | `quality="fast"` (gemini-3.1-flash-lite-image, 1K only) |
| Text in the image, faces, brand assets | `quality="quality"` (gemini-3-pro-image at 2K) |
| 2K or 4K on the default model | `model="gemini-nano-banana-2.1"`, `image_size="2K"` or `"4K"` |
| Panoramic ratio (`1:4`, `4:1`, `1:8`, `8:1`) | `model="gemini-nano-banana-2.1"` (or `gemini-3.1-flash-image`) |
| Small 0.5K drafts | `model="gemini-3.1-flash-image"`, `image_size="0.5K"` |
| Style or character references | `reference_images=[...]`; see `pixelforge-characters` |
| Complex layout with many elements | `quality="quality"` (gemini-3-pro-image, thinking always on); or try `thinking_budget=8192` on gemini-nano-banana-2.1 and drop it if the call fails |

A preset fixes the tier: `balanced` is always 1K and `quality` always 2K, whatever
`image_size` says. To choose a tier yourself, pass `model` and `image_size`.

## Several images

- Same prompt, up to 4 variations: `generate_image(..., number_of_images=4)`. Files get
  `_1` to `_4` suffixes and run in parallel.
- Different prompts, up to 10: `batch_generate(prompts=[...])`. One `model` or `quality` and
  one `aspect_ratio` for the whole batch; no `image_size`, `reference_images` or
  `output_filename`; `quality` overrides `model`. Outputs are `batch_<n>_...` files.
- A set that must look like one family (same character, mascot or palette): call
  `generate_image` sequentially with `reference_images`, never in parallel. Parallel
  calls drift in identity.

## Gotchas

- No transparency, ever. Output is opaque even as PNG; a prompt asking for a transparent
  background yields white, a flat colour or a drawn checkerboard. For a cutout, generate
  on a solid key colour the subject does not contain and key it outside the server.
- Text in images needs `gemini-3-pro-image`, the exact words in double quotes inside the
  prompt, a font description, and an `extract_text` round-trip afterwards. Draft the copy
  before the prompt; fix one wrong label per `edit_image` pass.
- Panoramic ratios are honoured only by `gemini-nano-banana-2.1` and
  `gemini-3.1-flash-image`. Other models, including both the `fast` and `quality`
  presets, render a standard shape with no error.
- `image_size` limits are enforced when `model` is set: `0.5K` only on `gemini-3.1-flash-image`;
  `gemini-3.1-flash-lite-image` is `1K` only, so `quality="fast"` with `image_size="2K"`
  is rejected; `2K` and `4K` need gemini-nano-banana-2.1, gemini-3.1-flash-image or
  gemini-3-pro-image.
- `gemini-3.1-flash-lite-image` is also not optimized for reference images or multi-turn
  edits; switch model when either matters.
- `generate_image` and `batch_generate` have no grounding; `use_grounding` exists only on
  the analysis tools. Put the facts you need in the prompt instead of asking the model to
  look them up.
- `thinking_budget` is sent to the API as a token budget (0-24576), not a level, and the
  tokens are billed. gemini-nano-banana-2.1 documents thinking as levels, so the API may
  reject the budget there; if the call fails, remove the parameter or switch to
  `gemini-3-pro-image`, where thinking is always on. Leave it unset for ordinary images.
- `reference_images` takes up to 14 paths on gemini-nano-banana-2.1 and
  gemini-3.1-flash-image and 6 on gemini-3-pro-image; each one is billed as input.
- "No image returned (finish_reason=...)" means the model answered with text or a filter
  stopped it. Start the prompt with "Create an image of" and remove ambiguous phrasing.
  "Content blocked by safety filter" on a harmless prompt: reword, then retry with
  `safety_setting="preset:relaxed"`.
- `person_generation` is accepted but not enforced; the API applies its own rules on
  identifiable people.
- Every call is billed, including the ones you discard. Iterate at 1K, then re-render the
  final once. `gemini-2.5-flash-image` is deprecated and shut-down preview IDs are rewritten
  to their replacements with a warning; pick from `list_available_models` instead.
- Every image carries a SynthID watermark; mention it when the user plans to sell or
  license the image as original work.
- Exact dimensions are not selectable (`1K`, `2K`, `4K` are tiers, not pixel sizes). For a
  fixed size, generate at the nearest ratio, then switch to the `pixelforge-web-assets`
  skill for the crop and resize step.

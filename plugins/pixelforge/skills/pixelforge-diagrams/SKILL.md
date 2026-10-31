---
name: pixelforge-diagrams
description: Produces diagrams, infographics, architecture visuals, flowcharts, timelines, posters and slide visuals with the PixelForge image tools, and first decides whether a code-rendered Mermaid or SVG diagram is the better answer. Use whenever the user says things like "make a diagram", "draw the architecture", "architecture diagram for the README", "create an infographic", "visualize this system", "flowchart of the pipeline", "slide visual", "hero image for this deck" or "turn this paper into a poster", and for any generated image that must carry exact labels, titles or numbers. Drafts the copy first, renders with a text-accurate model, OCR-checks every label and fixes drift one label per pass.
allowed-tools: mcp__plugin_pixelforge_pixelforge__generate_image mcp__plugin_pixelforge_pixelforge__edit_image mcp__plugin_pixelforge_pixelforge__extract_text mcp__plugin_pixelforge_pixelforge__analyze_image mcp__plugin_pixelforge_pixelforge__compare_images mcp__plugin_pixelforge_pixelforge__estimate_cost mcp__plugin_pixelforge_pixelforge__list_available_models
---

# PixelForge diagrams and infographics

Make a diagram, infographic, architecture visual, poster or slide visual that carries exact text, using the PixelForge MCP server. The image model paints letters rather than typesetting them, so every label is something to verify, and plain diagrams are usually better rendered in code.

## 1. Decide how to render

Pick the first row that fits. Only the last two rows call PixelForge.

| The user needs | Render with |
|---|---|
| Boxes and arrows, flowchart, sequence, ER, state or org chart, dependency graph, anything that will be edited or diffed later, or more than about 7 labels | Mermaid, Graphviz or hand-written SVG in the repo. No image call. Cheaper, exact text, version-controlled. |
| An illustrated or stylized piece where the look is the point (isometric architecture, infographic with icons and scenes, poster, slide hero, textured or 3D rendering) and the text is short | The image model, workflow below. |
| Dense copy (paragraphs, tables, many numbers) on an illustrated background | Hybrid. Generate the illustration with negative space and no text, then overlay the text in SVG, HTML or Pillow. Read `references/prompt-patterns.md` for the negative-space recipe. |

When in doubt, offer Mermaid first and the image model as the upgrade.

## 2. Workflow for an image-model diagram

1. **Draft the copy first.** You are the text model. Write the title, every node label, every arrow label, every number with its unit, and the legend entries as a plain list. This list is the ground truth for step 7. Keep it to about 7 labels of 1 to 3 words; explanations go in the caption or slide notes, not in the image.
2. **Check it fits.** `prompt` is capped at 2000 characters on `generate_image` and `edit_image`. If the quoted labels plus the layout description will not fit, split into panels (one image per panel) or switch to hybrid rendering.
3. **Pick model, ratio and size.** Defaults:
   - Text-carrying diagram or slide: `quality="quality"`, which is `gemini-3-pro-image` at `2K`, the strongest text rendering. Ratios: slides `16:9`, portrait poster `9:16` or `3:4`, square infographic `1:1`, wide system map `21:9`.
   - Cheaper draft, or a crowded layout that needs reasoning: `model="gemini-nano-banana-2.1"`, `image_size="2K"`, `thinking_budget=8192` (range 0 to 24576; it is sent only when above 0). Do not add a `quality` preset to this call, see Gotchas.
   - Banner timeline or storyboard strip at `4:1` or `8:1`: `model="gemini-nano-banana-2.1"`. Pro does not honour panoramic ratios.
   - Never `quality="fast"` for text work. It is `gemini-3.1-flash-lite-image`, 1K only, with the weakest text rendering.
4. **Gate the cost** when the job is more than a couple of images (a deck, a panel series). Call `estimate_cost(operation="generate", model=..., number_of_images=N)`. It prices the 1K tier only; for 2K or 4K read the `pricing` block from `list_available_models` and multiply yourself. Tell the user the figure before generating.
5. **Write the prompt.** Order: image type and purpose; the layout spelled out step by step (left to right, top to bottom); each node with its label in double quotes; each arrow with its direction and quoted label; typography ("large, bold, sans-serif labels with high contrast"); the palette by hue name; and a semantic negative ("a clean layout with no other text, no watermark, no decorative icons"). Numbers go in as digits with units, quoted. Do not run `optimize_prompt` on this prompt. Per-type templates: read `references/prompt-patterns.md`.
6. **Generate.** `generate_image(prompt=..., quality="quality", aspect_ratio="16:9")`. The result's `images[0].path` is the file to check. Use `number_of_images=2` when the layout is uncertain and keep the better one; it costs twice.
7. **OCR round-trip.** Call `extract_text(image_path=<path>, model="gemini-3.8-flash")`. Normalise case and whitespace, then diff against the step 1 list: every label must appear exactly (spelling, digits, units), proper nouns and brand names also match in case when you look at the image, and nothing else may appear (invented words, a fake signature, gibberish in small type). The result carries `text` and `blocks`; each block has `text` and a `confidence` of high, medium or low, and a `low` block means look at the image yourself. `blocks` is empty when the server could not parse the model's JSON, so diff against `text` then.
8. **Fix or regenerate.** One or two wrong labels: `edit_image(input_image_path=<path>, model="gemini-3-pro-image", prompt='Change the label "ACME" to "Acme AI". Keep everything else exactly the same, preserving the layout, colors, and all other text.')`. One label per pass, chain the previous output as the next `input_image_path`, then repeat step 7. Three or more wrong labels, or anything still wrong after three passes: regenerate from a corrected prompt with fewer or larger labels, or go hybrid. Every edit re-renders the whole image.
9. **Confirm nothing else moved.** `compare_images(image_paths=[v1, v2], prompt="List every difference other than the corrected label")`. For a final review run `analyze_image(image_path=<path>, prompt=...)` with the checklist in `references/prompt-patterns.md` (arrow directions, overlaps, legibility at thumbnail size).

## 3. Legibility and palette

- Few words, large type. Ask for labels that stay readable when the image is shown at a quarter of its size. A label that needs more than three words belongs in the caption.
- Name the hues and keep them colourblind-safe: blue, orange, sky blue, bluish green, vermilion, yellow and grey (the Okabe-Ito set; hex values in the reference). Never separate two states by red versus green alone; add a shape or label difference as well. Hex codes in a prompt are a hint, not a guarantee.
- Keep a margin. For slides ask for "about 15 percent empty margin on all sides"; for hybrid work describe the empty region where the text will go.
- Output is always a painted background, never an alpha channel. For a transparent overlay generate on a solid key colour and key it out in code, or use SVG.

## Gotchas

- `quality="balanced"` pins `image_size` to `1K`; the preset wins over the value you pass, silently. For `gemini-nano-banana-2.1` at 2K pass `model` and `image_size` with no `quality`. Passing `quality` and `model` together is rejected.
- `generate_image` accepts `1:4`, `4:1`, `1:8` and `8:1` for any model, but only `gemini-nano-banana-2.1` and `gemini-3.1-flash-image` honour them; `gemini-3-pro-image` renders a standard shape with no error. `21:9` is not panoramic and works on every model.
- `optimize_prompt` rewrites to under 500 characters and adds lighting and texture language, so it drops quoted labels. Skip it for anything that carries text.
- `list_templates(category="architecture")` is building photography (`exterior_photo`, `interior_design`), not software architecture. There is no diagram template; write the prompt yourself.
- `edit_image` has no `reference_images`, `aspect_ratio` or `image_size`; the output follows the source and each call is independent. Pass the previous output path forward.
- `batch_generate` has no `image_size` and no `reference_images`; `quality="quality"` is the only way to get 2K from it. For a deck that must look consistent, loop `generate_image` with `reference_images=[first_slide_path]` instead of batching.
- Leave `thinking_budget` unset on `gemini-3-pro-image`; its thinking is always on. Raise it on `gemini-nano-banana-2.1` for crowded layouts.
- Unspecified numbers get invented. If a figure must be exact, quote it; if no figure should appear, say so in the semantic negative.
- Prices and model IDs change. Take them from `list_available_models` and `estimate_cost` at run time, never from memory.

## References

- Read `references/prompt-patterns.md` when writing the prompt for a specific diagram type (architecture, flowchart, timeline, comparison, infographic, slide hero), for the hybrid negative-space recipe, the palette hex values, the OCR diff rules and the review checklist.

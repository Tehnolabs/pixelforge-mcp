---
name: pixelforge-analyze
description: Understands image files with the PixelForge MCP server. Describes a picture or screenshot, writes alt text, runs OCR with per-block confidence, finds objects with bounding boxes, and compares two to ten images for QA. Use it whenever the user asks what is in an image, to describe or summarise a screenshot, to write or audit alt text for accessibility, to extract or OCR text, to find where something sits in a picture (logo, face, product, button), to check whether a screenshot matches a mockup, to list what changed between two versions, or to verify that a generated image matches its brief including the spelling of rendered text. Reach for it after any pixelforge generation or edit to confirm the result before reporting it.
allowed-tools: mcp__plugin_pixelforge_pixelforge__analyze_image mcp__plugin_pixelforge_pixelforge__extract_text mcp__plugin_pixelforge_pixelforge__detect_objects mcp__plugin_pixelforge_pixelforge__compare_images mcp__plugin_pixelforge_pixelforge__estimate_cost mcp__plugin_pixelforge_pixelforge__list_available_models mcp__plugin_pixelforge_pixelforge__transform_image
---

# PixelForge image analysis

Four read-only tools on the PixelForge MCP server turn an image file into text: `analyze_image` (description or any question), `extract_text` (OCR with per-block confidence), `detect_objects` (labels with bounding boxes) and `compare_images` (2 to 10 images in one call). None of them returns pixels, so the answer is only as good as the question. Ask narrow questions and demand a fixed output shape.

## Pick the tool

| Need | Call | Read in the result |
|---|---|---|
| Describe, classify, answer a question, verify against a brief | `analyze_image` with a targeted `prompt` | `analysis` (free text) |
| Read text, check the spelling of rendered text | `extract_text` | `text`, `blocks[].text`, `blocks[].confidence` (`high`, `medium`, `low`) |
| Locate, count, or crop to a subject | `detect_objects` with an `objects` phrase | `detections[].label`, `detections[].box_2d`, `detections[].confidence`, `count` |
| Screenshot vs mockup, before vs after, drift across variants | `compare_images` with a `prompt` that names each image by position | `comparison` (free text) |

If the client shows you images natively, look at the file yourself for a one-off impression. Use these tools when you need structured output (confidence, boxes), a repeatable pass over many files, a second opinion on your own reading, or when the client cannot display the image at all.

## Procedure

1. Check the input. Every tool takes a local file path that exists and ends in `.png`, `.jpg`, `.jpeg` or `.webp`. Convert anything else (SVG, GIF, PDF pages, BMP, TIFF) to PNG first; URLs and base64 are rejected.
2. Choose the model. Leave `model` unset (`gemini-3.5-flash-lite`) for descriptions, alt text and clean screenshots. Pass `model="gemini-3.8-flash"` for small or stylised text, dense UI, counting, precise boxes, reasoning across several images, or as the retry after a `low` confidence or empty result. Those two IDs are the only valid values; the server does not validate the field, so a typo or an image-model ID fails at the API.
3. Write the prompt as a checklist with a fixed answer shape, under 2,000 characters. The default `analyze_image` prompt is a general description and is rarely what you want. Example for checking a generated image against its brief:
   `Answer in JSON with keys subject, text_visible, background, problems. subject = is there exactly one red bicycle, yes or no, with a reason. text_visible = quote every word you can read, in reading order. background = one sentence. problems = list anything that contradicts a clean product shot.`
4. Leave `use_grounding` off. Turn it on only when the answer depends on facts outside the pixels (identify a landmark, species, product or logo); it adds a Google Search call, latency and cost, and it does not make the model see small details any better.
5. Call the tool and parse the named field. On `success: false`, read `message`; it carries Google's own text for quota (429), permission (403), unknown model and safety blocks (`Content blocked by safety filter`). Retrying a safety block or a quota `limit: 0` does not help.
6. Escalate once, not in a loop: rerun with `gemini-3.8-flash`, a narrower `objects` phrase, or a sharper prompt. If two passes disagree, report both instead of picking one.

Cost gate: before a pass over more than a dozen files, call `estimate_cost(operation="analyze")` (optionally with `model`) for a per-call approximation, or read `analysis_models[].pricing_per_1m_tokens` from `list_available_models`. Never hardcode prices.

## Bounding boxes to pixels

`detect_objects` returns `box_2d` as `[y_min, x_min, y_max, x_max]` on a 0-1000 scale, y first, independent of the image size. To use a box:

1. Read the real width `W` and height `H` of the file (`file image.png` or Pillow); no PixelForge tool reports them.
2. Convert and round to integers: `x = x_min * W / 1000`, `y = y_min * H / 1000`, `width = (x_max - x_min) * W / 1000`, `height = (y_max - y_min) * H / 1000`.
3. Crop with `transform_image(image_path=..., operation="crop", x=..., y=..., width=..., height=...)`, padding each side by a few percent because boxes are approximate.

`count: 0` with an empty `detections` list also happens when the model answered in prose instead of JSON, so treat it as "unknown", not "absent": retry once with `gemini-3.8-flash` and a concrete `objects` phrase such as `"the company logo"`.

## Recipes

**Verify a generated image against its brief.** Call `analyze_image` with the brief turned into yes/no questions plus "quote every word you can read". For any image that carries text, also run `extract_text` and diff `text` against the intended copy word by word; a plausible-looking render often has a misspelt, missing or extra word. Hand the misspelt labels to the pixelforge-edit skill (one fix per pass), then re-run `extract_text` here until the diff is empty; this skill never calls `edit_image` itself.

**OCR a screenshot or document.** Call `extract_text` first. Empty `blocks` with non-empty `text` means the reply was not parsed as JSON and `text` is the raw model output; empty `text` and empty `blocks` means the model found nothing. Re-run with `gemini-3.8-flash` in either case before concluding. Treat every `low` block as unverified; confidence is the model's self-report, not a calibrated score.

**Screenshot vs mockup QA.** `compare_images(image_paths=[screenshot, mockup], prompt="Image 1 is the rendered page, image 2 is the design. List every difference in layout, spacing, alignment, colour, typography and copy as a numbered list, most visible first. Ignore anti-aliasing and compression noise.")`. Images are sent in list order, so always say which is which. Run one call per viewport width rather than packing many screenshots into one.

**Variant drift.** For a character, mascot or product rendered several times, pass the master reference first and up to nine variants, and ask for a per-image table of what deviates (face, colours, proportions, logo). Ten images is the hard cap per call; split larger sets and keep the master in every call.

**Alt text and accessibility audit.** Read `references/alt-text.md` when the user asks for alt text or an audit over a folder or site. It carries the decision tree, the prompt, the length rule and the CSV layout for batch runs.

## Gotchas

- Box order is `[y_min, x_min, y_max, x_max]`, not x first; swapping the axes silently crops the wrong region.
- `detect_objects` wraps a single returned object into a one-element list; `count` is the list length and nothing more.
- The `prompt` on `analyze_image` and `compare_images` is capped at 2,000 characters; `objects` is free text with no cap but works best as a short noun phrase.
- The tools never see each other's results. Paste `extract_text` output into the next `analyze_image` prompt yourself when both matter.
- `analysis` and `comparison` come back as one string; ask for JSON or a numbered list whenever you intend to parse the answer.
- `extract_text` is the dependable spelling check for generated text; `analyze_image` will happily call a misspelt headline "readable".
- Grounding changes what the model knows, not what it sees. For hard pixels, change the model, not the grounding flag.
- All four tools read the file at call time, so run them after the generation or edit has finished writing, and pass the path the generation result reported.

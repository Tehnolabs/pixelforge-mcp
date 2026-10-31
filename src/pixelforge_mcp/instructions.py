"""Server instructions sent to every MCP client at initialization.

Clients such as Claude Code load this text at session start, before any
tool description, so it carries the routing rules that keep an agent from
wasting calls. Keep it under 4,096 characters: Claude Code truncates longer
instructions. Model IDs and enum values here are checked against
``validation`` by the unit tests.
"""

SERVER_INSTRUCTIONS = """\
PixelForge generates, edits, analyzes and transforms images with Google \
Gemini image models. Tools write files to the output directory and return \
their paths; work with the saved paths rather than inline image data.

Tool selection:
- New image from text: generate_image. quality="fast" for drafts \
(gemini-3.1-flash-lite-image, 1K only), "balanced" for most work \
(gemini-nano-banana-2.1, the default), "quality" for text-heavy or brand \
assets (gemini-3-pro-image at 2K). Set model only when no preset fits; \
quality and model are mutually exclusive.
- Change an existing image: transform_image for exact geometry (crop, \
resize, rotate, flip, blur, sharpen, grayscale, watermark); edit_image for \
prompt-driven changes; advanced_edit for inpaint_removal, inpaint_insertion, \
outpaint, background_swap, style_transfer and product_image (optional mask, \
white = region to change, interpreted semantically); remove_background \
returns a flat background, never transparency; upscale_image re-renders at \
the next 1K/2K/4K tier (generative, not pixel-exact; use transform_image \
resize for exact scaling).
- Understand an image: analyze_image (custom prompt, optional grounding), \
extract_text (OCR with confidence), detect_objects (bounding boxes), \
compare_images (2-10 images). Analysis defaults to gemini-3.5-flash-lite; \
pass model="gemini-3.8-flash" for hard OCR or reasoning.
- Plan: optimize_prompt before generating, list_templates and \
apply_template for curated prompts, estimate_cost before batches, \
list_available_models for live capabilities and prices, get_server_info for \
the active backend and setup hints.

Rules that avoid wasted calls:
- Aspect ratios: 1:1, 3:2, 2:3, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9; the \
panoramic 1:4, 4:1, 1:8, 8:1 need gemini-nano-banana-2.1 or \
gemini-3.1-flash-image. image_size: 1K (default), 2K, 4K; 0.5K only on \
gemini-3.1-flash-image; gemini-3.1-flash-lite-image is 1K only. Unsupported \
combinations are rejected before any API call.
- Exact pixel sizes (1200x630 cards, 512 px icons): generate at the nearest \
ratio, then transform_image crop and resize.
- Text inside images: write the exact copy first, quote it in the prompt, \
prefer gemini-3-pro-image, verify with extract_text.
- Consistent characters: pass the same reference_images on every call and \
generate sequentially; batch_generate shares one model and ratio and does \
not preserve identity.
- Edits: one change per pass, state what must stay the same, verify with \
compare_images.
- Cost: every call is billed, including discarded results. Check \
estimate_cost (1K prices) or list_available_models (per-tier prices) before \
generating more than a few images.

Credentials: an API key (GOOGLE_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY or \
GEMINI_API_KEY) selects the Gemini API; GOOGLE_CLOUD_PROJECT with \
Application Default Credentials selects Vertex AI; the key wins when both \
are set unless GOOGLE_GENAI_USE_VERTEXAI=true. On "No credentials found", \
a 403 saying the API is disabled, a 429 with "limit: 0", or a 404 \
"Publisher model ... not found", call get_server_info and report its setup \
hint instead of retrying.
"""

MAX_INSTRUCTIONS_CHARS = 4096

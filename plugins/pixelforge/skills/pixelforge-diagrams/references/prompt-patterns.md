# Prompt patterns for diagrams, infographics and slide visuals

Companion to `SKILL.md`. Every pattern assumes the copy was drafted first and fits inside the 2000-character `prompt` limit. Replace the bracketed parts; keep the quotation marks around every label.

## Prompt skeleton

Write the prompt in this order. Each line maps to one thing the model gets wrong when it is missing.

1. Image type and purpose. `A clean technical infographic for a README that explains [system].`
2. Layout, step by step, in reading order. `From left to right, three rounded boxes connected by arrows.`
3. Every node with its quoted label. `The first box is labeled "Client". The second box is labeled "API Gateway". The third box is labeled "Postgres".`
4. Every arrow with direction and quoted label. `An arrow from "Client" to "API Gateway" labeled "HTTPS". An arrow from "API Gateway" to "Postgres" labeled "SQL".`
5. Typography. `All labels in a large, bold, sans-serif typeface with high contrast against their background.`
6. Palette by hue name. `Boxes in blue and orange on a white background, arrows in dark grey.`
7. Semantic negative. `A clean layout with no other text, no watermark, no logos, no decorative icons.`

## Google's own templates

Copied from the Gemini API image-generation guide at https://ai.google.dev/gemini-api/docs/image-generation (page text is CC BY 4.0).

Accurate text in images:

> Create a [image type] for [brand/concept] with the text "[text to render]" in a [font style]. The design should be [style description], with a [color scheme].

Minimalist and negative space (for the hybrid recipe, where text is overlaid in code):

> A minimalist composition featuring a single [subject] positioned in the [bottom-right/top-left/etc.] of the frame. The background is a vast, empty [color] canvas, creating significant negative space. Soft, subtle lighting. [Aspect ratio].

Google's own rule for text: first generate the text, then ask for an image with that text, with the words enclosed in quotes and the font described.

## Templates by diagram type

**Isometric architecture illustration** (`quality="quality"`, `aspect_ratio="16:9"` or `"21:9"`)

> An isometric 3D illustration of a cloud architecture on a white background. In the top left, a laptop labeled "Client". In the center, a server rack labeled "API". In the bottom right, a database cylinder labeled "Postgres". Thin dark-grey arrows connect "Client" to "API" and "API" to "Postgres". Flat shading, blue and orange accents, large bold sans-serif labels. No other text, no logos, no watermark.

**Process or pipeline flow** (`quality="quality"`, `aspect_ratio="16:9"`)

> A horizontal flow diagram with four rounded rectangles connected by right-pointing arrows, left to right: "Ingest", "Validate", "Transform", "Publish". Each box has a simple line icon above its label. Boxes in sky blue, arrows in dark grey, white background, large bold sans-serif labels. No other text.

**Timeline strip** (`model="gemini-nano-banana-2.1"`, `aspect_ratio="4:1"`; Pro cannot do this ratio)

> A horizontal timeline banner on a cream background. A single dark line runs left to right with five evenly spaced dots. Above each dot, one label in order: "Kickoff", "Alpha", "Beta", "Launch", "Scale". Below each dot, one short caption in order: "Week 0", "Week 6", "Week 12", "Week 20", "Week 40". Large bold sans-serif text in dark blue, orange dots. No other text.

**Comparison or two-column infographic** (`quality="quality"`, `aspect_ratio="4:3"` or `"1:1"`)

> A two-column comparison infographic. The left column is headed "Before" and the right column is headed "After". Under each heading, three short rows with one icon and one quoted label each, left column: "Manual deploys", "45 min builds", "Weekly releases"; right column: "One-click deploys", "6 min builds", "Daily releases". Left column in grey, right column in bluish green, white background, large bold sans-serif text. No other text.

**Stat callout infographic** (`quality="quality"`, `aspect_ratio="1:1"` or `"9:16"`)

> A poster-style infographic with a title at the top reading "Q3 at a glance" and three large stat cards below it. Card one shows the number "99.9%" with the caption "Uptime". Card two shows "4.2M" with the caption "Requests per day". Card three shows "120 ms" with the caption "p95 latency". Numbers in very large bold sans-serif type, captions smaller, cards in blue, orange and vermilion on a white background. No other text, no decorative charts.

**Slide hero, hybrid** (`quality="quality"`, `aspect_ratio="16:9"`, no text in the image)

> A minimalist composition featuring a single abstract network of glowing nodes positioned in the right third of the frame. The left two thirds are an empty, dark navy canvas creating significant negative space for a headline. Soft, subtle lighting. 16:9.

Then overlay the title and bullet text in the slide tool, SVG or HTML. The image never has to pass OCR.

## Hybrid overlay recipe

1. Generate the illustration with the negative-space template above. State which side stays empty and roughly how much ("the left two thirds").
2. Check with `analyze_image(image_path=<path>, prompt="Is the left two thirds of this image free of objects and detail, suitable for overlaying white text? Answer yes or no and explain.")`.
3. Place the text in code. SVG or HTML for the web, Pillow `ImageDraw.text` for a flat PNG, the deck tool for slides. Use a real font and the exact copy from the draft.
4. If the illustration bleeds into the text zone, run `edit_image` with "Remove all objects from the left two thirds of the image and leave that area a plain dark navy. Keep everything else exactly the same."

## Colourblind-safe palette (Okabe-Ito)

Name the hue in the prompt and treat the hex as a hint the model will approximate.

| Hue name to write | Hex |
|---|---|
| orange | #E69F00 |
| sky blue | #56B4E9 |
| bluish green | #009E73 |
| yellow | #F0E442 |
| blue | #0072B2 |
| vermilion | #D55E00 |
| reddish purple | #CC79A7 |
| black or dark grey | #000000 |

Pick at most four hues per image. Never encode a status pair as red versus green only; pair vermilion with blue, and add a shape or label difference too.

## OCR diff rules

Run `extract_text(image_path=<path>, model="gemini-3.8-flash")` and compare against the drafted copy.

- Normalise both sides: lowercase, collapse whitespace, strip trailing punctuation.
- A label is a miss when any character differs, including digits, units, decimal separators and currency symbols. Once the normalised diff passes, check case by eye on proper nouns and brand names, since lowercasing hides a "ACME" rendered for "Acme".
- Extra text is a miss too: words that are not in the draft, a signature, a caption, or illegible strings in small type.
- A block with `confidence` of `low` is not a verdict either way; open the image and check that label yourself.
- If `blocks` is empty, the server could not parse the model's JSON; diff against the `text` field instead.
- Record the misses as pairs of wrong and intended text; each pair becomes one `edit_image` pass.

Edit prompt per miss, on `model="gemini-3-pro-image"`:

> Change the label "[wrong text]" to "[intended text]". Keep everything else exactly the same, preserving the layout, colors, icons, and all other text.

Final review prompt for `analyze_image`:

> Review this diagram as a technical editor. List any arrow that points the wrong way or connects the wrong boxes, any labels that overlap or are cut off, any text that would be unreadable at a quarter of this size, any element that relies on red versus green alone, and any text or object that looks unintended. Answer with a numbered list or "No issues".

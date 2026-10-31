# Alt text and accessibility audits with PixelForge

Read this when the user asks for alt text for one image, for a page, or for every image in a folder or repository. The PixelForge tools supply the description; the role of each image comes from the code that uses it, so inspect the markup first.

## 1. Decide the role before describing

| Role | How to recognise it | What to write |
|---|---|---|
| Decorative | CSS background, spacer, divider, purely aesthetic illustration next to text that already says the same thing | `alt=""` (empty, present). Do not call the tool. |
| Informative | Photo, illustration or icon that adds meaning the text does not carry | One sentence, under 125 characters, leading with the subject |
| Functional | Inside a link or button, or acting as one (logo that links home, icon-only button) | The action or destination (`Search`, `Home`), never the picture |
| Image of text | Banner, badge, screenshot whose point is the words | The words verbatim; take them from `extract_text`, not from memory |
| Complex | Chart, diagram, map, infographic, screenshot of a UI state | Short alt naming the type and topic, plus a longer description in the page (figcaption, adjacent text or a details element) |
| Group | Gallery or a series meant to be read together | One alt on the first image that covers the set, `alt=""` on the rest, or one alt per image if each matters |

## 2. Writing rules

- Lead with the subject; drop "image of", "picture of", "photo of".
- Keep informative alt under about 125 characters, the ceiling most accessibility guides use; longer content belongs in a long description.
- Use the language of the page, not the language of the text in the picture.
- Take product names, prices, sizes and other attributes from the data the page already has, not from the photo. Alt text that invents a product name is worse than a generic one.
- Mark anything the image cannot prove with a `[CHECK]` placeholder for the human reviewer (a person's name, a location, a brand).
- Do not repeat a caption that sits next to the image.
- Match the page's purpose: the same photo of a bicycle is "Red commuter bicycle leaning against a brick wall" on a lifestyle blog and "Model name [CHECK] in red, side view" on a product page.

## 3. Prompt for one informative image

Call `analyze_image` with a prompt in this shape, filling in the page context you found in the code:

```
Write alt text for this image as used on a product listing page for a bicycle shop.
Rules: one sentence, maximum 125 characters, no leading "image of", lead with the main subject,
mention colour and setting only if they matter on a shop page, do not guess brand or model names.
Then on a second line answer decorative or informative with one reason.
```

For an image of text, call `extract_text` instead and use `text` verbatim; re-run with `model="gemini-3.8-flash"` when any block is `low`.

For a complex image, call `analyze_image` with `model="gemini-3.8-flash"` and ask for two parts: a one-line alt and a plain-language long description of what the chart or diagram shows, including the numbers that matter.

## 4. Folder or site audit

1. Enumerate images from the markup and templates, not from the filesystem alone, and record where each is used (`<img>`, CSS `background-image`, `<svg>` inline, `<picture>` sources).
2. Assign a role from the table in section 1 using the code context. Decorative images get `alt=""` without a tool call.
3. Before calling, gate the spend: `estimate_cost(operation="analyze", number_of_images=N)` and confirm with the user when N is large. `number_of_images` accepts 1 to 100; for a larger audit call `estimate_cost(operation="analyze", number_of_images=1)` and multiply `unit_cost_usd` by the image count yourself.
4. For each remaining image, convert to PNG, JPEG or WebP if needed, then call `analyze_image` (or `extract_text` for images of text). Run the calls one after another or in small parallel groups; a burst of parallel calls on a fresh project trips rate limits.
5. Write one CSV row per image with these columns: `path`, `used_in`, `role`, `current_alt`, `proposed_alt`, `chars`, `needs_check`, `notes`. Set `needs_check` to `yes` whenever the proposed alt contains `[CHECK]`, exceeds 125 characters, or the tool returned `success: false`.
6. Hand the CSV to a human before committing any alt attribute; the model's description is a draft, not a decision.

## 5. Quick checks on the result

- An alt that only lists colours and shapes usually means the prompt lacked page context; rerun with the context sentence.
- Two different images with identical alt text are a flag; make the alt say what distinguishes them or mark one decorative.
- Alt text longer than 125 characters belongs in a long description, not in the attribute.

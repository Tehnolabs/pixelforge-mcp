# Writing prompts for Gemini image models

Read this when writing a `generate_image` prompt from scratch, or when a template or an
`optimize_prompt` result needs tightening. The templates in section 3 and 4 are adapted
from Google's Gemini API image generation documentation (text licensed CC BY 4.0).

## Contents

1. Structure
2. Vocabulary
3. Generation templates
4. Follow-up edit phrasing
5. Worked example

## 1. Structure

Describe the scene as a short narrative, not a keyword list. Cover, in this order:

1. Subject: who or what, with specific attributes ("ornate elven plate armor, etched with
   silver leaf patterns, with a high collar", not "fantasy armor").
2. Action: what the subject is doing.
3. Setting: location, time of day, weather, era.
4. Composition: shot type, camera angle, lens, where the subject sits in the frame, where
   empty space must stay.
5. Style: medium, art movement, film stock, colour grading, mood.

Rules that change the output:

- Start with a strong verb that names the operation: "Create an image of", "A
  photorealistic photograph of", "Make a sticker of". A prompt that reads like a question
  can come back as text with no image.
- State context and intent: "a logo for a high-end, minimalist skincare brand" beats "a
  logo".
- For complex scenes, give steps: "First, create a background of ... Then, in the
  foreground, place ... Finally, add ...".
- Use semantic negatives. There is no negative-prompt parameter; "an empty, deserted
  street with no signs of traffic" works, "no cars" does not.
- Put every word that must appear in the image in double quotes and describe its font;
  keep it short. Draft the copy before the prompt.
- Name colours as hex codes when the palette matters ("#1E3A8A navy").
- Ask for one image per call. "Three versions" inside the prompt is unreliable; use
  `number_of_images` or `batch_generate` instead.
- Write the prompt in one language and name the target language separately when the
  rendered text must be in another one. Google lists EN, ar-EG, de-DE, es-MX, fr-FR,
  hi-IN, id-ID, it-IT, ja-JP, ko-KR, pt-BR, ru-RU, ua-UA, vi-VN and zh-CN as best
  supported.
- Keep the whole prompt under 2000 characters; the server rejects longer ones.

## 2. Vocabulary

| Dimension | Words the model responds to |
|---|---|
| Shot | wide-angle shot, close-up, macro shot, full-body shot, overhead flat lay, low-angle perspective, eye-level, bird's-eye view |
| Lens and optics | 85mm portrait lens, 24mm wide lens, f/1.8 shallow depth of field, deep focus, bokeh, tilt-shift |
| Lighting | three-point softbox setup, golden hour backlighting, overcast soft light, chiaroscuro, neon rim light, key light from the left, dappled sunlight |
| Materiality | brushed aluminium, navy blue tweed, matte ceramic, frosted glass, worn leather, wet asphalt |
| Style | photorealistic, cel-shaded, flat vector, isometric, watercolor wash, 35mm film grain, Kodak Portra colour grade, risograph print, pixel art, kawaii sticker |
| Composition | rule of thirds, centered subject, subject in the bottom-right third, generous negative space on the left for a headline, symmetrical, leading lines |
| Mood | serene, tense, playful, luxurious, austere, nostalgic |

## 3. Generation templates

Photorealistic scene:

> A photorealistic [type of shot] of a [subject description] in a [setting description].
> [Description of the light]. Shot from a [camera angle] with a [lens type].

Stylized illustration or sticker:

> A [style] of a [subject, with details about accessories or actions] doing [activity].
> The design features [visual qualities, e.g., bold outlines, cel-shading, etc.] and
> [color/background preference].

For a sticker add "The background must be white." The output is still opaque; key it
outside the server if a cutout is needed.

Accurate text in an image:

> Create a [image type] for [brand/concept] with the text "[text to render]" in a [font
> style]. The design should be [style description], with a [color scheme].

Product mockup or commercial photography:

> A high-resolution, studio-lit product photograph of a [product description] on a
> [background surface/description]. The lighting is a [lighting setup, e.g., three-point
> softbox setup] to [lighting purpose]. The camera angle is a [angle type] to showcase
> [specific feature]. Ultra-realistic, with sharp focus on [key detail]. [Aspect ratio].

Minimalist with negative space (for a headline overlaid in code):

> A minimalist composition featuring a single [subject] positioned in the
> [bottom-right/top-left/etc.] of the frame. The background is a vast, empty [color]
> canvas, creating significant negative space. Soft, subtle lighting. [Aspect ratio].

Where a template says "[Aspect ratio]", describe the framing in words ("wide 16:9
framing") and also pass `aspect_ratio` to the tool; the parameter is what sets the canvas.

## 4. Follow-up edit phrasing

Use these with `edit_image`, one change per call.

Change one element:

> Using the provided image, change only the [specific element] to [new
> element/description]. Keep everything else in the image exactly the same, preserving the
> original style, lighting, and composition.

Add or remove one element:

> Using the provided image of [subject], please [add/remove/modify] [element] to/from the
> scene. Ensure the change is [description of how the change should integrate].

## 5. Worked example

Brief: README hero for a command-line tool, dark theme, headline added in code later,
16:9, no words in the image.

Prompt:

> Create a minimalist composition featuring a single glowing terminal window, slightly
> tilted in 3D, positioned in the right third of the frame. The window is blank apart from
> a blinking cursor. The background is a vast, empty #0B1020 canvas with a faint grid,
> creating significant negative space on the left for a headline. Soft blue rim light from
> the top right, subtle film grain. Wide 16:9 framing.

Calls: `generate_image(prompt=..., quality="balanced", aspect_ratio="16:9",
output_filename="hero-dark.png")`, then `analyze_image` with the brief as the prompt, then
one `edit_image` pass per fix.

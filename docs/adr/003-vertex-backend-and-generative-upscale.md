# ADR-003: Vertex AI Backend and Generative Upscale/Edit

**Date:** 2026-10-29
**Status:** Accepted
**Deciders:** Project maintainers

## Context

PixelForge 0.6.0 ran `upscale_image` on `imagen-4.0-upscale-preview` and `advanced_edit` on `imagen-3.0-capability-001`, through a Vertex AI client that existed only for those two tools and defaulted to `us-central1`. Both tools returned 404 for every caller.

### Imagen is retired on both backends

| Source | Models | Status | Replacement named by Google |
|--------|--------|--------|-----------------------------|
| Vertex AI "Model versions and lifecycle" page | `imagen-4.0-generate-001`, `imagen-4.0-ultra-generate-001`, `imagen-3.0-generate-001`, `imagen-3.0-generate-002`, `imagegeneration@002` to `@006` | Retired June 30, 2026 | `gemini-2.5-flash-image` |
| Google Cloud notice (March 2026), Vertex edit page, Firebase model table | `imagen-3.0-capability-001` (edit, inpaint, outpaint) | Retired June 30, 2026 | `gemini-2.5-flash-image` |
| Vertex AI | `imagen-4.0-upscale-preview` | Not listed on the lifecycle page; its model page returns 404 | none |
| Gemini API deprecations page (updated 2026-10-22) | `imagen-4.0-generate-001`, `imagen-4.0-fast-generate-001`, `imagen-4.0-ultra-generate-001` | Shut down August 17, 2026 | `gemini-nano-banana-2.1` |

Live checks on 2026-10-29 agree with the pages:

- Every Imagen ID, including `imagen-4.0-upscale-preview` and `imagen-3.0-capability-001`, returns 404 `Publisher Model ... was not found` on Vertex AI in `us-central1`, `global`, and `europe-west4`.
- The Vertex AI publisher catalog visible to the project lists 131 models and no Imagen entry.
- The Gemini API returns "Model is not found" for the same IDs.

### It is not a project or permission limitation

The same credentials, in the same session, reached `gemini-2.5-flash-image` and `text-embedding-005` at `us-central1` and all four Gemini image models at `global`. A project missing an entitlement, a disabled API, or a bad key would fail those calls too. Forum reports from August 2026 describe the same pattern on new projects (Imagen 404, Gemini image generation works) and received no staff answer; the lifecycle pages above explain it.

### Other facts the design depends on

- Location: the Vertex AI model cards list `gemini-nano-banana-2.1`, `gemini-3.1-flash-lite-image`, and `gemini-3-pro-image` under `global` only; `gemini-3.1-flash-image` adds the `us` and `eu` multi-regions. The analysis models `gemini-3.5-flash-lite` and `gemini-3.8-flash` returned 404 at `us-central1` and worked at `global`. The old default of `us-central1` would break every tool on Vertex AI.
- Image size: Vertex AI rejects `"0.5K"` with INVALID_ARGUMENT and accepts `"512px"` on `gemini-3.1-flash-image` (512x512 output). The Gemini docs call the tier "512px (0.5K)". `gemini-3.1-flash-lite-image` is 1K only; `gemini-nano-banana-2.1` and `gemini-3-pro-image` have no 512px tier.
- Quota: a fresh Vertex AI project returned 429 RESOURCE_EXHAUSTED after about six rapid calls to `gemini-nano-banana-2.1`. google-genai does not retry unless `HttpRetryOptions` is configured; with 4 attempts and a 5 s initial delay the burst recovered.
- Response shape: a Vertex AI response can carry `candidates[0].content.parts == None`. The 0.6.0 code iterated `parts` and raised `TypeError`.
- Verified on Vertex AI `global` through `generate_content`: all four image models, temperature, thinking config, safety settings, Google Search grounding, panoramic ratios, reference-image editing, TEXT+IMAGE modalities, and the analysis text models. Everything PixelForge does through the Gemini API works through Vertex AI with the same SDK calls.

## Decision

### 1. One client, two backends

`ImagenAPIClient` builds one `genai.Client`: `genai.Client(vertexai=True, project=..., location=...)` for Vertex AI, or `genai.Client(api_key=...)` for the Gemini API. Every tool uses it. The separate Vertex client and the `google-cloud-aiplatform` dependency are removed; the `vertex` extra stays, empty, so `pip install pixelforge-mcp[vertex]` keeps working.

Backend selection (`Config.backend`):

1. `GOOGLE_GENAI_USE_VERTEXAI=true` selects Vertex AI.
2. Otherwise `GOOGLE_CLOUD_PROJECT` set and no API key selects Vertex AI.
3. Otherwise the Gemini API.

The API key wins when both are configured because it is the common single-user setup and keeps existing installs unchanged. Vertex AI credentials come from Application Default Credentials, so a service account key in `GOOGLE_APPLICATION_CREDENTIALS`, `gcloud auth application-default login`, and attached identities all work without PixelForge-specific code. `get_server_info` reports the active backend.

### 2. Default location `global`

`VertexConfig.location` defaults to `global`; `GOOGLE_CLOUD_LOCATION` overrides it. A `Publisher model ... was not found` error (matched case-insensitively; both casings have been observed) is rewritten: a regional location gets a hint pointing at `GOOGLE_CLOUD_LOCATION=global`, and `global` gets a note that the model ID is unknown or retired.

### 3. Generative upscale

`upscale_image` re-renders the source with a Gemini image model (`gemini-nano-banana-2.1` by default, `model` to override). The target tier is the smallest of 1K/2K/4K whose longest edge covers `source long edge x factor`, capped at 4K with a warning; a source already at or above 4K is an error. The aspect ratio is the nearest ratio the model supports, compared in log space so wide and tall sources are treated alike; models without panoramic output get a standard ratio and a warning. The request is `[source image, UPSCALE_INSTRUCTION]`, where the instruction asks for an exact reproduction with added detail only. Models that output 1K only (`gemini-3.1-flash-lite-image`) are rejected. The response reports `method: "generative"`, `image_size`, `source_size`, `output_size`, and `warnings`.

### 4. Prompt-driven edit modes with mask guidance

`advanced_edit` keeps its six modes. Each mode wraps the user prompt in an instruction template (`EDIT_MODE_INSTRUCTIONS`). An optional mask is sent as a second image together with `MASK_INSTRUCTION` (white marks the region to change). The output size matches the source tier (1K/2K/4K, clamped to what the model supports) and the aspect ratio is pinned to the nearest ratio the model supports (with a warning when that differs from the source) except for `outpaint`. The response reports `model`, `method`, `mask_used`, `image_size`, and `warnings`.

### 5. Retry with backoff

The client sets `HttpRetryOptions(attempts=5, initial_delay=2, max_delay=60, exp_base=2, jitter=1, http_status_codes=[408, 429, 500, 502, 503, 504])` on both backends. This covers the observed 429 bursts and transient 5xx responses without hiding permanent errors. Error messages keep Google's text, so a quota of 0 or a disabled API (a project limit, not a model problem) stays visible. The Gemini API client is created with `vertexai=False` so the SDK's own `GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_GENAI_USE_ENTERPRISE` switches cannot reroute an API key; PixelForge reads both switches itself to select Vertex AI when a project is configured.

### 6. `0.5K` maps to `512px`; sizes are validated per model

Tools keep accepting `0.5K`. The client sends `512px`. `MODEL_IMAGE_SIZES` lists what each model can output; `generate_image` and `upscale_image` reject unsupported combinations before calling the API, and `advanced_edit` clamps to the model's largest supported tier.

### 7. Tolerate responses without parts

`_extract_images` handles missing candidates, content, or parts and reports the finish reason (for example `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`, `NO_IMAGE`) instead of raising.

## Consequences

- `upscale_image` and `advanced_edit` work again, on both backends, and no longer need a Google Cloud project.
- Pixel fidelity: a generative upscale is a re-render. Composition, subjects, and colors are preserved by instruction, but fine detail can differ from the source, and the output has the model's tier dimensions rather than exactly `factor x` the source. Every result carries a warning saying so; `transform_image(operation="resize")` remains the exact-scaling option.
- Masks guide the model semantically. The edited region is not pixel-exact and unmasked areas are not guaranteed untouched. Google names no pixel-exact replacement for Imagen's mask-based editing.
- Pricing: an upscale or edit costs one generated image at the chosen tier, for example $0.0504 at 2K and $0.113 at 4K on `gemini-nano-banana-2.1` (list prices from ADR-002), instead of Imagen's per-operation rates.
- Data residency: the current models are served from `global`, so callers who need regional processing cannot use Vertex AI for them until Google adds regional endpoints; `GOOGLE_CLOUD_LOCATION` is honoured when that happens.
- Fresh Vertex AI projects rate-limit bursts. Retries absorb short bursts; batch callers should expect slower completion rather than failures. Integration tests make one call per tool on the cheapest model.
- CI: the `integration` job authenticates with a `GCP_SERVICE_ACCOUNT_KEY` secret (service account JSON, through `google-github-actions/auth`) or a `GOOGLE_API_KEY` secret, skips when neither is set, runs for pull requests only from the same repository, and masks the project ID, project number, and service account email in logs.

# ADR-002: Migrate to Stable Gemini Image Models and Retire Imagen 4

**Date:** 2026-10-28
**Status:** Accepted
**Deciders:** Ahmed Aleryani

## Context

Google retired most of the models PixelForge 0.5.x shipped with. Per the Gemini API deprecations page (updated 2026-10-22) and a live `models.list()` call on 2026-10-28:

| Model | Status | Replacement |
|-------|--------|-------------|
| `gemini-3-pro-image-preview` | Shut down 2026-06-25 | `gemini-3-pro-image` |
| `gemini-3.1-flash-image-preview` | Shut down 2026-06-25 | `gemini-nano-banana-2.1` |
| `imagen-4.0-generate-001` | Shut down 2026-08-17, returns 404 | `gemini-nano-banana-2.1` |
| `imagen-4.0-fast-generate-001` | Shut down 2026-08-17, returns 404 | `gemini-nano-banana-2.1` |
| `imagen-4.0-ultra-generate-001` | Shut down 2026-08-17, returns 404 | `gemini-nano-banana-2.1` |
| `gemini-2.5-flash-image` | Deprecated, shutdown announced | `gemini-3.1-flash-lite-image` |
| `gemini-2.5-flash` (analysis) | No longer available to new API users | `gemini-3.5-flash-lite` |

Imagen pricing no longer appears on the Gemini API pricing page, and the image generation guide describes Imagen as shut down. The Gemini API now has four stable image models, all served through `generate_content`:

| Model | Nickname | Sizes | Per image (1K / 2K / 4K) | Notes |
|-------|----------|-------|--------------------------|-------|
| `gemini-nano-banana-2.1` | Nano Banana 2.1 | 1K, 2K, 4K | $0.0336 / $0.0504 / $0.113 | Google's recommended default. Web and image search grounding, thinking levels, panoramic ratios |
| `gemini-3.1-flash-lite-image` | Nano Banana 2 Lite | 1K | $0.0336 | Fastest and cheapest. 10 standard ratios, no panoramic, no grounding |
| `gemini-3.1-flash-image` | Nano Banana 2 | 0.5K, 1K, 2K, 4K | $0.067 / $0.101 / $0.151 | Previous generation workhorse. Panoramic ratios, grounding |
| `gemini-3-pro-image` | Nano Banana Pro | 1K, 2K, 4K | $0.134 / $0.134 / $0.24 | Highest quality, best text rendering, up to 6 object and 3 style reference images |

## Decision

### 1. Ship the four stable models; make `gemini-nano-banana-2.1` the default

The default model and the `balanced` quality preset move to `gemini-nano-banana-2.1`. `fast` maps to `gemini-3.1-flash-lite-image` at 1K and `quality` to `gemini-3-pro-image` at 2K (same price as 1K on that model). Panoramic ratios auto-route to `gemini-nano-banana-2.1`.

### 2. Keep `gemini-2.5-flash-image` as a deprecated legacy entry

The model still resolves on the API. It stays in `list_available_models` flagged `deprecated: true` with its replacement so existing configs keep working until Google removes it. It is no longer the default.

### 3. Alias retired preview IDs to their stable replacements

`gemini-3-pro-image-preview` and `gemini-3.1-flash-image-preview` are rewritten to the replacements Google names on the deprecations page. The response carries a `warnings` entry naming the substitution. These are the same model families, so output quality is preserved.

### 4. Reject Imagen 4 IDs with a migration error

Imagen used a separate API (`generate_images`) with different semantics, including API-level `person_generation` enforcement. Silently swapping an Imagen request to a Gemini model is what #14 introduced and #15 reverted. Imagen IDs now return a clear error pointing to `gemini-nano-banana-2.1`, and the Imagen code path and its tests are removed.

### 5. Analysis default moves to `gemini-3.5-flash-lite`

`gemini-2.5-flash` returns 404 for new API users, which broke `analyze_image`, `extract_text`, `detect_objects`, `compare_images`, and `optimize_prompt` for new installs. `gemini-3.5-flash-lite` has the same list price ($0.30 input, $2.50 output per 1M tokens). Every analysis tool accepts a `model` override, documented with `gemini-3.8-flash` as the higher-capability option.

## Consequences

- Configs or prompts referencing Imagen 4 models fail with an actionable error instead of a 404 from Google.
- `person_generation` is accepted but not enforced on any model. The warning text no longer recommends Imagen.
- Vertex AI tools (`upscale_image`, `advanced_edit`) are unchanged. They use the Vertex client, which is not covered by the Gemini API shutdowns.
- Live generation could not be exercised during this change: the development API key is on the free tier, which has an image generation quota of 0. Routing and metadata are covered by unit tests. The first paid-tier call should be `generate_image(prompt="...", quality="balanced")`.

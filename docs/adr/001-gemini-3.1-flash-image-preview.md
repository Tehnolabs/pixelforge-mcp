# ADR-001: Add gemini-3.1-flash-image-preview and Panoramic Aspect Ratios

**Date:** 2026-02-28
**Status:** Accepted
**Deciders:** Ahmed Aleryani

## Context

Google released `gemini-3.1-flash-image-preview` (Nano Banana 2) on Feb 26, 2026. This model combines Pro-level image quality with Flash-tier speed and introduces 4 panoramic aspect ratios (1:4, 4:1, 1:8, 8:1). PixelForge previously supported only 2 models and 10 aspect ratios.

### Key findings from research

- **`gemini-imagen` library (0.6.6):** Passes model names as opaque strings to `google-genai` — no client-side validation or enum. The same approach we used to add `gemini-3-pro-image-preview` (which also wasn't in the library's own docs).
- **No `gemini-3.1-pro-image-preview` exists.** The 3.1 Pro is text-only. The Pro image model remains `gemini-3-pro-image-preview`.
- **Imagen 4** uses a different API surface (`generate_images()` vs `generate_content()`) and is out of scope for this change.
- **`google-generativeai` SDK** is deprecated (EOL Nov 2025). Our dependency chain goes through `gemini-imagen` which uses `google-genai` (latest: 1.65.0).

## Decision

### 1. Add panoramic ratios globally (not model-gated)

Panoramic ratios (1:4, 4:1, 1:8, 8:1) are added to the global `ASPECT_RATIOS` list in `validation.py` rather than being restricted to specific models.

**Rationale:** If an older model receives a panoramic ratio, the Gemini API returns a descriptive error. Our existing exception handling in `api_client.py` already catches this and returns it cleanly to the MCP client. Adding model-specific validation would:
- Couple validation logic to model capabilities that change frequently
- Require maintenance every time Google updates model support
- Duplicate validation the API already performs with better error messages

### 2. Add model as metadata only (no code changes to generation flow)

The new model is added to `list_models()` metadata, docstrings, and server info. No changes to the generation/editing flow itself — `gemini-imagen` passes the model name through transparently.

### 3. Distinct model guidance (no overlapping recommendations)

Each model has a clear, non-overlapping role in the guidance:

| Feature | Flash (2.5) | Flash (3.1) | Pro (3.0) |
|---------|------------|-------------|-----------|
| **Primary role** | Fast iterations | Panoramic, grounded | Max text fidelity |
| **Speed** | Fastest | 4-6s | 8-12s |
| **Text accuracy** | ~80% | ~90% | ~94% |
| **Panoramic ratios** | No | Yes | No |
| **Grounding** | No | Web + Image Search | Web only |
| **512px tier** | No | Yes | No |
| **Multi-turn editing** | Basic | Good | Advanced |

This avoids ambiguous "use either" guidance — each model wins on specific axes.

### 4. Version bump 0.1.5 -> 0.1.6

Patch version bump for additive feature (new model + new ratios, no breaking changes).

## Consequences

### Positive
- Users get access to the new model immediately
- Panoramic ratios enable new creative use cases (banners, scrolling content, landscapes)
- No breaking changes — existing workflows continue unchanged
- Minimal code surface area touched

### Negative
- Panoramic ratios will fail with an API error on older models — but the error message from Google is descriptive
- Model metadata is hardcoded — if Google changes capabilities, we need a manual update

### Risks
- `gemini-3.1-flash-image-preview` is a preview model — it may be renamed or removed by Google
- `gemini-imagen` library is in maintenance mode (last update Nov 2025) — eventual migration to direct `google-genai` SDK may be needed

## Alternatives Considered

### Model-gated aspect ratios
Restrict panoramic ratios to only `gemini-3.1-flash-image-preview` via validation logic. Rejected because it adds maintenance burden and the API already provides clear error messages.

### Dynamic model discovery via API
Query the Gemini API at runtime for available models and capabilities. Rejected because it adds latency, requires API calls on server startup, and the model list changes infrequently.

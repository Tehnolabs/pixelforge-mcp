"""Server instructions stay short and agree with the validation tables."""

import re

from pixelforge_mcp.instructions import MAX_INSTRUCTIONS_CHARS, SERVER_INSTRUCTIONS
from pixelforge_mcp.server import mcp
from pixelforge_mcp.utils.validation import (
    ADVANCED_EDIT_MODES,
    ANALYSIS_MODELS,
    ASPECT_RATIOS,
    IMAGE_MODELS,
    PANORAMIC_MODELS,
    PANORAMIC_RATIOS,
    QUALITY_PRESETS,
    TRANSFORM_OPERATIONS,
)


def test_instructions_fit_the_client_limit() -> None:
    assert 0 < len(SERVER_INSTRUCTIONS) <= MAX_INSTRUCTIONS_CHARS


def test_server_is_created_with_instructions() -> None:
    assert mcp.instructions == SERVER_INSTRUCTIONS


def test_every_mentioned_model_exists() -> None:
    mentioned = {
        m.rstrip(".,;")
        for m in re.findall(r"gemini-(?:\d|nano)[a-z0-9.\-]*", SERVER_INSTRUCTIONS)
    }
    known = set(IMAGE_MODELS) | set(ANALYSIS_MODELS)
    assert mentioned <= known, mentioned - known


def test_enums_are_listed_completely() -> None:
    for value in ADVANCED_EDIT_MODES:
        assert value in SERVER_INSTRUCTIONS
    for value in TRANSFORM_OPERATIONS:
        assert value in SERVER_INSTRUCTIONS
    for value in QUALITY_PRESETS:
        assert f'"{value}"' in SERVER_INSTRUCTIONS
    for ratio in ASPECT_RATIOS:
        assert ratio in SERVER_INSTRUCTIONS
    for ratio in PANORAMIC_RATIOS:
        assert ratio in SERVER_INSTRUCTIONS
    for model in PANORAMIC_MODELS:
        assert model in SERVER_INSTRUCTIONS

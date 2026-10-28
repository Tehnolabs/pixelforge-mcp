"""Unit tests for the prompt template library."""

from pixelforge_mcp.utils.templates import TemplateLibrary
from pixelforge_mcp.utils.validation import ASPECT_RATIOS, IMAGE_MODELS


class TestTemplateRecommendedModels:
    """Template model recommendations must point at supported models."""

    def test_recommended_models_are_supported_image_models(self):
        """Every recommended_model is a listed image model."""
        library = TemplateLibrary()
        recommended = [
            (t["name"], t["recommended_model"])
            for t in library.list_templates()
            if t.get("recommended_model")
        ]

        assert recommended, "expected at least one template with a recommendation"
        for name, model in recommended:
            assert model in IMAGE_MODELS, f"{name} recommends unsupported {model}"

    def test_recommended_models_are_not_deprecated(self):
        """No template recommends the deprecated legacy model."""
        library = TemplateLibrary()
        for template in library.list_templates():
            assert template.get("recommended_model") != "gemini-2.5-flash-image"

    def test_panoramic_templates_recommend_panoramic_models(self):
        """Panoramic ratios need a model that supports them."""
        panoramic_ratios = {"1:4", "4:1", "1:8", "8:1"}
        panoramic_models = {"gemini-nano-banana-2.1", "gemini-3.1-flash-image"}
        library = TemplateLibrary()

        for template in library.list_templates():
            ratio = template.get("recommended_aspect_ratio")
            if ratio in panoramic_ratios:
                model = template.get("recommended_model")
                assert model in panoramic_models, template["name"]

    def test_recommended_aspect_ratios_are_valid(self):
        """Recommended aspect ratios are supported by validation."""
        library = TemplateLibrary()
        for template in library.list_templates():
            ratio = template.get("recommended_aspect_ratio")
            if ratio:
                assert ratio in ASPECT_RATIOS, template["name"]

    def test_apply_template_returns_stable_recommendation(self):
        """apply_template surfaces the updated recommended_model."""
        library = TemplateLibrary()

        result = library.apply_template("product_hero", {"subject": "a watch"})

        assert result is not None
        assert result["recommended_model"] == "gemini-3-pro-image"

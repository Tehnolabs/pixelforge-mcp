"""Prompt template library for image generation."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class TemplateLibrary:
    """Manages curated prompt templates."""

    def __init__(self):
        self._templates: List[Dict[str, Any]] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load templates from YAML on first access."""
        if self._loaded:
            return
        templates_path = Path(__file__).parent.parent / "templates" / "prompts.yaml"
        if templates_path.exists():
            with open(templates_path) as f:
                data = yaml.safe_load(f)
                self._templates = data.get("templates", [])
        self._loaded = True

    def list_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all templates, optionally filtered by category."""
        self._ensure_loaded()
        if category:
            return [t for t in self._templates if t.get("category") == category.lower()]
        return list(self._templates)

    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a template by name."""
        self._ensure_loaded()
        for t in self._templates:
            if t.get("name") == name:
                return t
        return None

    def apply_template(
        self, name: str, variables: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Apply a template with variable substitution."""
        template = self.get_template(name)
        if not template:
            return None

        rendered = template["template"]
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{key}}}", value)

        return {
            "name": template["name"],
            "category": template.get("category"),
            "rendered_prompt": rendered,
            "recommended_model": template.get("recommended_model"),
            "recommended_aspect_ratio": template.get("recommended_aspect_ratio"),
        }

    def list_categories(self) -> List[str]:
        """Get all unique categories."""
        self._ensure_loaded()
        return sorted(set(t.get("category", "uncategorized") for t in self._templates))


# Module-level singleton
_library: Optional[TemplateLibrary] = None


def get_template_library() -> TemplateLibrary:
    """Get the global template library instance."""
    global _library
    if _library is None:
        _library = TemplateLibrary()
    return _library

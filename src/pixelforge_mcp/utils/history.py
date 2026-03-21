"""Generation history tracking."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GenerationHistory:
    """Append-only JSON log of all generation operations."""

    def __init__(self, output_dir: Path):
        self._log_path = output_dir / ".pixelforge_history.json"
        self._entries: Optional[List[Dict[str, Any]]] = None

    def _ensure_loaded(self) -> List[Dict[str, Any]]:
        """Load history from disk on first access."""
        if self._entries is None:
            if self._log_path.exists():
                try:
                    with open(self._log_path) as f:
                        self._entries = json.load(f)
                except (json.JSONDecodeError, OSError):
                    self._entries = []
            else:
                self._entries = []
        return self._entries

    def _save(self) -> None:
        """Persist history to disk."""
        entries = self._ensure_loaded()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w") as f:
            json.dump(entries, f, indent=2, default=str)

    def log(
        self,
        tool: str,
        params: Dict[str, Any],
        success: bool,
        output_paths: Optional[List[str]] = None,
        model: Optional[str] = None,
        error: Optional[str] = None,
    ) -> str:
        """Log a generation operation. Returns the entry ID."""
        entry_id = str(uuid.uuid4())[:8]
        entry = {
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "tool": tool,
            "params": params,
            "success": success,
            "output_paths": output_paths or [],
            "model": model,
            "error": error,
        }
        entries = self._ensure_loaded()
        entries.append(entry)
        try:
            self._save()
        except OSError as e:
            logger.warning(f"Failed to save history: {e}")
        return entry_id

    def list_entries(
        self,
        limit: int = 20,
        offset: int = 0,
        tool_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List history entries with pagination and optional tool filter."""
        entries = self._ensure_loaded()
        if tool_filter:
            entries = [e for e in entries if e.get("tool") == tool_filter]
        # Return newest first
        entries = list(reversed(entries))
        return entries[offset : offset + limit]

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific history entry by ID."""
        for entry in self._ensure_loaded():
            if entry.get("id") == entry_id:
                return entry
        return None

    def count(self, tool_filter: Optional[str] = None) -> int:
        """Count total history entries."""
        entries = self._ensure_loaded()
        if tool_filter:
            return sum(1 for e in entries if e.get("tool") == tool_filter)
        return len(entries)


# Module-level singleton
_history: Optional[GenerationHistory] = None


def get_history() -> GenerationHistory:
    """Get the global history instance."""
    global _history
    if _history is None:
        from pixelforge_mcp.config import get_config

        config = get_config()
        _history = GenerationHistory(config.storage.output_dir)
    return _history

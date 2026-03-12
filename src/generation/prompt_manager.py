"""
Prompt Manager — loads versioned prompts from YAML configuration.
Prompts are treated as code: versioned, tracked, and auditable.
"""

import os
from typing import Any, Dict, Optional

import yaml


class PromptManager:
    """Manages versioned prompts loaded from YAML config."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to config/prompts.yaml relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            )))
            config_path = os.path.join(project_root, "config", "prompts.yaml")

        self.config_path = config_path
        self._prompts: Dict[str, Any] = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from the YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Prompt config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._prompts = config.get("prompts", {})

    def get_prompt(self, name: str) -> Dict[str, Any]:
        """
        Get a prompt definition by name.

        Args:
            name: Prompt name (e.g., 'rag_system', 'rag_query')

        Returns:
            Dict with 'version', 'description', 'template', etc.
        """
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found. Available: {list(self._prompts.keys())}")
        return self._prompts[name]

    def get_template(self, name: str) -> str:
        """Get just the template string for a prompt."""
        return self.get_prompt(name)["template"]

    def get_version(self, name: str) -> str:
        """Get the version string for a prompt."""
        return self.get_prompt(name)["version"]

    def format_prompt(self, name: str, **kwargs) -> str:
        """
        Render a prompt template with the given variables.

        Args:
            name: Prompt name
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string
        """
        template = self.get_template(name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing variable {e} for prompt '{name}'")

    def get_all_versions(self) -> Dict[str, str]:
        """Return a map of prompt name → version for tracing/auditing."""
        return {
            name: info.get("version", "unknown")
            for name, info in self._prompts.items()
        }

    def reload(self):
        """Reload prompts from disk (useful for hot-reloading during development)."""
        self._load_prompts()

"""
Obsidian REST API client module.
Provides methods to interact with Obsidian vault through REST API.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import requests
from requests.exceptions import RequestException


@dataclass(frozen=True)
class ObsidianConfig:
    """Configuration for Obsidian REST API client."""
    base_url: str = "http://localhost:27777"
    timeout: int = 30
    api_key: Optional[str] = None


class ObsidianClient:
    """Client for interacting with Obsidian REST API."""

    def __init__(self, config: ObsidianConfig) -> None:
        self._config = config
        self._session = requests.Session()
        if config.api_key:
            self._session.headers.update({"Authorization": f"Bearer {config.api_key}"})

    def list_vaults(self) -> list[dict[str, Any]]:
        """List all available vaults."""
        response = self._session.get(
            f"{self._config.base_url}/vaults",
            timeout=self._config.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_files(self, vault_path: str) -> list[str]:
        """List all markdown files in vault."""
        response = self._session.get(
            f"{self._config.base_url}/files",
            params={"vault": vault_path},
            timeout=self._config.timeout,
        )
        response.raise_for_status()
        return response.json()

    def read_file(self, vault_path: str, file_path: str) -> str:
        """Read content of a markdown file."""
        response = self._session.get(
            f"{self._config.base_url}/file",
            params={"vault": vault_path, "path": file_path},
            timeout=self._config.timeout,
        )
        response.raise_for_status()
        return response.text

    def create_file(
        self,
        vault_path: str,
        file_path: str,
        content: str,
    ) -> bool:
        """Create a new markdown file."""
        response = self._session.post(
            f"{self._config.base_url}/file",
            params={"vault": vault_path, "path": file_path},
            json={"content": content},
            timeout=self._config.timeout,
        )
        response.raise_for_status()
        return response.json().get("success", False)

    def update_file(
        self,
        vault_path: str,
        file_path: str,
        content: str,
    ) -> bool:
        """Update an existing markdown file."""
        response = self._session.put(
            f"{self._config.base_url}/file",
            params={"vault": vault_path, "path": file_path},
            json={"content": content},
            timeout=self._config.timeout,
        )
        response.raise_for_status()
        return response.json().get("success", False)

    def search_notes(
        self,
        vault_path: str,
        query: str,
    ) -> list[dict[str, Any]]:
        """Search notes in vault by query."""
        response = self._session.post(
            f"{self._config.base_url}/search",
            params={"vault": vault_path},
            json={"query": query},
            timeout=self._config.timeout,
        )
        response.raise_for_status()
        return response.json()

    def append_to_file(
        self,
        vault_path: str,
        file_path: str,
        content: str,
    ) -> bool:
        """Append content to existing file."""
        existing = self.read_file(vault_path, file_path)
        updated = f"{existing}\n\n{content}" if existing else content
        return self.update_file(vault_path, file_path, updated)

    def close(self) -> None:
        """Close the client session."""
        self._session.close()

    def __enter__(self) -> "ObsidianClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def create_obsidian_client(
    base_url: str = "http://localhost:27777",
    api_key: Optional[str] = None,
) -> ObsidianClient:
    """Factory function to create Obsidian client."""
    config = ObsidianConfig(base_url=base_url, api_key=api_key)
    return ObsidianClient(config)
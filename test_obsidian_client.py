"""
Tests for Obsidian REST API client module.
"""

import pytest
from unittest.mock import Mock, patch

from obsidian_client import ObsidianClient, ObsidianConfig, create_obsidian_client


class TestObsidianConfig:
    def test_default_config(self) -> None:
        config = ObsidianConfig()
        assert config.base_url == "http://localhost:27777"
        assert config.timeout == 30
        assert config.api_key is None

    def test_custom_config(self) -> None:
        config = ObsidianConfig(
            base_url="http://custom:8080",
            timeout=60,
            api_key="test-key",
        )
        assert config.base_url == "http://custom:8080"
        assert config.timeout == 60
        assert config.api_key == "test-key"


class TestObsidianClient:
    def test_context_manager(self) -> None:
        with ObsidianClient(ObsidianConfig()) as client:
            assert client is not None

    @patch("obsidian_client.requests.Session")
    def test_list_vaults(self, mock_session_class: Mock) -> None:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.return_value.json.return_value = [
            {"name": "Vault1", "path": "/path1"},
            {"name": "Vault2", "path": "/path2"},
        ]

        client = ObsidianClient(ObsidianConfig())
        result = client.list_vaults()

        assert len(result) == 2
        assert result[0]["name"] == "Vault1"
        mock_session.get.assert_called_once()

    @patch("obsidian_client.requests.Session")
    def test_list_files(self, mock_session_class: Mock) -> None:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.return_value.json.return_value = ["note1.md", "note2.md"]

        client = ObsidianClient(ObsidianConfig())
        result = client.list_files("/vault/path")

        assert result == ["note1.md", "note2.md"]

    @patch("obsidian_client.requests.Session")
    def test_read_file(self, mock_session_class: Mock) -> None:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.return_value.text = "# Test Note\nContent here"

        client = ObsidianClient(ObsidianConfig())
        result = client.read_file("/vault/path", "note.md")

        assert result == "# Test Note\nContent here"

    @patch("obsidian_client.requests.Session")
    def test_create_file(self, mock_session_class: Mock) -> None:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.post.return_value.json.return_value = {"success": True}

        client = ObsidianClient(ObsidianConfig())
        result = client.create_file("/vault/path", "new.md", "# New Note")

        assert result is True

    @patch("obsidian_client.requests.Session")
    def test_update_file(self, mock_session_class: Mock) -> None:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.put.return_value.json.return_value = {"success": True}

        client = ObsidianClient(ObsidianConfig())
        result = client.update_file("/vault/path", "note.md", "Updated content")

        assert result is True

    @patch("obsidian_client.requests.Session")
    def test_search_notes(self, mock_session_class: Mock) -> None:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.post.return_value.json.return_value = [
            {"path": "note1.md", "score": 0.95},
        ]

        client = ObsidianClient(ObsidianConfig())
        result = client.search_notes("/vault/path", "query")

        assert len(result) == 1
        assert result[0]["path"] == "note1.md"

    @patch("obsidian_client.requests.Session")
    def test_append_to_file(self, mock_session_class: Mock) -> None:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get.return_value.text = "Existing content"
        mock_session.put.return_value.json.return_value = {"success": True}

        client = ObsidianClient(ObsidianConfig())
        result = client.append_to_file("/vault/path", "note.md", "New content")

        assert result is True
        # Verify that put was called with combined content
        call_args = mock_session.put.call_args
        assert "Existing content" in call_args.kwargs["json"]["content"]
        assert "New content" in call_args.kwargs["json"]["content"]


class TestCreateObsidianClient:
    def test_factory_default(self) -> None:
        client = create_obsidian_client()
        assert client._config.base_url == "http://localhost:27777"
        assert client._config.api_key is None

    def test_factory_custom(self) -> None:
        client = create_obsidian_client(
            base_url="http://custom:9000",
            api_key="secret-key",
        )
        assert client._config.base_url == "http://custom:9000"
        assert client._config.api_key == "secret-key"
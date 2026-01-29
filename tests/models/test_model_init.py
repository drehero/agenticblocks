"""Tests for Model class initialization, provider inference, and API key resolution."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestProviderInference:
    """Tests for automatic provider inference from api_url."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_defaults_to_openai_provider(self, mock_openai_class):
        """Provider should default to 'openai' when not specified."""
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4")
        assert model.provider == "openai"
        assert model.client_provider == "openai"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_infers_openrouter_from_url(self, mock_openai_class):
        """Should infer 'openrouter' provider from api_url."""
        from agenticblocks.models import Model

        model = Model(
            model_name="anthropic/claude-3",
            api_url="https://openrouter.ai/api/v1",
            web_search=False,
        )
        assert model.provider == "openrouter"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_infers_anthropic_from_url(self, mock_anthropic_module):
        """Should infer 'anthropic' provider from api_url."""
        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
            from agenticblocks.models import Model

            model = Model(
                model_name="claude-3-opus",
                api_url="https://api.anthropic.com",
            )
            assert model.provider == "anthropic"
            assert model.client_provider == "anthropic"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_infers_google_from_url(self, mock_google_module):
        """Should infer 'google' provider from api_url."""
        mock_google = MagicMock()
        mock_google.genai = mock_google_module
        with patch.dict(
            sys.modules, {"google": mock_google, "google.genai": mock_google_module}
        ):
            from agenticblocks.models import Model

            model = Model(
                model_name="gemini-pro",
                api_url="https://generativelanguage.googleapis.com/v1beta",
            )
            assert model.provider == "google"
            assert model.client_provider == "google"

    @patch.dict(os.environ, {"XAI_API_KEY": "test-key"})
    def test_infers_xai_from_url(self, mock_xai_module):
        """Should use 'xai' provider when explicitly set."""
        mock_xai, mock_chat_module = mock_xai_module
        with patch.dict(
            sys.modules, {"xai_sdk": mock_xai, "xai_sdk.chat": mock_chat_module}
        ):
            from agenticblocks.models import Model

            # xai provider must be explicitly set since "x.ai" doesn't contain "xai"
            model = Model(
                model_name="grok-2",
                provider="xai",
                web_search=False,
            )
            assert model.provider == "xai"
            assert model.client_provider == "xai"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_explicit_provider_overrides_inference(self, mock_openai_class):
        """Explicit provider parameter should override URL inference."""
        from agenticblocks.models import Model

        model = Model(
            model_name="gpt-4",
            provider="openai",
            api_url="https://custom.example.com/v1",
        )
        assert model.provider == "openai"


class TestApiKeyResolution:
    """Tests for API key resolution from environment variables."""

    @patch("openai.OpenAI")
    def test_uses_explicit_api_key(self, mock_openai_class):
        """Should use explicitly provided api_key."""
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", api_key="explicit-key")
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["api_key"] == "explicit-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"})
    @patch("openai.OpenAI")
    def test_uses_provider_specific_env_var(self, mock_openai_class):
        """Should use provider-specific environment variable."""
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", provider="openai")
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["api_key"] == "env-openai-key"

    @patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "env-anthropic-key"},
        clear=False,
    )
    def test_anthropic_uses_anthropic_api_key(self, mock_anthropic_module):
        """Anthropic provider should use ANTHROPIC_API_KEY."""
        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
            from agenticblocks.models import Model

            model = Model(model_name="claude-3-opus", provider="anthropic")
            mock_anthropic_module.Anthropic.assert_called_once()
            call_kwargs = mock_anthropic_module.Anthropic.call_args[1]
            assert call_kwargs["api_key"] == "env-anthropic-key"

    def test_raises_without_api_key(self, clean_env):
        """Should raise ValueError when no API key is available."""
        from agenticblocks.models import Model

        with pytest.raises(ValueError, match="No API key provided"):
            Model(model_name="gpt-4")


class TestBaseUrlResolution:
    """Tests for API base URL resolution."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_uses_explicit_api_url(self, mock_openai_class):
        """Should use explicitly provided api_url."""
        from agenticblocks.models import Model

        model = Model(
            model_name="gpt-4",
            api_url="https://custom.example.com/v1",
        )
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["base_url"] == "https://custom.example.com/v1"

    @patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_API_URL": "https://env.url"}
    )
    @patch("openai.OpenAI")
    def test_uses_provider_env_url(self, mock_openai_class):
        """Should use provider-specific environment URL variable."""
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4")
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["base_url"] == "https://env.url"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_uses_default_openai_url(self, mock_openai_class, clean_env):
        """Should use default OpenAI URL when none specified."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4")
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["base_url"] == "https://api.openai.com/v1"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_uses_default_openrouter_url(self, mock_openai_class):
        """Should use default OpenRouter URL."""
        from agenticblocks.models import Model

        model = Model(model_name="model", provider="openrouter")
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"


class TestOpenRouterOnlineSuffix:
    """Tests for OpenRouter :online suffix handling."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_adds_online_suffix_when_web_search_true(self, mock_openai_class):
        """Should add :online suffix when web_search=True."""
        from agenticblocks.models import Model

        model = Model(
            model_name="anthropic/claude-3",
            provider="openrouter",
            web_search=True,
        )
        assert model.model_name == "anthropic/claude-3:online"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_no_duplicate_suffix(self, mock_openai_class):
        """Should not add duplicate :online suffix."""
        from agenticblocks.models import Model

        model = Model(
            model_name="anthropic/claude-3:online",
            provider="openrouter",
            web_search=True,
        )
        assert model.model_name == "anthropic/claude-3:online"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_removes_suffix_when_web_search_false(self, mock_openai_class):
        """Should remove :online suffix when web_search=False."""
        from agenticblocks.models import Model

        model = Model(
            model_name="anthropic/claude-3:online",
            provider="openrouter",
            web_search=False,
        )
        assert model.model_name == "anthropic/claude-3"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_no_suffix_when_web_search_false(self, mock_openai_class):
        """Should not add suffix when web_search=False."""
        from agenticblocks.models import Model

        model = Model(
            model_name="anthropic/claude-3",
            provider="openrouter",
            web_search=False,
        )
        assert model.model_name == "anthropic/claude-3"


class TestModelRepr:
    """Tests for Model __repr__."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_repr_format(self, mock_openai_class):
        """__repr__ should return Model(model_name)."""
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4")
        assert repr(model) == "Model('gpt-4')"

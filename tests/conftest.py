"""Shared pytest fixtures and mock factories for all tests."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_global_model_stats():
    """Auto-reset GlobalModelStats before each test."""
    from agenticblocks.models.stats import GLOBAL_MODEL_STATS

    # Store original values
    original_cost = GLOBAL_MODEL_STATS._cost
    original_calls = GLOBAL_MODEL_STATS._n_calls
    original_cost_limit = GLOBAL_MODEL_STATS.cost_limit
    original_call_limit = GLOBAL_MODEL_STATS.call_limit

    # Reset for test
    GLOBAL_MODEL_STATS._cost = 0.0
    GLOBAL_MODEL_STATS._n_calls = 0
    GLOBAL_MODEL_STATS.cost_limit = 0
    GLOBAL_MODEL_STATS.call_limit = 0

    yield

    # Restore original values
    GLOBAL_MODEL_STATS._cost = original_cost
    GLOBAL_MODEL_STATS._n_calls = original_calls
    GLOBAL_MODEL_STATS.cost_limit = original_cost_limit
    GLOBAL_MODEL_STATS.call_limit = original_call_limit


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with standard response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.to_dict.return_value = {
        "choices": [{"message": {"content": "mock response"}}],
        "usage": {"cost": 0.001},
    }
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_module():
    """Create a mock anthropic module."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.Anthropic.return_value = mock_client

    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="mock response")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_client.messages.create.return_value = mock_response

    return mock_module


@pytest.fixture
def mock_google_module():
    """Create a mock google.genai module."""
    mock_genai = MagicMock()
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_response = MagicMock()
    mock_response.text = "mock response"
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=10, candidates_token_count=5
    )
    mock_client.models.generate_content.return_value = mock_response

    # Mock types for web search
    mock_types = MagicMock()
    mock_genai.types = mock_types

    return mock_genai


@pytest.fixture
def mock_xai_module():
    """Create a mock xai_sdk module."""
    mock_xai = MagicMock()
    mock_client = MagicMock()
    mock_xai.Client.return_value = mock_client

    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "mock response"
    mock_chat.sample.return_value = mock_response
    mock_client.chat.create.return_value = mock_chat

    # Mock chat helper functions
    mock_chat_module = MagicMock()
    mock_chat_module.system = lambda x: {"role": "system", "content": x}
    mock_chat_module.user = lambda x: {"role": "user", "content": x}
    mock_chat_module.assistant = lambda x: {"role": "assistant", "content": x}

    return mock_xai, mock_chat_module


@pytest.fixture
def clean_env():
    """Fixture to temporarily clear API key environment variables."""
    env_vars = [
        "OPENAI_API_KEY",
        "OPENAI_API_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_URL",
        "GOOGLE_API_KEY",
        "GOOGLE_API_URL",
        "XAI_API_KEY",
        "XAI_API_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_URL",
        "MSWEA_GLOBAL_COST_LIMIT",
        "MSWEA_GLOBAL_CALL_LIMIT",
        "MSWEA_SILENT_STARTUP",
    ]
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


def create_mock_model(return_value="mock response"):
    """Factory function to create a mock model callable."""
    mock_model = MagicMock(return_value=return_value)
    mock_model.__repr__ = lambda self: "MockModel()"
    return mock_model

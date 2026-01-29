"""Tests for Model conversation history management."""
import os
from unittest.mock import MagicMock, patch

import pytest


class TestKeepHistory:
    """Tests for keep_history functionality."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_messages_accumulated_with_keep_history(
        self, mock_openai_class, mock_openai_client
    ):
        """Messages should accumulate when keep_history=True."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", keep_history=True, web_search=False)
        model._warned_missing_cost = True

        model("First message")
        model("Second message")

        # Should have user and assistant messages for each call
        user_messages = [m for m in model.messages if m["role"] == "user"]
        assistant_messages = [m for m in model.messages if m["role"] == "assistant"]

        assert len(user_messages) == 2
        assert len(assistant_messages) == 2
        assert user_messages[0]["content"] == "First message"
        assert user_messages[1]["content"] == "Second message"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_messages_not_accumulated_without_keep_history(
        self, mock_openai_class, mock_openai_client
    ):
        """Messages should not accumulate when keep_history=False."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", keep_history=False, web_search=False)
        model._warned_missing_cost = True

        model("First message")
        model("Second message")

        # Messages list should only contain system messages (if any)
        assert len(model.messages) == 0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_system_prompt_preserved_in_history(
        self, mock_openai_class, mock_openai_client
    ):
        """System prompt should be included in history."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.models import Model

        model = Model(
            model_name="gpt-4",
            system_prompt="You are helpful.",
            keep_history=True,
            web_search=False,
        )
        model._warned_missing_cost = True

        model("Hello")

        system_messages = [m for m in model.messages if m["role"] == "system"]
        assert len(system_messages) == 1
        assert system_messages[0]["content"] == "You are helpful."


class TestResetHistory:
    """Tests for reset_history method."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_reset_clears_conversation(self, mock_openai_class, mock_openai_client):
        """reset_history should clear user/assistant messages."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", keep_history=True, web_search=False)
        model._warned_missing_cost = True

        model("First message")
        model("Second message")

        assert len(model.messages) == 4  # 2 user + 2 assistant

        model.reset_history()

        assert len(model.messages) == 0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_reset_preserves_system_prompt(self, mock_openai_class, mock_openai_client):
        """reset_history should preserve the system prompt."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.models import Model

        model = Model(
            model_name="gpt-4",
            system_prompt="You are helpful.",
            keep_history=True,
            web_search=False,
        )
        model._warned_missing_cost = True

        model("Hello")
        model("Goodbye")

        model.reset_history()

        # Should only have system message
        assert len(model.messages) == 1
        assert model.messages[0]["role"] == "system"
        assert model.messages[0]["content"] == "You are helpful."

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_reset_with_no_system_prompt(self, mock_openai_class, mock_openai_client):
        """reset_history with no system prompt should result in empty messages."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", keep_history=True, web_search=False)
        model._warned_missing_cost = True

        model("Hello")
        model.reset_history()

        assert len(model.messages) == 0


class TestAddMessage:
    """Tests for add_message method."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_add_message_basic(self, mock_openai_class):
        """add_message should add a message with role and content."""
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", keep_history=True, web_search=False)

        model.add_message("user", "Test message")

        assert len(model.messages) == 1
        assert model.messages[0]["role"] == "user"
        assert model.messages[0]["content"] == "Test message"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_add_message_includes_timestamp(self, mock_openai_class):
        """add_message should include a timestamp."""
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", keep_history=True, web_search=False)

        model.add_message("user", "Test")

        assert "timestamp" in model.messages[0]
        assert isinstance(model.messages[0]["timestamp"], float)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_add_message_extra_kwargs(self, mock_openai_class):
        """add_message should accept extra kwargs."""
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", keep_history=True, web_search=False)

        model.add_message("user", "Test", custom_field="custom_value")

        assert model.messages[0]["custom_field"] == "custom_value"


class TestModelCallCounting:
    """Tests for call counting and cost tracking on Model instance."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_n_calls_increments(self, mock_openai_class, mock_openai_client):
        """n_calls should increment with each call."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", web_search=False)
        model._warned_missing_cost = True

        assert model.n_calls == 0

        model("First")
        assert model.n_calls == 1

        model("Second")
        assert model.n_calls == 2

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_cost_accumulates(self, mock_openai_class):
        """cost should accumulate from usage data."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "choices": [{"message": {"content": "response"}}],
            "usage": {"cost": 0.01},
        }
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", web_search=False)

        model("First")
        assert model.cost == pytest.approx(0.01)

        model("Second")
        assert model.cost == pytest.approx(0.02)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_cost_defaults_to_zero_when_not_reported(self, mock_openai_class):
        """cost should be 0 when provider doesn't report cost."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Return response without cost in usage
        mock_response.to_dict.return_value = {
            "choices": [{"message": {"content": "response"}}],
            "usage": {},  # No cost field
        }
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", web_search=False)

        # First call will warn, subsequent won't
        with pytest.warns(UserWarning, match="Cost tracking is not available"):
            model("Hello")

        assert model.cost == 0.0

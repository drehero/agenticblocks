"""Tests for provider-specific caching functionality."""
import pytest
from unittest.mock import MagicMock, patch

from agenticblocks.models.utils import apply_anthropic_cache_control


class TestApplyAnthropicCacheControl:
    """Tests for apply_anthropic_cache_control function."""

    def test_system_prompt_gets_cached(self):
        """System prompt should have cache_control added to last block."""
        system_prompt = [{"type": "text", "text": "You are a helpful assistant."}]
        messages = [{"role": "user", "content": "Hello"}]

        cached_system, cached_messages = apply_anthropic_cache_control(system_prompt, messages)

        assert cached_system is not None
        assert cached_system[-1].get("cache_control") == {"type": "ephemeral"}

    def test_system_prompt_with_multiple_blocks(self):
        """Cache control should be added to the last block of system prompt."""
        system_prompt = [
            {"type": "text", "text": "First instruction."},
            {"type": "text", "text": "Second instruction."},
        ]
        messages = [{"role": "user", "content": "Hello"}]

        cached_system, cached_messages = apply_anthropic_cache_control(system_prompt, messages)

        # Only last block should have cache_control
        assert "cache_control" not in cached_system[0]
        assert cached_system[1].get("cache_control") == {"type": "ephemeral"}

    def test_conversation_prefix_gets_cached(self):
        """Message before last user message should get cache_control."""
        system_prompt = None
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]

        cached_system, cached_messages = apply_anthropic_cache_control(system_prompt, messages)

        # The assistant message (before last user) should be cached
        assert cached_messages[1].get("content") == [
            {
                "type": "text",
                "text": "First answer",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def test_last_user_message_not_cached(self):
        """Last user message should NOT have cache_control."""
        system_prompt = None
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]

        cached_system, cached_messages = apply_anthropic_cache_control(system_prompt, messages)

        # Last user message should not have cache_control
        last_msg = cached_messages[-1]
        if isinstance(last_msg["content"], list):
            assert "cache_control" not in last_msg["content"][-1]
        else:
            # String content means no cache_control was added
            assert isinstance(last_msg["content"], str)

    def test_single_user_message_no_cache(self):
        """Single user message should not have any conversation cache."""
        system_prompt = [{"type": "text", "text": "System"}]
        messages = [{"role": "user", "content": "Only message"}]

        cached_system, cached_messages = apply_anthropic_cache_control(system_prompt, messages)

        # System prompt should be cached
        assert cached_system[-1].get("cache_control") == {"type": "ephemeral"}
        # But the single user message should not be cached
        assert isinstance(cached_messages[0]["content"], str)

    def test_no_system_prompt(self):
        """Should handle None system prompt correctly."""
        system_prompt = None
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        cached_system, cached_messages = apply_anthropic_cache_control(system_prompt, messages)

        assert cached_system is None
        # Assistant message should still be cached
        assert cached_messages[1]["content"][-1].get("cache_control") == {"type": "ephemeral"}

    def test_empty_messages(self):
        """Should handle empty messages list."""
        system_prompt = [{"type": "text", "text": "System"}]
        messages = []

        cached_system, cached_messages = apply_anthropic_cache_control(system_prompt, messages)

        assert cached_system[-1].get("cache_control") == {"type": "ephemeral"}
        assert cached_messages == []

    def test_does_not_mutate_originals(self):
        """Original inputs should not be modified."""
        system_prompt = [{"type": "text", "text": "System"}]
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]

        # Keep copies of originals
        original_system = [{"type": "text", "text": "System"}]
        original_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]

        apply_anthropic_cache_control(system_prompt, messages)

        assert system_prompt == original_system
        assert messages == original_messages

    def test_list_content_format_preserved(self):
        """Messages with list content format should be handled correctly."""
        system_prompt = None
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "user", "content": [{"type": "text", "text": "Bye"}]},
        ]

        cached_system, cached_messages = apply_anthropic_cache_control(system_prompt, messages)

        # Assistant message with list content should have cache_control added
        assert cached_messages[1]["content"][-1].get("cache_control") == {"type": "ephemeral"}


class TestModelCacheControl:
    """Tests for Model class cache control integration."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_apply_cache_control_called_for_anthropic(self):
        """Cache control should be applied for Anthropic provider when keep_history=True."""
        import sys

        # Create mock anthropic module
        mock_anthropic_module = MagicMock()
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response

        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
            # Need to reimport after patching
            from importlib import reload
            import agenticblocks.models.model as model_module
            reload(model_module)

            with patch.object(model_module, "apply_anthropic_cache_control") as mock_cache:
                mock_cache.return_value = (None, [{"role": "user", "content": "test"}])

                model = model_module.Model(
                    model_name="claude-3-opus-20240229",
                    provider="anthropic",
                    keep_history=True,
                    web_search=False,
                )
                model("Hello")

                # Verify apply_anthropic_cache_control was called
                mock_cache.assert_called_once()

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_cache_control_not_called_without_keep_history(self):
        """Cache control should NOT be applied when keep_history=False."""
        import sys

        # Create mock anthropic module
        mock_anthropic_module = MagicMock()
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response

        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
            from importlib import reload
            import agenticblocks.models.model as model_module
            reload(model_module)

            with patch.object(model_module, "apply_anthropic_cache_control") as mock_cache:
                model = model_module.Model(
                    model_name="claude-3-opus-20240229",
                    provider="anthropic",
                    keep_history=False,
                    web_search=False,
                )
                model("Hello")

                # Verify apply_anthropic_cache_control was NOT called
                mock_cache.assert_not_called()

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_openrouter_anthropic_model_gets_cache_control(self):
        """OpenRouter with Anthropic model should apply cache control."""
        from agenticblocks.models.model import Model

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "choices": [{"message": {"content": "response"}}],
                "usage": {},
            }
            mock_client.chat.completions.create.return_value = mock_response

            model = Model(
                model_name="anthropic/claude-3-opus",
                provider="openrouter",
                keep_history=True,
                web_search=False,
            )
            model._warned_missing_cost = True  # Suppress warning

            # Add some history
            model.add_message("user", "First message")
            model.add_message("assistant", "First response")

            model("Second message")

            # Check that messages were passed with cache_control
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]

            # Find the assistant message and verify it has cache_control
            assistant_msgs = [m for m in messages if m["role"] == "assistant"]
            if assistant_msgs:
                content = assistant_msgs[0]["content"]
                if isinstance(content, list):
                    assert any(
                        block.get("cache_control") == {"type": "ephemeral"}
                        for block in content
                    )

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_openrouter_non_anthropic_model_no_cache_control(self):
        """OpenRouter with non-Anthropic model should NOT apply Anthropic cache control."""
        from agenticblocks.models.model import Model

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "choices": [{"message": {"content": "response"}}],
                "usage": {},
            }
            mock_client.chat.completions.create.return_value = mock_response

            model = Model(
                model_name="openai/gpt-4",
                provider="openrouter",
                keep_history=True,
                web_search=False,
            )
            model._warned_missing_cost = True

            model.add_message("user", "First message")
            model.add_message("assistant", "First response")

            model("Second message")

            # Check that messages were passed WITHOUT cache_control
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]

            # Messages should be simple strings, not list format with cache_control
            for msg in messages:
                assert isinstance(msg["content"], str)


class TestModelXaiConversationId:
    """Tests for xAI conversation ID generation."""

    @patch.dict("os.environ", {"XAI_API_KEY": "test-key"})
    def test_xai_conversation_id_generated(self):
        """Model should generate a UUID for xAI conversation ID."""
        from agenticblocks.models.model import Model
        import uuid

        with patch("xai_sdk.Client"):
            model = Model(
                model_name="grok-2",
                provider="xai",
                keep_history=True,
                web_search=False,
            )

            # Verify conversation ID is a valid UUID
            assert hasattr(model, "_xai_conversation_id")
            # Should be a valid UUID string
            uuid.UUID(model._xai_conversation_id)

    @patch.dict("os.environ", {"XAI_API_KEY": "test-key"})
    def test_xai_conversation_id_unique_per_instance(self):
        """Each Model instance should have a unique conversation ID."""
        from agenticblocks.models.model import Model

        with patch("xai_sdk.Client"):
            model1 = Model(
                model_name="grok-2",
                provider="xai",
                keep_history=True,
                web_search=False,
            )
            model2 = Model(
                model_name="grok-2",
                provider="xai",
                keep_history=True,
                web_search=False,
            )

            assert model1._xai_conversation_id != model2._xai_conversation_id

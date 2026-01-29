"""Tests for ChainOfThought block."""
import os
from unittest.mock import MagicMock, patch

import pytest


class TestChainOfThoughtBlock:
    """Tests for ChainOfThought template formatting."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_default_template_applied(self, mock_openai_class, mock_openai_client):
        """ChainOfThought should apply default 'think step by step' template."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import ChainOfThought

        cot = ChainOfThought(model="gpt-4")
        cot.model._warned_missing_cost = True
        cot.model.web_search = False

        cot("What is 2+2?")

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        user_message = [m for m in messages if m["role"] == "user"][0]

        assert "What is 2+2?" in user_message["content"]
        assert "Let's think step by step" in user_message["content"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_custom_template(self, mock_openai_class, mock_openai_client):
        """ChainOfThought should support custom templates."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import ChainOfThought

        cot = ChainOfThought(
            model="gpt-4",
            template="Question: {prompt}\nReason through this carefully:",
        )
        cot.model._warned_missing_cost = True
        cot.model.web_search = False

        cot("What is the capital of France?")

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        user_message = [m for m in messages if m["role"] == "user"][0]

        assert "Question: What is the capital of France?" in user_message["content"]
        assert "Reason through this carefully:" in user_message["content"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_returns_model_output(self, mock_openai_class, mock_openai_client):
        """ChainOfThought should return the model's output."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import ChainOfThought

        cot = ChainOfThought(model="gpt-4")
        cot.model._warned_missing_cost = True
        cot.model.web_search = False

        result = cot("test")

        assert result == "mock response"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_passes_kwargs_to_model(self, mock_openai_class, mock_openai_client):
        """ChainOfThought should pass kwargs to model."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import ChainOfThought

        cot = ChainOfThought(model="gpt-4")
        cot.model._warned_missing_cost = True
        cot.model.web_search = False

        cot("test", temperature=0.8, max_tokens=500)

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("temperature") == 0.8
        assert call_kwargs.get("max_tokens") == 500

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_accepts_model_instance(self, mock_openai_class, mock_openai_client):
        """ChainOfThought should accept a Model instance."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import ChainOfThought
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", web_search=False)
        cot = ChainOfThought(model=model)

        assert cot.model is model


class TestChainOfThoughtRepr:
    """Tests for ChainOfThought __repr__."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_repr_format(self, mock_openai_class):
        """__repr__ should show ChainOfThought(Model(...))."""
        from agenticblocks.blocks import ChainOfThought

        cot = ChainOfThought(model="gpt-4")

        assert repr(cot) == "ChainOfThought(Model('gpt-4'))"

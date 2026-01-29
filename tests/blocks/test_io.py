"""Tests for IO block."""
import os
from unittest.mock import MagicMock, patch

import pytest


class TestIOBlock:
    """Tests for IO block pass-through behavior."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_io_passes_prompt_to_model(self, mock_openai_class, mock_openai_client):
        """IO should pass prompt directly to model."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import IO

        io = IO(model="gpt-4")
        io.model._warned_missing_cost = True
        io.model.web_search = False

        result = io("test prompt")

        # Check that the model was called
        assert mock_openai_client.chat.completions.create.called
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert any(m["content"] == "test prompt" for m in messages)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_io_returns_model_output(self, mock_openai_class, mock_openai_client):
        """IO should return the model's output."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import IO

        io = IO(model="gpt-4")
        io.model._warned_missing_cost = True
        io.model.web_search = False

        result = io("test")

        assert result == "mock response"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_io_passes_kwargs_to_model(self, mock_openai_class, mock_openai_client):
        """IO should pass kwargs to model."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import IO

        io = IO(model="gpt-4")
        io.model._warned_missing_cost = True
        io.model.web_search = False

        io("test", temperature=0.5)

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("temperature") == 0.5

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_io_accepts_model_instance(self, mock_openai_class, mock_openai_client):
        """IO should accept a Model instance."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import IO
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", web_search=False)
        model._warned_missing_cost = True
        io = IO(model=model)

        result = io("test")

        assert result == "mock response"
        assert io.model is model

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_io_accepts_model_string(self, mock_openai_class, mock_openai_client):
        """IO should accept a model name string."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import IO
        from agenticblocks.models import Model

        io = IO(model="gpt-4")

        assert isinstance(io.model, Model)
        assert io.model.model_name == "gpt-4"


class TestIORepr:
    """Tests for IO __repr__."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_repr_format(self, mock_openai_class):
        """__repr__ should show IO(Model(...))."""
        from agenticblocks.blocks import IO

        io = IO(model="gpt-4")

        assert repr(io) == "IO(Model('gpt-4'))"

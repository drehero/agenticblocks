"""Tests for SelfRefine block."""
import os
from unittest.mock import MagicMock, patch, call

import pytest


class TestSelfRefineBlock:
    """Tests for SelfRefine critique/refine iterations."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_initial_response_generated(self, mock_openai_class, mock_openai_client):
        """Should generate initial response from prompt."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import SelfRefine

        sr = SelfRefine(model="gpt-4", iterations=0)
        sr.model._warned_missing_cost = True
        sr.model.web_search = False

        result = sr("Write a poem")

        # With 0 iterations, just returns initial response
        assert result == "mock response"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_critique_and_refine_iterations(self, mock_openai_class):
        """Should perform critique and refine for each iteration."""
        mock_client = MagicMock()
        responses = [
            # Initial response
            {"choices": [{"message": {"content": "Initial draft"}}], "usage": {}},
            # Iteration 1 critique
            {"choices": [{"message": {"content": "Could be better"}}], "usage": {}},
            # Iteration 1 refine
            {"choices": [{"message": {"content": "Improved v1"}}], "usage": {}},
            # Iteration 2 critique
            {"choices": [{"message": {"content": "Still some issues"}}], "usage": {}},
            # Iteration 2 refine
            {"choices": [{"message": {"content": "Final improved"}}], "usage": {}},
        ]
        mock_response = MagicMock()
        mock_response.to_dict.side_effect = responses
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        from agenticblocks.blocks import SelfRefine

        sr = SelfRefine(model="gpt-4", iterations=2)
        sr.model.web_search = False

        result = sr("Write something")

        # Should call: initial + 2*(critique + refine) = 5 calls
        assert mock_client.chat.completions.create.call_count == 5
        assert result == "Final improved"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_critique_template_used(self, mock_openai_class, mock_openai_client):
        """Critique template should be used."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import SelfRefine

        custom_critique = "Task: {prompt}\n\nAnswer: {response}\n\nWhat's wrong?"
        sr = SelfRefine(model="gpt-4", iterations=1, critique_template=custom_critique)
        sr.model._warned_missing_cost = True
        sr.model.web_search = False

        sr("my task")

        # Second call should be critique
        calls = mock_openai_client.chat.completions.create.call_args_list
        critique_call = calls[1]
        messages = critique_call[1]["messages"]
        user_msg = [m for m in messages if m["role"] == "user"][0]

        assert "Task: my task" in user_msg["content"]
        assert "What's wrong?" in user_msg["content"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_refine_template_used(self, mock_openai_class):
        """Refine template should be used."""
        mock_client = MagicMock()
        responses = [
            {"choices": [{"message": {"content": "Initial"}}], "usage": {}},
            {"choices": [{"message": {"content": "Critique text"}}], "usage": {}},
            {"choices": [{"message": {"content": "Refined"}}], "usage": {}},
        ]
        mock_response = MagicMock()
        mock_response.to_dict.side_effect = responses
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        from agenticblocks.blocks import SelfRefine

        custom_refine = "Task: {prompt}\nPrevious: {response}\nFeedback: {critique}\nImprove it:"
        sr = SelfRefine(model="gpt-4", iterations=1, refine_template=custom_refine)
        sr.model.web_search = False

        sr("my task")

        # Third call should be refine
        calls = mock_client.chat.completions.create.call_args_list
        refine_call = calls[2]
        messages = refine_call[1]["messages"]
        user_msg = [m for m in messages if m["role"] == "user"][0]

        assert "Task: my task" in user_msg["content"]
        assert "Previous: Initial" in user_msg["content"]
        assert "Feedback: Critique text" in user_msg["content"]
        assert "Improve it:" in user_msg["content"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_passes_kwargs_to_model(self, mock_openai_class, mock_openai_client):
        """Should pass kwargs to all model calls."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import SelfRefine

        sr = SelfRefine(model="gpt-4", iterations=1)
        sr.model._warned_missing_cost = True
        sr.model.web_search = False

        sr("test", temperature=0.3, max_tokens=200)

        for call_args in mock_openai_client.chat.completions.create.call_args_list:
            assert call_args[1].get("temperature") == 0.3
            assert call_args[1].get("max_tokens") == 200

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_accepts_model_instance(self, mock_openai_class, mock_openai_client):
        """Should accept a Model instance."""
        mock_openai_class.return_value = mock_openai_client
        from agenticblocks.blocks import SelfRefine
        from agenticblocks.models import Model

        model = Model(model_name="gpt-4", web_search=False)
        sr = SelfRefine(model=model, iterations=0)

        assert sr.model is model


class TestSelfRefineRepr:
    """Tests for SelfRefine __repr__."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_repr_format(self, mock_openai_class):
        """__repr__ should show model and iterations."""
        from agenticblocks.blocks import SelfRefine

        sr = SelfRefine(model="gpt-4", iterations=3)

        assert "SelfRefine" in repr(sr)
        assert "iterations=3" in repr(sr)

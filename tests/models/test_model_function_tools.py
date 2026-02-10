"""Tests for OpenAI function tool-calling integration."""
import os
from unittest.mock import MagicMock, patch

import pytest


def _mock_response(payload):
    response = MagicMock()
    response.to_dict.return_value = payload
    return response


class TestOpenAIFunctionTools:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_function_tool_loop_executes_tool_until_text(self, mock_openai_class):
        """Model should execute function tools and continue until assistant text is returned."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "add_numbers",
                                            "arguments": "{\"a\": 2, \"b\": 3}",
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"cost": 0.01},
                }
            ),
            _mock_response(
                {
                    "choices": [{"message": {"content": "The result is 5."}}],
                    "usage": {"cost": 0.01},
                }
            ),
        ]

        from agenticblocks.models import Model

        def add_numbers(a: int, b: int) -> int:
            return a + b

        model = Model("gpt-4", function_tools=[add_numbers], web_search=False)
        result = model("What is 2 + 3?")

        assert result == "The result is 5."
        assert mock_client.chat.completions.create.call_count == 2

        first_call_kwargs = mock_client.chat.completions.create.call_args_list[0].kwargs
        assert "tools" in first_call_kwargs
        assert any(
            tool.get("type") == "function"
            and tool.get("function", {}).get("name") == "add_numbers"
            for tool in first_call_kwargs["tools"]
        )

        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
        tool_messages = [msg for msg in second_call_kwargs["messages"] if msg.get("role") == "tool"]
        assert tool_messages
        assert tool_messages[-1]["content"] == "5"
        assert tool_messages[-1]["tool_call_id"] == "call_1"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_tool_messages_are_kept_in_history(self, mock_openai_class):
        """When keep_history=True, assistant and tool messages from tool-calling are persisted."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "echo_number",
                                            "arguments": "{\"value\": 7}",
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"cost": 0.01},
                }
            ),
            _mock_response(
                {
                    "choices": [{"message": {"content": "done"}}],
                    "usage": {"cost": 0.01},
                }
            ),
        ]

        from agenticblocks.models import Model

        def echo_number(value: int) -> int:
            return value

        model = Model(
            "gpt-4",
            function_tools=[echo_number],
            keep_history=True,
            web_search=False,
        )
        out = model("Use tool")

        assert out == "done"
        roles = [m["role"] for m in model.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert model.messages[2]["content"] == "7"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_call_rejects_function_tool_config_kwargs(self, mock_openai_class):
        """Tool config should be initialization-only, not per-call."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"cost": 0.01},
            }
        )

        from agenticblocks.models import Model

        model = Model("gpt-4", web_search=False)

        with pytest.raises(TypeError, match="function_tools must be set when constructing Model"):
            model("test", function_tools=[])

        with pytest.raises(TypeError, match="max_tool_rounds must be set when constructing Model"):
            model("test", max_tool_rounds=3)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_max_tool_rounds_can_be_set_on_init(self, mock_openai_class):
        """max_tool_rounds should be configurable at initialization."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "echo_value",
                                            "arguments": "{\"value\": 1}",
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"cost": 0.01},
                }
            ),
            _mock_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_2",
                                        "type": "function",
                                        "function": {
                                            "name": "echo_value",
                                            "arguments": "{\"value\": 2}",
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"cost": 0.01},
                }
            ),
        ]

        from agenticblocks.models import Model

        def echo_value(value: int) -> int:
            return value

        model = Model(
            "gpt-4",
            function_tools=[echo_value],
            max_tool_rounds=1,
            web_search=False,
        )
        with pytest.warns(UserWarning, match="max_tool_rounds"):
            result = model("loop")

        assert result == "Function tool loop reached max_tool_rounds."

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_max_tool_rounds_must_be_positive(self, mock_openai_class):
        """max_tool_rounds must be >= 1."""
        from agenticblocks.models import Model

        with pytest.raises(ValueError, match="max_tool_rounds must be >= 1"):
            Model("gpt-4", max_tool_rounds=0)

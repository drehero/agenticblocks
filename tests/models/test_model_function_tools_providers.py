"""Tests for function tools across non-OpenAI providers."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _mock_openai_response(payload):
    response = MagicMock()
    response.to_dict.return_value = payload
    return response


class FakeXaiChat:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.appended = []

    def append(self, message):
        self.appended.append(message)

    def sample(self):
        return next(self._responses)


class TestFunctionToolsOtherProviders:
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_openrouter_function_tools(self, mock_openai_class):
        """OpenRouter should support the same function tool loop as OpenAI."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _mock_openai_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "sum_numbers", "arguments": "{\"a\": 4, \"b\": 5}"},
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"cost": 0.01},
                }
            ),
            _mock_openai_response(
                {
                    "choices": [{"message": {"content": "9"}}],
                    "usage": {"cost": 0.01},
                }
            ),
        ]

        from agenticblocks.models import Model

        def sum_numbers(a: int, b: int) -> int:
            return a + b

        model = Model("anthropic/claude-3", provider="openrouter", web_search=False, function_tools=[sum_numbers])
        result = model("4+5?")

        assert result == "9"
        assert mock_client.chat.completions.create.call_count == 2

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_anthropic_function_tools(self):
        """Anthropic provider should execute function tools and send tool_result blocks."""
        mock_client = MagicMock()
        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        response_1 = MagicMock()
        response_1.content = [
            SimpleNamespace(type="tool_use", id="toolu_1", name="sum_numbers", input={"a": 2, "b": 3}),
        ]
        response_1.usage = SimpleNamespace(input_tokens=10, output_tokens=5)

        response_2 = MagicMock()
        response_2.content = [SimpleNamespace(type="text", text="5")]
        response_2.usage = SimpleNamespace(input_tokens=10, output_tokens=5)

        mock_client.messages.create.side_effect = [response_1, response_2]

        with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
            from agenticblocks.models import Model

            def sum_numbers(a: int, b: int) -> int:
                return a + b

            model = Model(
                "claude-3-opus",
                provider="anthropic",
                web_search=False,
                function_tools=[sum_numbers],
            )
            result = model("2+3?")

        assert result == "5"
        assert mock_client.messages.create.call_count == 2

        second_call = mock_client.messages.create.call_args_list[1].kwargs
        second_messages = second_call["messages"]
        assert any(
            msg.get("role") == "user"
            and isinstance(msg.get("content"), list)
            and any(block.get("type") == "tool_result" for block in msg["content"] if isinstance(block, dict))
            for msg in second_messages
        )

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_google_function_tools(self):
        """Google provider should loop over function calls and function responses."""
        mock_client = MagicMock()

        class FakeFunctionDeclaration(dict):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        class FakeTool(dict):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        class FakeGenerateContentConfig:
            def __init__(self, tools=None):
                self.tools = tools or []

        fake_types = SimpleNamespace(
            FunctionDeclaration=FakeFunctionDeclaration,
            Tool=FakeTool,
            GenerateContentConfig=FakeGenerateContentConfig,
            GoogleSearch=lambda: None,
        )

        mock_genai = SimpleNamespace(
            Client=lambda api_key: mock_client,
            types=fake_types,
        )
        mock_google_module = SimpleNamespace(genai=mock_genai)

        response_1 = SimpleNamespace(
            text="",
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                function_call=SimpleNamespace(name="echo_value", args={"value": 7})
                            )
                        ]
                    )
                )
            ],
        )
        response_2 = SimpleNamespace(
            text="done",
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
        )
        mock_client.models.generate_content.side_effect = [response_1, response_2]

        with patch.dict(
            sys.modules,
            {"google": mock_google_module, "google.genai": mock_genai},
        ):
            from agenticblocks.models import Model

            def echo_value(value: int) -> int:
                return value

            model = Model(
                "gemini-2.0-flash",
                provider="google",
                web_search=False,
                function_tools=[echo_value],
            )
            result = model("call tool")

        assert result == "done"
        assert mock_client.models.generate_content.call_count == 2
        second_contents = mock_client.models.generate_content.call_args_list[1].kwargs["contents"]
        assert any(
            any("function_response" in part for part in item.get("parts", []))
            for item in second_contents
        )

    @patch.dict(os.environ, {"XAI_API_KEY": "test-key"})
    def test_xai_function_tools(self):
        """xAI provider should support function tools through the loop."""
        mock_client = MagicMock()
        mock_xai_chat_module = SimpleNamespace(
            system=lambda x: {"role": "system", "content": x},
            user=lambda x: {"role": "user", "content": x},
            assistant=lambda x: {"role": "assistant", "content": x},
        )
        mock_xai_module = SimpleNamespace(
            Client=lambda api_key: mock_client,
            chat=mock_xai_chat_module,
        )

        chat_1 = FakeXaiChat(
            [
                SimpleNamespace(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "double_value", "arguments": "{\"value\": 3}"},
                        }
                    ],
                )
            ]
        )
        chat_2 = FakeXaiChat([SimpleNamespace(content="6")])
        mock_client.chat.create.side_effect = [chat_1, chat_2]

        with patch.dict(
            sys.modules,
            {"xai_sdk": mock_xai_module, "xai_sdk.chat": mock_xai_chat_module},
        ):
            from agenticblocks.models import Model

            def double_value(value: int) -> int:
                return value * 2

            model = Model(
                "grok-2",
                provider="xai",
                web_search=False,
                function_tools=[double_value],
            )
            result = model("double 3")

        assert result == "6"
        assert mock_client.chat.create.call_count == 2

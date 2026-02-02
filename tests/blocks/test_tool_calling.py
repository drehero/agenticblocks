"""Tests for ToolCalling block."""
from __future__ import annotations

from agenticblocks.blocks import ToolCalling


class DummyModel:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.client_provider = "openai"

    def _create_chat_completion(self, *, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.responses.pop(0)


def test_tool_calling_executes_tool_and_returns_final():
    calls: list[tuple[int, int]] = []

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls.append((a, b))
        return a + b

    model = DummyModel(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "add",
                                        "arguments": '{"a": 1, "b": 2}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "done"}}]},
        ]
    )

    block = ToolCalling(model=model, tools=[add])
    result = block("Add numbers.")

    assert result == "done"
    assert calls == [(1, 2)]
    assert model.calls[0]["kwargs"]["tool_choice"] == "auto"
    tool_message = model.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["content"] == "3"


def test_tool_calling_reports_unknown_tool():
    model = DummyModel(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "missing", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "ok"}}]},
        ]
    )

    def ping() -> str:
        """Ping."""
        return "pong"

    block = ToolCalling(model=model, tools=[ping])
    result = block("Try tool.")

    assert result == "ok"
    tool_message = model.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "Unknown tool" in tool_message["content"]

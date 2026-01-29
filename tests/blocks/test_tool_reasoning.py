"""Tests for ToolReasoning block."""
from __future__ import annotations

from agenticblocks.blocks import ToolReasoning
from agenticblocks.trace import trace


class DummyModel:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.keep_history = False
        self.messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content})

    def reset_history(self, keep_system: bool = True):
        if keep_system:
            self.messages = [msg for msg in self.messages if msg["role"] == "system"][-1:]
        else:
            self.messages = []

    def __call__(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return '{"final":"done"}'


def test_invalid_json_retry_then_tool_call_and_final():
    calls: list[tuple[int, int]] = []

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls.append((a, b))
        return a + b

    model = DummyModel(
        [
            "not json",
            '{"tool":"add","kwargs":{"a":1,"b":2}}',
            '{"final":"ok"}',
        ]
    )

    block = ToolReasoning(model=model, tools=[add], max_time=None)
    result = block("Add two numbers.")

    assert result == "ok"
    assert calls == [(1, 2)]
    assert "ToolResult: 3" in model.prompts[-1]


def test_tool_error_is_reported_to_model():
    def boom() -> str:
        """Always raise an error."""
        raise ValueError("bad")

    model = DummyModel(
        [
            '{"tool":"boom","kwargs":{}}',
            '{"final":"handled"}',
        ]
    )

    block = ToolReasoning(model=model, tools=[boom], max_time=None)
    result = block("Trigger error.")

    assert result == "handled"
    assert "ValueError: bad" in model.prompts[-1]


def test_max_steps_caps_invalid_json_loop():
    def no_op() -> str:
        """No-op tool."""
        return "ok"

    model = DummyModel(["nope", "nope", "nope"])
    block = ToolReasoning(model=model, tools=[no_op], max_steps=2, max_time=None)
    result = block("Keep trying.")

    assert result == "Unable to complete the task within the budget."
    assert len(model.prompts) == 3


def test_tool_reasoning_traces_block():
    def no_op() -> str:
        """No-op tool."""
        return "ok"

    model = DummyModel(['{"final":"ok"}'])
    block = ToolReasoning(model=model, tools=[no_op])

    with trace() as t:
        result = block("Trace this.")

    assert result == "ok"
    assert len(t.root_spans) == 1
    assert t.root_spans[0].kind == "block"
    assert "ToolReasoning" in t.root_spans[0].name
    assert t.root_spans[0].output == "ok"

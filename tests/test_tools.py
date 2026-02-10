"""Tests for Tool wrapper behavior."""
import pytest

from agenticblocks.tools import Tool, normalize_function_tools


def test_tool_from_callable_infers_schema():
    def add(a: int, b: int = 1) -> int:
        """Add two integers."""
        return a + b

    tool = Tool.from_callable(add)
    schema = tool.to_openai_tool()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert schema["function"]["description"] == "Add two integers."
    assert schema["function"]["parameters"]["required"] == ["a"]
    assert schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert schema["function"]["parameters"]["properties"]["b"]["default"] == 1


def test_tool_execute_serializes_non_string_output():
    def payload() -> dict:
        return {"ok": True}

    tool = Tool.from_callable(payload)
    assert tool.execute({}) == "{\"ok\": true}"


def test_normalize_function_tools_rejects_duplicate_names():
    def one() -> str:
        return "one"

    def two() -> str:
        return "two"

    same_name = Tool.from_callable(two, name="one")
    with pytest.raises(ValueError, match="Duplicate tool name"):
        normalize_function_tools([one, same_name])

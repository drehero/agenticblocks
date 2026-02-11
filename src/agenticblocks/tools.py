from __future__ import annotations

import inspect
import json
import types
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union, get_args, get_origin


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Map a Python annotation to a basic JSON schema fragment."""
    if annotation is inspect._empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        if annotation is str:
            return {"type": "string"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        if annotation is bool:
            return {"type": "boolean"}
        if annotation in (dict,):
            return {"type": "object"}
        if annotation in (list, tuple, set):
            return {"type": "array"}
        return {"type": "string"}

    if origin in (list, tuple, set):
        item_schema = _annotation_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin in (dict,):
        return {"type": "object"}

    if origin in (types.UnionType, Union):
        non_none = [arg for arg in args if arg is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return _annotation_to_json_schema(non_none[0])
        return {"type": "string"}

    if origin is Literal:
        return {"enum": [arg for arg in args]}

    return {"type": "string"}


def _infer_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(f"Tool function '{func.__name__}' cannot use *args or **kwargs.")
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(f"Tool function '{func.__name__}' cannot use positional-only args.")

        schema = _annotation_to_json_schema(param.annotation)
        if param.default is not inspect._empty:
            try:
                json.dumps(param.default)
                schema["default"] = param.default
            except TypeError:
                pass
        else:
            required.append(name)
        properties[name] = schema

    payload: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        payload["required"] = required
    return payload


@dataclass(frozen=True)
class Tool:
    """Python callable wrapper that can be exposed as an OpenAI function tool."""

    func: Callable[..., Any]
    name: str
    description: str
    parameters: dict[str, Any]

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> "Tool":
        resolved_name = name or func.__name__
        doc = inspect.getdoc(func) or ""
        resolved_description = description or (doc.splitlines()[0] if doc else f"Execute {resolved_name}.")
        resolved_parameters = parameters or _infer_parameters_schema(func)
        return cls(
            func=func,
            name=resolved_name,
            description=resolved_description,
            parameters=resolved_parameters,
        )

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        result = self.func(**arguments)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


def normalize_function_tools(
    tools: list[Tool | Callable[..., Any]] | None,
) -> list[Tool]:
    if not tools:
        return []

    normalized: list[Tool] = []
    seen: set[str] = set()
    for tool in tools:
        parsed = tool if isinstance(tool, Tool) else Tool.from_callable(tool)
        if parsed.name in seen:
            raise ValueError(f"Duplicate tool name '{parsed.name}'.")
        seen.add(parsed.name)
        normalized.append(parsed)
    return normalized

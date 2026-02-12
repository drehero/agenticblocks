# Based on mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent)
# Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez
# Licensed under the MIT License
import os
import time
import uuid
import warnings
import json
from typing import Any, Callable, Literal
from urllib.parse import urlparse

import openai

from agenticblocks.models.stats import GLOBAL_MODEL_STATS
from agenticblocks.models.utils import apply_anthropic_cache_control, format_openai_messages
from agenticblocks.tools import Tool, normalize_function_tools
from agenticblocks.trace import span



class Model:
    """OpenAI-compatible model wrapper with optional history and tracing.

    Takes text as input and outputs text. Handles conversation history,
    provider inference, optional web search, and cost tracking when the
    provider reports it.

    Args:
        model_name: Model name or ID to use.
        model_kwargs: Default kwargs passed to the API on each call.
        system_prompt: Optional system prompt to prepend to conversations.
        keep_history: If True, maintains conversation history across calls.
            When enabled, provider-specific caching is automatically applied.
        web_search: If True, enables provider web search when available. For
            OpenRouter, toggles the ":online" model suffix.
        provider: Provider to use. If None, inferred from `api_url` or defaults
            to "openai". Supported providers: "openrouter", "openai", "google",
            "anthropic", "xai".
        api_url: API base URL. Defaults to provider base URL, or
            `{PROVIDER}_API_URL`, or `OPENAI_API_URL` for OpenAI.
        api_key: API key. Defaults to `{PROVIDER}_API_KEY` or `OPENAI_API_KEY`
            for OpenAI.
        function_tools: Optional list of Python callables or Tool objects
            available for OpenAI function-calling.
        max_tool_rounds: Maximum number of assistant->tool exchange rounds
            before the function loop exits.
        cost_tracking: Reserved for compatibility. Cost tracking only uses
            provider-reported values when available; otherwise cost=0.0 and a
            one-time warning is emitted.

    Raises:
        ValueError: If an API key is missing or the provider is unsupported.

    Example:
        >>> model = Model("openai/gpt-4o-mini")
        >>> model("Hello")
        '...'
    """

    def __init__(
            self,
            model_name: str,
            model_kwargs: dict[str, Any] = {},
            system_prompt: str | None = None,
            keep_history: bool = False,
            web_search: bool = True,
            provider: None | Literal["openrouter", "openai", "google", "anthropic", "xai"] = None,
            api_url: str | None = None,
            api_key: str | None = None,
            function_tools: list[Tool | Callable[..., Any]] | None = None,
            max_tool_rounds: int = 8,
            cost_tracking: Literal["default", "ignore_errors"] = "default",
        ) -> None:
        self.model_name = model_name
        self.web_search = web_search

        self.provider = provider.lower() if provider is not None else None
        if self.provider is None:
            self.provider = "openai"
            if api_url is not None:
                for provider in ["openrouter", "google", "anthropic", "xai"]:
                    if provider in api_url.lower():
                        self.provider = provider
                        break

        if self.provider == "openrouter":
            self.model_name = self._apply_openrouter_online_suffix(self.model_name)

        base_url = self._resolve_base_url(api_url)
        api_key = api_key or os.getenv(f"{self.provider.upper()}_API_KEY")
        if api_key is None and self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")

        if api_key is None:
            raise ValueError(f"No API key provided for provider {self.provider}. Please set the {self.provider.upper()}_API_KEY environment variable or pass it as the api_key argument.")

        self._init_client(base_url=base_url, api_key=api_key)
        self.base_url = base_url

        self.model_kwargs = model_kwargs
        self.cost_tracking = cost_tracking
        self.function_tools = normalize_function_tools(function_tools)
        self.max_tool_rounds = max_tool_rounds
        if self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be >= 1.")
        # Generate a unique conversation ID for xAI caching
        self._xai_conversation_id = str(uuid.uuid4())
        self.keep_history = keep_history
        self.cost = 0.0
        self.n_calls = 0
        self.messages = []
        self._warned_missing_cost = False

        if system_prompt is not None:
            self.add_message("system", system_prompt)


    def add_message(self, role: str, content: Any, **kwargs):
        """Append a message to the internal history."""
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def reset_history(self, keep_system: bool = True):
        """Reset message history.

        Args:
            keep_system: If True, keeps the last system message.
        """
        if keep_system:
            self.messages = [msg for msg in self.messages if msg["role"] == "system"][-1:]
        else:
            self.messages = []

    def __repr__(self):
        return f"Model({self.model_name!r})"

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the model with a prompt.

        Args:
            prompt: Input prompt text.
            **kwargs: Provider-specific request parameters.

        Returns:
            The model's response text.
        """
        trace_kwargs = dict(kwargs)
        if self.function_tools:
            trace_kwargs["function_tools"] = [tool.name for tool in self.function_tools]
            trace_kwargs["max_tool_rounds"] = self.max_tool_rounds
        with span(kind="model", name=repr(self), input=prompt, kwargs=trace_kwargs) as sp:
            resolved_tools = self.function_tools

            if self.keep_history:
                self.add_message("user", prompt)
                messages = self.messages
            else:
                # Keep a per-request transient history so tool-calling can
                # continue across rounds even when persistent history is off.
                messages = self.messages + [{"role": "user", "content": prompt}]

            content = ""
            total_cost = 0.0

            function_tool_loop = bool(resolved_tools)
            if function_tool_loop:
                content, total_cost = self._run_function_tool_loop(
                    messages=messages,
                    function_tools=resolved_tools,
                    max_tool_rounds=self.max_tool_rounds,
                    **kwargs,
                )
            else:
                response = self._create_chat_completion(messages=messages, **kwargs)
                total_cost = self._extract_response_cost(response)
                content = response["choices"][0]["message"]["content"] or ""
                if self.keep_history:
                    self.add_message("assistant", content)

            self.n_calls += 1
            self.cost += total_cost
            GLOBAL_MODEL_STATS.add(total_cost)

            sp.output = content
            return content

    def _append_conversation_message(
        self,
        messages: list[dict[str, Any]],
        role: str,
        content: Any,
        **kwargs: Any,
    ) -> None:
        if self.keep_history and messages is self.messages:
            self.add_message(role, content, **kwargs)
            return
        messages.append({"role": role, "content": content, **kwargs})

    def _extract_response_cost(self, response: dict[str, Any]) -> float:
        usage = response.get("usage", {})
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and cost > 0:
            return float(cost)
        if not self._warned_missing_cost:
            warnings.warn(
                "Cost tracking is not available for this provider/model. "
                "Continuing with cost=0.0.",
                stacklevel=2,
            )
            self._warned_missing_cost = True
        return 0.0

    def _run_function_tool_loop(
        self,
        *,
        messages: list[dict[str, Any]],
        function_tools: list[Tool],
        max_tool_rounds: int,
        **kwargs: Any,
    ) -> tuple[str, float]:
        tool_map = {tool.name: tool for tool in function_tools}
        total_cost = 0.0
        round_count = 0

        while True:
            response = self._create_chat_completion(
                messages=messages,
                function_tools=function_tools,
                **kwargs,
            )
            total_cost += self._extract_response_cost(response)

            choice = response.get("choices", [{}])[0] or {}
            assistant_message = choice.get("message", {}) or {}
            assistant_content = assistant_message.get("content") or ""
            tool_calls = assistant_message.get("tool_calls") or []

            assistant_extra: dict[str, Any] = {}
            if tool_calls:
                assistant_extra["tool_calls"] = tool_calls
            self._append_conversation_message(
                messages,
                "assistant",
                assistant_content,
                **assistant_extra,
            )

            if not tool_calls:
                return assistant_content, total_cost

            round_count += 1
            if round_count > max_tool_rounds:
                warnings.warn(
                    "Function tool loop reached max_tool_rounds; returning latest assistant text.",
                    stacklevel=2,
                )
                if assistant_content:
                    return assistant_content, total_cost
                return "Function tool loop reached max_tool_rounds.", total_cost

            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
                function_payload = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                tool_name = function_payload.get("name")
                raw_arguments = function_payload.get("arguments", "{}")

                tool_output = self._execute_function_tool(
                    tool_map=tool_map,
                    tool_name=tool_name,
                    raw_arguments=raw_arguments,
                )
                tool_kwargs = {}
                if tool_call_id:
                    tool_kwargs["tool_call_id"] = tool_call_id
                if tool_name:
                    tool_kwargs["name"] = tool_name
                self._append_conversation_message(
                    messages,
                    "tool",
                    tool_output,
                    **tool_kwargs,
                )

    @staticmethod
    def _execute_function_tool(
        *,
        tool_map: dict[str, Tool],
        tool_name: str | None,
        raw_arguments: Any,
    ) -> str:
        trace_input = Model._format_tool_trace_input(raw_arguments)
        trace_kwargs = {"tool_name": tool_name}
        trace_name = f"Tool({tool_name or 'unknown'})"

        with span(kind="model", name=trace_name, input=trace_input, kwargs=trace_kwargs) as sp:
            if not tool_name:
                error_payload = json.dumps({"error": "Tool call missing function name."})
                sp.error = "ToolError: Tool call missing function name."
                sp.output = error_payload
                return error_payload

            tool = tool_map.get(tool_name)
            if tool is None:
                error_payload = json.dumps({"error": f"Unknown tool '{tool_name}'."})
                sp.error = f"ToolError: Unknown tool '{tool_name}'."
                sp.output = error_payload
                return error_payload

            arguments: Any = raw_arguments
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    error_payload = json.dumps({"error": f"Invalid JSON arguments for '{tool_name}': {exc}"})
                    sp.error = f"JSONDecodeError: {exc}"
                    sp.output = error_payload
                    return error_payload
            if not isinstance(arguments, dict):
                error_payload = json.dumps({"error": f"Arguments for '{tool_name}' must be a JSON object."})
                sp.error = f"ToolError: Arguments for '{tool_name}' must be a JSON object."
                sp.output = error_payload
                return error_payload

            try:
                output = tool.execute(arguments)
                sp.output = output
                return output
            except Exception as exc:  # noqa: BLE001
                error_payload = json.dumps({"error": f"Tool '{tool_name}' failed: {exc}"})
                sp.error = f"{type(exc).__name__}: {exc}"
                sp.output = error_payload
                return error_payload

    @staticmethod
    def _format_tool_trace_input(raw_arguments: Any) -> str:
        if isinstance(raw_arguments, str):
            return raw_arguments
        try:
            return json.dumps(raw_arguments, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            return str(raw_arguments)

    def _resolve_base_url(self, api_url: str | None) -> str | None:
        default_base_urls = {
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "xai": "https://api.x.ai/v1",
            "anthropic": "https://api.anthropic.com",
            "google": "https://generativelanguage.googleapis.com/v1beta",
        }
        base_url = api_url or os.getenv(f"{self.provider.upper()}_API_URL")
        if base_url is None and self.provider == "openai":
            base_url = os.getenv("OPENAI_API_URL")
        return base_url or default_base_urls.get(self.provider)

    def _init_client(self, *, base_url: str | None, api_key: str) -> None:
        if self.provider in ["openai", "openrouter"]:
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
            self.client_provider = "openai"
            return
        if self.provider == "xai":
            from xai_sdk import Client  # type: ignore[import-not-found]
            self.client = Client(api_key=api_key)
            self.client_provider = "xai"
            return
        if self.provider == "google":
            from google import genai  # type: ignore[import-not-found]
            self.client = genai.Client(api_key=api_key)
            self.client_provider = "google"
            return
        if self.provider == "anthropic":
            import anthropic  # type: ignore[import-not-found]
            if base_url is None:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
            self.client_provider = "anthropic"
            return
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _create_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        function_tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.client_provider == "xai":
            merged_kwargs = self._apply_xai_web_search_kwargs(self.model_kwargs | kwargs)
            merged_kwargs = self._apply_openai_function_tools_kwargs(
                merged_kwargs,
                function_tools=function_tools,
            )
            chat_kwargs = self._filter_xai_kwargs(merged_kwargs)
            chat = self.client.chat.create(model=self.model_name, **chat_kwargs)
            self._append_xai_messages(chat, messages)
            response = chat.sample()
            content_text = getattr(response, "content", "") or ""
            raw_tool_calls = getattr(response, "tool_calls", None)
            if raw_tool_calls is None and isinstance(response, dict):
                raw_tool_calls = response.get("tool_calls")
            message: dict[str, Any] = {"content": content_text}
            tool_calls = self._normalize_openai_tool_calls(raw_tool_calls)
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {"choices": [{"message": message}], "usage": {}}
        if self.client_provider == "openai":
            merged_kwargs = self._apply_openai_web_search_kwargs(self.model_kwargs | kwargs)
            merged_kwargs = self._apply_openai_function_tools_kwargs(
                merged_kwargs,
                function_tools=function_tools,
            )
            formatted_messages = self._format_openai_messages(messages)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                **merged_kwargs,
            )
            return response.to_dict()
        if self.client_provider == "google":
            contents = self._build_google_contents(messages)
            merged_kwargs = self._apply_google_web_search_kwargs(self.model_kwargs | kwargs)
            merged_kwargs = self._apply_google_function_tools_kwargs(
                merged_kwargs,
                function_tools=function_tools,
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                **merged_kwargs,
            )
            usage = {}
            usage_metadata = getattr(response, "usage_metadata", None)
            if usage_metadata is not None:
                usage = {
                    "input_tokens": getattr(usage_metadata, "prompt_token_count", None),
                    "output_tokens": getattr(usage_metadata, "candidates_token_count", None),
                }
            content_text = getattr(response, "text", "") or ""
            message: dict[str, Any] = {"content": content_text}
            tool_calls = self._extract_google_tool_calls(response)
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {"choices": [{"message": message}], "usage": usage}
        if self.client_provider == "anthropic":
            system_prompt, cleaned_messages = self._build_anthropic_messages(messages)
            # Apply cache control when keep_history is enabled
            if self.keep_history:
                system_prompt, cleaned_messages = apply_anthropic_cache_control(
                    system_prompt, cleaned_messages
                )
            merged_kwargs = self._apply_anthropic_web_search_kwargs(self.model_kwargs | kwargs)
            merged_kwargs = self._apply_anthropic_function_tools_kwargs(
                merged_kwargs,
                function_tools=function_tools,
            )
            request_kwargs = {
                "model": self.model_name,
                "messages": cleaned_messages,
                **merged_kwargs,
            }
            if system_prompt is not None:
                request_kwargs["system"] = system_prompt
            response = self.client.messages.create(**request_kwargs)
            usage = {}
            if getattr(response, "usage", None) is not None:
                usage = {
                    "input_tokens": getattr(response.usage, "input_tokens", None),
                    "output_tokens": getattr(response.usage, "output_tokens", None),
                }
            content_text = self._extract_anthropic_text(response)
            message: dict[str, Any] = {"content": content_text}
            tool_calls = self._extract_anthropic_tool_calls(response)
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {"choices": [{"message": message}], "usage": usage}
        raise RuntimeError(f"Unsupported client provider: {self.client_provider}")

    def _build_google_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        system_messages = [msg["content"] for msg in messages if msg["role"] == "system" and isinstance(msg["content"], str)]
        system_prefix = ""
        if system_messages:
            system_prefix = "\n".join(system_messages)

        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "user" if msg["role"] in {"user", "tool"} else "model"
            parts: list[dict[str, Any]] = []

            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
                if role == "user" and system_prefix and msg["role"] == "user":
                    text = f"{system_prefix}\n\n{text}"
                    system_prefix = ""
                if text:
                    parts.append({"text": text})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        parts.append(part)

            if msg["role"] == "assistant":
                for tool_call in msg.get("tool_calls") or []:
                    function_payload = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    name = function_payload.get("name")
                    raw_arguments = function_payload.get("arguments", "{}")
                    arguments = self._parse_json_object(raw_arguments)
                    if name:
                        parts.append({"function_call": {"name": name, "args": arguments}})

            if msg["role"] == "tool":
                name = msg.get("name") or "tool"
                tool_response = msg.get("content", "")
                try:
                    parsed_response = json.loads(tool_response) if isinstance(tool_response, str) else tool_response
                except json.JSONDecodeError:
                    parsed_response = {"content": str(tool_response)}
                if not isinstance(parsed_response, dict):
                    parsed_response = {"content": parsed_response}
                parts = [{"function_response": {"name": name, "response": parsed_response}}]

            if not parts:
                parts = [{"text": ""}]
            contents.append({"role": role, "parts": parts})

        if system_prefix:
            contents.insert(0, {"role": "user", "parts": [{"text": system_prefix}]})
        return contents

    def _build_anthropic_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
        system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
        system_prompt = system_messages[-1] if system_messages else None
        system_blocks = None
        if system_prompt:
            system_blocks = [{"type": "text", "text": system_prompt}]

        cleaned_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                continue

            role = msg["role"]
            content = msg.get("content", "")

            if role == "tool":
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": str(content),
                }
                cleaned_messages.append({"role": "user", "content": [tool_result_block]})
                continue

            if role == "assistant":
                blocks: list[dict[str, Any]] = []
                if isinstance(content, str) and content:
                    blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    blocks.extend([block for block in content if isinstance(block, dict)])

                for tool_call in msg.get("tool_calls") or []:
                    function_payload = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    name = function_payload.get("name")
                    if not name:
                        continue
                    arguments = self._parse_json_object(function_payload.get("arguments", "{}"))
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id"),
                            "name": name,
                            "input": arguments,
                        }
                    )
                if not blocks:
                    blocks = [{"type": "text", "text": ""}]
                cleaned_messages.append({"role": "assistant", "content": blocks})
                continue

            cleaned_messages.append({"role": "user", "content": content})
        return system_blocks, cleaned_messages

    def _format_openai_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Format messages for OpenAI-compatible API.

        For OpenRouter with Anthropic models, applies cache control when keep_history is enabled.
        """
        return format_openai_messages(
            messages,
            keep_history=self.keep_history,
            apply_anthropic_cache=self._is_openrouter_anthropic_model(),
        )

    def _is_openrouter_anthropic_model(self) -> bool:
        """Check if using OpenRouter with an Anthropic model."""
        if self.provider != "openrouter":
            return False
        model_lower = self.model_name.lower()
        return "anthropic" in model_lower or "claude" in model_lower

    def _apply_openrouter_online_suffix(self, model_name: str) -> str:
        suffix = ":online"
        if self.web_search:
            return model_name if model_name.endswith(suffix) else f"{model_name}{suffix}"
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
        return model_name

    def _apply_openai_web_search_kwargs(self, merged_kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self.web_search or self.provider != "openai":
            return merged_kwargs
        if "web_search_options" in merged_kwargs:
            return merged_kwargs
        if not self._is_official_openai_base_url():
            return merged_kwargs
        merged_kwargs["web_search_options"] = {}
        return merged_kwargs

    def _is_official_openai_base_url(self) -> bool:
        if not self.base_url:
            return True
        host = urlparse(self.base_url).hostname
        if not host:
            return False
        return host == "api.openai.com" or host.endswith(".openai.com")

    @staticmethod
    def _apply_openai_function_tools_kwargs(
        merged_kwargs: dict[str, Any],
        *,
        function_tools: list[Tool] | None,
    ) -> dict[str, Any]:
        if not function_tools:
            return merged_kwargs
        function_tool_specs = [tool.to_openai_tool() for tool in function_tools]
        existing_tools = merged_kwargs.get("tools")
        if existing_tools is None:
            merged_kwargs["tools"] = function_tool_specs
            return merged_kwargs
        if isinstance(existing_tools, list):
            existing_tools.extend(function_tool_specs)
        return merged_kwargs

    def _apply_xai_web_search_kwargs(self, merged_kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self.web_search:
            return merged_kwargs
        try:
            from xai_sdk.tools import web_search  # type: ignore[import-not-found]
        except Exception:
            return merged_kwargs
        tools = merged_kwargs.get("tools")
        tool_instance = web_search()
        if tools is None:
            merged_kwargs["tools"] = [tool_instance]
        elif isinstance(tools, list):
            if not any(self._is_xai_web_search_tool(t) for t in tools):
                tools.append(tool_instance)
        return merged_kwargs

    @staticmethod
    def _filter_xai_kwargs(merged_kwargs: dict[str, Any]) -> dict[str, Any]:
        allowed = {"tools", "tool_choice", "temperature", "max_tokens", "top_p", "seed"}
        return {key: value for key, value in merged_kwargs.items() if key in allowed}

    @staticmethod
    def _append_xai_messages(chat: Any, messages: list[dict[str, Any]]) -> None:
        from xai_sdk import chat as xai_chat  # type: ignore[import-not-found]

        system_fn = getattr(xai_chat, "system", None)
        user_fn = getattr(xai_chat, "user", None)
        assistant_fn = getattr(xai_chat, "assistant", None)

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system" and system_fn is not None:
                chat.append(system_fn(content))
            elif role == "user" and user_fn is not None:
                chat.append(user_fn(content))
            elif (
                role == "assistant"
                and assistant_fn is not None
                and not msg.get("tool_calls")
                and isinstance(content, str)
            ):
                chat.append(assistant_fn(content))
            else:
                payload = {"role": role, "content": content}
                for key in ("tool_calls", "tool_call_id", "name"):
                    if key in msg:
                        payload[key] = msg[key]
                chat.append(payload)

    @staticmethod
    def _is_xai_web_search_tool(tool: Any) -> bool:
        tool_type = getattr(tool, "type", None)
        if tool_type is not None:
            return tool_type == "web_search"
        return getattr(tool, "name", None) == "web_search"

    def _apply_google_web_search_kwargs(self, merged_kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self.web_search:
            return merged_kwargs
        from google.genai import types  # type: ignore[import-not-found]

        tool = types.Tool(google_search=types.GoogleSearch())
        config = merged_kwargs.get("config")
        if config is None:
            merged_kwargs["config"] = types.GenerateContentConfig(tools=[tool])
            return merged_kwargs

        tools = None
        if hasattr(config, "tools"):
            tools = list(config.tools or [])
        elif isinstance(config, dict):
            tools = list(config.get("tools") or [])

        if tools is not None and not any(getattr(t, "google_search", None) for t in tools):
            tools.append(tool)
            if hasattr(config, "tools"):
                config.tools = tools
            elif isinstance(config, dict):
                config["tools"] = tools
        return merged_kwargs

    def _apply_google_function_tools_kwargs(
        self,
        merged_kwargs: dict[str, Any],
        *,
        function_tools: list[Tool] | None,
    ) -> dict[str, Any]:
        if not function_tools:
            return merged_kwargs

        from google.genai import types  # type: ignore[import-not-found]

        declarations = []
        for tool in function_tools:
            try:
                declarations.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )
                )
            except Exception:
                declarations.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                )

        try:
            function_tool = types.Tool(function_declarations=declarations)
        except Exception:
            function_tool = {"function_declarations": declarations}

        config = merged_kwargs.get("config")
        if config is None:
            try:
                merged_kwargs["config"] = types.GenerateContentConfig(tools=[function_tool])
            except Exception:
                merged_kwargs["config"] = {"tools": [function_tool]}
            return merged_kwargs

        tools = None
        if hasattr(config, "tools"):
            tools = list(config.tools or [])
        elif isinstance(config, dict):
            tools = list(config.get("tools") or [])

        if tools is not None:
            tools.append(function_tool)
            if hasattr(config, "tools"):
                config.tools = tools
            elif isinstance(config, dict):
                config["tools"] = tools
        return merged_kwargs

    def _apply_anthropic_web_search_kwargs(self, merged_kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self.web_search:
            return merged_kwargs
        return self._ensure_tool(
            merged_kwargs,
            {"type": "web_search_20250305", "name": "web_search"},
            tools_key="tools",
        )

    @staticmethod
    def _apply_anthropic_function_tools_kwargs(
        merged_kwargs: dict[str, Any],
        *,
        function_tools: list[Tool] | None,
    ) -> dict[str, Any]:
        if not function_tools:
            return merged_kwargs
        anthropic_tools = [tool.to_anthropic_tool() for tool in function_tools]
        existing_tools = merged_kwargs.get("tools")
        if existing_tools is None:
            merged_kwargs["tools"] = anthropic_tools
            return merged_kwargs
        if isinstance(existing_tools, list):
            existing_tools.extend(anthropic_tools)
        return merged_kwargs

    @staticmethod
    def _ensure_tool(
        merged_kwargs: dict[str, Any],
        tool: dict[str, Any],
        *,
        tools_key: str = "tools",
    ) -> dict[str, Any]:
        tools = merged_kwargs.get(tools_key)
        if tools is None:
            merged_kwargs[tools_key] = [tool]
            return merged_kwargs
        if not isinstance(tools, list):
            return merged_kwargs
        for existing in tools:
            if not isinstance(existing, dict) or existing.get("type") != tool.get("type"):
                continue
            for key, value in tool.items():
                if key not in existing:
                    existing[key] = value
            return merged_kwargs
        tools.append(tool)
        return merged_kwargs

    @staticmethod
    def _extract_anthropic_text(response: Any) -> str:
        content = getattr(response, "content", None)
        if not content:
            return ""
        text_parts = []
        for block in content:
            block_type = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
            if block_type != "text":
                continue
            block_text = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
            if block_text:
                text_parts.append(block_text)
        return "".join(text_parts)

    def _extract_anthropic_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        content = getattr(response, "content", None)
        if not content:
            return []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            block_type = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
            if block_type != "tool_use":
                continue
            tool_name = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
            if not tool_name:
                continue
            tool_call_id = getattr(block, "id", None) if not isinstance(block, dict) else block.get("id")
            if not tool_call_id:
                tool_call_id = f"tool_{uuid.uuid4().hex[:8]}"
            tool_input = getattr(block, "input", None) if not isinstance(block, dict) else block.get("input")
            if not isinstance(tool_input, dict):
                tool_input = {"value": self._coerce_jsonable(tool_input)}
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_input, ensure_ascii=False, default=str),
                    },
                }
            )
        return tool_calls

    def _extract_google_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []

        raw_function_calls = getattr(response, "function_calls", None)
        if raw_function_calls is None and isinstance(response, dict):
            raw_function_calls = response.get("function_calls")
        if raw_function_calls:
            for idx, function_call in enumerate(raw_function_calls, start=1):
                parsed = self._normalize_google_function_call(function_call, idx=idx)
                if parsed is not None:
                    tool_calls.append(parsed)
            return tool_calls

        candidates = getattr(response, "candidates", None)
        if candidates is None and isinstance(response, dict):
            candidates = response.get("candidates")
        if not candidates:
            return tool_calls
        for candidate in candidates:
            content = getattr(candidate, "content", None) if not isinstance(candidate, dict) else candidate.get("content")
            if content is None:
                continue
            parts = getattr(content, "parts", None) if not isinstance(content, dict) else content.get("parts")
            if not parts:
                continue
            for part in parts:
                function_call = getattr(part, "function_call", None) if not isinstance(part, dict) else part.get("function_call")
                if function_call is None:
                    continue
                parsed = self._normalize_google_function_call(function_call, idx=len(tool_calls) + 1)
                if parsed is not None:
                    tool_calls.append(parsed)
        return tool_calls

    def _normalize_google_function_call(self, function_call: Any, *, idx: int) -> dict[str, Any] | None:
        if isinstance(function_call, dict):
            name = function_call.get("name")
            arguments = function_call.get("args", {})
        else:
            name = getattr(function_call, "name", None)
            arguments = getattr(function_call, "args", {})
        if not name:
            return None
        plain_arguments = self._coerce_jsonable(arguments)
        if not isinstance(plain_arguments, dict):
            plain_arguments = {"value": plain_arguments}
        return {
            "id": f"google_call_{idx}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(plain_arguments, ensure_ascii=False, default=str),
            },
        }

    @staticmethod
    def _normalize_openai_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
        if not raw_tool_calls:
            return []
        if not isinstance(raw_tool_calls, list):
            return []

        normalized: list[dict[str, Any]] = []
        for idx, tool_call in enumerate(raw_tool_calls, start=1):
            if isinstance(tool_call, dict):
                function_payload = tool_call.get("function", {})
                name = function_payload.get("name")
                arguments = function_payload.get("arguments", "{}")
                if not name:
                    continue
                if isinstance(arguments, (dict, list)):
                    arguments = json.dumps(arguments, ensure_ascii=False, default=str)
                normalized.append(
                    {
                        "id": tool_call.get("id", f"call_{idx}"),
                        "type": tool_call.get("type", "function"),
                        "function": {"name": name, "arguments": arguments},
                    }
                )
                continue

            function_payload = getattr(tool_call, "function", None)
            if function_payload is None:
                continue
            name = getattr(function_payload, "name", None)
            arguments = getattr(function_payload, "arguments", "{}")
            if not name:
                continue
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False, default=str)
            normalized.append(
                {
                    "id": getattr(tool_call, "id", f"call_{idx}"),
                    "type": getattr(tool_call, "type", "function"),
                    "function": {"name": name, "arguments": arguments},
                }
            )
        return normalized

    @staticmethod
    def _parse_json_object(raw_value: Any) -> dict[str, Any]:
        if isinstance(raw_value, dict):
            return raw_value
        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def _coerce_jsonable(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): cls._coerce_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._coerce_jsonable(item) for item in value]
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return cls._coerce_jsonable(to_dict())
            except Exception:
                return str(value)
        return str(value)

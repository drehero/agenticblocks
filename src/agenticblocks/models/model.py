# Based on mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent)
# Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez
# Licensed under the MIT License
import os
import time
import uuid
import warnings
from typing import Any, Literal

import openai
import httpx

from agenticblocks.models.stats import GLOBAL_MODEL_STATS
from agenticblocks.models.utils import apply_anthropic_cache_control, format_openai_messages
from agenticblocks.trace import span



class Model:
    f"""A wrapper for OpenAI-compatible API models.

    Takes text as input and outputs text. Handles conversation history,
    cost tracking (when available), and automatic cache control.

    Args:
        model_name: The name/ID of the model to use.
        model_kwargs: Default kwargs passed to the API on each call.
        system_prompt: Optional system prompt to prepend to conversations.
        keep_history: If True, maintains conversation history across calls.
            When enabled, provider-specific caching is automatically applied.
        web_search: If True, enables web search. For OpenRouter, toggles the
            ":online" model suffix. For other providers, enables their
            web search tool where supported.
        provider: The provider to use. If None or api_url is provided, the provider is inferred from the api_url otherwise it behaves like openai.
            Supported providers: "openrouter", "openai", "google", "anthropic", "xai".
        api_url: API base URL. Defaults to the provider's base URL, {{PROVIDER_NAME}}_API_URL env var or the OPENAI_API_URL env var.
        api_key: API key. Defaults to {{PROVIDER_NAME}}_API_KEY env var or OPENAI_API_KEY env var.
        cost_tracking: Reserved for compatibility. Cost tracking only uses
            provider-reported values when available; otherwise cost=0.0 and a
            one-time warning is emitted.
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

        self.model_kwargs = model_kwargs
        self.cost_tracking = cost_tracking
        # Generate a unique conversation ID for xAI caching
        self._xai_conversation_id = str(uuid.uuid4())
        self.keep_history = keep_history
        self.cost = 0.0
        self.n_calls = 0
        self.messages = []
        self._warned_missing_cost = False

        if system_prompt is not None:
            self.add_message("system", system_prompt)


    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def reset_history(self, keep_system: bool = True):
        """Reset message history, keeping only the system messages if keep_system is True."""
        if keep_system:
            self.messages = [msg for msg in self.messages if msg["role"] == "system"][-1:]
        else:
            self.messages = []

    def __repr__(self):
        return f"Model({self.model_name!r})"

    def __call__(self, prompt: str, **kwargs) -> str:
        with span(kind="model", name=repr(self), input=prompt, kwargs=dict(kwargs)) as sp:
            if self.keep_history:
                self.add_message("user", prompt)
                messages = self.messages
            else:
                messages = self.messages + [{"role": "user", "content": prompt}]

            response = self._create_chat_completion(messages=messages, **kwargs)

            usage = response.get("usage", {})
            cost = usage.get("cost")
            if not isinstance(cost, (int, float)) or cost <= 0:
                cost = 0.0
                if not self._warned_missing_cost:
                    warnings.warn(
                        "Cost tracking is not available for this provider/model. "
                        "Continuing with cost=0.0.",
                        stacklevel=2,
                    )
                    self._warned_missing_cost = True

            self.n_calls += 1
            self.cost += cost
            GLOBAL_MODEL_STATS.add(cost)

            content = response["choices"][0]["message"]["content"] or ""

            if self.keep_history:
                self.add_message("assistant", content)

            sp.output = content
            return content

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

    def _create_chat_completion(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if self.client_provider == "xai":
            merged_kwargs = self._apply_xai_web_search_kwargs(self.model_kwargs | kwargs)
            chat_kwargs = self._filter_xai_kwargs(merged_kwargs)
            chat = self.client.chat.create(model=self.model_name, **chat_kwargs)
            self._append_xai_messages(chat, messages)
            response = chat.sample()
            content_text = getattr(response, "content", "") or ""
            return {"choices": [{"message": {"content": content_text}}], "usage": {}}
        if self.client_provider == "openai":
            merged_kwargs = self._apply_openai_web_search_kwargs(self.model_kwargs | kwargs)
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
            return {"choices": [{"message": {"content": content_text}}], "usage": usage}
        if self.client_provider == "anthropic":
            system_prompt, cleaned_messages = self._build_anthropic_messages(messages)
            # Apply cache control when keep_history is enabled
            if self.keep_history:
                system_prompt, cleaned_messages = apply_anthropic_cache_control(
                    system_prompt, cleaned_messages
                )
            merged_kwargs = self._apply_anthropic_web_search_kwargs(self.model_kwargs | kwargs)
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
            return {"choices": [{"message": {"content": content_text}}], "usage": usage}
        raise RuntimeError(f"Unsupported client provider: {self.client_provider}")

    def _build_google_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
        system_prefix = ""
        if system_messages:
            system_prefix = "\n".join(system_messages)

        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "user" if msg["role"] == "user" else "model"
            text = msg["content"]
            if role == "user" and system_prefix:
                text = f"{system_prefix}\n\n{text}"
                system_prefix = ""
            contents.append({"role": role, "parts": [{"text": text}]})

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
        cleaned_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] != "system"
        ]
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
        if not self.web_search or self.provider == "openrouter":
            return merged_kwargs
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
            elif role == "assistant" and assistant_fn is not None:
                chat.append(assistant_fn(content))
            else:
                chat.append({"role": role, "content": content})

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

    def _apply_anthropic_web_search_kwargs(self, merged_kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self.web_search:
            return merged_kwargs
        return self._ensure_tool(
            merged_kwargs,
            {"type": "web_search_20250305", "name": "web_search"},
            tools_key="tools",
        )

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


class LocalModel:
    """A local model wrapper for OpenAI-compatible runtimes like Ollama and vLLM."""

    def __init__(
        self,
        model_name: str,
        provider: Literal["ollama", "vllm"] = "ollama",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        keep_history: bool = False,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.base_url = base_url or self._default_base_url(provider)
        self.api_key = api_key or self._default_api_key(provider)
        self.model_kwargs = model_kwargs or {}
        self.keep_history = keep_history
        self.messages: list[dict[str, Any]] = []
        self.cost = 0.0
        self.n_calls = 0

        self._client = client or httpx.Client(timeout=timeout)
        self._server_checked = False

        if system_prompt is not None:
            self.add_message("system", system_prompt)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self.messages.append({"role": role, "content": content, **kwargs})

    def reset_history(self, keep_system: bool = True) -> None:
        if keep_system:
            self.messages = [msg for msg in self.messages if msg["role"] == "system"][-1:]
        else:
            self.messages = []

    def __repr__(self) -> str:
        return f"LocalModel({self.model_name!r}, provider={self.provider!r})"

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        with span(kind="model", name=repr(self), input=prompt, kwargs=dict(kwargs)) as sp:
            self._ensure_server_available()
            if self.keep_history:
                self.add_message("user", prompt)
                messages = self.messages
            else:
                messages = self.messages + [{"role": "user", "content": prompt}]

            response = self._create_chat_completion(messages=messages, **kwargs)

            self.n_calls += 1
            GLOBAL_MODEL_STATS.add(0.0)

            content = response["choices"][0]["message"]["content"] or ""

            if self.keep_history:
                self.add_message("assistant", content)

            sp.output = content
            return content

    def _create_chat_completion(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        formatted_messages = format_openai_messages(messages)
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            **self.model_kwargs,
            **kwargs,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self._chat_completions_url()
        response = self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def _chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        return f"{base}/chat/completions"

    def _ensure_server_available(self) -> None:
        if self._server_checked:
            return
        url = self._health_check_url()
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # noqa: BLE001
            if self.provider == "ollama":
                hint = (
                    "Start Ollama with: `ollama serve` and ensure the model is pulled "
                    f"with: `ollama run {self.model_name}`."
                )
            elif self.provider == "vllm":
                hint = (
                    "Start vLLM's OpenAI-compatible server, e.g.: "
                    f"`python -m vllm.entrypoints.openai.api_server --model {self.model_name}`."
                )
            else:
                hint = "Start your local model server and verify the base URL."
            raise ConnectionError(
                "Local model server is not reachable. "
                f"{hint}\n"
                f"Health check failed for: {url}"
            ) from exc
        self._server_checked = True

    def _health_check_url(self) -> str:
        base = self.base_url.rstrip("/")
        if self.provider == "ollama":
            root = base[:-3] if base.endswith("/v1") else base
            return f"{root}/api/tags"
        if self.provider == "vllm":
            return f"{base}/models"
        raise ValueError(f"Unsupported local provider: {self.provider}")

    @staticmethod
    def _default_base_url(provider: str) -> str:
        env_url = os.getenv(f"{provider.upper()}_API_URL")
        if env_url:
            return env_url
        if provider == "ollama":
            return "http://localhost:11434/v1"
        if provider == "vllm":
            return "http://localhost:8000/v1"
        raise ValueError(f"Unsupported local provider: {provider}")

    @staticmethod
    def _default_api_key(provider: str) -> str:
        env_key = os.getenv(f"{provider.upper()}_API_KEY")
        if env_key:
            return env_key
        if provider == "ollama":
            return "ollama"
        if provider == "vllm":
            return "EMPTY"
        raise ValueError(f"Unsupported local provider: {provider}")

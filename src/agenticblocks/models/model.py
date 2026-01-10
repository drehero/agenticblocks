# Based on mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent)
# Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez
# Licensed under the MIT License
import os
import time
from typing import Any, Literal

import openai

from agenticblocks.models.stats import GLOBAL_MODEL_STATS
from agenticblocks.models.utils import set_cache_control
from agenticblocks.trace import span


class Model:
    """A wrapper for OpenAI-compatible API models.

    Takes text as input and outputs text. Handles conversation history,
    cost tracking, and cache control.

    Args:
        model_name: The name/ID of the model to use.
        model_kwargs: Default kwargs passed to the API on each call.
        system_prompt: Optional system prompt to prepend to conversations.
        keep_history: If True, maintains conversation history across calls.
        api_url: API base URL. Defaults to OPENAI_API_URL env var.
        api_key: API key. Defaults to OPENAI_API_KEY env var.
        set_cache_control: Cache control mode for prompt caching.
        cost_tracking: How to handle cost tracking. "default" raises on missing cost,
            "ignore_errors" silently continues.
    """

    def __init__(
            self,
            model_name: str,
            model_kwargs: dict[str, Any] = {},
            system_prompt: str | None = None,
            keep_history: bool = False,
            api_url: str | None = None,
            api_key: str | None = None,
            set_cache_control: Literal["default_end"] | None = None,
            cost_tracking: Literal["default", "ignore_errors"] = "default",
        ) -> None:
        self.model_name = model_name
        self.client = openai.OpenAI(
            base_url=api_url if api_url is not None else os.getenv("OPENAI_API_URL"),
            api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"),
        )
        self.model_kwargs = model_kwargs
        self.set_cache_control = set_cache_control
        self.cost_tracking = cost_tracking
        self.keep_history = keep_history
        self.cost = 0.0
        self.n_calls = 0
        self.messages = []

        if system_prompt is not None:
            self.add_message("system", system_prompt)


    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def reset_history(self):
        """Reset message history, keeping only the system message if present."""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages[-1:] if system_messages else []

    def __repr__(self):
        return f"Model({self.model_name!r})"

    def __call__(self, prompt: str, **kwargs) -> str:
        with span(kind="model", name=repr(self), input=prompt, kwargs=dict(kwargs)) as sp:
            if self.keep_history:
                self.add_message("user", prompt)
                messages = self.messages
            else:
                messages = self.messages + [{"role": "user", "content": prompt}]

            if self.set_cache_control:
                messages = set_cache_control(messages, mode=self.set_cache_control)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
                **(self.model_kwargs | kwargs),
            ).to_dict()

            usage = response.get("usage", {})
            cost = usage.get("cost", 0.0)
            if cost <= 0.0 and self.cost_tracking != "ignore_errors":
                raise RuntimeError(
                    f"No valid cost information available from the API response for model {self.model_name}: "
                    f"Usage {usage}, cost {cost}. Cost must be > 0.0. Set cost_tracking: 'ignore_errors'"
                )

            self.n_calls += 1
            self.cost += cost
            GLOBAL_MODEL_STATS.add(cost)

            content = response["choices"][0]["message"]["content"] or ""

            if self.keep_history:
                self.add_message("assistant", content)

            sp.output = content
            return content

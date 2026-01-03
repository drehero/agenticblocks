# Based on mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent)
# Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez
# Licensed under the MIT License
import os
import time
from typing import Any, Literal

import openai

from agenticblocks.models.stats import GLOBAL_MODEL_STATS
from agenticblocks.models.utils import set_cache_control


class Model:
    def __init__(
            self,
            model_name: str,
            model_kwargs: dict[str, Any] = {},
            system_prompt: str | None = None,
            build_message_history: bool = False,
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
        self.build_message_history = build_message_history
        self.cost = 0.0
        self.n_calls = 0
        self.messages = []

        if system_prompt is not None:
            self.add_message("system", system_prompt)


    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})


    def __call__(self, prompt, **kwargs) -> dict:
        if self.build_message_history:
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
                f"Usage {usage}, cost {cost}. Cost must be > 0.0. Set cost_tracking: 'ignore_errors' in your config file or "
            )

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        content = response["choices"][0]["message"]["content"] or ""

        if self.build_message_history:
            self.add_message("assistant", content)

        return {
            "content": content,
            "extra": {
                "response": response,  # already is json
            },
        }

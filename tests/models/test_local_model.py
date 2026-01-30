"""Tests for LocalModel."""
from __future__ import annotations

import json

import httpx

from agenticblocks.models.model import LocalModel


def test_local_model_posts_to_default_ollama_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "http://localhost:11434/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "llama3.1"
        assert payload["messages"][-1]["content"] == "hello"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    def router(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://localhost:11434/api/tags":
            return httpx.Response(200, json={"models": []})
        return handler(request)

    transport = httpx.MockTransport(router)
    client = httpx.Client(transport=transport)
    model = LocalModel("llama3.1", provider="ollama", client=client)

    assert model("hello") == "ok"


def test_local_model_keep_history_tracks_messages():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    def router(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://localhost:11434/api/tags":
            return httpx.Response(200, json={"models": []})
        return handler(request)

    transport = httpx.MockTransport(router)
    client = httpx.Client(transport=transport)
    model = LocalModel("llama3.1", provider="ollama", client=client, keep_history=True)

    model("first")
    model("second")

    # user/assistant per call
    assert [m["role"] for m in model.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]

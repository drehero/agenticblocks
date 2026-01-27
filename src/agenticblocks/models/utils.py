# Based on mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent)
# Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez
# Licensed under the MIT License
import copy


def apply_anthropic_cache_control(
    system_prompt: list[dict] | None,
    messages: list[dict],
) -> tuple[list[dict] | None, list[dict]]:
    """Apply Anthropic cache control to system prompt and messages.

    Uses 2 cache breakpoints:
    1. System prompt (if present)
    2. Message before the last user message (conversation prefix)

    The last user message is NOT cached to allow cache hits on the prefix.

    Args:
        system_prompt: The system prompt blocks (list of dicts with type/text),
                      or None if no system prompt.
        messages: List of message dicts with role and content.

    Returns:
        Tuple of (cached_system_prompt, cached_messages).
    """
    # Deep copy to avoid mutating originals
    system_prompt = copy.deepcopy(system_prompt) if system_prompt else None
    messages = copy.deepcopy(messages)

    # Cache breakpoint 1: System prompt
    if system_prompt:
        # Add cache_control to the last block in system prompt
        system_prompt[-1]["cache_control"] = {"type": "ephemeral"}

    # Cache breakpoint 2: Message before last user message
    # Find the index of the last user message
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    # Cache the message before the last user message (if it exists)
    if last_user_idx is not None and last_user_idx > 0:
        prefix_idx = last_user_idx - 1
        _add_cache_control_to_message(messages[prefix_idx])

    return system_prompt, messages


def _add_cache_control_to_message(message: dict) -> None:
    """Add cache_control to a message, handling both string and list content formats."""
    content = message.get("content")

    if isinstance(content, str):
        # Convert string content to list format with cache_control
        message["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list) and content:
        # Add cache_control to the last content block
        content[-1]["cache_control"] = {"type": "ephemeral"}

from __future__ import annotations

from typing import Any, Dict, Optional

from agenticblocks.trace import span


class Block:
    """Base class for blocks with opt-in tracing.

    A Block is a callable unit that transforms a prompt into a response.
    Subclasses implement `forward` and can rely on `__call__` to wrap the
    invocation in a trace span.

    Subclasses should be pure functions of their inputs unless they
    intentionally manage state (e.g., internal history).

    Example:
        >>> class Echo(Block):
        ...     def forward(self, prompt: str, **kwargs: Any) -> str:
        ...         return prompt
        >>> Echo()("hello")
        'hello'
    """

    def forward(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover
        """Run the block logic.

        Args:
            prompt: Input prompt text.
            **kwargs: Optional parameters forwarded to the block logic.

        Returns:
            The generated response text.
        """
        raise NotImplementedError

    @property
    def trace_name(self) -> str:
        """Name used in tracing spans."""
        # Prefer repr to include key configuration (e.g., model name).
        try:
            return repr(self)
        except Exception:  # noqa: BLE001
            return self.__class__.__name__

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Call the block and emit a trace span if tracing is active."""
        trace_kwargs: Dict[str, Any] = dict(kwargs)
        with span(kind="block", name=self.trace_name, input=prompt, kwargs=trace_kwargs) as sp:
            out = self.forward(prompt, **kwargs)
            sp.output = out
            return out

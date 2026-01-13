from __future__ import annotations

from typing import Any, Dict, Optional

from agenticblocks.trace import span


class Block:
    """Base class for blocks with opt-in tracing.

    Users should implement `forward(prompt, **kwargs)`.
    """

    def forward(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError

    @property
    def trace_name(self) -> str:
        # Prefer repr to include key configuration (e.g., model name).
        try:
            return repr(self)
        except Exception:  # noqa: BLE001
            return self.__class__.__name__

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        trace_kwargs: Dict[str, Any] = dict(kwargs)
        with span(kind="block", name=self.trace_name, input=prompt, kwargs=trace_kwargs) as sp:
            out = self.forward(prompt, **kwargs)
            sp.output = out
            return out


from __future__ import annotations

import contextlib
import contextvars
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional

SpanKind = Literal["block", "model", "tool"]


@dataclass
class TraceSpan:
    """A single traced span for a block or model call.

    Attributes:
        id: Unique span ID.
        name: Span name (typically repr of the block/model).
        kind: "block", "model", or "tool".
        start_time: Unix timestamp at start.
        end_time: Unix timestamp at end (None if still running).
        input: Optional input prompt.
        output: Optional output text.
        kwargs: Optional call kwargs captured at start.
        error: Optional error string if an exception occurred.
        children: Nested spans.
    """
    id: str
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    input: Optional[str] = None
    output: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    children: List["TraceSpan"] = field(default_factory=list)

    @property
    def duration_s(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self, *, keys: List[str] | None = None) -> Dict[str, Any]:
        """Serialize the span to a dict.

        Args:
            keys: Optional list of keys to include (children are always included).

        Returns:
            A dict representation of the span.
        """
        payload = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.duration_s,
            "input": self.input,
            "output": self.output,
            "kwargs": self.kwargs,
            "error": self.error,
            "children": [c.to_dict(keys=keys) for c in self.children],
        }
        if keys is None:
            return payload
        return {
            key: value
            for key, value in payload.items()
            if key in keys or key == "children"
        }


@dataclass
class Trace:
    """A collection of root spans captured within a trace context."""
    root_spans: List[TraceSpan] = field(default_factory=list)

    def to_dict(self, *, keys: List[str] | None = None) -> Dict[str, Any]:
        """Serialize the trace to a dict.

        Args:
            keys: Optional list of keys to include on spans.

        Returns:
            A dict with a single "spans" key.
        """
        payload = {"spans": [s.to_dict(keys=keys) for s in self.root_spans]}
        return payload

    def to_json(self, *, indent: int = 2, keys: List[str] | None = None) -> str:
        """Serialize the trace to JSON."""
        return json.dumps(self.to_dict(keys=keys), indent=indent, ensure_ascii=False)

    def pretty(self) -> str:
        """Render a human-readable tree of spans."""
        lines: List[str] = []

        def _fmt(span: TraceSpan, depth: int) -> None:
            dur = span.duration_s
            dur_txt = f"{dur:.3f}s" if dur is not None else "running"
            head = f"{'  ' * depth}- [{span.kind}] {span.name} ({dur_txt})"
            lines.append(head)
            if span.error:
                lines.append(f"{'  ' * (depth + 1)}error: {span.error}")
            if span.input is not None:
                lines.append(f"{'  ' * (depth + 1)}in: {span.input}")
            if span.output is not None:
                lines.append(f"{'  ' * (depth + 1)}out: {span.output}")
            for child in span.children:
                _fmt(child, depth + 1)

        for s in self.root_spans:
            _fmt(s, 0)
        return "\n".join(lines)


_active_trace: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar("agenticblocks_active_trace", default=None)
_span_stack: contextvars.ContextVar[List[TraceSpan]] = contextvars.ContextVar("agenticblocks_span_stack", default=[])


def get_active_trace() -> Optional[Trace]:
    """Return the active trace for the current context, if any."""
    return _active_trace.get()


def _new_span_id() -> str:
    return uuid.uuid4().hex


def _push_span(span: TraceSpan) -> None:
    stack = _span_stack.get()
    if stack:
        stack[-1].children.append(span)
    else:
        trace = _active_trace.get()
        if trace is not None:
            trace.root_spans.append(span)
    stack.append(span)


def _pop_span() -> Optional[TraceSpan]:
    stack = _span_stack.get()
    if not stack:
        return None
    return stack.pop()


@contextlib.contextmanager
def span(
    *,
    kind: SpanKind,
    name: str,
    input: Optional[str] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Iterator[TraceSpan]:
    """Create a traced span iff tracing is active; otherwise yields a dummy span."""
    trace = _active_trace.get()
    if trace is None:
        dummy = TraceSpan(id="__not_traced__", name=name, kind=kind, start_time=time.time())
        try:
            yield dummy
        finally:
            pass
        return

    sp = TraceSpan(
        id=_new_span_id(),
        name=name,
        kind=kind,
        start_time=time.time(),
        input=input,
        kwargs=kwargs or {},
    )
    _push_span(sp)
    try:
        yield sp
    except Exception as e:  # noqa: BLE001
        sp.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        sp.end_time = time.time()
        _pop_span()


@contextlib.contextmanager
def trace() -> Iterator[Trace]:
    """Opt-in tracing context for block/model calls.

    Example:
        >>> with trace() as t:
        ...     model = Model("openai/gpt-4o-mini")  # doctest: +SKIP
        ...     model("Hello")  # doctest: +SKIP
        >>> isinstance(t, Trace)
        True
    """
    t = Trace()
    token_trace = _active_trace.set(t)
    token_stack = _span_stack.set([])
    try:
        yield t
    finally:
        _span_stack.reset(token_stack)
        _active_trace.reset(token_trace)

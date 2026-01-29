"""Tests for base Block class."""
from unittest.mock import MagicMock

import pytest

from agenticblocks.block import Block


class ConcreteBlock(Block):
    """Concrete implementation of Block for testing."""

    def __init__(self, return_value="test output"):
        self.return_value = return_value
        self.forward_calls = []

    def forward(self, prompt: str, **kwargs) -> str:
        self.forward_calls.append((prompt, kwargs))
        return self.return_value

    def __repr__(self):
        return f"ConcreteBlock({self.return_value!r})"


class TestBlockBase:
    """Tests for Block base class."""

    def test_forward_not_implemented(self):
        """forward() should raise NotImplementedError in base class."""
        block = Block()
        with pytest.raises(NotImplementedError):
            block.forward("test")

    def test_call_invokes_forward(self):
        """__call__ should invoke forward."""
        block = ConcreteBlock(return_value="hello")

        result = block("test prompt")

        assert result == "hello"
        assert len(block.forward_calls) == 1
        assert block.forward_calls[0][0] == "test prompt"

    def test_call_passes_kwargs_to_forward(self):
        """__call__ should pass kwargs to forward."""
        block = ConcreteBlock()

        block("prompt", temperature=0.5, max_tokens=100)

        assert block.forward_calls[0][1] == {"temperature": 0.5, "max_tokens": 100}


class TestBlockTraceName:
    """Tests for trace_name property."""

    def test_trace_name_uses_repr(self):
        """trace_name should use __repr__ when available."""
        block = ConcreteBlock(return_value="test")

        assert block.trace_name == "ConcreteBlock('test')"

    def test_trace_name_fallback_to_class_name(self):
        """trace_name should fallback to class name if repr raises."""

        class BrokenReprBlock(Block):
            def forward(self, prompt, **kwargs):
                return "output"

            def __repr__(self):
                raise ValueError("broken repr")

        block = BrokenReprBlock()
        assert block.trace_name == "BrokenReprBlock"


class TestBlockTracing:
    """Tests for Block tracing integration."""

    def test_call_creates_span_when_tracing(self):
        """__call__ should create a span when tracing is active."""
        from agenticblocks.trace import trace

        block = ConcreteBlock(return_value="traced output")

        with trace() as t:
            result = block("traced input")

        assert result == "traced output"
        assert len(t.root_spans) == 1

        span = t.root_spans[0]
        assert span.kind == "block"
        assert span.name == "ConcreteBlock('traced output')"
        assert span.input == "traced input"
        assert span.output == "traced output"

    def test_call_records_kwargs_in_span(self):
        """__call__ should record kwargs in span."""
        from agenticblocks.trace import trace

        block = ConcreteBlock()

        with trace() as t:
            block("prompt", temperature=0.7, top_p=0.9)

        span = t.root_spans[0]
        assert span.kwargs == {"temperature": 0.7, "top_p": 0.9}

    def test_no_span_without_tracing(self):
        """__call__ should not fail when tracing is not active."""
        block = ConcreteBlock(return_value="no trace")

        # Should work without trace context
        result = block("test")

        assert result == "no trace"

    def test_error_recorded_in_span(self):
        """Exceptions should be recorded in span."""
        from agenticblocks.trace import trace

        class ErrorBlock(Block):
            def forward(self, prompt, **kwargs):
                raise ValueError("test error")

        block = ErrorBlock()

        with trace() as t:
            with pytest.raises(ValueError, match="test error"):
                block("input")

        span = t.root_spans[0]
        assert span.error == "ValueError: test error"

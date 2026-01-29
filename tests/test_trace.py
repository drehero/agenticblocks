"""Tests for tracing functionality."""
import time

import pytest

from agenticblocks.trace import (
    Trace,
    TraceSpan,
    span,
    trace,
    get_active_trace,
)


class TestTraceSpan:
    """Tests for TraceSpan dataclass."""

    def test_span_creation(self):
        """TraceSpan should be created with required fields."""
        sp = TraceSpan(
            id="test-id",
            name="test-span",
            kind="block",
            start_time=1000.0,
        )

        assert sp.id == "test-id"
        assert sp.name == "test-span"
        assert sp.kind == "block"
        assert sp.start_time == 1000.0
        assert sp.end_time is None
        assert sp.input is None
        assert sp.output is None
        assert sp.error is None
        assert sp.children == []
        assert sp.kwargs == {}

    def test_duration_none_when_not_ended(self):
        """duration_s should be None when end_time is not set."""
        sp = TraceSpan(
            id="test", name="test", kind="block", start_time=1000.0
        )

        assert sp.duration_s is None

    def test_duration_calculated_correctly(self):
        """duration_s should calculate end_time - start_time."""
        sp = TraceSpan(
            id="test",
            name="test",
            kind="block",
            start_time=1000.0,
            end_time=1002.5,
        )

        assert sp.duration_s == pytest.approx(2.5)

    def test_to_dict_all_fields(self):
        """to_dict should include all fields."""
        sp = TraceSpan(
            id="span-1",
            name="my-span",
            kind="model",
            start_time=100.0,
            end_time=105.0,
            input="input text",
            output="output text",
            kwargs={"temp": 0.7},
            error="some error",
        )

        d = sp.to_dict()

        assert d["id"] == "span-1"
        assert d["name"] == "my-span"
        assert d["kind"] == "model"
        assert d["start_time"] == 100.0
        assert d["end_time"] == 105.0
        assert d["duration_s"] == pytest.approx(5.0)
        assert d["input"] == "input text"
        assert d["output"] == "output text"
        assert d["kwargs"] == {"temp": 0.7}
        assert d["error"] == "some error"
        assert d["children"] == []

    def test_to_dict_with_children(self):
        """to_dict should include nested children."""
        child = TraceSpan(
            id="child", name="child-span", kind="block", start_time=101.0
        )
        parent = TraceSpan(
            id="parent",
            name="parent-span",
            kind="block",
            start_time=100.0,
            children=[child],
        )

        d = parent.to_dict()

        assert len(d["children"]) == 1
        assert d["children"][0]["id"] == "child"

    def test_to_dict_with_keys_filter(self):
        """to_dict with keys should only include specified fields."""
        sp = TraceSpan(
            id="test",
            name="test-span",
            kind="block",
            start_time=100.0,
            input="input",
            output="output",
        )

        d = sp.to_dict(keys=["id", "name"])

        assert "id" in d
        assert "name" in d
        assert "children" in d  # children always included
        assert "input" not in d
        assert "output" not in d


class TestTrace:
    """Tests for Trace class."""

    def test_empty_trace(self):
        """New Trace should have empty root_spans."""
        t = Trace()
        assert t.root_spans == []

    def test_to_dict(self):
        """to_dict should return spans list."""
        sp = TraceSpan(id="1", name="span", kind="block", start_time=100.0)
        t = Trace(root_spans=[sp])

        d = t.to_dict()

        assert "spans" in d
        assert len(d["spans"]) == 1
        assert d["spans"][0]["id"] == "1"

    def test_to_json(self):
        """to_json should return valid JSON string."""
        sp = TraceSpan(id="1", name="span", kind="block", start_time=100.0)
        t = Trace(root_spans=[sp])

        import json
        json_str = t.to_json()
        parsed = json.loads(json_str)

        assert parsed["spans"][0]["id"] == "1"

    def test_to_json_with_keys(self):
        """to_json should respect keys filter."""
        sp = TraceSpan(
            id="1", name="span", kind="block", start_time=100.0, input="test"
        )
        t = Trace(root_spans=[sp])

        import json
        json_str = t.to_json(keys=["id", "name"])
        parsed = json.loads(json_str)

        assert "id" in parsed["spans"][0]
        assert "name" in parsed["spans"][0]
        assert "input" not in parsed["spans"][0]

    def test_pretty_format(self):
        """pretty() should return readable tree format."""
        child = TraceSpan(
            id="child",
            name="child-span",
            kind="model",
            start_time=101.0,
            end_time=102.0,
            input="child input",
            output="child output",
        )
        parent = TraceSpan(
            id="parent",
            name="parent-span",
            kind="block",
            start_time=100.0,
            end_time=103.0,
            children=[child],
        )
        t = Trace(root_spans=[parent])

        pretty = t.pretty()

        assert "[block] parent-span" in pretty
        assert "[model] child-span" in pretty
        assert "child input" in pretty
        assert "child output" in pretty


class TestSpanContextManager:
    """Tests for span() context manager."""

    def test_span_without_trace_yields_dummy(self):
        """span() without active trace should yield a dummy span."""
        with span(kind="block", name="test") as sp:
            assert sp.id == "__not_traced__"

    def test_span_with_trace_records_span(self):
        """span() with active trace should record the span."""
        with trace() as t:
            with span(kind="block", name="my-span") as sp:
                pass

        assert len(t.root_spans) == 1
        assert t.root_spans[0].name == "my-span"

    def test_span_sets_start_and_end_time(self):
        """span() should set start_time and end_time."""
        with trace() as t:
            before = time.time()
            with span(kind="block", name="test") as sp:
                time.sleep(0.01)  # Small delay
            after = time.time()

        recorded = t.root_spans[0]
        assert recorded.start_time >= before
        assert recorded.end_time <= after
        assert recorded.duration_s > 0

    def test_span_captures_input_and_kwargs(self):
        """span() should capture input and kwargs."""
        with trace() as t:
            with span(
                kind="model",
                name="test",
                input="my input",
                kwargs={"temp": 0.5},
            ):
                pass

        recorded = t.root_spans[0]
        assert recorded.input == "my input"
        assert recorded.kwargs == {"temp": 0.5}

    def test_span_output_can_be_set(self):
        """Output can be set on the span inside context."""
        with trace() as t:
            with span(kind="block", name="test") as sp:
                sp.output = "result"

        assert t.root_spans[0].output == "result"

    def test_span_captures_error(self):
        """span() should capture exceptions."""
        with trace() as t:
            with pytest.raises(ValueError):
                with span(kind="block", name="test") as sp:
                    raise ValueError("test error")

        recorded = t.root_spans[0]
        assert recorded.error == "ValueError: test error"
        assert recorded.end_time is not None  # Should still end


class TestSpanNesting:
    """Tests for nested span behavior."""

    def test_nested_spans_create_tree(self):
        """Nested spans should create parent-child relationships."""
        with trace() as t:
            with span(kind="block", name="parent") as parent_sp:
                with span(kind="model", name="child1"):
                    pass
                with span(kind="model", name="child2"):
                    pass

        assert len(t.root_spans) == 1
        parent = t.root_spans[0]
        assert parent.name == "parent"
        assert len(parent.children) == 2
        assert parent.children[0].name == "child1"
        assert parent.children[1].name == "child2"

    def test_deep_nesting(self):
        """Should support deeply nested spans."""
        with trace() as t:
            with span(kind="block", name="level1"):
                with span(kind="block", name="level2"):
                    with span(kind="model", name="level3"):
                        pass

        level1 = t.root_spans[0]
        level2 = level1.children[0]
        level3 = level2.children[0]

        assert level1.name == "level1"
        assert level2.name == "level2"
        assert level3.name == "level3"

    def test_multiple_root_spans(self):
        """Multiple top-level spans should all be root_spans."""
        with trace() as t:
            with span(kind="block", name="first"):
                pass
            with span(kind="block", name="second"):
                pass
            with span(kind="block", name="third"):
                pass

        assert len(t.root_spans) == 3
        assert t.root_spans[0].name == "first"
        assert t.root_spans[1].name == "second"
        assert t.root_spans[2].name == "third"


class TestTraceContextManager:
    """Tests for trace() context manager."""

    def test_trace_isolates_spans(self):
        """Each trace() context should have isolated spans."""
        with trace() as t1:
            with span(kind="block", name="span1"):
                pass

        with trace() as t2:
            with span(kind="block", name="span2"):
                pass

        assert len(t1.root_spans) == 1
        assert t1.root_spans[0].name == "span1"
        assert len(t2.root_spans) == 1
        assert t2.root_spans[0].name == "span2"

    def test_get_active_trace_returns_current(self):
        """get_active_trace() should return current trace."""
        assert get_active_trace() is None

        with trace() as t:
            assert get_active_trace() is t

        assert get_active_trace() is None

    def test_nested_trace_not_supported(self):
        """Nested trace() should create a new context (replacing outer)."""
        with trace() as outer:
            with span(kind="block", name="outer-span"):
                pass

            with trace() as inner:
                with span(kind="block", name="inner-span"):
                    pass

                assert get_active_trace() is inner

            # After inner exits, outer should be restored
            assert get_active_trace() is outer

        # outer trace should only have outer-span
        assert len(outer.root_spans) == 1
        assert outer.root_spans[0].name == "outer-span"

        # inner trace should only have inner-span
        assert len(inner.root_spans) == 1
        assert inner.root_spans[0].name == "inner-span"

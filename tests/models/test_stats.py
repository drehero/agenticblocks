"""Tests for GlobalModelStats cost/call tracking, limits, and thread safety."""
import os
import threading
from unittest.mock import patch

import pytest


class TestGlobalModelStatsBasic:
    """Basic tests for GlobalModelStats."""

    def test_initial_state(self):
        """Stats should start at zero."""
        from agenticblocks.models.stats import GlobalModelStats

        with patch.dict(os.environ, {"MSWEA_SILENT_STARTUP": "1"}):
            stats = GlobalModelStats()

        assert stats.cost == 0.0
        assert stats.n_calls == 0

    def test_add_increments_cost_and_calls(self):
        """add() should increment cost and call count."""
        from agenticblocks.models.stats import GlobalModelStats

        with patch.dict(os.environ, {"MSWEA_SILENT_STARTUP": "1"}):
            stats = GlobalModelStats()

        stats.add(0.01)
        assert stats.cost == pytest.approx(0.01)
        assert stats.n_calls == 1

        stats.add(0.02)
        assert stats.cost == pytest.approx(0.03)
        assert stats.n_calls == 2

    def test_cost_property_readonly(self):
        """cost property should be read-only."""
        from agenticblocks.models.stats import GlobalModelStats

        with patch.dict(os.environ, {"MSWEA_SILENT_STARTUP": "1"}):
            stats = GlobalModelStats()

        with pytest.raises(AttributeError):
            stats.cost = 100

    def test_n_calls_property_readonly(self):
        """n_calls property should be read-only."""
        from agenticblocks.models.stats import GlobalModelStats

        with patch.dict(os.environ, {"MSWEA_SILENT_STARTUP": "1"}):
            stats = GlobalModelStats()

        with pytest.raises(AttributeError):
            stats.n_calls = 100


class TestGlobalModelStatsLimits:
    """Tests for cost and call limits."""

    def test_cost_limit_from_env(self):
        """Should read cost limit from MSWEA_GLOBAL_COST_LIMIT."""
        from agenticblocks.models.stats import GlobalModelStats

        with patch.dict(
            os.environ,
            {"MSWEA_GLOBAL_COST_LIMIT": "5.0", "MSWEA_SILENT_STARTUP": "1"},
        ):
            stats = GlobalModelStats()

        assert stats.cost_limit == 5.0

    def test_call_limit_from_env(self):
        """Should read call limit from MSWEA_GLOBAL_CALL_LIMIT."""
        from agenticblocks.models.stats import GlobalModelStats

        with patch.dict(
            os.environ,
            {"MSWEA_GLOBAL_CALL_LIMIT": "100", "MSWEA_SILENT_STARTUP": "1"},
        ):
            stats = GlobalModelStats()

        assert stats.call_limit == 100

    def test_cost_limit_exceeded_raises(self):
        """Should raise RuntimeError when cost limit exceeded."""
        from agenticblocks.models.stats import GlobalModelStats

        with patch.dict(
            os.environ,
            {"MSWEA_GLOBAL_COST_LIMIT": "0.05", "MSWEA_SILENT_STARTUP": "1"},
        ):
            stats = GlobalModelStats()

        stats.add(0.03)  # OK
        stats.add(0.02)  # OK, at limit

        with pytest.raises(RuntimeError, match="cost/call limit exceeded"):
            stats.add(0.01)  # Exceeds limit

    def test_call_limit_exceeded_raises(self):
        """Should raise RuntimeError when call limit exceeded."""
        from agenticblocks.models.stats import GlobalModelStats

        with patch.dict(
            os.environ, {"MSWEA_GLOBAL_CALL_LIMIT": "3", "MSWEA_SILENT_STARTUP": "1"}
        ):
            stats = GlobalModelStats()

        stats.add(0.01)  # Call 1
        stats.add(0.01)  # Call 2

        # Call 3 triggers the limit check (checks if next call would exceed)
        with pytest.raises(RuntimeError, match="cost/call limit exceeded"):
            stats.add(0.01)  # Call 3, triggers limit

    def test_zero_limit_means_no_limit(self):
        """Limit of 0 should mean no limit (unlimited)."""
        from agenticblocks.models.stats import GlobalModelStats

        with patch.dict(
            os.environ,
            {
                "MSWEA_GLOBAL_COST_LIMIT": "0",
                "MSWEA_GLOBAL_CALL_LIMIT": "0",
                "MSWEA_SILENT_STARTUP": "1",
            },
        ):
            stats = GlobalModelStats()

        # Should not raise even with many calls
        for _ in range(100):
            stats.add(100.0)

        assert stats.n_calls == 100
        assert stats.cost == pytest.approx(10000.0)


class TestGlobalModelStatsThreadSafety:
    """Tests for thread safety of GlobalModelStats."""

    def test_concurrent_adds_are_thread_safe(self):
        """Multiple threads adding simultaneously should not lose data."""
        from agenticblocks.models.stats import GlobalModelStats

        with patch.dict(os.environ, {"MSWEA_SILENT_STARTUP": "1"}):
            stats = GlobalModelStats()

        num_threads = 10
        adds_per_thread = 100
        cost_per_add = 0.01

        def add_costs():
            for _ in range(adds_per_thread):
                stats.add(cost_per_add)

        threads = [threading.Thread(target=add_costs) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_calls = num_threads * adds_per_thread
        expected_cost = expected_calls * cost_per_add

        assert stats.n_calls == expected_calls
        assert stats.cost == pytest.approx(expected_cost)


class TestGlobalModelStatsSingleton:
    """Tests for GLOBAL_MODEL_STATS singleton behavior."""

    def test_global_stats_shared_across_models(self, mock_openai_client):
        """GLOBAL_MODEL_STATS should be shared across Model instances."""
        import os
        from unittest.mock import patch as mock_patch

        with mock_patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with mock_patch("openai.OpenAI", return_value=mock_openai_client):
                from agenticblocks.models import Model, GLOBAL_MODEL_STATS

                model1 = Model(model_name="gpt-4", web_search=False)
                model1._warned_missing_cost = True
                model2 = Model(model_name="gpt-4", web_search=False)
                model2._warned_missing_cost = True

                initial_calls = GLOBAL_MODEL_STATS.n_calls

                model1("Hello")
                model2("World")

                assert GLOBAL_MODEL_STATS.n_calls == initial_calls + 2

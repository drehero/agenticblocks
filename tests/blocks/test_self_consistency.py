"""Tests for SelfConsistency block."""
from unittest.mock import MagicMock, call

import pytest


class TestSelfConsistencyBlock:
    """Tests for SelfConsistency N iterations and aggregation."""

    def test_runs_n_iterations(self):
        """SelfConsistency should run block N times."""
        mock_block = MagicMock(return_value="response")

        from agenticblocks.blocks import SelfConsistency

        sc = SelfConsistency(block=mock_block, n=5)
        sc(prompt="test prompt")

        assert mock_block.call_count == 5

    def test_passes_temperature_to_block(self):
        """SelfConsistency should pass configured temperature."""
        mock_block = MagicMock(return_value="response")

        from agenticblocks.blocks import SelfConsistency

        sc = SelfConsistency(block=mock_block, n=3, temperature=0.9)
        sc(prompt="test")

        for call_args in mock_block.call_args_list:
            assert call_args[1]["temperature"] == 0.9

    def test_passes_prompt_to_block(self):
        """SelfConsistency should pass the original prompt to block."""
        mock_block = MagicMock(return_value="response")

        from agenticblocks.blocks import SelfConsistency

        sc = SelfConsistency(block=mock_block, n=2)
        sc(prompt="my question")

        for call_args in mock_block.call_args_list:
            assert call_args[0][0] == "my question"

    def test_returns_concatenated_responses_without_aggregator(self):
        """Without aggregator, should return concatenated responses."""
        responses = ["Answer A", "Answer B", "Answer C"]
        mock_block = MagicMock(side_effect=responses)

        from agenticblocks.blocks import SelfConsistency

        sc = SelfConsistency(block=mock_block, n=3, aggregator=None)
        result = sc(prompt="test")

        assert "Answer A" in result
        assert "Answer B" in result
        assert "Answer C" in result

    def test_aggregator_receives_formatted_responses(self):
        """Aggregator should receive formatted responses."""
        responses = ["One", "Two", "Three"]
        mock_block = MagicMock(side_effect=responses)
        mock_aggregator = MagicMock(return_value="Final answer")

        from agenticblocks.blocks import SelfConsistency

        sc = SelfConsistency(block=mock_block, n=3, aggregator=mock_aggregator)
        result = sc(prompt="test")

        # Aggregator should be called once
        mock_aggregator.assert_called_once()
        agg_input = mock_aggregator.call_args[0][0]

        # Input should contain all responses and the template
        assert "One" in agg_input
        assert "Two" in agg_input
        assert "Three" in agg_input
        assert "most common answer" in agg_input

    def test_aggregator_returns_final_result(self):
        """Should return aggregator's result when aggregator is provided."""
        mock_block = MagicMock(return_value="response")
        mock_aggregator = MagicMock(return_value="aggregated result")

        from agenticblocks.blocks import SelfConsistency

        sc = SelfConsistency(block=mock_block, n=3, aggregator=mock_aggregator)
        result = sc(prompt="test")

        assert result == "aggregated result"

    def test_custom_aggregator_template(self):
        """Should use custom aggregator template."""
        mock_block = MagicMock(return_value="response")
        mock_aggregator = MagicMock(return_value="final")

        from agenticblocks.blocks import SelfConsistency

        custom_template = "Responses:\n{responses}\n\nPick the best one."
        sc = SelfConsistency(
            block=mock_block,
            n=2,
            aggregator=mock_aggregator,
            aggregator_template=custom_template,
        )
        sc(prompt="test")

        agg_input = mock_aggregator.call_args[0][0]
        assert "Pick the best one." in agg_input

    def test_passes_extra_kwargs_to_block(self):
        """Should pass extra kwargs to block calls."""
        mock_block = MagicMock(return_value="response")

        from agenticblocks.blocks import SelfConsistency

        sc = SelfConsistency(block=mock_block, n=2, temperature=0.7)
        sc(prompt="test", max_tokens=100)

        for call_args in mock_block.call_args_list:
            assert call_args[1]["max_tokens"] == 100


class TestSelfConsistencyRepr:
    """Tests for SelfConsistency __repr__."""

    def test_repr_format(self):
        """__repr__ should show block and n."""
        mock_block = MagicMock()
        mock_block.__repr__ = lambda self: "MockBlock()"

        from agenticblocks.blocks import SelfConsistency

        sc = SelfConsistency(block=mock_block, n=5)

        assert "SelfConsistency" in repr(sc)
        assert "n=5" in repr(sc)

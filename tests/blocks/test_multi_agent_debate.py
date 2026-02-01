"""Tests for MultiAgentDebate block."""
from unittest.mock import MagicMock, call

import pytest


class TestMultiAgentDebateBlock:
    """Tests for MultiAgentDebate rounds, history, and moderator."""

    def test_initial_round_all_agents_respond(self):
        """All blocks should respond in initial round."""
        agent1 = MagicMock(return_value="Agent 1 initial")
        agent2 = MagicMock(return_value="Agent 2 initial")
        agent3 = MagicMock(return_value="Agent 3 initial")

        from agenticblocks.blocks import MultiAgentDebate

        debate = MultiAgentDebate(
            blocks=[agent1, agent2, agent3],
            rounds=0,  # Only initial round, no debate rounds
        )

        # Mock the final synthesis call on last agent
        agent3.side_effect = ["Agent 3 initial", "Final synthesis"]

        result = debate(prompt="What is the answer?")

        # Each agent called once for initial + last agent for final
        assert agent1.call_count == 1
        assert agent2.call_count == 1
        assert agent3.call_count == 2  # initial + final synthesis

    def test_debate_rounds_executed(self):
        """Should execute configured number of debate rounds."""
        agent1 = MagicMock(return_value="response")
        agent2 = MagicMock(return_value="response")

        from agenticblocks.blocks import MultiAgentDebate

        debate = MultiAgentDebate(blocks=[agent1, agent2], rounds=2)
        debate(prompt="test")

        # Initial round: 2 calls (one per agent)
        # Debate round 1: 2 calls
        # Debate round 2: 2 calls
        # Final synthesis: 1 call (last agent)
        # Total per agent: 1 (initial) + 2 (debate rounds) = 3
        # agent2 also does final: +1 = 4
        assert agent1.call_count == 3
        assert agent2.call_count == 4

    def test_debate_history_accumulated(self):
        """Debate history should be passed to agents in later rounds."""
        agent1 = MagicMock(return_value="Agent 1 says X")
        agent2 = MagicMock(return_value="Agent 2 says Y")

        from agenticblocks.blocks import MultiAgentDebate

        debate = MultiAgentDebate(blocks=[agent1, agent2], rounds=1)
        debate(prompt="What is 2+2?")

        # Check that debate round calls include history
        # The third call to agent1 (first debate round) should include prior responses
        debate_call = agent1.call_args_list[1]  # Second call is debate round
        prompt_with_history = debate_call[0][0]

        assert "Agent 1" in prompt_with_history
        assert "Agent 2" in prompt_with_history

    def test_moderator_used_for_final_synthesis(self):
        """When moderator is provided, it should synthesize final answer."""
        agent1 = MagicMock(return_value="Agent 1")
        agent2 = MagicMock(return_value="Agent 2")
        moderator = MagicMock(return_value="Moderator's final answer")

        from agenticblocks.blocks import MultiAgentDebate

        debate = MultiAgentDebate(
            blocks=[agent1, agent2], rounds=1, moderator=moderator
        )

        result = debate(prompt="test")

        assert result == "Moderator's final answer"
        moderator.assert_called_once()

    def test_last_agent_used_when_no_moderator(self):
        """Without moderator, last agent should do final synthesis."""
        agent1 = MagicMock(return_value="Agent 1")
        agent2 = MagicMock(side_effect=["Agent 2 initial", "Agent 2 debate", "Final"])

        from agenticblocks.blocks import MultiAgentDebate

        debate = MultiAgentDebate(blocks=[agent1, agent2], rounds=1, moderator=None)

        result = debate(prompt="test")

        assert result == "Final"

    def test_debate_template_used(self):
        """Debate template should be used in debate rounds."""
        agent1 = MagicMock(return_value="response")
        agent2 = MagicMock(return_value="response")

        from agenticblocks.blocks import MultiAgentDebate

        custom_template = "Topic: {prompt}\n\nHistory:\n{history}\n\nYour turn:"
        debate = MultiAgentDebate(
            blocks=[agent1, agent2], rounds=1, debate_template=custom_template
        )
        debate(prompt="my topic")

        # Check debate round call uses template
        debate_call = agent1.call_args_list[1][0][0]
        assert "Topic: my topic" in debate_call
        assert "Your turn:" in debate_call

    def test_final_template_used(self):
        """Final template should be used for synthesis."""
        agent1 = MagicMock(return_value="response")
        moderator = MagicMock(return_value="final")

        from agenticblocks.blocks import MultiAgentDebate

        custom_final = "Question: {prompt}\n\nDebate:\n{debate_history}\n\nSummarize:"
        debate = MultiAgentDebate(
            blocks=[agent1],
            rounds=0,
            moderator=moderator,
            final_template=custom_final,
        )
        debate(prompt="my question")

        final_call = moderator.call_args[0][0]
        assert "Question: my question" in final_call
        assert "Summarize:" in final_call

    def test_passes_kwargs_to_agents(self):
        """Should pass kwargs to all agent calls."""
        agent1 = MagicMock(return_value="response")
        agent2 = MagicMock(return_value="response")

        from agenticblocks.blocks import MultiAgentDebate

        debate = MultiAgentDebate(blocks=[agent1, agent2], rounds=0)
        debate(prompt="test", temperature=0.5)

        for agent_call in agent1.call_args_list:
            assert agent_call[1].get("temperature") == 0.5


class TestMultiAgentDebateRepr:
    """Tests for MultiAgentDebate __repr__."""

    def test_repr_format(self):
        """__repr__ should show blocks and rounds."""
        agent1 = MagicMock()
        agent2 = MagicMock()

        from agenticblocks.blocks import MultiAgentDebate

        debate = MultiAgentDebate(blocks=[agent1, agent2], rounds=3)

        assert "MultiAgentDebate" in repr(debate)
        assert "rounds=3" in repr(debate)

from __future__ import annotations

from typing import Any, Callable, Optional

from agenticblocks.block import Block
from agenticblocks.models import Model


class IO(Block):
    """IO block - simple pass-through to the model."""

    def __init__(self, model: Model | str):
        self.model = Model(model) if isinstance(model, str) else model

    def __repr__(self):
        return f"IO({self.model!r})"

    def forward(self, prompt: str, **kwargs: Any) -> str:
        return self.model(prompt, **kwargs)


class ChainOfThought(Block):
    """Chain of Thought block - prompts the model to think step by step."""

    def __init__(self, model: Model | str, template: str = "{prompt}\nLet's think step by step."):
        self.model = Model(model) if isinstance(model, str) else model
        self.template = template

    def __repr__(self):
        return f"ChainOfThought({self.model!r})"

    def forward(self, prompt: str, **kwargs: Any) -> str:
        return self.model(self.template.format(prompt=prompt), **kwargs)


class SelfConsistency(Block):
    """Self-Consistency block - runs a block N times and aggregates responses."""

    def __init__(
        self,
        block: Callable[..., str],
        n: int = 5,
        temperature: float = 0.7,
        aggregator=None,
        aggregator_template: str = "{responses}\nGiven the responses above. Output the most common answer.",
    ):
        self.block = block
        self.n = n
        self.temperature = temperature
        self.aggregator = aggregator
        self.aggregator_template = aggregator_template

    def __repr__(self):
        return f"SelfConsistency({self.block!r}, n={self.n})"

    def forward(self, prompt: str, **kwargs: Any) -> str:
        results = []
        for _ in range(self.n):
            result = self.block(prompt, temperature=self.temperature, **kwargs)
            results.append(result)

        responses_text = "\n\n".join(results)

        if self.aggregator is None:
            return responses_text

        return self.aggregator(self.aggregator_template.format(responses=responses_text))


class MultiAgentDebate(Block):
    """Multi-Agent Debate block - multiple agents debate to reach a consensus."""

    def __init__(
        self,
        agents: list[Callable[..., str]],
        rounds: int = 2,
        moderator=None,
        debate_template: str = "Question: {prompt}\n\nPrevious responses:\n{history}\n\nProvide your response, considering the perspectives above:",
        final_template: str = "Question: {prompt}\n\nDebate summary:\n{debate_history}\n\nBased on this debate, provide the final answer:",
    ):
        self.agents = agents
        self.rounds = rounds
        self.moderator = moderator
        self.debate_template = debate_template
        self.final_template = final_template

    def __repr__(self):
        return f"MultiAgentDebate({self.agents!r}, rounds={self.rounds})"

    def forward(self, prompt: str, **kwargs: Any) -> str:
        all_results = []
        debate_history = []

        # Initial round - each agent responds to the prompt
        for i, agent in enumerate(self.agents):
            result = agent(prompt, **kwargs)
            all_results.append({"agent": i, "round": 0, "result": result})
            debate_history.append(f"Agent {i + 1}: {result}")

        # Debate rounds
        for round_num in range(self.rounds):
            history_text = "\n\n".join(debate_history)
            round_responses = []

            for i, agent in enumerate(self.agents):
                result = agent(
                    self.debate_template.format(prompt=prompt, history=history_text),
                    **kwargs,
                )
                all_results.append({"agent": i, "round": round_num + 1, "result": result})
                round_responses.append(f"Agent {i + 1}: {result}")

            debate_history.extend(round_responses)

        # Final synthesis
        full_history = "\n\n".join(debate_history)

        if self.moderator is not None:
            final_result = self.moderator(
                self.final_template.format(prompt=prompt, debate_history=full_history),
                **kwargs,
            )
        else:
            # Use last agent as moderator if none provided
            final_result = self.agents[-1](
                self.final_template.format(prompt=prompt, debate_history=full_history),
                **kwargs,
            )

        return final_result


class SelfRefine(Block):
    """Self-Refine block - iteratively critiques and improves responses."""

    def __init__(
        self,
        model: Model | str,
        iterations: int = 2,
        critique_template: str = "Task: {prompt}\n\nResponse:\n{response}\n\nCritique this response. What are its weaknesses? How can it be improved?",
        refine_template: str = "Task: {prompt}\n\nPrevious response:\n{response}\n\nCritique:\n{critique}\n\nProvide an improved response addressing the critique:",
    ):
        self.model = Model(model) if isinstance(model, str) else model
        self.iterations = iterations
        self.critique_template = critique_template
        self.refine_template = refine_template

    def __repr__(self):
        return f"SelfRefine({self.model!r}, iterations={self.iterations})"

    def forward(self, prompt: str, **kwargs: Any) -> str:
        # Generate initial response
        response = self.model(prompt, **kwargs)

        # Iterative refinement
        for _ in range(self.iterations):
            # Critique
            critique_result = self.model(
                self.critique_template.format(prompt=prompt, response=response),
                **kwargs,
            )

            # Refine
            refine_result = self.model(
                self.refine_template.format(prompt=prompt, response=response, critique=critique_result),
                **kwargs,
            )
            response = refine_result

        return response

from __future__ import annotations

from typing import Any, Callable

from agenticblocks.block import Block
from agenticblocks.models import Model


class IO(Block):
    """Simple pass-through block that calls a model directly.

    Args:
        model: A model name or Model instance.

    Example:
        >>> io = IO("openai/gpt-4o-mini")
        >>> io("Hello")
        '...'
    """

    def __init__(self, model: Model | str):
        self.model = Model(model) if isinstance(model, str) else model

    def __repr__(self):
        return f"IO({self.model!r})"

    def forward(self, prompt: str, **kwargs: Any) -> str:
        return self.model(prompt, **kwargs)


class ChainOfThought(Block):
    """Prompt a model to reason step by step.

    Args:
        model: A model name or Model instance.
        template: Prompt template with a `{prompt}` placeholder.

    Example:
        >>> cot = ChainOfThought("openai/gpt-4o-mini")
        >>> cot("How many r's are in strawberry?")
        '...'
    """

    def __init__(self, model: Model | str, template: str = "{prompt}\nLet's think step by step."):
        self.model = Model(model) if isinstance(model, str) else model
        self.template = template

    def __repr__(self):
        return f"ChainOfThought({self.model!r})"

    def forward(self, prompt: str, **kwargs: Any) -> str:
        return self.model(self.template.format(prompt=prompt), **kwargs)


class SelfConsistency(Block):
    """Run a block N times and optionally aggregate the responses.

    If `aggregator` is None, returns the concatenated responses.

    Args:
        block: A callable block or function to invoke.
        n: Number of runs.
        temperature: Temperature passed to each run.
        aggregator: Optional callable that combines responses.
        aggregator_template: Template used when calling the aggregator.

    Example:
        >>> sc = SelfConsistency(ChainOfThought("openai/gpt-4o-mini"), n=3)
        >>> sc("Question?")
        '...'
    """

    def __init__(
        self,
        block: Callable[..., str],
        n: int = 5,
        temperature: float = 0.7,
        aggregator: Callable[[str], str] | None = None,
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
    """Run a multi-block debate and synthesize a final answer.

    Each block responds, then debates for multiple rounds. A moderator
    (or the final block if none is provided) synthesizes the final answer.

    Args:
        blocks: List of callable blocks.
        rounds: Number of debate rounds after the initial responses.
        moderator: Optional callable to synthesize the final answer.
        debate_template: Template for debate rounds.
        final_template: Template for final synthesis.

    Example:
        >>> debate = MultiAgentDebate([IO("openai/gpt-4o-mini")], rounds=1)
        >>> debate("Question?")
        '...'
    """

    def __init__(
        self,
        blocks: list[Callable[..., str]],
        rounds: int = 2,
        moderator: Callable[[str], str] | None = None,
        debate_template: str = "Question: {prompt}\n\nPrevious responses:\n{history}\n\nProvide your response, considering the perspectives above:",
        final_template: str = "Question: {prompt}\n\nDebate summary:\n{debate_history}\n\nBased on this debate, provide the final answer:",
    ):
        self.blocks = blocks
        self.rounds = rounds
        self.moderator = moderator
        self.debate_template = debate_template
        self.final_template = final_template

    def __repr__(self):
        return f"MultiAgentDebate({self.blocks!r}, rounds={self.rounds})"

    def forward(self, prompt: str, **kwargs: Any) -> str:
        all_results = []
        debate_history = []

        # Initial round - each block responds to the prompt
        for i, block in enumerate(self.blocks):
            result = block(prompt, **kwargs)
            all_results.append({"agent": i, "round": 0, "result": result})
            debate_history.append(f"Agent {i + 1}: {result}")

        # Debate rounds
        for round_num in range(self.rounds):
            history_text = "\n\n".join(debate_history)
            round_responses = []

            for i, block in enumerate(self.blocks):
                result = block(
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
            # Use last block as moderator if none provided
            final_result = self.blocks[-1](
                self.final_template.format(prompt=prompt, debate_history=full_history),
                **kwargs,
            )

        return final_result


class SelfRefine(Block):
    """Iteratively critique and refine a response.

    Args:
        model: A model name or Model instance.
        iterations: Number of critique/refine cycles.
        critique_template: Template for critique prompts.
        refine_template: Template for refinement prompts.

    Example:
        >>> refine = SelfRefine("openai/gpt-4o-mini", iterations=2)
        >>> refine("Explain recursion.")
        '...'
    """

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

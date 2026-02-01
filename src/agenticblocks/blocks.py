from __future__ import annotations

import inspect
import time
import traceback
from string import Template
from typing import Any, Callable

from agenticblocks.block import Block
from agenticblocks.models import Model
from agenticblocks.utils import extract_json_obj


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
    """Run a multi-agent debate and synthesize a final answer.

    Each agent responds, then debates for multiple rounds. A moderator
    (or the final agent if none is provided) synthesizes the final answer.

    Args:
        agents: List of callable agents.
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
        agents: list[Callable[..., str]],
        rounds: int = 2,
        moderator: Callable[[str], str] | None = None,
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


class ReAct(Block):
    """Tool-using block that enforces JSON tool calls.

    Tools must be callables with docstrings; their signatures and docstrings
    are surfaced to the model. The model is instructed to return either a
    tool call JSON object or a final JSON response.

    Args:
        model: A model name or Model instance for reasoning.
        tools: A callable or list of callables exposed as tools.
        max_time: Optional time budget in seconds.
        max_steps: Optional step budget for tool-calling iterations.

    Example:
        >>> def add(a, b):  # doctest: +SKIP
        ...     "Add two numbers."
        ...     return a + b
        >>> agent = ReAct("openai/gpt-4o-mini", tools=[add], max_steps=3)
        >>> agent("What is 2 + 2?")
        '4'
    """

    SYSTEM_TEMPLATE = Template(
        "You are a tool-using agent. You must respond with a single JSON object and nothing else.\n"
        "Use one of these formats:\n"
        '- Tool call: {{"tool": "<name>", "kwargs": {{...}}, "thought": "<optional>"}}\n'
        '- Final answer: {{"final": "<answer>", "thought": "<optional>"}}\n'
        "Rules:\n"
        "- Use only the tools listed below.\n"
        "- If a tool fails, you will receive an error and can try again.\n"
        "\n"
        "Available tools:\n"
        "$tools\n"
    )

    def __init__(
        self,
        model: Model | str,
        tools: Callable[..., Any] | list[Callable[..., Any]],
        max_time: float | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Initialize a tool-using reasoning block.

        Args:
            model: A model name or Model instance to use for reasoning.
            tools: A callable or list of callables. Docstrings and type annotations
                of the callable are used to tell the model how the tool should be invoked.
            max_time: Optional time budget in seconds.
            max_steps: Optional step budget for tool-calling iterations.
        """
        self.max_time = max_time
        self.max_steps = max_steps
        self.tools = tools if isinstance(tools, list) else [tools]
        if not self.tools:
            raise ValueError("At least one tool must be provided.")
        descriptions = []
        for tool in self.tools:
            if not callable(tool):
                raise TypeError(f"Tool {tool!r} must be callable.")
            doc = inspect.getdoc(tool)
            if doc is None:
                raise ValueError(
                    f"Tool {getattr(tool, '__name__', tool.__class__.__name__)} "
                    "must define a docstring."
                )
            name = getattr(tool, "__name__", tool.__class__.__name__)
            signature = ""
            try:
                signature = str(inspect.signature(tool))
            except Exception:  # noqa: BLE001
                signature = ""
            if signature:
                descriptions.append(f"- {name}{signature}: {doc}")
            else:
                descriptions.append(f"- {name}: {doc}")
        self._system_prompt = self.SYSTEM_TEMPLATE.substitute(
            tools="\n".join(descriptions)
        )
        if self.max_time is not None or self.max_steps is not None:
            self._system_prompt += "\n\nBudget:\n"
            if self.max_time is not None:
                if not isinstance(self.max_time, (int, float)) or self.max_time <= 0:
                    raise ValueError("max_time must be a positive number")
                self._system_prompt += f"- You have a budget of {self.max_time} seconds to complete the task.\n"
            if self.max_steps is not None:
                if not isinstance(self.max_steps, int) or self.max_steps <= 0:
                    raise ValueError("max_steps must be a positive integer")
                self._system_prompt += f"- You have a budget of a total of {self.max_steps} reasoning / tool calling steps to complete the task.\n"

        if isinstance(model, str):
            self.model = Model(model)
        else:
            self.model = model

    def __repr__(self):
        tool_names = [getattr(t, "__name__", t.__class__.__name__) for t in self.tools]
        tools_repr = ", ".join(tool_names)
        return (
            f"ReAct({self.model!r}, tools=[{tools_repr}], "
            f"max_time={self.max_time}, max_steps={self.max_steps})"
        )

    def forward(self, prompt: str, **kwargs: Any) -> str:
        start = time.monotonic()
        steps = 0
        stop = False
        
        # handle global history of the model
        original_keep_history = self.model.keep_history
        original_messages = list(self.model.messages)
        try:
            self.model.keep_history = True
            self.model.reset_history(keep_system=False)
            self.model.add_message("system", self._system_prompt)
            current_prompt = prompt

            while True:
                budget_lines = []
                if self.max_steps is not None:
                    remaining_steps = max(0, self.max_steps - steps)
                    budget_lines.append(f"- Remaining steps: {remaining_steps}")
                    if remaining_steps == 0:
                        budget_lines.append(
                            "Step limit reached. Provide a final answer now. Do not call any tools."
                        )
                        stop = True
                if self.max_time is not None:
                    remaining_time = max(0, self.max_time - (time.monotonic() - start))
                    budget_lines.append(f"- Remaining time (seconds): {remaining_time:.2f}")
                    if remaining_time == 0:
                        budget_lines.append(
                            "Time limit reached. Provide a final answer now. Do not call any tools."
                        )
                        stop = True
                if budget_lines:
                    current_prompt = "Budget:\n" + "\n".join(budget_lines) + "\n\n" + current_prompt

                steps += 1
                response = self.model(current_prompt, **kwargs)

                try:
                    parsed = extract_json_obj(response)
                    if stop or "final" in parsed:
                        if "thought" in parsed:
                            return f"{parsed['thought']}\n{parsed['final']}".strip()
                        return str(parsed["final"])

                    if "tool" in parsed:
                        tool = None
                        for t in self.tools:
                            if getattr(t, "__name__", t.__class__.__name__) == parsed["tool"]:
                                tool = t
                                break
                        if tool is None:
                            current_prompt = f"ToolError: Unknown tool {parsed['tool']}"
                        else:
                            current_prompt = f"ToolResult: {tool(**parsed.get('kwargs', {}))}"
                    
                    else:
                        current_prompt = f"Invalid JSON response: It should be a JSON object with a 'tool' or 'final' key."

                except Exception:
                    if stop:
                        return "Unable to complete the task within the budget."
                    current_prompt = (
                        "An exception occurred. Try again.\n\nTraceback:\n"
                        f"{traceback.format_exc()}"
                    )
        finally:
            self.model.keep_history = original_keep_history
            if self.model.keep_history:
                self.model.messages = original_messages + self.model.messages
            else:
                self.model.messages = original_messages

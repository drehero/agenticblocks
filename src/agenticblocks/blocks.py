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


class ToolReasoning(Block):
    """Tool-integrated reasoning block using strict JSON tool calls."""

    SYSTEM_TEMPLATE = Template(
        "You are a tool-using agent. You must respond with a single JSON object and nothing else.\n"
        "Use one of these formats:\n"
        '- Tool call: {{"tool": "<name>", "kwargs": {{...}}}}\n'
        '- Final answer: {{"final": "<answer>"}}\n'
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
            f"ToolReasoning({self.model!r}, tools=[{tools_repr}], "
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

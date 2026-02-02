from __future__ import annotations

import inspect
import json
import time
import traceback
from string import Template
from types import UnionType
from typing import Any, Callable, Literal, Union, get_args, get_origin

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
        if hasattr(self.model, "client_provider") and self.model.client_provider not in ("openai",):
            raise NotImplementedError(
                "ToolCalling currently supports OpenAI-compatible chat APIs only."
            )

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


class ToolCalling(Block):
    """Native tool-calling block for OpenAI-compatible chat APIs.

    This block uses provider tool-calling (e.g., OpenAI tool_calls) instead of
    a JSON-only protocol. Tools are exposed as function schemas.

    Args:
        model: A model name or Model instance for reasoning.
        tools: A callable or list of callables exposed as tools.
        system_prompt: Optional system prompt to prepend.
        tool_choice: Tool choice directive (e.g., "auto", "none", or a dict).
        max_time: Optional time budget in seconds.
        max_steps: Optional step budget for tool-calling iterations.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a tool-using assistant. Use tools when helpful and respond "
        "with a final answer when done."
    )

    def __init__(
        self,
        model: Model | str,
        tools: Callable[..., Any] | list[Callable[..., Any]],
        *,
        system_prompt: str | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        max_time: float | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.max_time = max_time
        self.max_steps = max_steps
        self.tool_choice = tool_choice
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.tools = tools if isinstance(tools, list) else [tools]
        if not self.tools:
            raise ValueError("At least one tool must be provided.")

        if isinstance(model, str):
            self.model = Model(model)
        else:
            self.model = model

        self._tool_map = {
            getattr(t, "__name__", t.__class__.__name__): t for t in self.tools
        }
        self._tool_schemas = [self._tool_schema(t) for t in self.tools]

    def __repr__(self) -> str:
        tool_names = [getattr(t, "__name__", t.__class__.__name__) for t in self.tools]
        tools_repr = ", ".join(tool_names)
        return (
            f"ToolCalling({self.model!r}, tools=[{tools_repr}], "
            f"max_time={self.max_time}, max_steps={self.max_steps})"
        )

    def forward(self, prompt: str, **kwargs: Any) -> str:
        start = time.monotonic()
        steps = 0
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        while True:
            if self.max_steps is not None and steps >= self.max_steps:
                return "Unable to complete the task within the budget."
            if self.max_time is not None and (time.monotonic() - start) >= self.max_time:
                return "Unable to complete the task within the budget."

            steps += 1
            request_kwargs = dict(kwargs)
            request_kwargs["tools"] = self._tool_schemas
            if self.tool_choice is not None:
                request_kwargs["tool_choice"] = self.tool_choice

            response = self.model._create_chat_completion(  # type: ignore[attr-defined]
                messages=messages,
                **request_kwargs,
            )
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if tool_calls:
                messages.append(message)
                for call in tool_calls:
                    tool_name = call.get("function", {}).get("name")
                    tool = self._tool_map.get(tool_name)
                    tool_call_id = call.get("id")
                    args_text = call.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_text) if args_text else {}
                    except json.JSONDecodeError as exc:
                        result = f"ToolError: JSONDecodeError: {exc}"
                    else:
                        if tool is None:
                            result = f"ToolError: Unknown tool {tool_name}"
                        else:
                            try:
                                result = tool(**args)
                            except Exception as exc:  # noqa: BLE001
                                result = f"ToolError: {type(exc).__name__}: {exc}"

                    messages.append(
                        self._tool_result_message(
                            tool_call_id=tool_call_id,
                            content=self._stringify_tool_result(result),
                        )
                    )
                continue

            content = message.get("content")
            return content or ""

    @staticmethod
    def _stringify_tool_result(result: Any) -> str:
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return str(result)

    @classmethod
    def _tool_schema(cls, tool: Callable[..., Any]) -> dict[str, Any]:
        if not callable(tool):
            raise TypeError(f"Tool {tool!r} must be callable.")
        doc = inspect.getdoc(tool) or ""
        name = getattr(tool, "__name__", tool.__class__.__name__)
        signature = inspect.signature(tool)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in signature.parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise ValueError(f"Tool {name} must not use *args or **kwargs.")
            annotation = param.annotation
            schema = cls._annotation_to_schema(annotation)
            properties[param.name] = schema
            if param.default is param.empty:
                required.append(param.name)
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": doc,
                "parameters": parameters,
            },
        }

    @classmethod
    def _annotation_to_schema(cls, annotation: Any) -> dict[str, Any]:
        if annotation is inspect._empty:
            return {"type": "string"}

        origin = get_origin(annotation)
        if origin is None:
            return cls._primitive_schema(annotation)

        if origin in (list,):
            args = get_args(annotation)
            items = cls._annotation_to_schema(args[0]) if args else {}
            return {"type": "array", "items": items}
        if origin in (dict,):
            return {"type": "object"}
        if origin is tuple:
            return {"type": "array"}
        if origin is type(None):
            return {"type": "null"}
        if origin in (Literal,):
            return {"enum": list(get_args(annotation))}
        if origin in (Union, UnionType):
            args = get_args(annotation)
            schemas = [cls._annotation_to_schema(arg) for arg in args]
            return {"anyOf": schemas}

        return {"type": "string"}

    @staticmethod
    def _primitive_schema(annotation: Any) -> dict[str, Any]:
        if annotation in (str,):
            return {"type": "string"}
        if annotation in (int,):
            return {"type": "integer"}
        if annotation in (float,):
            return {"type": "number"}
        if annotation in (bool,):
            return {"type": "boolean"}
        if annotation in (dict,):
            return {"type": "object"}
        if annotation in (list, tuple):
            return {"type": "array"}
        return {"type": "string"}

    @staticmethod
    def _tool_result_message(*, tool_call_id: str | None, content: str) -> dict[str, Any]:
        message = {"role": "tool", "content": content}
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        return message

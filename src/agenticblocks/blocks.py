class ChainOfThought:
    """Chain of Thought block - prompts the model to think step by step."""

    def __init__(self, model, template: str = "{prompt}\nLet's think step by step."):
        self.model = model
        self.template = template

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.model(self.template.format(prompt=prompt), **kwargs)


class SelfConsistency:
    """Self-Consistency block - runs a block N times and aggregates responses."""

    def __init__(
        self,
        block,
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

    def __call__(self, prompt: str, **kwargs) -> dict:
        responses = []
        for _ in range(self.n):
            result = self.block(prompt, temperature=self.temperature, **kwargs)
            responses.append(result["content"])

        responses_text = "\n\n".join(responses)

        if self.aggregator is None:
            return {"content": responses_text, "extra": {"responses": responses}}

        return self.aggregator(self.aggregator_template.format(responses=responses_text))


class MultiAgentDebate:
    """Multi-Agent Debate block - multiple agents debate to reach a consensus."""

    def __init__(
        self,
        agents: list,
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

    def __call__(self, prompt: str, **kwargs) -> dict:
        debate_history = []

        # Initial round - each agent responds to the prompt
        for i, agent in enumerate(self.agents):
            result = agent(prompt, **kwargs)
            debate_history.append(f"Agent {i + 1}: {result['content']}")

        # Debate rounds
        for _ in range(self.rounds):
            history_text = "\n\n".join(debate_history)
            round_responses = []

            for i, agent in enumerate(self.agents):
                result = agent(
                    self.debate_template.format(prompt=prompt, history=history_text),
                    **kwargs,
                )
                round_responses.append(f"Agent {i + 1}: {result['content']}")

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

        return {
            "content": final_result["content"],
            "extra": {"debate_history": debate_history},
        }


class SelfRefine:
    """Self-Refine block - iteratively critiques and improves responses."""

    def __init__(
        self,
        model,
        iterations: int = 2,
        critique_template: str = "Task: {prompt}\n\nResponse:\n{response}\n\nCritique this response. What are its weaknesses? How can it be improved?",
        refine_template: str = "Task: {prompt}\n\nPrevious response:\n{response}\n\nCritique:\n{critique}\n\nProvide an improved response addressing the critique:",
    ):
        self.model = model
        self.iterations = iterations
        self.critique_template = critique_template
        self.refine_template = refine_template

    def __call__(self, prompt: str, **kwargs) -> dict:
        # Generate initial response
        result = self.model(prompt, **kwargs)
        response = result["content"]

        history = [{"response": response, "critique": None}]

        # Iterative refinement
        for _ in range(self.iterations):
            # Critique
            critique_result = self.model(
                self.critique_template.format(prompt=prompt, response=response),
                **kwargs,
            )
            critique = critique_result["content"]

            # Refine
            refine_result = self.model(
                self.refine_template.format(prompt=prompt, response=response, critique=critique),
                **kwargs,
            )
            response = refine_result["content"]

            history.append({"response": response, "critique": critique})

        return {
            "content": response,
            "extra": {"history": history},
        }

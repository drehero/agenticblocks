# agenticblocks

## What is agenticblocks?

agenticblocks offers two concepts: models and blocks (tools and environments coming soon).

Combining these lets you build powerful multi agent systems.

## Installation

```bash
pip install git+https://github.com/drehero/agenticblocks.git              # Core + OpenAI/OpenRouter support
pip install "git+https://github.com/drehero/agenticblocks.git#egg=agenticblocks[anthropic]"   # + Anthropic support
pip install "git+https://github.com/drehero/agenticblocks.git#egg=agenticblocks[google]"      # + Google Gemini support
pip install "git+https://github.com/drehero/agenticblocks.git#egg=agenticblocks[xai]"         # + xAI/Grok support
pip install "git+https://github.com/drehero/agenticblocks.git#egg=agenticblocks[all]"         # All providers
```

### Models

A model takes text as input and outputs text:

```python
import agenticblocks as ab
model = ab.Model(model_name)
model("How many r's are in strawberry?")
```

agenticblocks supports all OpenAI API compatible providers (more coming soon).

### Blocks

A block defines how a model is called. For example a Chain of Thought (CoT) block might look something like this:

```python
class CoT:
    def __init__(self, model):
        self.model = model
        self.template = "{prompt}\nLet's think step by step." 

    def __call__(self, prompt, **kwargs):
        return self.model(self.template.format(prompt=prompt), **kwargs)

cot_block = CoT(model)
cot_block("How many r's are in strawberry?")
```

Under the hood the model class takes care of things like conversation history and limits. This lets you focus on combining the agentic blocks to more complex workflows.

For example into a Chain of Thought with Self Consistency (CoT-SC) block:

```python
class CoTSC:
    def __init__(self, cot_block, N=5, temperature=0.7):
        self.cot_block = cot_block
        self.N = N
        self.temperature = temperature
        self.aggregator = ab.Model(model_name)
        self.template = "{responses}\nGiven the responses above. Output the most common answer."
    
    def __call__(self, prompt):
        responses = []
        for _ in range(self.N):
            responses += [self.cot_block(prompt, temperature=self.temperature)]
        responses = "\n\n".join(responses)
        return self.aggregator(self.template.format(responses=responses))

cotsc_block = CoTSC(cot_block)
cotsc_block("How many r's are in strawberry?")
```

Many commonly used blocks are already built into agenticblocks. For example here we implement a multi agent debate (MAD) using built in blocks:

```python
import agenticblocks as ab

cot = ab.ChainOfThought(ab.Model(cot_model_name))
cotsc = ab.SelfConsistency(ab.Model(cotsc_model_name))
refine = ab.SelfRefine(ab.Model(refine_model_name))

mad = ab.MultiAgentDebate([cot, cotsc, refine])
mad("How many r's are in strawberry?")
```

Using built in and custom blocks we can define powerful reasoning flows which produce better solutions than standard LLM calls.

### Tool-Integrated Reasoning

You can also add arbitrary Python tools (functions or classes) with a tool-aware reasoning block:

```python
import agenticblocks as ab

def add(a, b):
    "Add two numbers."
    return a + b

tools = [add]
agent = ab.ReAct(ab.Model(model_name), tools=tools, max_steps=5, max_time=30)
agent("What is 12 + 30?")
```

### But why stop here?

agenticblocks makes building workflows so simple we can use it to create an agent which uses the framework to build its own solution to a task.
And by giving the agent feedback from the environment on how well its workflow performed it can update and refine its creation.
Now we have built a self evolving and life long learning system.

For more details on how to do this check out the [examples](examples/).

## Roadmap

- [ ] Implement tests and improve model class and built-in blocks
- [ ] Create PyPI package (v0)
- [ ] Add more built-in blocks and examples
- [ ] Add documentation
- [ ] Implement support for local models
- [ ] Implement optimizers and search
- [ ] Implement tool use
- [ ] Implement a way to share discovered blocks


## Acknowledgments

This project includes code from [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 
by Kilian A. Lieret and Carlos E. Jimenez, licensed under the MIT License.

It is inspired by [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), [live-swe-agent](https://github.com/OpenAutoCoder/live-swe-agent) and other self evolving agent research, in particular:
* [Automated Design of Agentic Systems (ADAS)](https://github.com/ShengranHu/ADAS)
* [MAS-Zero: Multi-Agent Systems with Zero Supervision](https://github.com/SalesforceAIResearch/MAS-Zero)
* a comprehensive overview of the self evolving agent research can be found [here](https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents).

# agenticblocks

Compose models and blocks to build multi-agent workflows.

## Quickstart

```python
import agenticblocks as ab

model = ab.Model("openai/gpt-4o-mini")
model("How many r's are in strawberry?")
```

## Blocks

```python
import agenticblocks as ab

cot = ab.ChainOfThought("openai/gpt-4o-mini")
cot("How many r's are in strawberry?")
```

## Local models

```python
import agenticblocks as ab

local = ab.LocalModel("llama3.1", provider="ollama")
local("How many r's are in strawberry?")
```

## Build the docs

```bash
pip install -e ".[docs]"
mkdocs serve
```

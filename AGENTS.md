# AGENTS.md

## Purpose
`agenticblocks` is a Python package for composing "models" and "blocks" to build multi‑agent workflows. The public API is exported from `src/agenticblocks/__init__.py`.

## Quick setup
- Python: 3.10+
- Install (editable, core):
  - `pip install -e .`
- Optional providers (extras):
  - `pip install -e ".[anthropic]"`
  - `pip install -e ".[google]"`
  - `pip install -e ".[xai]"`
  - `pip install -e ".[all]"`
- Dev dependencies:
  - `pip install -e ".[dev]"`

## Tests
- Run all tests: `pytest`
- Skip slow tests: `pytest -m "not slow"`

## Repo layout
- `src/agenticblocks/`: core package code
  - `block.py`: base Block class
  - `blocks.py`: built-in blocks (e.g., ChainOfThought, SelfConsistency, MultiAgentDebate)
  - `models/`: model implementations and utilities
  - `tools.py`: tool definitions and helpers
  - `trace.py`: tracing utilities
- `tests/`: pytest suite
- `examples/`: notebooks and usage examples

## Conventions
- Keep public API in sync with `src/agenticblocks/__init__.py` when adding/removing exports.
- Prefer small, composable blocks and provider‑agnostic model behavior.

## Examples
Minimal usage with a model:
```python
import agenticblocks as ab

model = ab.Model("openai/gpt-4o-mini")
model("How many r's are in strawberry?")
```

Composing a built-in block:
```python
import agenticblocks as ab

model = ab.Model("openai/gpt-4o-mini")
cot = ab.ChainOfThought(model)
cot("How many r's are in strawberry?")
```

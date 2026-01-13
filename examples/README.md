# `agenticblocks` Examples

This folder contains runnable notebooks demonstrating core `agenticblocks` concepts: models, blocks, and building agentic workflows.

## Notebooks

1. [Getting Started](01_getting_started.ipynb)  
   TODO A minimal introduction to `agenticblocks.Model` and how to wrap model calls into simple blocks (e.g., Chain-of-Thought) and composite flows (e.g., Self-Consistency). Also shows basic provider setup (e.g., OpenRouter via `OPENROUTER_API_KEY`).

2. Intro to Self Evolving Agents with agenticblocks
    TODO shows how to use agenticblocks to automatically design agentic blocks

3. [Automated Design of Agentic Systems (ADAS)](03_automated_design_of_agentic_systems_adas.ipynb)  
   Implements ideas from Hu et al.’s ADAS: a meta-agent generates new block designs (code), evaluates them on a small dataset (MMLU-STEM), and iterates based on feedback.

4. [MAS-Zero: Multi-Agent Systems with Zero Supervision](04_multi_agent_systems_with_zero_supervision_mas-zero.ipynb)  
   Implements MAS-Zero (Ke et al.): run built-in blocks to generate candidates (MAS-Init), iteratively refine via a meta-design/meta-feedback loop (MAS-Evolve), then select a final answer via majority vote + verification (MAS-Verify). Demonstrates tracing (`with ab.trace()`) to inspect intermediate inputs/outputs.
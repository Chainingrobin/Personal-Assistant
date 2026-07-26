# Personal Assistant

This repo is being reset toward a modular personal-assistant stack with a separate synthetic evaluation layer for local LLM testing.

## Recommended local test path

Use Ollama with `qwen2.5:1.5b` as the first gate before any real API integration. The goal is to validate three things in a fully synthetic environment:

1. Mail parsing quality: can the model extract sender, intent, priority, dates, and action items?
2. Calendar handling: can it turn text into structured event objects and detect scheduling conflicts?
3. Event logging: can it emit clean tool calls or event records for downstream storage?

### Minimal flow

1. Start Ollama locally.
2. Pull `qwen2.5:1.5b`.
3. Run the synthetic harness in `tests/agentic/` against JSON fixtures in `tests/synthetic_data/`.
4. Score output for structure, correctness, and tool-call validity.
5. Only after the model is stable do you wire real Gmail/Calendar APIs.

## Clean architecture

Keep the repo split into four layers:

1. `services/llm_orchestrator/` for prompt assembly, tool schema definitions, and LLM transport.
2. `services/actions/` for external side effects, but behind adapter interfaces.
3. `services/rag_store/` for retrieval only, with no orchestration logic.
4. `tests/agentic/` for synthetic evaluation, fixture loading, and scoring.

The test harness should never call real APIs. It should only talk to the local Ollama endpoint and compare the model output against expected synthetic results.

## What was redundant

The current tree contains a few accidental placeholder names such as comma-separated file and folder names. Those are not part of the target structure and should be removed in favor of the clean modules above.

See [docs/synthetic_llm_testing.md](docs/synthetic_llm_testing.md) for the runnable skeleton and test plan.

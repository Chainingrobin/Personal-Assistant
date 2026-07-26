# System Overview

This repo now has two clear paths:

1. `services/` for runtime boundaries and adapters.
2. `tests/agentic/` for synthetic local-model evaluation.

## Canonical runtime boundary

- `services/llm_orchestrator/` owns the request/response contract for the model.
- `services/actions/` owns side effects such as webhooks, but only through adapters.
- `services/rag_store/` owns retrieval only.
- `services/identity/` and `services/sensing/` remain future runtime layers.

## Canonical evaluation boundary

- `tests/agentic/` builds prompts from synthetic cases.
- `tests/synthetic_data/` stores the fixtures.
- The runner talks to Ollama locally and scores structure, extraction, and tool intent.

## Rule of separation

Do not mix evaluation logic into runtime adapters. The synthetic harness should stay isolated so you can change prompts, fixtures, and scoring without risking real integrations.

## Related docs

- [Synthetic LLM testing](./synthetic_llm_testing.md)
- [Diagrams](./diagrams.md)

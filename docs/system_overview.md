# System Overview

This repo now focuses on the runtime path:

1. `orchestrate.py` and `tools/` for the tool-calling assistant.
2. `config.py` and `credentials/` for runtime configuration and Google OAuth.

## Canonical runtime boundary

- `services/llm_orchestrator/` owns the request/response contract for the model.
- `services/actions/` owns side effects such as webhooks, but only through adapters.
- `services/rag_store/` owns retrieval only.
- `services/identity/` and `services/sensing/` remain future runtime layers.

## Rule of separation

Do not mix evaluation logic into runtime adapters. The synthetic harness should stay isolated so you can change prompts, fixtures, and scoring without risking real integrations.

## Related docs

- [Synthetic LLM testing](./synthetic_llm_testing.md)

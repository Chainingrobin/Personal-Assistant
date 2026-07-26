# Architecture & Data Flow Diagrams

## 1. Service Interaction Sequence Diagram

````
┌─────────┐     ┌──────────┐     ┌─────────────┐     ┌────────┐
│ Sensing │────▶│Identity  │     │   Actions    │◀────┤ n8n   │
│ (Wake)  │◀────│Service   │     │   Service    │     │ Webhook│
└─────────┘     └────┬─────┘     └──────┬──────┘     └────────┘
                     │                  │
        # Diagrams

        ## Synthetic evaluation loop

        ```mermaid
        flowchart LR
            Fixtures[tests/synthetic_data/] --> Runner[tests/agentic/runner.py]
            Runner --> Ollama[Local Ollama API]
            Ollama --> Model[qwen2.5:1.5b]
            Runner --> Score[Structured scoring]
            Score --> Report[Pass/fail report]
        ```

        ## Runtime boundary

        ```mermaid
        flowchart TB
            Orchestrator[services/llm_orchestrator/] --> Actions[services/actions/]
            Orchestrator --> Rag[services/rag_store/]
            Sensing[services/sensing/] --> Orchestrator
            Identity[services/identity/] --> Orchestrator
        ```

        These diagrams intentionally stay small. The detailed test strategy lives in [synthetic_llm_testing.md](./synthetic_llm_testing.md).
    participant LLM as Ollama/LLM
````

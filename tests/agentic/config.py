from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    api_path: str = "/api/chat"
    provider: str = "ollama"

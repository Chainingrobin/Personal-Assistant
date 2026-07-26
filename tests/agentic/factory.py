from .client import ChatModelClient
from .config import AgentConfig
from .ollama_client import OllamaClient


def create_client(config: AgentConfig) -> ChatModelClient:
    provider = config.provider.lower().strip()
    if provider == "ollama":
        return OllamaClient(config)
    raise NotImplementedError(f"Unsupported provider: {config.provider}")

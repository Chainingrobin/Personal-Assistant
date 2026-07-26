import json
from urllib import error, request

from .config import AgentConfig
from .client import ChatModelClient


class OllamaClient(ChatModelClient):
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
            "format": "json",
        }
        url = f"{self.config.base_url.rstrip('/')}{self.config.api_path}"
        http_request = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        if isinstance(body, dict):
            if "message" in body and isinstance(body["message"], dict):
                content = body["message"].get("content", "")
                return str(content)
            if "response" in body:
                return str(body["response"])

        raise RuntimeError("Unexpected Ollama response shape")

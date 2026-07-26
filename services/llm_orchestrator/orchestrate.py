import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentRequest:
	messages: list[dict[str, str]]
	tools: list[dict[str, Any]] = field(default_factory=list)
	response_format: str = "json"


@dataclass(frozen=True)
class AgentResponse:
	raw_output: str
	parsed_output: dict[str, Any] | None


class LLMTransport(Protocol):
	def chat(self, messages: list[dict[str, str]]) -> str:
		raise NotImplementedError


class Orchestrator:
	def __init__(self, transport: LLMTransport) -> None:
		self.transport = transport

	def run(self, request: AgentRequest) -> AgentResponse:
		raw_output = self.transport.chat(request.messages)
		return AgentResponse(raw_output=raw_output, parsed_output=self._parse_json(raw_output))

	def _parse_json(self, raw_output: str) -> dict[str, Any] | None:
		try:
			parsed = json.loads(raw_output)
		except json.JSONDecodeError:
			return None
		return parsed if isinstance(parsed, dict) else None


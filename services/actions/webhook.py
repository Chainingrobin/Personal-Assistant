from typing import Any


class WebhookActionAdapter:
	def dispatch(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
		raise NotImplementedError("Wire this to n8n or another webhook backend later")


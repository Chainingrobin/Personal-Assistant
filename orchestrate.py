import json
import ollama
from dataclasses import dataclass, field
from typing import Any, Protocol, Callable

# ─── Tool imports ─────────────────────────────────────────────────────────────
from tools.read_email import read_email
from tools.get_calendar_events import get_calendar_events
from tools.add_calendar_event import add_calendar_event
from tools.draft_email import draft_email
from tools.query_rag import query_rag

# ─── Data contracts (keep these — clean architecture) ─────────────────────────

@dataclass(frozen=True)
class AgentRequest:
    messages: list[dict[str, str]]
    tools: list[Callable] = field(default_factory=list)

@dataclass
class AgentResponse:
    raw_output: str
    tool_called: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    attempts: int = 1
    success: bool = False

# ─── LLM Transport Protocol (swap Ollama for anything later) ──────────────────

class LLMTransport(Protocol):
    def chat(self, messages: list[dict], tools: list[Callable]) -> Any:
        raise NotImplementedError

class OllamaTransport:
    """Concrete implementation of LLMTransport using Ollama."""
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def chat(self, messages: list[dict], tools: list[Callable]) -> Any:
        return ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            options={"temperature": 0.0}
        )

# ─── Tool Registry ────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Callable] = {
    "read_email": read_email,
    "get_calendar_events": get_calendar_events,
    "add_calendar_event": add_calendar_event,
    "draft_email": draft_email,
    "query_rag": query_rag,
}

# ─── Orchestrator ─────────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(self, transport: LLMTransport, max_retries: int = 5):
        self.transport = transport
        self.max_retries = max_retries

    def run(self, request: AgentRequest) -> AgentResponse:
        messages = list(request.messages)
        tools = request.tools

        for attempt in range(1, self.max_retries + 1):
            print(f"\n--- Attempt {attempt} of {self.max_retries} ---")
            response = self.transport.chat(messages, tools)
            msg = response.get("message", {})

            if msg.get("tool_calls"):
                # ✅ Model called a tool
                call = msg["tool_calls"][0]
                name = call["function"]["name"]
                args = call["function"]["arguments"]

                print(f"✅ Tool called: {name}({args})")

                # Dispatch to the registered Python function
                fn = TOOL_REGISTRY.get(name)
                if fn is None:
                    result = f"[ERROR] Unknown tool: {name}"
                else:
                    result = fn(**args)

                print(f"📦 Tool result: {result}")

                return AgentResponse(
                    raw_output=str(msg),
                    tool_called=name,
                    tool_args=args,
                    tool_result=result,
                    attempts=attempt,
                    success=True,
                )

            else:
                # ❌ Model replied with text instead of a tool call
                text = msg.get("content", "")
                print(f"❌ Text reply (no tool call): '{text[:100]}'")

                messages.append(msg)
                messages.append({
                    "role": "user",
                    "content": "CORRECTION: You must use one of your available tools. Do not respond in text. Call the appropriate tool now."
                })

        # Exhausted retries
        return AgentResponse(
            raw_output="",
            success=False,
            attempts=self.max_retries,
        )


# ─── Entry point for manual testing ──────────────────────────────────────────

if __name__ == "__main__":
    transport = OllamaTransport(model="qwen2.5:3b")
    orchestrator = Orchestrator(transport, max_retries=5)

    request = AgentRequest(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict routing assistant for a student productivity system. "
                    "You have access to tools for email, calendar, and knowledge retrieval. "
                    "When the user asks about any of these, you MUST call the appropriate tool. "
                    "Never respond with plain text when a tool is available."
                ),
            },
            {"role": "user", "content": "What do I have coming up this week?"},
        ],
        tools=[read_email, get_calendar_events, add_calendar_event, draft_email, query_rag],
    )

    result = orchestrator.run(request)

    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(f"Success:      {result.success}")
    print(f"Tool called:  {result.tool_called}")
    print(f"Arguments:    {result.tool_args}")
    print(f"Tool result:  {result.tool_result}")
    print(f"Attempts:     {result.attempts}")
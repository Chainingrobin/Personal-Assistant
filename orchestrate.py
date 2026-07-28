import logging

import ollama
from dataclasses import dataclass, field
from typing import Any, Protocol, Callable

from tools.read_email import read_email
from tools.get_calendar_events import get_calendar_events
from tools.add_calendar_event import add_calendar_event
from tools.draft_email import draft_email
from tools.query_rag import query_rag


logger = logging.getLogger(__name__)

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
    model_reasoning: str | None = None   # NEW: any text the model produced pre-dispatch
    attempts: int = 1
    success: bool = False


class LLMTransport(Protocol):
    def chat(self, messages: list[dict], tools: list[Callable]) -> Any:
        raise NotImplementedError


class OllamaTransport:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def chat(self, messages: list[dict], tools: list[Callable]) -> Any:
        return ollama.chat(model=self.model, messages=messages, tools=tools,
                            options={"temperature": 0.0})


TOOL_REGISTRY: dict[str, Callable] = {
    "read_email": read_email,
    "get_calendar_events": get_calendar_events,
    "add_calendar_event": add_calendar_event,
    "draft_email": draft_email,
    "query_rag": query_rag,
}


class Orchestrator:
    def __init__(self, transport: LLMTransport, max_retries: int = 5, verbose: bool = True):
        self.transport = transport
        self.max_retries = max_retries
        self.verbose = verbose

    def _log(self, *parts):
        if self.verbose:
            print(*parts)

    def run(self, request: AgentRequest, current_user_id: str = "youssef") -> AgentResponse:
        messages = list(request.messages)
        tools = request.tools

        for attempt in range(1, self.max_retries + 1):
            self._log(f"\n--- Attempt {attempt} of {self.max_retries} ---")
            response = self.transport.chat(messages, tools)
            msg = response.get("message", {})

            # ── ALWAYS show the model's raw reasoning text, tool call or not ──
            reasoning_text = msg.get("content", "").strip()
            #if reasoning_text:
                #self._log(f"💭 Model reasoning/text: '{reasoning_text}'")

            if msg.get("tool_calls"):
                call = msg["tool_calls"][0]
                name = call["function"]["name"]
                args = dict(call["function"].get("arguments") or {})

                self._log(f"✅ Tool selected: {name}")
                self._log(f"📋 Arguments passed: {args if args else '(empty — check reasoning above)'}")

                fn = TOOL_REGISTRY.get(name)
                if fn:
                    # The LLM chooses the tool and task arguments only; user identity comes from the face/voice layer.
                    args["user_id"] = current_user_id
                    try:
                        tool_result = fn(**args)
                    except Exception as exc:
                        logger.exception("Tool %s failed", name)
                        tool_result = f"[ERROR] Tool {name} failed: {exc}"
                else:
                    tool_result = f"[ERROR] Unknown tool: {name}"

                self._log(f"📦 Raw tool result:\n{tool_result}")

                messages.append(msg)
                messages.append({"role": "tool", "content": str(tool_result)})

                final = self.transport.chat(messages, [])
                final_text = final.get("message", {}).get("content", "")
                

                return AgentResponse(
                    raw_output=final_text,
                    tool_called=name,
                    tool_args=args,
                    tool_result=str(tool_result),
                    model_reasoning=reasoning_text or None,
                    attempts=attempt,
                    success=True,
                )

            else:
                # ── NEW: Allow the model to just chat without forcing a tool ──
                self._log("💬 No tool needed — model replied with conversational text.")
                return AgentResponse(
                    raw_output=reasoning_text,
                    model_reasoning=reasoning_text or None,
                    attempts=attempt,
                    success=True,
                )

        return AgentResponse(raw_output="", success=False, attempts=self.max_retries)


if __name__ == "__main__":
    transport = OllamaTransport(model="qwen2.5:3b")
    orchestrator = Orchestrator(transport, max_retries=5, verbose=True)

    request = AgentRequest(
        messages=[
            {"role": "system", "content": (
                "You are a strict routing assistant for a student productivity system. "
                "You have access to tools for email, calendar, and knowledge retrieval. "
                "When the user asks about any of these, you MUST call the appropriate tool. "
                "Never respond with plain text when a tool is available."
            )},
            {"role": "user", "content": ""},
        ],
        tools=[read_email, get_calendar_events, add_calendar_event, draft_email, query_rag],
    )

    result = orchestrator.run(request, current_user_id="youssef")

    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY")
    print("="*50)
    print(f"Success:          {result.success}")
    print(f"Tool called:      {result.tool_called}")
    print(f"Arguments:        {result.tool_args}")
    print(f"Model reasoning:  {result.model_reasoning}")
    print(f"Attempts needed:  {result.attempts}")
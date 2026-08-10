"""
orchestrate.py — single request/response cycle for the Jarvis/Aegis assistant.

Optimizations applied vs. the original:
  1. num_ctx capped at 2048 — reduces attention compute on every token.
  2. num_predict capped at 512 — avoids runaway generation.
  3. keep_alive=-1 passed per-request — model stays loaded between turns even
     if the systemd env var wasn't set (belt-and-suspenders).
  4. Second (summarization) LLM call uses a minimal 2-message context instead
     of the full conversation history — cuts its token count dramatically.
  5. Streaming used on the summarization call — first token prints immediately
     so the user sees output start ~1-2 s into a 10-15 s inference instead of
     waiting for the whole thing.

Nothing in main.py needs to change — the public API (Orchestrator, OllamaTransport,
AgentRequest, AgentResponse) is identical to the original.
"""

import logging
import ollama
from dataclasses import dataclass, field
from typing import Any, Protocol, Callable
from datetime import datetime


from tools.read_email import read_email
from tools.get_calendar_events import get_calendar_events
from tools.add_calendar_event import add_calendar_event
from tools.draft_email import draft_email
from tools.query_rag import query_rag

logger = logging.getLogger(__name__)


# Build dynamic system prompt before calling Ollama
today_str = datetime.now().strftime("%A, %B %d, %Y")

# ---------------------------------------------------------------------------
# Shared Ollama options — applied to every request so you only tune one place.
# ---------------------------------------------------------------------------
from config import AGENT_CONFIG

def _build_options(profile: str, temperature: float) -> dict[str, Any]:
    """
    Returns Ollama inference options tuned for the active hardware profile.
    Add new profiles here as needed — orchestrate.py itself never checks
    the profile string directly, so adding "jetson" or "mac" later is trivial.
    """
    base = {"temperature": temperature, "keep_alive": -1}

    if profile == "pi":
        return {
            **base,
            "num_ctx": 2048,    # caps attention compute — scales quadratically
            "num_predict": 512, # tool calls and summaries are short
        }

    # desktop / any other profile: let Ollama use its own defaults
    # num_ctx and num_predict are intentionally omitted so the 4060
    # can use its full context window and generate freely
    return base

_BASE_OPTIONS = _build_options(AGENT_CONFIG.hardware_profile, AGENT_CONFIG.temperature)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
    model_reasoning: str | None = None
    attempts: int = 1
    success: bool = False


# ---------------------------------------------------------------------------
# Transport layer
# ---------------------------------------------------------------------------

class LLMTransport(Protocol):
    def chat(self, messages: list[dict], tools: list[Callable]) -> Any:
        raise NotImplementedError

    def chat_stream(self, messages: list[dict]) -> str:
        raise NotImplementedError


class OllamaTransport:
    def __init__(self, model: str = "qwen3:1.7b"):
        self.model = model

    def chat(self, messages: list[dict], tools: list[Callable]) -> Any:
        """Blocking chat — used for the tool-selection pass where we need the
        full response before we can do anything."""
        return ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            options=_BASE_OPTIONS,
        )

    def chat_stream(self, messages: list[dict]) -> str:
        """Streaming chat — used for the summarization pass so the user sees
        the first token immediately rather than waiting for full inference.
        Returns the complete assembled string for logging / AgentResponse."""
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options=_BASE_OPTIONS,
        )
        parts: list[str] = []
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                print(token, end="", flush=True)
                parts.append(token)
        print()   # newline after the streamed response finishes
        return "".join(parts)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable] = {
    "read_email":           read_email,
    "get_calendar_events":  get_calendar_events,
    "add_calendar_event":   add_calendar_event,
    "draft_email":          draft_email,
    "query_rag":            query_rag,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(
        self,
        transport: LLMTransport,
        max_retries: int = 5,
        verbose: bool = True,
    ):
        self.transport = transport
        self.max_retries = max_retries
        self.verbose = verbose

    def _log(self, *parts):
        if self.verbose:
            print(*parts)

    def run(
        self,
        request: AgentRequest,
        current_user_id: str ,
    ) -> AgentResponse:
        messages = list(request.messages)
        tools = request.tools

        for attempt in range(1, self.max_retries + 1):
            self._log(f"\n--- Attempt {attempt} of {self.max_retries} ---")

            # ── Pass 1: tool selection (blocking, full context) ──────────────
            response = self.transport.chat(messages, tools)
            msg = response.get("message", {})
            reasoning_text = msg.get("content", "").strip()

            if msg.get("tool_calls"):
                call = msg["tool_calls"][0]
                name = call["function"]["name"]
                args = dict(call["function"].get("arguments") or {})

                self._log(f"✅ Tool selected: {name}")
                self._log(f"📋 Arguments: {args if args else '(empty)'}")

                fn = TOOL_REGISTRY.get(name)
                if fn:
                    args["user_id"] = current_user_id
                    try:
                        tool_result = fn(**args)
                    except Exception as exc:
                        logger.exception("Tool %s failed", name)
                        tool_result = f"[ERROR] Tool {name} failed: {exc}"
                else:
                    tool_result = f"[ERROR] Unknown tool: {name}"

                self._log(f"📦 Raw tool result:\n{tool_result}")

                # ── Pass 2: summarization (streaming, minimal context) ───────
                # Only send what the model actually needs — the tool name and
                # its result.  Sending the full conversation history here is
                # wasted tokens: the model produced a natural-language summary
                # either way, and the extra context doesn't improve quality for
                # this task while costing significant compute on a Pi.
                summary_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a concise assistant. Summarize the tool result "
                            "below in one to three natural sentences for the user. "
                            "Do not repeat the raw data verbatim; highlight what matters."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Tool used: {name}\n"
                            f"Result:\n{tool_result}"
                        ),
                    },
                ]

                self._log("🖨️  Streaming response:")
                final_text = self.transport.chat_stream(summary_messages)

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
                # ── Conversational turn — no tool needed ────────────────────
                self._log("💬 No tool needed — streaming conversational reply:")

                if reasoning_text:
                    for char in reasoning_text:
                        print(char, end="", flush=True)
                    print()
                    return AgentResponse(
                        raw_output=reasoning_text,
                        model_reasoning=reasoning_text,
                        attempts=attempt,
                        success=True,
                    )
                else:
                    # FALLBACK: If pass 1 returned no tool call and empty text,
                    # force a streaming chat call so the user always gets a response.
                    final_text = self.transport.chat_stream(messages)
                    return AgentResponse(
                        raw_output=final_text,
                        attempts=attempt,
                        success=bool(final_text),
                    )
                
        return AgentResponse(raw_output="", success=False, attempts=self.max_retries)


# ---------------------------------------------------------------------------
# Self-test — python orchestrate.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = OllamaTransport(model="qwen2.5:3b")
    orchestrator = Orchestrator(transport, max_retries=5, verbose=True)

    request = AgentRequest(
        messages=[
            {
                "role": "system",
                "content": (
                    """You are a strict routing assistant for a student productivity system. 
                    You have access to tools for email, calendar, and knowledge retrieval.
                    When the user asks about any of these, you MUST call the appropriate tool.
                    Never respond with plain text when a tool is available.
                    Current Date: {today_str}

                    DIRECT CONVERSATION RULES:
                    - For general knowledge, date/time questions ("what's today's date?", "what day is it?"), answer DIRECTLY without calling any tools.

                    TOOL USAGE RULES:
                    - ONLY call `get_calendar_events` when the user explicitly asks about their schedule, appointments, calendar, or events.
                    - Do NOT call `get_calendar_events` just to check today's date."""
                ),
            },
            {"role": "user", "content": "What emails do I have?"},
        ],
        tools=[read_email, get_calendar_events, add_calendar_event, draft_email, query_rag],
    )

    result = orchestrator.run(request, current_user_id="youssef")

    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"Success:          {result.success}")
    print(f"Tool called:      {result.tool_called}")
    print(f"Arguments:        {result.tool_args}")
    print(f"Model reasoning:  {result.model_reasoning}")
    print(f"Attempts needed:  {result.attempts}")
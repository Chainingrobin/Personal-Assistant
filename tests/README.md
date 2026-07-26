# Personal Assistant Testing Suite - Agentic Tests for LLMs

## Overview

This repository provides synthetic testing environments to evaluate the Qwen2.5:1.5b model (via local Ollama) on agentic tasks such as email parsing and calendar event extraction **before** connecting to real APIs like Gmail/Outlook or Google Calendar.

## Quick Start with Local Ollama + qwen2.5:1.5b

### Step 1: Ensure Ollama is Running

```bash
docker-compose up ollama
# Then in a separate terminal:
ollama pull qwen2.5:1.5b
ollama show qwen2.5:1.5b
```

### Step 2: Run Synthetic Email Parsing Tests

```python
from tests.agentic.mail.parser import MailParserAgent, mail_parser_tests
import asyncio

# Configure Ollama connection
mail_parser = MailParserAgent(base_url="http://localhost:11434", model="qwen2.5:1.5b")

async def run():
    results = await mail_parser.run(mail_parser_tests())

for result in results.results[:10]:  # View first 10 test results
    print(result)
```

### Step 3: Run Synthetic Calendar Event Tests

```python
from tests.agentic.calendar import CalendarEventAgent, calendar_test_suite

calendar_agent = CalendarEventAgent(base_url="http://localhost:11434", model="qwen2.5:1.5b")

# Execute full test suite
test_suite_result = await calendar_agent.run(calendar_test_suite())
print(test_suite_result.summary)
```

## Architecture Diagram for Agentic Testing

```
┌─────────────────────────────────────────────────────────────┐
│                    Synthetic Data Store                       │
│  ┌──────────────────┐      ┌──────────────────┐              │
│  │ Mock Emails JSON │◄────►│ Calendar Events   │              │
│  └──────────────────┘      └──────────────────┘              │
└─────────────────────────┬─────────────────────────────────────┤
                          │                                        │
                          ▼                                        │
┌─────────────────────────────────────────────────────────────┐
│          LLM Orchestrator (Qwen2.5 via Ollama)               │
│  ┌──────────────────┴─────────────────────────┐              │
│  │ Agent Registry                              │              │
│  │ • MailParserAgent                           │◄────────────┤
│  │ • CalendarEventAgent                        │              │
│  │ • EventLoggerAgent                          │              │
│  └──────────────────┬─────────────────────────┘              │
└─────────────────────┼─────────────────────────────────────────┤
                      ▼                                         │
        ┌────────────────────────────────────┐                  │
        │   Test Harness & Metrics           │◄────────────────┤
        │ • Tool Calling Evaluation          │  (Ollama API)    │
        │ • Response Accuracy Measurement    │                   │
        │ • Error Handling Assessment        │                   │
        └─────────────────────────────────────┘                  │
                          ▲                                       │
                          │                                        │
            ┌─────────────┴──────────┐                           │
            │                         │  Evaluation Results       │
            ▼                         ▼                            │
    ┌─────────────────┐     ┌─────────────────┐                   │
    │   Pass/Fail     │     │   Detailed      │                   │
    │ Report          │     │ Reports         │                   │
    └─────────────────┘     └─────────────────┘                   │
```

## Directory Structure for Agentic Tests

```
tests/
  agentic/                    # Core agent implementations and tests
    __init__.py
    mail/                     # Email parsing agents/tests
      parser.py              # MailParserAgent class
      parser_test_data.json  # Synthetic email fixtures
      test_parser.py         # Unit/integration tests for mail parsing
    calendar/                 # Calendar event extraction agents/tests
      extractor.py           # CalendarEventExtractor agent
      schedule_agent.py      # Schedule creation/planning agent
      fixture_events.json    # Mock calendar events
      test_calendar.py       # Tests for calendar operations
  synthetic_data/             # All mock data fixtures
    emails/
      inbox_sample.json      # Synthetic email corpus (100+ messages)
      important_emails.json   # Test case: critical emails to detect
      spam_emails.json        # Edge cases and noise handling
    calendars/
      google_calendar_format.json  # Mock Google Calendar events
      outlook_calendar_format.json # Microsoft Outlook format mockups
      recurring_events.json        # Complex recurrence patterns
    logs/                    # Event logging synthetic data
      user_activity_stream.json   # Test case: activity extraction
    mixed_scenarios/         # Combined scenarios testing multiple agents
      morning_routine_scenario.json  # Tests multi-agent collaboration
      workday_summary_scenario.json  # Email digest + calendar sync tests
  test_harnesses/             # Evaluation frameworks and metrics
    __init__.py
    base_agent_test.py       # Abstract base class for agent evaluation
    email_parser_metrics.py  # Specific metrics: extraction_rate, false_positives
    calendar_event_metrics.py# Metrics: event_accuracy, slot_filling_score
    tool_calling_evaluator.py # Evaluates function/tool call correctness
  fixtures/                   # Shared test configuration and utilities
    __init__.py
    ollama_client.py         # Ollama connection wrapper with retry logic
    prompt_templates.py      # Test prompts for agent evaluation
    validators.py            # JSON schema validators for LLM outputs
```

---

**Note:** Run `tests/run_all_synthetic_tests.sh` after implementing all agents to execute the full test suite.

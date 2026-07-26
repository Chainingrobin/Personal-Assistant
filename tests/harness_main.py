from argparse import ArgumentParser
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.agentic.config import AgentConfig
from tests.agentic.factory import create_client
from tests.agentic.runner import AgenticTestRunner


def parse_args() -> AgentConfig:
    parser = ArgumentParser(description="Run synthetic agentic tests against a local chat model.")
    parser.add_argument("--provider", default="ollama", help="Chat backend provider name")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Chat backend base URL")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Model name to test")
    parser.add_argument("--timeout-seconds", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--api-path", default="/api/chat", help="HTTP path for the chat endpoint")
    args = parser.parse_args()
    return AgentConfig(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        api_path=args.api_path,
        provider=args.provider,
    )


def main() -> None:
    config = parse_args()
    client = create_client(config)
    report = AgenticTestRunner(config=config, client=client).run_all()
    print(
        f"cases={report.total_cases} passed={report.passed_cases} "
        f"failed={report.failed_cases} avg={report.average_score:.2f}"
    )
    for result in report.results:
        print(f"- {result.case_id}: score={result.score:.2f} passed={result.passed}")


if __name__ == "__main__":
    main()

import json
from statistics import mean
from typing import Any

from .config import AgentConfig
from .client import ChatModelClient
from .ollama_client import OllamaClient
from .schemas import CaseResult, RunReport, SyntheticCase
from .scenarios import build_cases

class AgenticTestRunner:
    def __init__(self, config: AgentConfig | None = None, client: ChatModelClient | None = None) -> None:
        self.config = config or AgentConfig()
        self.client = client or OllamaClient(self.config)

    def run_case(self, case: SyntheticCase) -> CaseResult:
        # 1. Dynamically give the model the exact JSON keys we expect
        expected_keys = list(case.expected.keys())
        schema_instruction = (
            f"You are a strict data extraction engine. Return ONLY valid JSON. "
            f"The JSON must contain exactly these keys: {expected_keys}"
        )
        
        messages = [
            {"role": "system", "content": schema_instruction},
            {"role": "user", "content": case.prompt},
        ]
        
        raw_output = self.client.chat(messages)
        parsed_output = self._parse_json(raw_output)
        score, notes = self._score_case(case.expected, parsed_output)
        
        # 2. Print a detailed debug evaluation to the terminal
        print("\n" + "="*60)
        print(f"CASE ID: {case.case_id} ({case.task_type})")
        print("-" * 60)
        print(f"PROMPT:\n{case.prompt}")
        print("-" * 60)
        print(f"EXPECTED JSON: {json.dumps(case.expected, indent=2)}")
        print(f"RAW MODEL OUTPUT:\n{raw_output}")
        print("-" * 60)
        print("GRADING NOTES:")
        for note in notes:
            print(f"  * {note}")
        print(f"SCORE: {score:.2f} | PASSED: {score >= 0.8}")
        print("="*60 + "\n")
        
        return CaseResult(
            case_id=case.case_id,
            task_type=case.task_type,
            score=score,
            passed=score >= 0.8,
            raw_output=raw_output,
            parsed_output=parsed_output,
            notes=notes,
        )

    def run_all(self) -> RunReport:
        results = [self.run_case(case) for case in build_cases()]
        scores = [result.score for result in results]
        passed_cases = sum(1 for result in results if result.passed)
        return RunReport(
            total_cases=len(results),
            passed_cases=passed_cases,
            failed_cases=len(results) - passed_cases,
            average_score=mean(scores) if scores else 0.0,
            results=results,
        )

    def _parse_json(self, raw_output: str) -> dict[str, Any] | None:
        try:
            # Strip markdown formatting in case the model wraps the JSON in ```json blocks
            clean_text = raw_output.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _score_case(self, expected: dict[str, Any], parsed: dict[str, Any] | None) -> tuple[float, list[str]]:
        if not parsed:
            return 0.0, ["CRITICAL: Model output was not valid JSON or was missing entirely."]

        matched = 0
        notes: list[str] = []
        for key, expected_value in expected.items():
            actual_value = parsed.get(key)
            
            # 3. Flexible soft-grading
            if actual_value is None:
                notes.append(f"FAIL (MISSING): Key '{key}' was not found in the output.")
                continue
                
            # Convert both to strings and lowercase to ignore formatting differences
            str_expected = str(expected_value).lower().strip()
            str_actual = str(actual_value).lower().strip()
            
            if str_expected == str_actual:
                matched += 1
                notes.append(f"PASS: '{key}' matched exactly.")
            elif str_expected in str_actual or str_actual in str_expected:
                # Give partial credit (0.5) if the answer is inside the model's text
                matched += 0.5
                notes.append(f"PARTIAL PASS: '{key}' soft matched. Expected: '{expected_value}', Got: '{actual_value}'")
            else:
                notes.append(f"FAIL: '{key}'. Expected: '{expected_value}', Got: '{actual_value}'")

        total = max(len(expected), 1)
        return matched / total, notes
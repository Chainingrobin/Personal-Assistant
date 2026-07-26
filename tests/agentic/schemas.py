from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    task_type: str
    prompt: str
    expected: dict[str, Any]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    task_type: str
    score: float
    passed: bool
    raw_output: str
    parsed_output: dict[str, Any] | None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunReport:
    total_cases: int
    passed_cases: int
    failed_cases: int
    average_score: float
    results: list[CaseResult]

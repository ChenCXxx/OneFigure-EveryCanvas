from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


_case_name: ContextVar[str | None] = ContextVar("benchmark_case", default=None)
_active_recorder: "BenchmarkRecorder | None" = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class BenchmarkRecorder:
    def __init__(
        self,
        output_dir: Path,
        pricing: dict[str, dict[str, float]],
    ) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.output_dir / "api_calls.jsonl"
        self.summary_path = self.output_dir / "benchmark_metrics.json"
        self.pricing = pricing
        self._events: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self.calls_path.write_text("", encoding="utf-8")

    def record(self, event: dict[str, object]) -> None:
        model = str(event.get("model", ""))
        input_tokens = event.get("input_tokens")
        output_tokens = event.get("output_tokens")
        model_price = self.pricing.get(model)
        cost_usd: float | None = None
        if (
            model_price is not None
            and isinstance(input_tokens, int)
            and isinstance(output_tokens, int)
        ):
            input_price = float(model_price.get("input", 0.0))
            output_price = float(model_price.get("output", 0.0))
            threshold = model_price.get("threshold_tokens")
            if threshold is not None and input_tokens > float(threshold):
                input_price = float(
                    model_price.get("input_above_threshold", input_price)
                )
                output_price = float(
                    model_price.get("output_above_threshold", output_price)
                )
            cost_usd = (
                input_tokens * input_price
                + output_tokens * output_price
            ) / 1_000_000

        row = {
            "timestamp": _timestamp(),
            "case": _case_name.get(),
            **event,
            "cost_usd": round(cost_usd, 10) if cost_usd is not None else None,
        }
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._events.append(row)
            with self.calls_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def write_summary(
        self,
        *,
        wall_clock_seconds: float,
        case_results: list[dict[str, object]],
    ) -> Path:
        with self._lock:
            events = list(self._events)

        total_calls = len(events)
        successful_calls = sum(event.get("success") is True for event in events)
        failed_calls = total_calls - successful_calls
        retry_calls = sum(event.get("is_retry") is True for event in events)
        input_tokens = sum(
            int(event["input_tokens"])
            for event in events
            if isinstance(event.get("input_tokens"), int)
        )
        output_tokens = sum(
            int(event["output_tokens"])
            for event in events
            if isinstance(event.get("output_tokens"), int)
        )
        priced_calls = [event for event in events if event.get("cost_usd") is not None]
        total_cost = sum(float(event["cost_usd"]) for event in priced_calls)
        attempted_cases = [
            result for result in case_results if result.get("status") != "skipped"
        ]
        failed_cases = [
            result
            for result in attempted_cases
            if result.get("status") in {"failed", "missing"}
        ]

        by_component: dict[str, dict[str, object]] = {}
        for event in events:
            component = str(event.get("component", "unknown"))
            row = by_component.setdefault(
                component,
                {
                    "calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "retry_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            row["calls"] = int(row["calls"]) + 1
            key = "successful_calls" if event.get("success") is True else "failed_calls"
            row[key] = int(row[key]) + 1
            if event.get("is_retry") is True:
                row["retry_calls"] = int(row["retry_calls"]) + 1
            if isinstance(event.get("input_tokens"), int):
                row["input_tokens"] = int(row["input_tokens"]) + int(
                    event["input_tokens"]
                )
            if isinstance(event.get("output_tokens"), int):
                row["output_tokens"] = int(row["output_tokens"]) + int(
                    event["output_tokens"]
                )
            if event.get("cost_usd") is not None:
                row["cost_usd"] = float(row["cost_usd"]) + float(event["cost_usd"])

        for row in by_component.values():
            row["cost_usd"] = round(float(row["cost_usd"]), 10)

        summary = {
            "generated_at": _timestamp(),
            "definitions": {
                "api_call": "One LLMClient.generate invocation (logical request).",
                "retry": "An agent validation retry with attempt > 1.",
                "wall_clock": "Elapsed real time for the whole batch.",
                "sdk_transport_retries_included": False,
            },
            "wall_clock_seconds": round(wall_clock_seconds, 3),
            "cases": {
                "attempted": len(attempted_cases),
                "completed": sum(
                    result.get("status") == "completed" for result in attempted_cases
                ),
                "failed": len(failed_cases),
                "failure_rate": (
                    round(len(failed_cases) / len(attempted_cases), 6)
                    if attempted_cases
                    else 0.0
                ),
            },
            "api": {
                "calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "failure_rate": (
                    round(failed_calls / total_calls, 6) if total_calls else 0.0
                ),
                "retry_calls": retry_calls,
                "retry_rate": (
                    round(retry_calls / total_calls, 6) if total_calls else 0.0
                ),
            },
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
            "cost": {
                "currency": "USD",
                "total": round(total_cost, 10),
                "priced_calls": len(priced_calls),
                "unpriced_calls": total_calls - len(priced_calls),
                "complete": len(priced_calls) == total_calls,
            },
            "by_component": by_component,
            "files": {
                "api_calls_jsonl": str(self.calls_path),
            },
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self.summary_path


def set_active_recorder(recorder: BenchmarkRecorder | None) -> None:
    global _active_recorder
    _active_recorder = recorder


def record_api_call(event: dict[str, object]) -> None:
    recorder = _active_recorder
    if recorder is not None:
        recorder.record(event)


@contextmanager
def benchmark_case(case_name: str) -> Iterator[None]:
    token = _case_name.set(case_name)
    try:
        yield
    finally:
        _case_name.reset(token)

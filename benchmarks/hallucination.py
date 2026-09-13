"""Hallucination benchmark evaluator."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from ..llm import LLMClient
    from ..schemas import BenchmarkCase, BenchmarkResult
    from ..utils import parse_json_text
except ImportError:  # Support direct imports when running from benchmark/
    from llm import LLMClient
    from schemas import BenchmarkCase, BenchmarkResult
    from utils import parse_json_text


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "halluc_prompt.txt"


class HallucinationBenchmark:
    """Compare each method image against the case reference image."""

    name = "hallucination"

    def __init__(self, llm: LLMClient, prompt_path: Path | None = None) -> None:
        self.llm = llm
        path = prompt_path or DEFAULT_PROMPT_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Hallucination prompt not found: {path}")
        self.prompt = path.read_text(encoding="utf-8")

    def run(self, case: BenchmarkCase) -> list[BenchmarkResult]:
        """Run benchmark for all methods"""
        return [
            self._evaluate_method(case, method.name, method.path)
            for method in case.method_images
        ]

    def _evaluate_method(
        self,
        case: BenchmarkCase,
        method_name: str,
        method_path: Path,
    ) -> BenchmarkResult:
        """Evaluate hallucination for a single method."""
        print(
            f"[{case.key}] [hallucination] [method={method_name}] start",
            flush=True,
        )
        metadata={"reference": str(case.reference_path), "candidate": str(method_path)}
        try:
            # Call the LLM
            response = self.llm.generate(
                prompt=self.prompt,
                images=[case.reference_path, method_path],
            )
            # Parse the response
            payload = parse_hallucination_response(response.text)
            # Add metadata
            metadata = {
                "reference": str(case.reference_path),
                "candidate": str(method_path),
                "model": response.model,
                "provider": response.provider,
            }
            # Return the result
            result = BenchmarkResult(self.name, "success", method_name, payload, metadata, None)
        except Exception as exc:  # ERROR: status="failed"
            result = BenchmarkResult(self.name, "failed", method_name, {}, metadata, f"{type(exc).__name__}: {exc}")

        print(
            f"[{case.key}] [hallucination] [method={method_name}] "
            f"done: {result.status}",
            flush=True,
        )
        return result


def parse_hallucination_response(text: str) -> dict[str, Any]:
    """
    Parse the LLM response. 
    Expected JSON format:
    {
        "correct": true,
        "has_hallucination": false,
        "differences": []
    }
    """
    # Parse the JSON text
    data = parse_json_text(text)
    # ERROR: empty data
    if not data:
        raise ValueError("Model response could not be parsed as JSON.")

    differences = data.get("differences", [])
    if not isinstance(differences, list):
        differences = [str(differences)]
    differences = [str(item) for item in differences]

    correct = bool(data.get("correct", False)) and not differences
    return {
        "correct": correct,
        "has_hallucination": not correct,
        "differences": differences,
    }

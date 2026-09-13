"""Relationship/connection consistency benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from ..llm import LLMClient
    from ..schemas import BenchmarkCase, BenchmarkResult
    from ..utils import parse_json_text
except ImportError:
    from llm import LLMClient
    from schemas import BenchmarkCase, BenchmarkResult
    from utils import parse_json_text


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rel_prompt.txt"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rel_system_prompt.txt"


class RelationshipBenchmark:
    """Compare structural connections in each candidate with the reference."""

    name = "relationship"

    def __init__(self, llm: LLMClient, prompt_path: Path | None = None, system_prompt_path: Path | None = None) -> None:
        self.llm = llm
        path = prompt_path or DEFAULT_PROMPT_PATH
        system_path = system_prompt_path or DEFAULT_SYSTEM_PROMPT_PATH
        if not path.is_file() or not system_path.is_file():
            raise FileNotFoundError(f"Relationship prompt not found: {path} or {system_path}")
        self.prompt = path.read_text(encoding="utf-8")
        self.system_prompt = system_path.read_text(encoding="utf-8")

    def run(self, case: BenchmarkCase) -> list[BenchmarkResult]:
        return [self._evaluate_method(case, method.name, method.path) for method in case.method_images]

    def _evaluate_method(self, case: BenchmarkCase, method_name: str, method_path: Path) -> BenchmarkResult:
        metadata = {"reference": str(case.reference_path), "candidate": str(method_path)}
        print(
            f"[{case.key}] [relationship] [method={method_name}] start",
            flush=True,
        )
        try:
            # Call the LLM
            response = self.llm.generate(
                prompt=self.prompt,
                system_prompt=self.system_prompt,
                images=[case.reference_path, method_path],
            )
            # Parse the response
            payload = parse_relationship_response(response.text)
            # Add metadata
            metadata.update({"model": response.model, "provider": response.provider})
            result = BenchmarkResult(self.name, "success", method_name, payload, metadata, None)
        except Exception as exc:
            result = BenchmarkResult(self.name, "failed", method_name, {}, metadata, f"{type(exc).__name__}: {exc}")

        print(
            f"[{case.key}] [relationship] [method={method_name}] "
            f"done: {result.status}",
            flush=True,
        )
        return result


def parse_relationship_response(text: str) -> dict[str, Any]:
    """
    Parse the LLM response. 
    Expected JSON format:
    {
        "correct": true,
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

    return {
        "correct": bool(data.get("correct", False)),
        "differences": differences,
    }

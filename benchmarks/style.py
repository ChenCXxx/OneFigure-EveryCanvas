"""Style benchmark: rank several candidate images against one reference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from ..llm import ImagePart, LLMClient, TextPart
    from ..schemas import BenchmarkCase, BenchmarkResult
    from ..utils import parse_json_text
except ImportError:  # Support direct imports from benchmark/
    from llm import ImagePart, LLMClient, TextPart
    from schemas import BenchmarkCase, BenchmarkResult
    from utils import parse_json_text


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
DEFAULT_PROMPT_PATH = PROMPT_DIR / "style_prompt.txt"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPT_DIR / "style_system_prompt.txt"
REFERENCE_DIR = (Path(__file__).resolve().parent / "space_reference")

class StyleBenchmark:
    """Rank all methods in one case by visual style similarity."""

    name = "style"

    def __init__(
        self,
        llm: LLMClient,
        prompt_path: Path | None = None,
        system_prompt_path: Path | None = None,
    ) -> None:
        self.llm = llm
        prompt_path = prompt_path or DEFAULT_PROMPT_PATH
        system_prompt_path = system_prompt_path or DEFAULT_SYSTEM_PROMPT_PATH

        if not prompt_path.is_file():
            raise FileNotFoundError(f"Style prompt not found: {prompt_path}")
        if not system_prompt_path.is_file():
            raise FileNotFoundError(
                f"Style system prompt not found: {system_prompt_path}"
            )

        self.prompt = prompt_path.read_text(encoding="utf-8")
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8")

    def run(self, case: BenchmarkCase) -> BenchmarkResult:
        if not case.method_images:
            raise ValueError(f"{case.key}: style requires at least one method")

        method_names = ", ".join(
            method.name for method in case.method_images
        )
        print(
            f"[{case.key}] [style] evaluating methods: {method_names}",
            flush=True,
        )

        parts = self._build_parts(case)
        metadata = {
            "reference": str(case.reference_path),
            "candidates": {
                method.name: str(method.path)
                for method in case.method_images
            },
        }

        try:
            response = self.llm.generate(
                system_prompt=self.system_prompt,
                parts=parts,
            )
            payload = parse_style_response(response.text, case)
            metadata.update({
                "model": response.model,
                "provider": response.provider,
            })
            result = BenchmarkResult(self.name, "success", None, payload, metadata, None)
        except Exception as exc:  # ERROR: status="failed"
            result = BenchmarkResult(self.name, "failed", None, {}, metadata, f"{type(exc).__name__}: {exc}")

        print(
            f"[{case.key}] [style] evaluation done: {result.status}",
            flush=True,
        )
        return result

    def _build_parts(self, case: BenchmarkCase) -> list[TextPart | ImagePart]:
        """Build the original ordering: reference is image 0, candidates 1..N."""
        parts: list[TextPart | ImagePart] = [
            TextPart("Image 0 - REFERENCE IMAGE:"),
            ImagePart(case.reference_path),
        ]

        for index, method in enumerate(case.method_images, start=1):
            parts.extend([
                TextPart(
                    f"Image {index} - CANDIDATE {index} "
                    f"({method.name}):"
                ),
                ImagePart(method.path),
            ])

        parts.append(TextPart(self.prompt))
        return parts


def parse_style_response(
    text: str,
    case: BenchmarkCase,
) -> dict[str, Any]:
    """Validate ranking JSON and map candidate indices back to method names."""
    data = parse_json_text(text)
    if not data:
        raise ValueError("Model response could not be parsed as JSON.")

    descriptions = data.get("descriptions")
    if not isinstance(descriptions, list):
        raise ValueError("Style response field 'descriptions' must be a list.")

    ranking: list[dict[str, Any]] = []
    seen_indices: set[int] = set()

    for item in descriptions:
        if not isinstance(item, dict):
            raise ValueError("Each style description must be an object.")

        candidate_index = item.get("candidate_index")
        rank = item.get("ranking")

        if not isinstance(candidate_index, int):
            raise ValueError("Style response has an invalid candidate_index.")
        if not 1 <= candidate_index <= len(case.method_images):
            raise ValueError(
                f"Style candidate_index out of range: {candidate_index}"
            )
        if candidate_index in seen_indices:
            raise ValueError(
                f"Style response repeats candidate_index: {candidate_index}"
            )
        if not isinstance(rank, int):
            raise ValueError("Style response has an invalid ranking.")

        seen_indices.add(candidate_index)
        method = case.method_images[candidate_index - 1]
        ranking.append({
            "rank": rank,
            "method": method.name,
            "path": str(method.path),
            "description": str(item.get("description", "")),
        })

    if len(seen_indices) != len(case.method_images):
        raise ValueError("Style response does not rank every candidate.")

    ranking.sort(key=lambda item: item["rank"])
    return {
        "ranking": ranking,
        "overall_summary": str(data.get("overall_summary", "")),
    }

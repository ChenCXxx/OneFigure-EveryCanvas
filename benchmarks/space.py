"""Space benchmark orchestration.

This module combines deterministic space checks for every candidate:

1. blank-area analysis and visualization;
2. OCR-based rotation detection;
3. reference/candidate LPIPS comparison.

The current run always writes its own metrics. A later run can optionally
read compatible metrics from ``previous_root``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from ..llm import ImagePart, LLMClient, TextPart
    from ..schemas import BenchmarkCase, BenchmarkResult
    from ..utils import load_json, parse_json_text, save_json
    from .space_tools import (
        analyze_image_rotation,
        analyze_visual_similarity,
        visualize_and_analyze_blank_area,
    )
except ImportError:
    from llm import ImagePart, LLMClient, TextPart
    from schemas import BenchmarkCase, BenchmarkResult
    from utils import load_json, parse_json_text, save_json
    from benchmarks.space_tools import (
        analyze_image_rotation,
        analyze_visual_similarity,
        visualize_and_analyze_blank_area,
    )

# ====================================
# Contants
# ====================================

METRICS_VERSION = 1
LLM_MAX_ATTEMPTS = 3
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
DEFAULT_PROMPT_PATH = PROMPT_DIR / "space_prompt.txt"
DEFAULT_SYSTEM_PROMPT_PATH = PROMPT_DIR / "space_system_prompt.txt"
REFERENCE_DIR = Path(__file__).resolve().parent / "space_reference"

SEVERE_ERROR_REFERENCES = {
    "compression.png": (
        "嚴重錯誤一：視覺平衡中文字壓縮參考圖。"
        "文字擠壓後變得有一點細長，且排版與細節仍與原圖相同，"
        "這種情況屬於壓縮流程圖錯誤。"
    ),
    "unclear_arrow_flow_reference.png": (
        "流向安排不好參考圖：箭頭流向不統一，且混雜意義不明、"
        "難以判斷連接目標的邊或箭頭。"
    ),
}


# ===================================
# Error Reference Images
# ===================================

def _severe_error_reference_images() -> list[tuple[Path, str]]:
    """Return optional calibration images that are not ranking candidates."""
    return [
        (path, description)
        for filename, description in SEVERE_ERROR_REFERENCES.items()
        if (path := REFERENCE_DIR / filename).is_file()
    ]


# ===================================
# Metrics helpers
# ===================================

def _file_sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics_match(
    saved: dict[str, Any],
    *,
    case: BenchmarkCase,
    method_name: str,
    method_path: Path,
) -> bool:
    """Check whether the saved metrics match the current case and method."""
    return (
        saved.get("metrics_version") == METRICS_VERSION
        and saved.get("case_key") == case.key
        and saved.get("method") == method_name
        and saved.get("reference_sha256") == _file_sha256(case.reference_path)
        and saved.get("candidate_sha256") == _file_sha256(method_path)
        and isinstance(saved.get("metrics"), dict)
    )


def build_prompt_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return compact deterministic signals intended for an LLM prompt."""
    blank = metrics.get("blank_area", {})
    ocr = metrics.get("ocr", {})
    similarity = metrics.get("similarity", {})
    return {
        "foreground_ratio": blank.get("foreground_ratio"),
        "empty_area_ratio": blank.get("empty_area_ratio"),
        "background_regions": blank.get("background_regions", []),
        "ocr_has_rotation": ocr.get("ocr_has_rotation"),
        "is_likely_direct_resize": similarity.get(
            "is_likely_direct_resize"
        ),
    }


# ===================================
# Validation and Parsing of LLM Response
# ===================================

def _validate_candidate_entries(
    data: dict[str, Any],
    field_name: str,
    candidate_count: int,
) -> list[dict[str, Any]]:
    """Require one valid candidate index for every candidate."""
    entries = data.get(field_name)
    
    # Check whether "rank" and "descriptions" are listed correctly
    if not isinstance(entries, list):  # Exists
        raise ValueError(
            f"Space response must contain a {field_name} list."
        )
    if len(entries) != candidate_count:  # Count
        raise ValueError(
            f"{field_name} count must equal the method count "
            f"({candidate_count}), got {len(entries)}."
        )

    expected = set(range(1, candidate_count + 1))
    indices: list[int] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Each {field_name} entry must be an object."
            )
        candidate_index = entry.get("candidate_index")
        if not isinstance(candidate_index, int):
            raise ValueError(
                f"{field_name}.candidate_index must be an integer."
            )
        indices.append(candidate_index)

    if set(indices) != expected:
        raise ValueError(
            f"{field_name}.candidate_index must contain exactly "
            f"1..{candidate_count}; got {indices}."
        )

    return entries


def parse_space_response(
    text: str,
    *,
    candidate_count: int,
) -> dict[str, Any]:
    """Parse the existing Space prompt response schema."""
    data = parse_json_text(text)
    if not data:
        raise ValueError("Space model response could not be parsed as JSON.")

    ranking = _validate_candidate_entries(
        data,
        "ranking",
        candidate_count,
    )
    descriptions = _validate_candidate_entries(
        data,
        "descriptions",
        candidate_count,
    )

    severe_errors = data.get("severe_layout_errors", [])
    if not isinstance(severe_errors, list):
        severe_errors = []
    if len(severe_errors) != candidate_count:
        raise ValueError(
            "severe_layout_errors count must equal the method count "
            f"({candidate_count}), got {len(severe_errors)}."
        )

    for entry in severe_errors:
        if not isinstance(entry, dict):
            raise ValueError(
                "Each severe_layout_errors entry must be an object."
            )
        candidate_index = entry.get("candidate_index")
        if not isinstance(candidate_index, int):
            raise ValueError(
                "severe_layout_errors.candidate_index must be an integer."
            )
        if not 1 <= candidate_index <= candidate_count:
            raise ValueError(
                "severe_layout_errors.candidate_index out of range: "
                f"{candidate_index}"
            )
        for field_name in (
            "orientation_error",
            "excessive_empty_space",
            "direct_resize",
        ):
            if not isinstance(entry.get(field_name), bool):
                raise ValueError(
                    "severe_layout_errors."
                    f"{field_name} must be a boolean."
                )

    severe_indices = [entry["candidate_index"] for entry in severe_errors]
    if set(severe_indices) != set(range(1, candidate_count + 1)):
        raise ValueError(
            "severe_layout_errors.candidate_index must contain exactly "
            f"1..{candidate_count}; got {severe_indices}."
        )

    return {
        "severe_layout_errors": severe_errors,
        "ranking": ranking,
        "descriptions": descriptions,
        "overall_summary": str(data.get("overall_summary", "")),
    }


# ==================================
# Space Benchmark
# ==================================
class SpaceBenchmark:
    """Run deterministic space metrics for one case."""

    name = "space"

    def __init__(
        self,
        output_root: Path,
        *,
        llm: LLMClient | None = None,
        previous_root: Path | None = None,
        prompt_path: Path | None = None,
        system_prompt_path: Path | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.llm = llm
        self.previous_root = (
            Path(previous_root)
            if previous_root is not None
            else None
        )
        self.prompt_path = prompt_path or DEFAULT_PROMPT_PATH
        self.system_prompt_path = (
            system_prompt_path or DEFAULT_SYSTEM_PROMPT_PATH
        )
        if self.llm is not None:
            if not self.prompt_path.is_file():
                raise FileNotFoundError(
                    f"Space prompt not found: {self.prompt_path}"
                )
            if not self.system_prompt_path.is_file():
                raise FileNotFoundError(
                    "Space system prompt not found: "
                    f"{self.system_prompt_path}"
                )
            self.prompt = self.prompt_path.read_text(encoding="utf-8")
            self.system_prompt = self.system_prompt_path.read_text(
                encoding="utf-8"
            )

    def run(self, case: BenchmarkCase) -> BenchmarkResult:
        # Create output dir for saving artifacts and metrics
        case_space_dir = self.output_root / case.key / "space"
        artifact_dir = case_space_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        metrics_by_method: dict[str, Any] = {}
        failures: dict[str, str] = {}

        for method in case.method_images:
            try:
                metrics, source = self._load_or_compute_metrics(
                    case,
                    method.name,
                    method.path,
                    artifact_dir,
                )
                metrics_by_method[method.name] = {
                    "status": "success",
                    "source": source,
                    "metrics": metrics,
                    "prompt_metrics": build_prompt_metrics(metrics),
                }
                print(
                    f"[{case.key}] [space] [method={method.name}] "
                    f"metrics done: {source}",
                    flush=True,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failures[method.name] = error
                metrics_by_method[method.name] = {
                    "status": "failed",
                    "error": error,
                    "metrics": {},
                    "prompt_metrics": {},
                }
                print(
                    f"[{case.key}] [space] [method={method.name}] "
                    f"metrics failed: {error}",
                    flush=True,
                )

        payload = {
            "reference": str(case.reference_path),
            "metrics": metrics_by_method,
            "failed_methods": failures,
            "llm_ranking": None,
        }
        metadata = {
            "case_key": case.key,
            "case_id": case.case_id,
            "ratio": case.ratio,
            "methods": [method.name for method in case.method_images],
        }
        llm_error = None
        if self.llm is not None and not failures:
            try:
                print(
                    f"[{case.key}] [space] LLM ranking start",
                    flush=True,
                )
                ranking, response_metadata = self._run_llm_ranking(
                    case,
                    metrics_by_method,
                )
                payload["llm_ranking"] = ranking
                metadata.update(response_metadata)
                print(
                    f"[{case.key}] [space] LLM ranking done",
                    flush=True,
                )
            except Exception as exc:
                llm_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[{case.key}] [space] LLM ranking failed: {llm_error}",
                    flush=True,
                )

        if failures:
            status = "failed"
        elif llm_error is not None:
            status = "error"
        else:
            status = "success"
        return BenchmarkResult(
            benchmark=self.name,
            method=None,
            status=status,
            payload=payload,
            metadata=metadata,
            error=llm_error or (
                f"Metrics failed for: {', '.join(failures)}"
                if failures
                else None
            ),
        )

    def run_and_save(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run the metrics and save the case-level space result JSON."""
        result = self.run(case)
        result_path = (
            self.output_root
            / case.key
            / "space"
            / "result.json"
        )
        save_json(result_path, result.to_dict())
        return result

    def _run_llm_ranking(
        self,
        case: BenchmarkCase,
        metrics_by_method: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.llm is None:
            raise RuntimeError("Space LLM client is not configured")

        parts = self._build_parts(case, metrics_by_method)
        last_error: Exception | None = None
        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            attempt_parts = list(parts)
            if attempt > 1:
                attempt_parts.append(
                    TextPart(
                        "The previous response was invalid: "
                        f"{last_error}. Return only the required JSON "
                        "with exactly one ranking and description entry "
                        "for every candidate."
                    )
                )

            try:
                response = self.llm.generate(
                    system_prompt=self.system_prompt,
                    parts=attempt_parts,
                )
                return (
                    parse_space_response(
                        response.text,
                        candidate_count=len(case.method_images),
                    ),
                    {
                        "model": response.model,
                        "provider": response.provider,
                        "llm_attempt": attempt,
                    },
                )
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Space LLM response remained invalid after "
            f"{LLM_MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def _build_parts(
        self,
        case: BenchmarkCase,
        metrics_by_method: dict[str, Any],
    ) -> list[TextPart | ImagePart]:
        """Build the ordered image, metrics, and prompt parts."""
       
        # 1. SEVERE ERROR - reference images
        parts: list[TextPart | ImagePart] = [
            TextPart(
                "The following are severe-error calibration examples only. "
                "They are not part of the current case and must not be ranked."
            ),
        ]
        for reference_image, description in (_severe_error_reference_images()):
            parts.extend([
                TextPart(
                    f"{description} "
                    f"This image is not a candidate. "
                    f"Filename: {reference_image.name}"
                ),
                ImagePart(reference_image),
            ])
  
        # 2. CASE - reference and candidates
        parts.append(TextPart("Now evaluate the actual case below."))
        parts.extend([
            TextPart(f"Image 0 - REFERENCE IMAGE ({case.reference_path.name}):"),
            ImagePart(case.reference_path),
        ])

        prompt_metrics = {}
        for index, method in enumerate(case.method_images, start=1):
            parts.extend([
                TextPart(
                    f"Image {index} - CANDIDATE {index} "
                    f"({method.name}):"
                ),
                ImagePart(method.path),
            ])
            prompt_metrics[method.name] = metrics_by_method[
                method.name
            ]["prompt_metrics"]

        # 3. Metrics
        parts.append(TextPart(
            "Deterministic checks for each candidate:\n"
            + json.dumps(
                prompt_metrics,
                ensure_ascii=False,
                indent=2,
            )
        ))
        
        # 4. Prompt
        parts.append(TextPart(self.prompt))
        return parts

    # ======================================
    # Compute / Load Metrics
    # ======================================
    def _load_or_compute_metrics(
        self,
        case: BenchmarkCase,
        method_name: str,
        method_path: Path,
        artifact_dir: Path,
    ) -> tuple[dict[str, Any], str]:
        """Load metrics from a previous run or compute them."""
        current_path = self._metrics_path(self.output_root, case, method_name)
        previous_path = self._metrics_path(self.previous_root, case, method_name)
       
        # If previous metrics exist
        if previous_path is not None and previous_path.is_file():
            saved = load_json(previous_path)
            
            # Check if the saved metrics match the current case and method
            if _metrics_match(saved, case=case, method_name=method_name, method_path=method_path):
                metrics = saved["metrics"]
                self._save_metrics(current_path, case, method_name, method_path, metrics)
                return metrics, f"previous:{previous_path}"

        # Otherwise, compute the metrics
        artifact_path = artifact_dir / f"{method_name}_empty_regions.png"
        # metrics1: blank area analysis
        blank_area = visualize_and_analyze_blank_area(
            image_path=method_path,
            output_path=artifact_path,
        )
        # metrics2: OCR-based rotation detection
        ocr = analyze_image_rotation(
            method_path,
        )
        # metrics3: reference/candidate LPIPS comparison
        similarity = analyze_visual_similarity(
            reference=case.reference_path,
            candidate=method_path,
        )

        metrics = {
            "blank_area": blank_area,
            "ocr": ocr,
            "similarity": similarity,
        }
        self._save_metrics(current_path, case, method_name, method_path, metrics)

        return metrics, "computed"


    # =======================================
    # Metrics file helpers
    # =======================================
    @staticmethod
    def _metrics_path(
        root: Path | None,
        case: BenchmarkCase,
        method_name: str,
    ) -> Path | None:
        if root is None:
            return None
        return (Path(root) / case.key / "space" / "metrics" / f"{method_name}.json")

    @staticmethod
    def _save_metrics(
        path: Path | None,
        case: BenchmarkCase,
        method_name: str,
        method_path: Path,
        metrics: dict[str, Any],
    ) -> None:
        if path is None:
            return
        save_json(
            path,
            {
                "metrics_version": METRICS_VERSION,
                "case_key": case.key,
                "method": method_name,
                "reference": str(case.reference_path),
                "candidate": str(method_path),
                "reference_sha256": _file_sha256(case.reference_path),
                "candidate_sha256": _file_sha256(method_path),
                "metrics": metrics,
            },
        )


__all__ = [
    "SpaceBenchmark",
    "build_prompt_metrics",
    "analyze_image_rotation",
    "analyze_visual_similarity",
    "visualize_and_analyze_blank_area",
]

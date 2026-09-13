"""Input and output data structures for the benchmark pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
CASE_DIR_RE = re.compile(r"^(?P<case_id>.+)_(?P<width>\d+)_(?P<height>\d+)$")

# ------------------------------------------
# Benchmark Input Data Structures
# ------------------------------------------

# Each case name and directory path
@dataclass(frozen=True)
class MethodImageInfo:
    name: str
    path: Path


@dataclass(frozen=True)
class BenchmarkCase:
    """
    Case directory structure:
    <case_id>_<ratio_width>_<ratio_height>/
        reference.png
        method1.png
        method2.png
        ...
    """
    case_id: str
    ratio_width: int
    ratio_height: int
    case_directory: Path
    reference_path: Path  # Path(".../reference.png")
    method_images: tuple[MethodImageInfo, ...]  # (MethodImageInfo(name="method1", path=Path(".../method1.png")), ...)

    @property
    def key(self) -> str:
        return self.case_directory.name

    @property
    def ratio(self) -> str:
        return f"{self.ratio_width}:{self.ratio_height}"

# ------------------------------------------
# Benchmark Output Data Structures
# ------------------------------------------

@dataclass
class BenchmarkResult:
    benchmark: str
    status: str = "success"  # "success" | "failed"
    method: str | None = None  # case method name
    payload: dict[str, Any] = field(default_factory=dict)  # benchmark fields
    metadata: dict[str, Any] = field(default_factory=dict)  # execution/input info
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "status": self.status,
            "method": self.method,
            "payload": self.payload,
            "metadata": self.metadata,
            "error": self.error,
        }

    def to_run_dict(self) -> dict[str, Any]:
        """Return the compact representation stored in run.json."""
        payload_fields = {
            "style": ("ranking", "overall_summary"),
            "space": ("llm_ranking",),
            "hallucination": (
                "correct",
                "has_hallucination",
                "differences",
            ),
            "relationship": ("correct", "differences"),
        }.get(self.benchmark)

        if payload_fields is None:
            payload = dict(self.payload)
        elif self.benchmark == "space":
            # Keep only the model's ranking output. The deterministic metrics
            # stay in space/metrics/*.json and never enter run.json.
            space_ranking = self.payload.get("llm_ranking")
            if isinstance(space_ranking, dict):
                space_ranking = dict(space_ranking)
                method_names = self.metadata.get("methods", [])
                if isinstance(method_names, list):
                    # show each metrics entry
                    severe_errors = space_ranking.get(
                        "severe_layout_errors"
                    )
                    if isinstance(severe_errors, list):
                        named_errors = []
                        for entry in severe_errors:
                            if not isinstance(entry, dict):
                                named_errors.append(entry)
                                continue
                            named_entry = dict(entry)
                            # get index
                            candidate_index = named_entry.get(
                                "candidate_index"
                            )
                            # get method name
                            if (
                                isinstance(candidate_index, int)
                                and 1 <= candidate_index <= len(method_names)
                            ):
                                named_entry["method"] = method_names[
                                    candidate_index - 1
                                ]
                            named_errors.append(named_entry)
                        space_ranking["severe_layout_errors"] = named_errors

                    ranking_entries = space_ranking.get("ranking")
                    description_entries = space_ranking.get("descriptions")
                    if (
                        isinstance(ranking_entries, list)
                        and isinstance(description_entries, list)
                    ):
                        descriptions_by_index = {
                            entry.get("candidate_index"): entry
                            for entry in description_entries
                            if isinstance(entry, dict)
                        }
                        merged_ranking = []
                        for entry in ranking_entries:
                            if not isinstance(entry, dict):
                                merged_ranking.append(entry)
                                continue

                            merged_entry = dict(entry)
                            candidate_index = merged_entry.get(
                                "candidate_index"
                            )
                            description = descriptions_by_index.get(
                                candidate_index,
                                {},
                            )
                            if isinstance(description, dict):
                                if "description" in description:
                                    merged_entry["description"] = (
                                        description["description"]
                                    )
                            if (
                                isinstance(candidate_index, int)
                                and 1 <= candidate_index <= len(method_names)
                            ):
                                merged_entry["method"] = method_names[
                                    candidate_index - 1
                                ]
                            merged_ranking.append(merged_entry)

                        space_ranking["ranking"] = merged_ranking
                        space_ranking.pop("descriptions", None)

            payload = {
                "llm_ranking": space_ranking,
            }
        else:
            payload = {
                field: self.payload[field]
                for field in payload_fields
                if field in self.payload
            }

        result: dict[str, Any] = {
            "benchmark": self.benchmark,
            "status": self.status,
            "payload": payload,
        }

        # Style and Space are case-level results. Relationship and
        # Hallucination return one result per method, so retain the method
        # name only when it is needed to identify the result.
        if self.method is not None:
            result["method"] = self.method

        # Keep failure diagnostics without adding metadata to successful
        # results.
        if self.error is not None:
            result["error"] = self.error

        return result


@dataclass
class CaseResult:  # Each benchmark case
    case_key: str
    case_id: str
    ratio: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_key": self.case_key,
            "case_id": self.case_id,
            "ratio": self.ratio,
            "results": [result.to_dict() for result in self.results],
        }

    def to_run_dict(self) -> dict[str, Any]:
        """Return the compact case representation stored in run.json."""
        compact_results = [
            result.to_run_dict()
            for result in self.results
        ]

        # Hallucination and Relationship produce one result per method.
        # Group those method results under one benchmark entry so run.json
        # does not repeat the benchmark metadata for every method.
        grouped_results: list[dict[str, Any]] = []
        grouped_by_benchmark: dict[str, dict[str, Any]] = {}
        per_method_benchmarks = {"hallucination", "relationship"}

        for result in compact_results:
            benchmark_name = result["benchmark"]
            if benchmark_name not in per_method_benchmarks:
                grouped_results.append(result)
                continue

            group = grouped_by_benchmark.get(benchmark_name)
            if group is None:
                group = {
                    "benchmark": benchmark_name,
                    "status": result["status"],
                    "result": [],
                }
                grouped_by_benchmark[benchmark_name] = group
                grouped_results.append(group)
            else:
                group["status"] = _merge_status(
                    group["status"],
                    result["status"],
                )

            method_result = {
                "method": result.get("method"),
                "payload": result.get("payload", {}),
            }
            if "error" in result:
                method_result["error"] = result["error"]
            group["result"].append(method_result)

        return {
            "case_key": self.case_key,
            "results": grouped_results,
        }


def _merge_status(current: str, new: str) -> str:
    """Aggregate method statuses into one benchmark status."""
    if "error" in {current, new}:
        return "error"
    if "failed" in {current, new}:
        return "failed"
    return "success"


@dataclass
class RunResult:  # Each run of the benchmark suite
    run_id: str
    benchmarks: list[str]
    provider: str
    model: str
    status: str = "running"  # "running" | "completed" | "failed"
    cases: list[CaseResult] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmarks": self.benchmarks,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cases": [case.to_run_dict() for case in self.cases],
        }


class InputDiscoveryError(ValueError):
    pass

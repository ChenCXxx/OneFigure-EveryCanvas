"""Discover cases and run the selected benchmark suite."""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
try:
    from .schemas import (
        BenchmarkCase,
        CaseResult,
        InputDiscoveryError,
        MethodImageInfo,
        RunResult,
    )
except ImportError:  # Support running pipeline.py directly.
    from schemas import (
        BenchmarkCase,
        CaseResult,
        InputDiscoveryError,
        MethodImageInfo,
        RunResult,
    )


BENCHMARK_NAMES = (
    "hallucination",
    "relationship",
    "style",
    "space",
)

# Edit this list to choose which benchmarks are executed by the pipeline.
# Example: ENABLED_BENCHMARKS = ("style", "space")
ENABLED_BENCHMARKS = (
    "style",
    "space",
    "relationship",
    "hallucination",
)


def discover_cases(
    input_root: Path,
    *,
    strict: bool = True,
) -> list[BenchmarkCase]:
    """Find ``{case_id}_{width}_{height}`` directories and their images."""
    input_root = input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise InputDiscoveryError(
            f"Input root does not exist: {input_root}"
        )

    case_pattern = re.compile(r"^(?P<case_id>.+)_(?P<width>\d+)_(?P<height>\d+)$")
    image_suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
    }
    cases: list[BenchmarkCase] = []
    errors: list[str] = []

    for directory in sorted(
        input_root.iterdir(),
        key=lambda path: path.name,
    ):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        
        # match pattern: <case_id>_<width>_<height>
        match = case_pattern.fullmatch(directory.name)
        if match is None:
            errors.append(
                f"{directory.name}: expected "
                "{case_id}_{width}_{height}"
            )
            continue
        
        # reference image path
        reference_path = next(
            (
                directory / name
                for name in (
                    "reference.png",
                    "reference.jpg",
                    "reference.jpeg",
                    "reference.webp",
                    "reference.bmp",
                )
                if (directory / name).is_file()
            ),
            None,
        )
        if reference_path is None:
            errors.append(
                f"{directory.name}: missing reference image"
            )
            continue

        image_paths = sorted(
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in image_suffixes
            )
        )
        
        # all other images (name, path) except reference.png
        methods = tuple(
            MethodImageInfo(path.stem, path)
            for path in image_paths
            if path != reference_path
        )
        if not methods:
            errors.append(
                f"{directory.name}: no method images found"
            )
            continue

        cases.append(
            BenchmarkCase(  # id, w, h, dir, ref path, method
                case_id=match.group("case_id"),
                ratio_width=int(match.group("width")),
                ratio_height=int(match.group("height")),
                case_directory=directory,
                reference_path=reference_path,
                method_images=methods,
            )
        )

    if errors and strict:
        raise InputDiscoveryError(
            "Invalid benchmark inputs:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    return cases


class BenchmarkPipeline:
    """Run one or more benchmarks for every discovered case."""

    def __init__(
        self,
        *,
        input_root: Path,
        output_root: Path,
        llm: LLMClient | None = None,
        previous_root: Path | None = None,
        benchmarks: Sequence[str] | None = None,
    ) -> None:
        if llm is None:
            try:
                from .llm import LLMClient
            except ImportError:  # Support running pipeline.py directly.
                from llm import LLMClient
            llm = LLMClient()

        selected = tuple(
            ENABLED_BENCHMARKS
            if benchmarks is None
            else benchmarks
        )
        invalid = sorted(set(selected) - set(BENCHMARK_NAMES))
        if invalid:
            raise ValueError(
                f"Unknown benchmarks: {', '.join(invalid)}"
            )

        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.benchmark_names = selected
        self.llm = llm
        self.previous_root = (
            Path(previous_root)
            if previous_root is not None
            else None
        )

        self._benchmarks = self._build_benchmarks()

    def _build_benchmarks(self) -> dict[str, Any]:
        # Import lazily: Space optionally imports LPIPS/Torch, so discovery
        # and unrelated benchmarks should not require those heavy packages.
        benchmarks: dict[str, Any] = {}

        if "hallucination" in self.benchmark_names:
            try:
                from .benchmarks.hallucination import HallucinationBenchmark
            except ImportError:
                from benchmarks.hallucination import HallucinationBenchmark

            benchmarks["hallucination"] = HallucinationBenchmark(self.llm)
        if "relationship" in self.benchmark_names:
            try:
                from .benchmarks.relationship import RelationshipBenchmark
            except ImportError:
                from benchmarks.relationship import RelationshipBenchmark

            benchmarks["relationship"] = RelationshipBenchmark(self.llm)
        if "style" in self.benchmark_names:
            try:
                from .benchmarks.style import StyleBenchmark
            except ImportError:
                from benchmarks.style import StyleBenchmark

            benchmarks["style"] = StyleBenchmark(self.llm)
        if "space" in self.benchmark_names:
            try:
                from .benchmarks.space import SpaceBenchmark
            except ImportError:
                from benchmarks.space import SpaceBenchmark

            benchmarks["space"] = SpaceBenchmark(
                output_root=self.output_root,
                llm=self.llm,
                previous_root=self.previous_root,
            )

        return benchmarks

    def run(self, *, strict_discovery: bool = True) -> RunResult:
        cases = discover_cases(
            self.input_root,
            strict=strict_discovery,
        )
        try:
            from .llm import load_config
            from .utils import save_json
        except ImportError:
            from llm import load_config
            from utils import save_json
        config = load_config()
        run_id = self.output_root.name
        
        # Initialize RunResult with run_id, benchmarks, provider, and model
        run_result = RunResult(
            run_id=run_id,
            benchmarks=list(self.benchmark_names),
            provider=config.provider,
            model=config.model,
        )

        # Add each case result to the RunResult object
        for case in cases:
            case_result = self._run_case(case)
            run_result.cases.append(case_result)

        # Determine the overall status of the run based on individual case results
        all_results = [
            result
            for case in run_result.cases
            for result in case.results
        ]
        run_result.status = (
            "completed"
            if all_results and all(result.status == "success" for result in all_results)
            else "failed"
        )
        run_result.finished_at = datetime.now(
            timezone.utc
        ).astimezone().isoformat(timespec="seconds")
        save_json(
            self.output_root / "run.json",
            run_result.to_dict(),
        )
        return run_result

    def _run_case(self, case: BenchmarkCase) -> CaseResult:
        case_result = CaseResult(
            case_key=case.key,
            case_id=case.case_id,
            ratio=case.ratio,
        )

        for name in self.benchmark_names:
            benchmark = self._benchmarks[name]
            method_names = ", ".join(
                method.name for method in case.method_images
            )
            print(
                f"[{case.key}] [{name}] start: methods={method_names}",
                flush=True,
            )

            raw_results = benchmark.run(case)

            if isinstance(raw_results, list):
                # Hallucination / Relationship return one result per method.
                results = raw_results
            else:
                # Style / Space return one case-level result.
                results = [raw_results]

            case_result.results.extend(results)
            self._save_benchmark_result(
                case,
                name,
                results,
            )

            statuses = ", ".join(result.status for result in results)
            print(
                f"[{case.key}] [{name}] done: {statuses}",
                flush=True,
            )

        return case_result

    def _save_benchmark_result(
        self,
        case: BenchmarkCase,
        benchmark_name: str,
        results: list[BenchmarkResult],
    ) -> None:
        path = (
            self.output_root
            / case.key
            / benchmark_name
            / "result.json"
        )
        try:
            from .utils import save_json
        except ImportError:
            from utils import save_json
        save_json(
            path,
            {
                "case_key": case.key,
                "case_id": case.case_id,
                "ratio": case.ratio,
                "benchmark": benchmark_name,
                "results": [
                    result.to_dict()
                    for result in results
                ],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Flowchart benchmark pipeline."
    )
    # structure: inputs/<case_id>_<width>_<height>/
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("inputs"),
        help="Directory containing case input folders.",
    )
    # structure: outputs/<run_name>/<case_id>_<width>_<height>/<benchmark_name>/result.json
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing all named runs.",
    )
    # output run directory name (default: datetime.now())
    parser.add_argument(
        "--name",
        default=datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"),
        help="Run directory name; defaults to the current local datetime.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARK_NAMES,
        default=None,
        metavar="BENCHMARK",
        help=(
            "Benchmarks to run. Choices: hallucination, relationship, "
            "style, space. Defaults to ENABLED_BENCHMARKS."
        ),
    )
    # use previous metrics(OCR, LPIPS, blank_area) to avoid re-computation
    parser.add_argument(
        "--reuse-metrics",
        action="store_true",
        help="Reuse compatible Space metrics from --previous-root.",
    )
    parser.add_argument(
        "--previous-root",
        type=Path,
        default=None,
        help="Previous run directory, for example benchmark/outputs/run1.",
    )
    parser.add_argument(
        "--allow-invalid-cases",
        action="store_true",
        help="Skip invalid case directories instead of failing discovery.",
    )
    args = parser.parse_args()

    if args.reuse_metrics and args.previous_root is None:
        parser.error("--reuse-metrics requires --previous-root <path>")

    output_root = args.output_dir / args.name
    previous_root = args.previous_root if args.reuse_metrics else None

    pipeline = BenchmarkPipeline(
        input_root=args.input_dir,
        output_root=output_root,
        previous_root=previous_root,
        benchmarks=args.benchmarks,
    )
    result = pipeline.run(
        strict_discovery=not args.allow_invalid_cases,
    )
    print(
        f"Completed {len(result.cases)} case(s) in {output_root}: "
        f"{', '.join(result.benchmarks)}"
    )


if __name__ == "__main__":
    main()

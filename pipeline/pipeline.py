from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from math import gcd
from pathlib import Path

from add_image import run_add_image_stage
from config import Settings, load_settings
from layout import run_layout_stage
from parse import run_parse_stage
from style import run_style_stage
from utils import ensure_bbox_json

def run_pipeline(
    settings: Settings,
    input_path: Path,
    output_dir: Path,
    *,
    bbox_path: Path | None = None,
    image_name: str | None = None,
    layout_output_dir: Path | None = None,
    log_case_id: str | None = None,
    parse_iterations: int = 3,
    style_iterations: int = 2,
    layout_iterations: int = 5,
    target_aspect_ratio: tuple[int, int] | None = None,
    skip_style: bool = False,
    skip_layout: bool = False,
    skip_overwrite: bool = False,
) -> Path:

    # get input image
    image_path = input_path.expanduser().resolve()
    if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        raise FileNotFoundError(f"Invalid input image: {image_path}")
    case_id = log_case_id or image_name or image_path.stem
    log_prefix = f"[{case_id}]"

    # Prefer bbox.json. If it is absent but bbox.txt exists beside the image,
    # convert the normalized YOLO rows and save bbox.json in the same directory.
    bbox_path = ensure_bbox_json(
        image_path=image_path,
        bbox_path=bbox_path,
    )
    if bbox_path is not None:
        print(f"{log_prefix}[pipeline] bbox={bbox_path}", flush=True)
    else:
        print(f"{log_prefix}[pipeline] bbox=none", flush=True)

    # get output dir
    run_dir = (
        output_dir.expanduser().resolve()
        / (image_name or image_path.stem)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_layout_dir = (
        layout_output_dir.expanduser().resolve()
        if layout_output_dir is not None
        else run_dir / "layout"
    )

    # record the timeline 
    pipeline_started = time.perf_counter()
    timeline_path = run_dir / "timeline.json"
    timeline: dict[str, object] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": str(image_path),
        "bbox": str(bbox_path) if bbox_path is not None else None,
        "run_dir": str(run_dir),
        "layout_dir": str(resolved_layout_dir),
        "stages": {},
    }
    _write_timeline(timeline_path, timeline)
    print(
        f"{log_prefix}[pipeline] input={image_path} output={run_dir} "
        f"provider={settings.model_provider}",
        flush=True,
    )

    # =========================================================
    # Parse Stage
    # =========================================================
    # Original Image Path : args.input_dir / "figure.png"
    # Bbox Path : args.bbox
    # Output Path : run_dir / "parse" / "final.xml"
    parse_output = run_dir / "parse" / "final.xml"
    if skip_overwrite and parse_output.is_file():
        parsed_xml = parse_output
        _mark_reused_stage(timeline, "parse", parsed_xml)
        print(
            f"{log_prefix}[pipeline] stage=parse reused existing output={parsed_xml}",
            flush=True,
        )
        _write_timeline(timeline_path, timeline)
    else:
        print(f"{log_prefix}[pipeline] stage=parse starting", flush=True)
        try:
            parsed_xml = run_parse_stage(
                settings=settings,
                image_path=image_path,
                output_dir=run_dir / "parse",
                bbox_path=bbox_path,
                iterations=parse_iterations,
                timeline=timeline,
                case_id=case_id,
            )
        finally:
            _write_timeline(timeline_path, timeline)
        print(f"{log_prefix}[pipeline] stage=parse completed output={parsed_xml}", flush=True)

    # Parse creates this file when no input bbox.json/bbox.txt was available.
    generated_bbox = run_dir / "parse" / "bbox" / "bbox.json"
    if bbox_path is None and generated_bbox.is_file():
        bbox_path = generated_bbox
        timeline["bbox"] = str(bbox_path)
        _write_timeline(timeline_path, timeline)

    # =========================================================
    # Style Stage
    # =========================================================
    # Parsed XML Path : run_dir / "parse" / "final.xml"
    # Original Image Path : args.input_dir / "figure.png"
    # Output Path : run_dir / "style" / "final.xml"
    styled_xml = parsed_xml
    if not skip_style:
        style_output = run_dir / "style" / "final.xml"
        if skip_overwrite and style_output.is_file():
            styled_xml = style_output
            _mark_reused_stage(timeline, "style", styled_xml)
            print(
                f"{log_prefix}[pipeline] stage=style reused existing output={styled_xml}",
                flush=True,
            )
            _write_timeline(timeline_path, timeline)
        else:
            print(f"{log_prefix}[pipeline] stage=style starting", flush=True)
            try:
                styled_xml = run_style_stage(
                    settings=settings,
                    input_xml=parsed_xml,
                    reference_image=image_path,
                    output_dir=run_dir / "style",
                    iterations=style_iterations,
                    timeline=timeline,
                    case_id=case_id,
                )
            finally:
                _write_timeline(timeline_path, timeline)
            print(f"{log_prefix}[pipeline] stage=style completed output={styled_xml}", flush=True)
    else:
        stages = timeline["stages"]
        if isinstance(stages, dict):
            stages["style"] = {
                "status": "skipped",
                "iterations": [],
                "total_seconds": 0.0,
            }
        _write_timeline(timeline_path, timeline)
        print(f"{log_prefix}[pipeline] stage=style skipped", flush=True)

    # =========================================================
    # Layout Stage
    # =========================================================
    final_xml = styled_xml
    if not skip_layout:
        layout_output = resolved_layout_dir / "final.xml"
        if skip_overwrite and layout_output.is_file():  # layout/final.xml exists
            final_xml = layout_output
            _mark_reused_stage(timeline, "layout", final_xml)
            print(
                f"{log_prefix}[pipeline] stage=layout reused existing output={final_xml}",
                flush=True,
            )
            _write_timeline(timeline_path, timeline)
        else:  # start layout stage
            print(
                f"{log_prefix}[pipeline] stage=layout starting "
                f"target_aspect_ratio={target_aspect_ratio or 'from-input'}",
                flush=True,
            )
            try:
                # run layout stage
                final_xml = run_layout_stage(
                    settings=settings,
                    input_xml=styled_xml,
                    output_dir=resolved_layout_dir,
                    iterations=layout_iterations,
                    target_aspect_ratio=target_aspect_ratio,
                    timeline=timeline,
                    case_id=case_id,
                )
            finally:
                _write_timeline(timeline_path, timeline)
            print(f"{log_prefix}[pipeline] stage=layout completed output={final_xml}", flush=True)
    else:  # skip layout stage
        stages = timeline["stages"]
        if isinstance(stages, dict):
            stages["layout"] = {
                "status": "skipped",
                "iterations": [],
                "total_seconds": 0.0,
            }
        _write_timeline(timeline_path, timeline)
        print(f"{log_prefix}[pipeline] stage=layout skipped", flush=True)

    # =========================================================
    # Add Image Stage
    # =========================================================
    # Crop coordinates must come from the first parser XML, while the final
    # positions and canvas come from layout/final.xml.
    image_final_png: Path | None = None
    if not skip_layout:
        parse_reference_xml = run_dir / "parse" / "iter_1" / "parse.xml"
        image_stage_dir = run_dir / "image"
        print(
            f"{log_prefix}[pipeline] stage=add_image starting "
            f"parse_xml={parse_reference_xml}",
            flush=True,
        )
        try:
            final_xml, image_final_png = run_add_image_stage(
                source_image=image_path,
                parse_xml=parse_reference_xml,
                layout_xml=final_xml,
                output_dir=run_dir,
                drawio_bin=settings.drawio_bin,
            )
            stages = timeline["stages"]
            if isinstance(stages, dict):
                stages["add_image"] = {
                    "status": "completed",
                    "parse_xml": str(parse_reference_xml),
                    "layout_xml": str(final_xml),
                    "output_dir": str(image_stage_dir),
                    "final_png": str(image_final_png),
                }
        finally:
            _write_timeline(timeline_path, timeline)
        print(
            f"{log_prefix}[pipeline] stage=add_image completed "
            f"output={image_final_png}",
            flush=True,
        )
    else:
        stages = timeline["stages"]
        if isinstance(stages, dict):
            stages["add_image"] = {
                "status": "skipped",
                "reason": "layout stage was skipped",
            }
        _write_timeline(timeline_path, timeline)
        print(f"{log_prefix}[pipeline] stage=add_image skipped", flush=True)

    manifest = {
        "input": str(image_path),
        "bbox": str(bbox_path) if bbox_path is not None else None,
        "run_dir": str(run_dir),
        "layout_dir": str(resolved_layout_dir),
        "parsed_xml": str(parsed_xml),
        "styled_xml": str(styled_xml),
        "final_xml": str(final_xml),
        "image_dir": str(run_dir / "image"),
        "image_final_png": str(image_final_png) if image_final_png else None,
        "parse_iterations": parse_iterations,
        "style_iterations": 0 if skip_style else style_iterations,
        "layout_iterations": 0 if skip_layout else layout_iterations,
        "skip_overwrite": skip_overwrite,
        "target_aspect_ratio": (
            {"width": target_aspect_ratio[0], "height": target_aspect_ratio[1]}
            if target_aspect_ratio is not None
            else None
        ),
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    timeline["completed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    timeline["total_seconds"] = round(
        time.perf_counter() - pipeline_started,
        3,
    )
    timeline["summary"] = _timeline_summary(timeline)
    _write_timeline(timeline_path, timeline)
    print(
        f"{log_prefix}[pipeline] timing "
        + json.dumps(timeline["summary"], ensure_ascii=False),
        flush=True,
    )
    print(f"{log_prefix}[pipeline] timeline saved to {timeline_path}", flush=True)
    print(f"{log_prefix}[pipeline] completed final_xml={final_xml}", flush=True)
    return final_xml


def _write_timeline(path: Path, timeline: dict[str, object]) -> None:
    path.write_text(
        json.dumps(timeline, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _mark_reused_stage(
    timeline: dict[str, object],
    stage_name: str,
    output_path: Path,
) -> None:
    stages = timeline.setdefault("stages", {})
    if isinstance(stages, dict):
        stages[stage_name] = {
            "status": "reused_existing",
            "output": str(output_path),
            "iterations": [],
            "total_seconds": 0.0,
        }


def _timeline_summary(timeline: dict[str, object]) -> dict[str, object]:
    stages = timeline.get("stages", {})
    stage_totals: dict[str, float] = {}
    agent_totals: dict[str, float] = {}
    critic_totals: dict[str, float] = {}

    if isinstance(stages, dict):
        for stage_name in ("parse", "style", "layout", "add_image"):
            stage = stages.get(stage_name, {})
            if not isinstance(stage, dict):
                continue
            stage_totals[stage_name] = float(stage.get("total_seconds", 0))
            iterations = stage.get("iterations", [])
            if not isinstance(iterations, list):
                continue
            agent_totals[stage_name] = round(
                sum(
                    float(row.get("agent_seconds", 0))
                    for row in iterations
                    if isinstance(row, dict)
                ),
                3,
            )
            critic_totals[stage_name] = round(
                sum(
                    float(row.get("critic_seconds", 0))
                    for row in iterations
                    if isinstance(row, dict)
                ),
                3,
            )

    return {
        "stage_seconds": stage_totals,
        "agent_seconds": agent_totals,
        "critic_seconds": {
            **critic_totals,
            "total": round(sum(critic_totals.values()), 3),
        },
        "pipeline_seconds": float(timeline.get("total_seconds", 0)),
    }


def parse_aspect_ratio(value: str) -> tuple[int, int]:
    separator = ":" if ":" in value else ","
    try:
        width_text, height_text = value.split(separator, maxsplit=1)
        width = int(width_text.strip())
        height = int(height_text.strip())
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Aspect ratio must use WIDTH:HEIGHT, for example 16:9."
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(
            "Aspect-ratio width and height must be positive integers."
        )
    divisor = gcd(width, height)
    return width // divisor, height // divisor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parse -> style -> layout pipeline.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing figure.png",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Output root directory")
    parser.add_argument("--bbox", type=Path, default=None, help="Optional bbox txt/json file")
    parser.add_argument(
        "--image_name",
        required=True,
        default=None,
        help="Output run folder name; this is not the input filename",
    )
    parser.add_argument("--parse_iterations", type=int, default=1)
    parser.add_argument("--style_iterations", type=int, default=1)
    parser.add_argument("--layout_iterations", type=int, default=1)
    parser.add_argument(
        "--aspect_ratio",
        type=parse_aspect_ratio,
        default=None,
        metavar="WIDTH:HEIGHT",
        help=(
            "Target layout ratio, for example 16:9. The layout canvas is "
            "computed from the generated XML area and rounded to integers."
        ),
    )
    parser.add_argument("--skip_style", action="store_true")
    parser.add_argument("--skip_layout", action="store_true")
    parser.add_argument(
        "--skip-overwrite",
        "--skip_overwrite",
        dest="skip_overwrite",
        action="store_true",
        help="Reuse existing parse/style/layout final.xml files stage by stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    final_xml = run_pipeline(
        settings=settings,
        input_path=args.input_dir / "figure.png", # for parser
        output_dir=args.output_dir, # all files are under output_dir/<image_name>/ 
        bbox_path=args.bbox, # auto-discovers bbox.json, then bbox.txt
        image_name=args.image_name,
        parse_iterations=args.parse_iterations, # default = 3
        style_iterations=args.style_iterations, # default = 2
        layout_iterations=args.layout_iterations, # default = 5
        target_aspect_ratio=args.aspect_ratio, # for relayout
        skip_style=args.skip_style, # default = False
        skip_layout=args.skip_layout, # default = False
        skip_overwrite=args.skip_overwrite,
    )
    print(f"Final XML saved to {final_xml}")


if __name__ == "__main__":
    main()

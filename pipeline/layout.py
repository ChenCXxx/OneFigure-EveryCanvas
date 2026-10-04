from __future__ import annotations

import json
import math
import re
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from config import Settings
from critic import (
    LayoutCritic,
    validate_xml,
)
from llm import LLMClient
from utils import (
    export_xml_to_image,
    extract_mxgraphmodel,
    format_prompt,
    load_prompt,
)

DECORATION_SHAPES = {
    "shape=mxgraph.flowchart.annotation_1": "annotation_1",
    "shape=curlyBracket": "curlyBracket",
}


# ==============================
# canvas and XML utilities
# ==============================

def set_canvas_size(xml_text: str, canvas: tuple[float, float]) -> str:
    width, height = canvas
    if width <= 0 or height <= 0:
        raise ValueError("Canvas width and height must be positive numbers.")
    root = ET.fromstring(xml_text)
    if root.tag != "mxGraphModel":
        raise ValueError("Expected mxGraphModel as the XML root.")
    root.set("pageWidth", _format_canvas_number(width))
    root.set("pageHeight", _format_canvas_number(height))
    return ET.tostring(root, encoding="unicode")


def canvas_for_same_area(
    source_canvas: tuple[float, float],
    target_aspect_ratio: tuple[int, int],
) -> tuple[int, int]:
    source_width, source_height = source_canvas
    ratio_width, ratio_height = target_aspect_ratio
    if min(source_width, source_height, ratio_width, ratio_height) <= 0:
        raise ValueError("Canvas and aspect-ratio values must be positive.")

    area = source_width * source_height
    ratio = ratio_width / ratio_height
    target_width = math.sqrt(area * ratio)
    target_height = target_width / ratio
    return _round_positive_half_up(target_width), _round_positive_half_up(target_height)


def _round_positive_half_up(value: float) -> int:
    return max(1, int(math.floor(value + 0.5)))


def _format_canvas_number(value: float) -> str:
    rounded = round(float(value), 6)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


# ==============================
# XML input parsing
# ===============================

def parse_xml_input(
    xml_path: str | Path,
    target_aspect_ratio: tuple[int, int] | None = None,
    target_canvas: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Read XML and build the structured input used by the layout agent.

    ``target_canvas`` and ``target_aspect_ratio`` are mutually exclusive.
    The target canvas preserves the original canvas area when an aspect ratio
    is provided, and block dimensions are scaled accordingly.
    """
    if target_canvas is not None and target_aspect_ratio is not None:
        raise ValueError("Use either target_canvas or target_aspect_ratio, not both.")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # original canvas
    original_canvas: dict[str, int] | None = None
    if root.tag == "mxGraphModel":
        width = root.get("pageWidth")
        height = root.get("pageHeight")
        if width is not None and height is not None:
            original_canvas = {
                "width": int(float(width)),
                "height": int(float(height)),
            }

    target_canvas_info: dict[str, int] | None = original_canvas
    if target_canvas is not None:
        target_canvas_info = {
            "width": int(target_canvas[0]),
            "height": int(target_canvas[1]),
        }
    if target_aspect_ratio is not None:
        if original_canvas is None:
            raise ValueError(
                "Cannot calculate target canvas without original canvas size."
            )
        target_width, target_height = canvas_for_same_area(
            (original_canvas["width"], original_canvas["height"]),
            target_aspect_ratio,
        )
        target_canvas_info = {
            "width": target_width,
            "height": target_height,
        }

    blocks: list[dict[str, object]] = []
    relationships: list[dict[str, object]] = []
    decorations: list[dict[str, object]] = []

    # Blocks and decorations from mxCell vertex="1".
    for cell in root.findall(".//mxCell"):
        if cell.get("vertex") != "1":
            continue

        block_id = cell.get("id")
        geometry_element = cell.find("mxGeometry")
        geometry: dict[str, float] = {}
        if geometry_element is not None:
            geometry = {
                "x": float(geometry_element.get("x", 0)),
                "y": float(geometry_element.get("y", 0)),
                "w": float(geometry_element.get("width", 0)),
                "h": float(geometry_element.get("height", 0)),
            }

        blocks.append({"id": block_id, "geometry": geometry})

        style = cell.get("style", "")
        for marker, decoration_type in DECORATION_SHAPES.items():
            if marker in style:
                decorations.append(
                    {"id": block_id, "type": decoration_type}
                )
                break

    # Relationships from mxCell edge="1" source/target.
    # two-way / forward / undirected
    # if startArrow=classic and endArrow=none => reverse source/target
    for cell in root.findall(".//mxCell"):
        if cell.get("edge") != "1":
            continue
        source = cell.get("source")
        target = cell.get("target")
        if source and target:
            style_values: dict[str, str] = {}
            # get style attributes
            for item in cell.get("style", "").split(";"):
                if "=" in item:
                    key, value = item.split("=", maxsplit=1)
                    style_values[key] = value

            start_arrow = style_values.get("startArrow", "none")
            end_arrow = style_values.get("endArrow", "none")
            start_has_arrow = start_arrow == "classic"
            end_has_arrow = end_arrow == "classic"

            if start_has_arrow and end_has_arrow:
                relationship_type = "Two-way"
                relationship_from = source
                relationship_to = target
            elif start_has_arrow:  # Reverse direction
                relationship_type = "Forward"
                relationship_from = target
                relationship_to = source
            elif end_has_arrow:
                relationship_type = "Forward"
                relationship_from = source
                relationship_to = target
            else:
                relationship_type = "Undirected"
                relationship_from = source
                relationship_to = target

            relationships.append(
                {
                    "from": relationship_from,
                    "to": relationship_to,
                    "type": relationship_type,
                }
            )

    if original_canvas and target_canvas_info:
        source_area = original_canvas["width"] * original_canvas["height"]
        target_area = (
            target_canvas_info["width"] * target_canvas_info["height"]
        )
        scale = math.sqrt(target_area / source_area) if source_area > 0 else 1.0
    else:
        scale = 1.0

    # only return blocks id, scaled width, scaled height (No x,y)
    agent_blocks = [
        {
            "id": block["id"],
            "w": max(1, int(round(block["geometry"].get("w", 0) * scale))),
            "h": max(1, int(round(block["geometry"].get("h", 0) * scale))),
        }
        for block in blocks
        if block.get("id") is not None
    ]

    return {
        "original_canvas": original_canvas,
        "target_canvas": target_canvas_info,
        "blocks": agent_blocks,
        "relationships": relationships,
        "decorations": decorations,
        "xml_text": ET.tostring(root, encoding="unicode"),
    }


# ==============================
# output normalization
# ==============================

def shrink_font_sizes(
    output_root: ET.Element,
    input_xml_path: str | Path,
    min_font_size: int = 6,
) -> None:
    """Adjust vertex fonts using the original XML block sizes."""
    input_root = ET.parse(input_xml_path).getroot()
    sizes: dict[str, tuple[float, float]] = {}
    for cell in input_root.findall(".//mxCell"):
        geom = cell.find("mxGeometry")
        cell_id = cell.get("id")
        if cell.get("vertex") == "1" and cell_id and geom is not None:
            sizes[cell_id] = (
                float(geom.get("width", 0)),
                float(geom.get("height", 0)),
            )

    for cell in output_root.findall(".//mxCell"):
        style = cell.get("style", "")
        value = cell.get("value")
        geom = cell.find("mxGeometry")
        old = sizes.get(cell.get("id", ""))
        font_match = re.search(r"fontSize=(\d+(?:\.\d+)?)", style)
        if cell.get("vertex") != "1" or geom is None or old is None:
            continue

        old_width, old_height = old
        new_width = float(geom.get("width", 0))
        new_height = float(geom.get("height", 0))
        if min(old_width, old_height, new_width, new_height) <= 0:
            continue

        scale = math.sqrt(
            (new_width * new_height) / (old_width * old_height)
        )
        if font_match:
            new_font_size = max(
                min_font_size,
                round(float(font_match.group(1)) * scale),
            )
            cell.set(
                "style",
                re.sub(
                    r"fontSize=\d+(?:\.\d+)?",
                    f"fontSize={new_font_size}",
                    style,
                ),
            )

        if value and "font-size" in value:
            def replace_html_font(match: re.Match[str]) -> str:
                new_font_size = max(
                    min_font_size,
                    round(float(match.group(1)) * scale),
                )
                return f"font-size: {new_font_size}px"

            cell.set(
                "value",
                re.sub(
                    r"font-size:\s*(\d+(?:\.\d+)?)px",
                    replace_html_font,
                    value,
                ),
            )


# ===============================
# Layout agent
# ==============================

class LayoutAgent:
    def __init__(self, settings: Settings, *, case_id: str = "") -> None:
        self.settings = settings
        self.log_prefix = f"[{case_id}]" if case_id else ""
        self.llm = LLMClient(settings)
        self.system_prompt = load_prompt("layout_system.txt")

    def arrange(
        self,
        *,
        xml_text: str,
        layout_input: dict[str, object],
        feedback: str = "",
    ) -> str:
        # put xml_text and layout_input into the prompt
        prompt = format_prompt(
            "layout_prompt.txt",
            xml_text=xml_text,
            layout_input=json.dumps(layout_input, ensure_ascii=False, indent=2),
            feedback=feedback,
        )
        last_error: Exception | None = None
        attempt_prompt = prompt
        for attempt in range(1, 4):
            time.sleep(1)
            try:
                # Call the LLM
                raw_response = self.llm.generate(
                    model=self.settings.layout_model,
                    system_prompt=self.system_prompt,
                    prompt=attempt_prompt,
                    component="layout_agent",
                    attempt=attempt,
                )
                output = extract_mxgraphmodel(raw_response)
                ET.fromstring(output)
                return output
            except (ET.ParseError, ValueError, RuntimeError) as exc:
                # if output is invalid, retry with feedback
                last_error = exc
                print(
                    f"{self.log_prefix}[layout] invalid output on attempt {attempt}/3: {exc}",
                    flush=True,
                )
                attempt_prompt = (
                    prompt
                    + "\n\nThe previous response was rejected for the following reason:\n"
                    + str(exc)
                    + "\nReturn a corrected complete mxGraphModel XML. Preserve all protected IDs, values, styles, parents, sources, and targets exactly."
                )
        # If all attempts fail, raise an error
        raise RuntimeError(f"Layout agent failed to return valid XML: {last_error}")


def run_layout_stage(
    settings: Settings,
    input_xml: Path,
    output_dir: Path,
    *,
    iterations: int = 5,
    canvas: tuple[float, float] | None = None,
    target_aspect_ratio: tuple[int, int] | None = None,
    timeline: dict[str, object] | None = None,
    case_id: str = "",
) -> Path:
    log_prefix = f"[{case_id}]" if case_id else ""
    stage_started = time.perf_counter()
    stage_timeline: dict[str, object] = {"iterations": []}
    if timeline is not None:
        stages = timeline.setdefault("stages", {})
        if isinstance(stages, dict):
            stages["layout"] = stage_timeline
    if not input_xml.is_file():
        raise FileNotFoundError(f"Layout input XML not found: {input_xml}")

    # copy input images to layout/
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_input = output_dir / "input.xml"
    shutil.copy2(input_xml, copied_input)

    original_xml = copied_input.read_text(encoding="utf-8")
    # Parse canvas, blocks, rel ... + Compute target canvas
    parsed_info = parse_xml_input(
        copied_input,
        target_aspect_ratio=target_aspect_ratio,
        target_canvas=canvas,
    )
    # target canvas size
    canvas = (
        float(parsed_info["target_canvas"]["width"]),
        float(parsed_info["target_canvas"]["height"]),
    )
    # only put target canvas, blocks info, rel for LLM agent
    layout_input = {
        "target_canvas": parsed_info["target_canvas"],
        "blocks": parsed_info["blocks"],
        "relationships": parsed_info["relationships"],
    }

    # current_xml is the XML that will be put into the next iter of the layout agent
    current_xml = original_xml
    original_decorations = parsed_info["decorations"]

    # Initialize agent and critic
    agent = LayoutAgent(settings, case_id=case_id)
    critic = LayoutCritic(settings)

    feedback = ""
    final_xml_path = output_dir / "final.xml"
    print(
        f"{log_prefix}[layout] starting model={settings.layout_model} "
        f"critic_model={settings.critic_model} iterations={max(1, iterations)} "
        f"canvas={parsed_info["original_canvas"]['width']}x{parsed_info["original_canvas"]['height']} "
        f"target_canvas={parsed_info['target_canvas']['width']}x{parsed_info['target_canvas']['height']} ",
        flush=True,
    )

    # Run layout iterations
    for index in range(1, max(1, iterations) + 1):
        # directory: layout/iter_1, layout/iter_2, ...
        iter_dir = output_dir / f"iter_{index}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"{log_prefix}[layout] iteration {index}/{max(1, iterations)} generating XML", flush=True)

        # Call the layout agent
        arranged_xml = agent.arrange(
            xml_text=current_xml,
            layout_input=layout_input,
            feedback=feedback,
        )
        # Make sure the output XML has the correct canvas size
        arranged_xml = set_canvas_size(arranged_xml, canvas)

        # Shrink font sizes
        arranged_root = ET.fromstring(arranged_xml)
        shrink_font_sizes(arranged_root, copied_input)
        arranged_xml = ET.tostring(arranged_root, encoding="unicode")

        # Save this iteration's XML and image
        xml_path = iter_dir / "layout.xml"
        rendered = iter_dir / "layout.png"
        xml_path.write_text(arranged_xml, encoding="utf-8")

        # Validate deterministic constraints
        xml_ok, reasons, deterministic_suggestion = validate_xml(
            arranged_xml,
            original_xml,
        )

        # Render the validated XML for the layout critic.
        export_xml_to_image(xml_path, rendered, drawio_bin=settings.drawio_bin)
        print(f"{log_prefix}[layout] XML saved to {xml_path} and rendered to {rendered}", flush=True)

        skip_critic = index == max(1, iterations)
        if skip_critic:  # Last iteration
            result = {
                "ok": xml_ok,
                "reasons": reasons,
                "suggestion": deterministic_suggestion,
            }
            print(f"{log_prefix}[layout] final iteration; critic skipped", flush=True)
        else:  # Else, call the critic agent
            print(f"{log_prefix}[layout] calling critic", flush=True)
            result = critic.critique(
                rendered_image=rendered,
                xml_text=arranged_xml,
                reasons=reasons,
                decorations=original_decorations,
            )

        suggestion = str(result.get("suggestion", "") or deterministic_suggestion)

        (iter_dir / "critic.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"{log_prefix}[layout] validation ok={result.get("ok") is True} "
            f"reasons={len(result.get('reasons', []))}",
            flush=True,
        )

        current_xml = arranged_xml
        final_xml_path.write_text(current_xml, encoding="utf-8")

        # Break if the critic accepts the output
        if result.get("ok") is True and not reasons:
            print(f"{log_prefix}[layout] accepted at iteration {index}", flush=True)
            break

        # Otherwise, prepare feedback for the next iteration
        feedback = "\n\n【驗證失敗原因】\n- " + "\n- ".join(reasons)
        if suggestion:
            feedback += "\n\n【修正建議】\n" + suggestion
        # DEBUG: save feedback txt
        xml_path.with_suffix(".txt").write_text(
            feedback.lstrip() + "\n",
            encoding="utf-8",
        )

    stage_seconds = time.perf_counter() - stage_started
    stage_timeline["status"] = "completed"
    stage_timeline["total_seconds"] = round(stage_seconds, 3)
    print(
        f"{log_prefix}[layout] completed output={final_xml_path} total={stage_seconds:.3f}s",
        flush=True,
    )
    return final_xml_path

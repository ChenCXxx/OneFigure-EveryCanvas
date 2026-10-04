from __future__ import annotations

import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from config import Settings
from critic import StyleCritic
from llm import LLMClient
from tools.block_color_hints import sample_block_color_hints
from utils import (
    export_xml_to_image,
    extract_mxgraphmodel,
    format_prompt,
    load_prompt,
    normalize_render_size_top_left,
    normalize_visible_text,
)


class StyleAgent:
    def __init__(self, settings: Settings, *, case_id: str = "") -> None:
        self.settings = settings
        self.log_prefix = f"[{case_id}]" if case_id else ""
        self.llm = LLMClient(settings)
        self.system_prompt = load_prompt("style_system.txt")

    def stylize(
        self,
        *,
        xml_text: str,
        reference_image: Path,
        current_render: Path,
        feedback: str = "",
        block_color_hints: list[dict[str, object]] | None = None,
    ) -> str:
        # get color hints for each block
        hints_section = (
            "Per-block color hints sampled from the matching reference image regions:\n"
            "block_color_hints="
            + json.dumps(
                block_color_hints,
                ensure_ascii=False,
                indent=2,
            )
            + "\n\n"
            if block_color_hints is not None
            else ""
        )
        prompt = format_prompt(
            "style_prompt.txt",
            xml_text=xml_text,
            hints_section=hints_section,
            feedback=feedback.strip(),
        )
        last_error: Exception | None = None
        for attempt in range(1, 4):
            time.sleep(1)
            try:
                output = extract_mxgraphmodel(
                    self._generate(
                        prompt,
                        reference_image,
                        current_render,
                        attempt,
                    )
                )
                _validate_style_only_change(xml_text, output)
                return output
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                print(f"{self.log_prefix}[style] invalid output on attempt {attempt}/3: {exc}")
        raise RuntimeError(f"Style agent failed to return style-only XML: {last_error}")

    def _generate(
        self,
        prompt: str,
        reference_image: Path,
        current_render: Path,
        attempt: int,
    ) -> str:
        return self.llm.generate(
            model=self.settings.style_model,
            system_prompt=self.system_prompt,
            prompt=prompt,
            images=[reference_image, current_render],
            component="style_agent",
            attempt=attempt,
        )


def run_style_stage(
    settings: Settings,
    input_xml: Path,
    reference_image: Path,
    output_dir: Path,
    *,
    iterations: int = 2,
    timeline: dict[str, object] | None = None,
    case_id: str = "",
) -> Path:
    log_prefix = f"[{case_id}]" if case_id else ""
    stage_started = time.perf_counter()
    stage_timeline: dict[str, object] = {"iterations": []}
    if timeline is not None:
        stages = timeline.setdefault("stages", {})
        if isinstance(stages, dict):
            stages["style"] = stage_timeline
    if not input_xml.is_file():
        raise FileNotFoundError(f"Style input XML not found: {input_xml}")
    if not reference_image.is_file():
        raise FileNotFoundError(f"Style reference image not found: {reference_image}")

    with Image.open(reference_image) as source_image:
        render_target_size = source_image.size

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_input = output_dir / "input.xml"
    shutil.copy2(input_xml, copied_input)
    current_xml = copied_input.read_text(encoding="utf-8")
    current_render = output_dir / "input.png"
    export_xml_to_image(
        copied_input, current_render, drawio_bin=settings.drawio_bin
    )
    normalize_render_size_top_left(current_render, render_target_size)

    agent = StyleAgent(settings, case_id=case_id)
    critic = StyleCritic(settings)
    feedback = ""
    final_xml_path = output_dir / "final.xml"
    print(
        f"{log_prefix}[style] starting model={settings.style_model} "
        f"critic_model={settings.critic_model} iterations={max(1, iterations)}",
        flush=True,
    )

    for index in range(1, max(1, iterations) + 1):
        iter_dir = output_dir / f"iter_{index}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        hints = sample_block_color_hints(current_xml, reference_image)
        print(
            f"{log_prefix}[style] iteration {index}/{max(1, iterations)} "
            f"generating XML color_hints={len(hints)}",
            flush=True,
        )
        styled_xml = agent.stylize(
            xml_text=current_xml,
            reference_image=reference_image,
            current_render=current_render,
            feedback=feedback,
            block_color_hints=hints,
        )
        xml_path = iter_dir / "style.xml"
        xml_path.write_text(styled_xml, encoding="utf-8")
        print(f"{log_prefix}[style] XML saved to {xml_path}", flush=True)
        rendered = iter_dir / "style.png"
        print(f"{log_prefix}[style] rendering {rendered}", flush=True)
        export_xml_to_image(
            xml_path, rendered, drawio_bin=settings.drawio_bin
        )
        normalize_render_size_top_left(rendered, render_target_size)
        result = critic.critique(
            reference_image=reference_image,
            rendered_image=rendered,
            styled_xml=styled_xml,
            block_color_hints=hints,
        )
        (iter_dir / "critic.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"{log_prefix}[style] critic ok={result.get('ok')} "
            f"reasons={len(result.get('reasons', []))}",
            flush=True,
        )
        current_xml = styled_xml
        current_render = rendered
        feedback = str(result.get("suggestion", ""))
        final_xml_path.write_text(current_xml, encoding="utf-8")
        if result.get("ok") is True:
            print(f"{log_prefix}[style] accepted at iteration {index}", flush=True)
            break
    stage_seconds = time.perf_counter() - stage_started
    stage_timeline["status"] = "completed"
    stage_timeline["total_seconds"] = round(stage_seconds, 3)
    print(
        f"{log_prefix}[style] completed output={final_xml_path} total={stage_seconds:.3f}s",
        flush=True,
    )
    return final_xml_path


def _validate_style_only_change(before_xml: str, after_xml: str) -> None:
    # Validate the XML structure
    try:
        before = ET.fromstring(before_xml)
        after = ET.fromstring(after_xml)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML: {exc}") from exc

    # Compare the two XML trees for style-only changes
    def cells_by_id(root: ET.Element) -> dict[str, ET.Element]:
        return {
            cell.get("id", ""): cell
            for cell in root.findall(".//mxCell")
            if cell.get("id")
        }

    # get all cells by ID for both before and after XML
    before_cells = cells_by_id(before)
    after_cells = cells_by_id(after)
    differences: list[str] = []

    # Prohibit to add or remove cells
    removed_ids = sorted(before_cells.keys() - after_cells.keys())
    added_ids = sorted(after_cells.keys() - before_cells.keys())
    if removed_ids:
        differences.append(f"removed cell IDs: {removed_ids}")
    if added_ids:
        differences.append(f"added cell IDs: {added_ids}")

    # List of attributes that should not be changed
    protected_attributes = (
        "parent",
        "vertex",
        "edge",
        "source",
        "target",
    )
    geometry_attributes = ("x", "y", "width", "height", "relative")

    for cell_id in sorted(before_cells.keys() & after_cells.keys()):
        old_cell = before_cells[cell_id]
        new_cell = after_cells[cell_id]

        # Prohibit to change the visible text
        old_text = normalize_visible_text(old_cell.get("value"))
        new_text = normalize_visible_text(new_cell.get("value"))
        if old_text != new_text:
            differences.append(
                f"cell {cell_id!r} changed visible text: "
                f"{old_text!r} -> {new_text!r} "
                f"(raw {old_cell.get('value')!r} -> "
                f"{new_cell.get('value')!r})"
            )

        # Prohibit to change the protected attributes
        for attribute in protected_attributes:
            old_value = old_cell.get(attribute)
            new_value = new_cell.get(attribute)
            if old_value != new_value:
                differences.append(
                    f"cell {cell_id!r} changed {attribute}: "
                    f"{old_value!r} -> {new_value!r}"
                )

        # Prohibit to change the geometry attributes
        old_geometry = old_cell.find("mxGeometry")
        new_geometry = new_cell.find("mxGeometry")
        if (old_geometry is None) != (new_geometry is None):
            differences.append(
                f"cell {cell_id!r} changed mxGeometry presence: "
                f"{old_geometry is not None} -> {new_geometry is not None}"
            )
        elif old_geometry is not None and new_geometry is not None:
            for attribute in geometry_attributes:
                old_value = old_geometry.get(attribute)
                new_value = new_geometry.get(attribute)
                if old_value != new_value:
                    differences.append(
                        f"cell {cell_id!r} changed geometry.{attribute}: "
                        f"{old_value!r} -> {new_value!r}"
                    )

        if len(differences) >= 20:
            break

    if differences:
        raise ValueError(
            "Style output changed protected content:\n- "
            + "\n- ".join(differences[:20])
        )

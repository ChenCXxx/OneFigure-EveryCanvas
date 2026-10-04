from __future__ import annotations

import json
import re
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from config import Settings
from critic import ParseCritic, validate_canvas
from llm import LLMClient
from tools.bbox_generator import generate_bbox_json
from utils import (
    ensure_bbox_json,
    export_xml_to_image,
    extract_mxgraphmodel,
    format_prompt,
    load_prompt,
    normalize_render_size_top_left,
)


class ParserAgent:
    def __init__(self, settings: Settings, *, case_id: str = "") -> None:
        self.settings = settings
        self.log_prefix = f"[{case_id}]" if case_id else ""
        self.llm = LLMClient(settings)
        self.system_prompt = load_prompt("parse_system.txt")

    def parse(
        self,
        image_path: Path,
        *,
        previous_xml: str = "",
        feedback: str = "",
        bbox_json: str = "",
    ) -> str:
        with Image.open(image_path) as img:
            width, height = img.width, img.height

        # construct the prompt
        prompt = format_prompt(
            "parse_prompt.txt",
            width=width,
            height=height,
            previous_xml=previous_xml,
            feedback=feedback,
            bbox_json=bbox_json,
        )

        last_errors: list[str] = []
        for attempt in range(1, 4):
            time.sleep(1)
            # call LLM to generate XML
            raw = self._generate(prompt, image_path, attempt)
            # extract math cells and normalize them
            xml_text = _normalize_math_cells(extract_mxgraphmodel(raw))
            # validate the XML syntax
            ok, errors = validate_drawio_xml_syntax(xml_text)
            if ok:
                print(f"{self.log_prefix}[parse] XML valid on attempt {attempt}/3")
                return xml_text
            last_errors = errors
            print(f"{self.log_prefix}[parse] XML invalid on attempt {attempt}/3")

        raise RuntimeError("Parser returned invalid XML:\n" + "\n".join(f"- {e}" for e in last_errors))

    def _generate(self, prompt: str, image_path: Path, attempt: int) -> str:
        return self.llm.generate(
            model=self.settings.parser_model,
            system_prompt=self.system_prompt,
            prompt=prompt,
            images=[image_path],
            component="parse_agent",
            attempt=attempt,
        )


def run_parse_stage(
    settings: Settings,
    image_path: Path,
    output_dir: Path,
    *,
    bbox_path: Path | None = None,
    iterations: int = 3,
    timeline: dict[str, object] | None = None,
    case_id: str = "",
) -> Path:
    log_prefix = f"[{case_id}]" if case_id else ""
    stage_started = time.perf_counter()
    stage_timeline: dict[str, object] = {"iterations": []}
    if timeline is not None:
        stages = timeline.setdefault("stages", {})
        if isinstance(stages, dict):
            stages["parse"] = stage_timeline
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output_dir / image_path.name)
    with Image.open(image_path) as source_image:
        render_target_size = source_image.size

    parser = ParserAgent(settings, case_id=case_id)
    critic = ParseCritic(settings)
    bbox_path = _resolve_bbox_path(
        image_path=image_path,
        output_dir=output_dir,
        bbox_path=bbox_path,
        log_prefix=log_prefix,
    )
    bbox_json = load_bbox_for_prompt(bbox_path)
    previous_xml = ""
    feedback = ""
    final_xml_path = output_dir / "final.xml"
    print(
        f"{log_prefix}[parse] starting model={settings.parser_model} "
        f"critic_model={settings.critic_model} iterations={max(1, iterations)}",
        flush=True,
    )

    # iteration loop
    for index in range(1, max(1, iterations) + 1):
        # make iter dir
        iter_dir = output_dir / f"iter_{index}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"{log_prefix}[parse] iteration {index}/{max(1, iterations)} generating XML",
            flush=True,
        )

        # parse the image to XML
        xml = parser.parse(
            image_path,
            previous_xml=previous_xml,
            feedback=feedback,
            bbox_json=bbox_json,
        )

        # save parse.xml and parse.jpg for this iteration
        xml_path = iter_dir / "parse.xml"
        rendered_path = iter_dir / "parse.jpg"
        xml_path.write_text(xml, encoding="utf-8")
        export_xml_to_image(
            xml_path,
            rendered_path,
            drawio_bin=settings.drawio_bin,
            fmt="jpg",
        )
        print(f"{log_prefix}[parse] XML saved to {xml_path} and rendered to {rendered_path}", flush=True)
        normalize_render_size_top_left(rendered_path, render_target_size)

        # validate the xml (deterministic checks) and call the critic
        canvas_reasons = validate_canvas(xml)
        print(
            f"{log_prefix}[parse] deterministic reasons={len(canvas_reasons)}; calling critic",
            flush=True,
        )
        # call the critic to critique the XML
        result = critic.critique(
            target_image=image_path,
            rendered_image=rendered_path,
            parsed_xml=xml,
            reasons=canvas_reasons,
            bbox_json=bbox_json,
        )
        (iter_dir / "critic.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"{log_prefix}[parse] critic ok={result.get('ok')} "
            f"reasons={len(result.get('reasons', []))}",
            flush=True,
        )

        previous_xml = xml
        feedback = str(result.get("suggestion", ""))
        final_xml_path.write_text(xml, encoding="utf-8")

        # if the critic return "ok" => break the loop
        if result.get("ok") is True and not canvas_reasons:
            print(f"{log_prefix}[parse] accepted at iteration {index}", flush=True)
            break

    stage_seconds = time.perf_counter() - stage_started
    stage_timeline["status"] = "completed"
    stage_timeline["total_seconds"] = round(stage_seconds, 3)
    print(
        f"{log_prefix}[parse] completed output={final_xml_path} total={stage_seconds:.3f}s",
        flush=True,
    )
    return final_xml_path


def _resolve_bbox_path(
    *,
    image_path: Path,
    output_dir: Path,
    bbox_path: Path | None,
    log_prefix: str,
) -> Path:
    """Use an existing bbox, or generate one once for this parse stage."""
    resolved = ensure_bbox_json(image_path=image_path, bbox_path=bbox_path)
    if resolved is not None:
        print(f"{log_prefix}[parse] using bbox={resolved}", flush=True)
        return resolved
    if bbox_path is not None:
        raise FileNotFoundError(f"Bounding box file not found: {bbox_path}")

    generated_dir = output_dir / "bbox"
    print(
        f"{log_prefix}[parse] bbox not found; generating with SAM3 in {generated_dir}",
        flush=True,
    )
    generated = generate_bbox_json(image_path, generated_dir)
    print(f"{log_prefix}[parse] generated bbox={generated}", flush=True)
    return generated


def load_bbox_for_prompt(bbox_path: Path) -> str:
    if not bbox_path or not bbox_path.exists():
        raise FileNotFoundError(f"Bounding box file not found: {bbox_path}")

    raw_text = bbox_path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw_text)

    if isinstance(parsed, dict) and isinstance(parsed.get("objects"), list):
        parsed = {"canvas": parsed.get("canvas"), "objects": parsed.get("objects")}
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def validate_drawio_xml_syntax(xml_text: str) -> tuple[bool, list[str]]:
    text = xml_text.strip()
    errors: list[str] = []
    # forbid <style> tags
    if re.search(r"<\s*/?\s*style\b", text, re.IGNORECASE):
        errors.append("Forbidden <style> tag detected. Use mxCell style attributes only.")
    # need to start with <mxGraphModel> and end with </mxGraphModel>
    if not text.startswith("<mxGraphModel"):
        errors.append("XML must start with <mxGraphModel>.")
    if not re.search(r"</mxGraphModel>\s*$", text):
        errors.append("Missing closing </mxGraphModel> tag.")
    try:
        # use ElementTree to parse the XML and check for well-formedness
        ET.fromstring(text)
    except ET.ParseError as exc:
        errors.append(f"XML parse error: {exc}.")
    return not errors, errors


def _normalize_math_cells(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return xml_text

    for cell in root.findall(".//mxCell"):
        value = cell.get("value", "")
        style = cell.get("style", "")
        has_math = bool(value) and any(token in value for token in (r"\(", "$$", r"\frac", r"\sqrt", "^", "_"))
        # add htm=1 and math=1 if math is detected
        if has_math:
            if "html=1" not in style:
                style += ";html=1"
            if "math=1" not in style:
                style += ";math=1"
            cell.set("style", re.sub(r";+", ";", style).strip(";"))
    return ET.tostring(root, encoding="unicode")

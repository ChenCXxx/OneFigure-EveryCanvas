from __future__ import annotations

import base64
import html
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT_DIR / "prompts"

YOLO_CLASS_TO_TYPE = {
    "0": "rect",
    "1": "image",
}


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() == "br":
            self.parts.append("\n")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() == "br":
            self.parts.append("\n")


def normalize_visible_text(value: str | None) -> str:
    """Normalize mxCell value markup while preserving visible text content."""
    text = value or ""
    # mxGraph values may contain nested entities such as "&amp;#xa;".
    for _ in range(3):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded

    # Treat a literal backslash-n from an LLM response as a line break.
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")

    parser = _VisibleTextParser()
    try:
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        # Fall back to the decoded source when malformed HTML-like text appears.
        pass

    # Line breaks, tabs, non-breaking spaces, and ordinary spaces are equivalent
    # for protected visible-text comparison.
    return re.sub(r"\s+", " ", text).strip()


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def format_prompt(name: str, **values: Any) -> str:
    template = load_prompt(name)
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def extract_mxgraphmodel(content: str) -> str:
    text = content.strip()
    match = re.search(r"(<mxGraphModel[\s\S]*?</mxGraphModel>)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    block = re.search(r"```(?:xml)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if block:
        return extract_mxgraphmodel(block.group(1))
    return text


def extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    candidates = getattr(response, "candidates", []) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", []) if content else []
        texts = [part.text for part in parts if getattr(part, "text", None)]
        if texts:
            return "\n".join(texts).strip()

    output_items = getattr(response, "output", []) or []
    texts = []
    for item in output_items:
        for content_item in getattr(item, "content", []) or []:
            text_item = getattr(content_item, "text", None)
            if text_item:
                texts.append(text_item)
    if texts:
        return "\n".join(texts).strip()

    raise RuntimeError("Model returned no text.")


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            return json.loads(match.group(1))
        raise


def wrap_as_drawio(graph_xml: str) -> str:
    # An XML declaration is only legal at the beginning of the document.
    # Stage XML files may contain ``<?xml ...?>``; remove it before nesting
    # the graph model inside the draw.io <diagram> element.
    graph_xml = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", graph_xml, count=1, flags=re.IGNORECASE)
    return (
        '<mxfile host="app.diagrams.net">\n'
        '  <diagram id="diagram-1" name="Page-1">\n'
        f"{graph_xml.strip()}\n"
        "  </diagram>\n"
        "</mxfile>\n"
    )


def _drawio_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    # Keep the subprocess environment portable. Users can prepend custom
    # locations through PATH or set DRAWIO_BIN in the component .env file.
    extra_paths: list[str] = []
    existing = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(
        extra_paths + ([existing] if existing else [])
    )
    return env


def resolve_drawio_bin(drawio_bin: str = "") -> str:
    if drawio_bin and Path(drawio_bin).exists():
        return drawio_bin

    found = shutil.which("drawio") or shutil.which("draw.io")
    if found:
        return found
    windows = Path(r"C:\Program Files\draw.io\draw.io.exe")
    if windows.exists():
        return str(windows)
    mac = Path("/Applications/draw.io.app/Contents/MacOS/draw.io")
    if mac.exists():
        return str(mac)
    raise RuntimeError("draw.io CLI not found. Set DRAWIO_BIN in .env.")


def export_xml_to_image(
    xml_path: Path,
    image_path: Path,
    *,
    drawio_bin: str = "",
    fmt: str = "png",
) -> Path:
    drawio_path = xml_path.with_suffix(".drawio")
    drawio_path.write_text(wrap_as_drawio(xml_path.read_text(encoding="utf-8")), encoding="utf-8")
    cmd = [
        resolve_drawio_bin(drawio_bin),
        "--export",
        "--format",
        fmt,
        "--output",
        str(image_path),
        str(drawio_path),
    ]
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_drawio_subprocess_env(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"draw.io export failed ({completed.returncode}) "
            f"with command {cmd}:\n{completed.stderr}"
        )
    return image_path

def openai_image_part(path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{guess_mime(path)};base64,{encoded}"}}


def vertex_image_part(path: Path) -> Any:
    from google.genai import types

    return types.Part.from_bytes(data=path.read_bytes(), mime_type=guess_mime(path))


def guess_mime(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed:
        return guessed

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    raise ValueError(f"Unsupported image type: {path}")


def resize_canvas_same_area(
    input_width: int,
    input_height: int,
    ratio_width: int,
    ratio_height: int,
) -> tuple[int, int]:
    area = input_width * input_height
    ratio = ratio_width / ratio_height

    output_width = round(math.sqrt(area * ratio))
    output_height = round(math.sqrt(area / ratio))

    return output_width, output_height


def convert_yolo_bbox_txt_to_json(
    bbox_txt_path: Path,
    image_path: Path,
    output_json_path: Path | None = None,
) -> Path:
    """Convert normalized YOLO bbox rows into the pipeline bbox JSON schema."""
    bbox_txt_path = bbox_txt_path.expanduser().resolve()
    image_path = image_path.expanduser().resolve()
    output_json_path = (
        output_json_path.expanduser().resolve()
        if output_json_path is not None
        else bbox_txt_path.with_suffix(".json")
    )

    if not bbox_txt_path.is_file():
        raise FileNotFoundError(f"BBox TXT file not found: {bbox_txt_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    raw_text = bbox_txt_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"BBox TXT file is empty: {bbox_txt_path}")

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    objects: list[dict[str, object]] = []
    warnings: list[str] = []
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 5:
            warnings.append(f"line {line_number}: expected at least 5 fields")
            continue

        label = parts[0]
        try:
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])
        except ValueError:
            warnings.append(f"line {line_number}: coordinates are not numeric")
            continue

        normalized_values = (
            x_center_norm,
            y_center_norm,
            width_norm,
            height_norm,
        )
        if not all(0.0 <= value <= 1.0 for value in normalized_values):
            warnings.append(f"line {line_number}: coordinates are outside 0..1")
            continue
        if width_norm == 0 or height_norm == 0:
            warnings.append(f"line {line_number}: width or height is zero")
            continue

        box_width = width_norm * image_width
        box_height = height_norm * image_height
        xmin = x_center_norm * image_width - box_width / 2
        ymin = y_center_norm * image_height - box_height / 2
        xmax = xmin + box_width
        ymax = ymin + box_height

        xmin = max(0.0, min(xmin, float(image_width)))
        ymin = max(0.0, min(ymin, float(image_height)))
        xmax = max(0.0, min(xmax, float(image_width)))
        ymax = max(0.0, min(ymax, float(image_height)))

        objects.append(
            {
                "type": YOLO_CLASS_TO_TYPE.get(label, label),
                "x": round(xmin, 1),
                "y": round(ymin, 1),
                "width": round(xmax - xmin, 1),
                "height": round(ymax - ymin, 1),
            }
        )

    for warning in warnings:
        print(f"[bbox] warning {bbox_txt_path.name} {warning}", flush=True)
    if not objects:
        raise ValueError(f"No valid bounding boxes found in {bbox_txt_path}")

    payload = {
        "canvas": {
            "width": float(image_width),
            "height": float(image_height),
        },
        "objects": objects,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[bbox] converted {bbox_txt_path} -> {output_json_path} "
        f"objects={len(objects)}",
        flush=True,
    )
    return output_json_path


def ensure_bbox_json(
    image_path: Path,
    bbox_path: Path | None = None,
) -> Path | None:
    """Resolve bbox JSON, converting a sibling bbox.txt when JSON is absent."""
    image_path = image_path.expanduser().resolve()
    default_json = image_path.parent / "bbox.json"
    default_txt = image_path.parent / "bbox.txt"

    if bbox_path is None:
        json_path = default_json
        txt_path = default_txt
    else:
        requested_path = bbox_path.expanduser().resolve()
        if requested_path.suffix.lower() == ".txt":
            txt_path = requested_path
            json_path = requested_path.with_suffix(".json")
        else:
            json_path = requested_path
            txt_path = requested_path.with_suffix(".txt")

    if json_path.is_file():
        return json_path
    if txt_path.is_file():
        return convert_yolo_bbox_txt_to_json(
            bbox_txt_path=txt_path,
            image_path=image_path,
            output_json_path=json_path,
        )
    return None

def normalize_render_size_top_left(
    image_path: Path,
    target_size: tuple[int, int],
    *,
    background: str = "white",
) -> None:
    from PIL import Image

    target_w, target_h = target_size

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        canvas = Image.new("RGB", (target_w, target_h), background)

        crop = img.crop((
            0,
            0,
            min(img.width, target_w),
            min(img.height, target_h),
        ))

        canvas.paste(crop, (0, 0))
        canvas.save(image_path)

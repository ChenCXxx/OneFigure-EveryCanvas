"""Shared image-processing and JSON helpers for the benchmark."""
from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


# ------------------------------------------
# Image helpers
# ------------------------------------------

def image_mime(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(path.suffix.lower(), "application/octet-stream")


def flatten_transparency(image: Image.Image) -> Image.Image:
    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_alpha:
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def preprocess_image(path: Path) -> tuple[bytes, str]:
    """Read, normalize, and encode an image as PNG bytes plus MIME type."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as source:
        image = flatten_transparency(source)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


# ------------------------------------------
# JSON helpers
# ------------------------------------------

def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_json_text(text: str) -> dict[str, Any]:
    """Parse plain or markdown-fenced JSON returned by an LLM."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}

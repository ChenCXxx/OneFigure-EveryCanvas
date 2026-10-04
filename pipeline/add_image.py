"""Restore image cells after layout.

The parser XML is used to locate image regions in the original image. Those
regions are cropped and then embedded into the final layout XML. This stage is
intentionally a pipeline-only API; it has no standalone CLI.
"""

from __future__ import annotations

import base64
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from utils import export_xml_to_image


def _absolute_coords(
    cell: ET.Element,
    cells_by_id: dict[str, ET.Element],
    visiting: set[str] | None = None,
) -> tuple[float, float]:
    """Return a cell's coordinates in the root canvas coordinate system."""
    geometry = cell.find("mxGeometry")
    if geometry is None:
        return 0.0, 0.0

    x = float(geometry.get("x", 0))
    y = float(geometry.get("y", 0))
    parent_id = cell.get("parent")
    if not parent_id or parent_id in {"0", "1"}:
        return x, y

    visiting = set() if visiting is None else visiting
    if parent_id in visiting:
        raise ValueError(f"Cycle detected in mxCell parents at {parent_id!r}")

    parent = cells_by_id.get(parent_id)
    # if no parent => return x, y
    if parent is None:
        return x, y
    
    # otherwise, recursively compute the parent's absolute coordinates
    visiting.add(parent_id)
    parent_x, parent_y = _absolute_coords(parent, cells_by_id, visiting)
    return x + parent_x, y + parent_y


def _is_image_cell(cell: ET.Element) -> bool:
    return "shape=image" in (cell.get("style") or "")


def crop_images_from_xml(
    *,
    source_image: Path,
    xml_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Crop image cells from ``source_image`` using geometry in ``xml_path``."""
    source_image = Path(source_image).expanduser().resolve()
    xml_path = Path(xml_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not source_image.is_file():
        raise FileNotFoundError(f"Source image not found: {source_image}")
    if not xml_path.is_file():
        raise FileNotFoundError(f"Parser XML not found: {xml_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    root = ET.parse(xml_path).getroot()
    cells_by_id = {
        cell_id: cell
        for cell in root.findall(".//mxCell")
        if (cell_id := cell.get("id"))
    }

    cropped_paths: list[Path] = []
    with Image.open(source_image) as opened_image:
        image = opened_image.convert("RGB")
        for cell in cells_by_id.values():
            # if not an image cell, skip
            if not _is_image_cell(cell):
                continue

            cell_id = cell.get("id")
            geometry = cell.find("mxGeometry")
            if not cell_id or geometry is None:
                continue

            width = float(geometry.get("width", 0))
            height = float(geometry.get("height", 0))
            if width <= 0 or height <= 0:
                continue
            
            # compute the cropping box in absolute coordinates
            x, y = _absolute_coords(cell, cells_by_id)
            cropped = image.crop((x, y, x + width, y + height))
            
            # save as image file
            output_path = output_dir / f"{cell_id}.png"
            cropped.save(output_path)
            cropped_paths.append(output_path)

    # return the list of cropped image paths
    return cropped_paths


def _embed_png_as_data_uri(image_path: Path) -> str:
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    # draw.io style values use semicolons as separators.  Omitting the
    # optional ``;base64`` token keeps the data URI from being split by the
    # style parser (draw.io still decodes the PNG payload correctly).
    return f"data:image/png,{data}"


def _replace_style_value(style: str, key: str, value: str) -> str:
    parts = [part for part in style.split(";") if part]
    # A previous data URI may have been split at ``;base64`` by the
    # draw.io style grammar, leaving a standalone ``base64,...`` fragment.
    kept = [
        part for part in parts
        if not part.startswith(f"{key}=") and not part.startswith("base64,")
    ]
    kept.append(f"{key}={value}")
    return ";".join(kept)


def embed_cropped_images(
    *,
    layout_xml: Path,
    crop_dir: Path,
) -> int:
    """Embed crop images into image cells in ``layout_xml`` in place."""
    layout_xml = Path(layout_xml).expanduser().resolve()
    crop_dir = Path(crop_dir).expanduser().resolve()

    tree = ET.parse(layout_xml)
    root = tree.getroot()
    embedded_count = 0

    for cell in root.findall(".//mxCell"):
        # if not an image cell, skip
        if not _is_image_cell(cell):
            continue

        cell_id = cell.get("id")
        if not cell_id:
            continue

        # find the corresponding cropped image
        image_path = crop_dir / f"{cell_id}.png"
        if not image_path.is_file():
            continue

        style = cell.get("style") or ""
        # replace "image=placeholder" with "image=data:image/png;base64,..."
        cell.set(
            "style",
            _replace_style_value(
                style,
                "image",
                _embed_png_as_data_uri(image_path),
            ),
        )
        embedded_count += 1

    tree.write(layout_xml, encoding="utf-8", xml_declaration=True)
    return embedded_count


def run_add_image_stage(
    *,
    source_image: Path,
    parse_xml: Path,
    layout_xml: Path,
    output_dir: Path,
    drawio_bin: str = "",
) -> tuple[Path, Path]:
    """Crop from parser XML, embed into layout XML, and render final PNG.

    Outputs are stored as follows::

        output_dir/image/crop_image/<cell_id>.png
        output_dir/image/final.xml
        output_dir/image/final.png
    """
    output_dir = Path(output_dir).expanduser().resolve()
    image_dir = output_dir / "image"
    crop_dir = image_dir / "crop_image"
    final_xml = image_dir / "final.xml"
    final_png = image_dir / "final.png"

    image_dir.mkdir(parents=True, exist_ok=True)
    if crop_dir.exists():
        shutil.rmtree(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    # get the list of cropped image paths
    cropped_paths = crop_images_from_xml(
        source_image=source_image,
        xml_path=parse_xml,
        output_dir=crop_dir,
    )
    # Keep layout/final.xml as the layout-only artifact and create the
    # self-contained image-stage XML under image/final.xml.
    shutil.copy2(layout_xml, final_xml)
    embedded_count = embed_cropped_images(
        layout_xml=final_xml,
        crop_dir=crop_dir,
    )
    # export the final layout XML to a PNG image
    export_xml_to_image(
        final_xml,
        final_png,
        drawio_bin=drawio_bin,
        fmt="png",
    )

    print(
        f"[add_image] cropped={len(cropped_paths)} "
        f"embedded={embedded_count} output={final_png}",
        flush=True,
    )
    return final_xml, final_png

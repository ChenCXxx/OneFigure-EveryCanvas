#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def parse_style(style_text: str) -> dict[str, str]:
    style: dict[str, str] = {}
    for part in style_text.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            style[key] = value
    return style


def absolute_vertex_geometries(xml_text: str) -> list[dict[str, object]]:
    root = ET.fromstring(xml_text)
    cell_map = {
        cell.get("id"): cell
        for cell in root.findall(".//mxCell")
        if cell.get("id")
    }
    geom_cache: dict[str, dict[str, float] | None] = {}

    def get_abs_geom(cell_id: str) -> dict[str, float] | None:
        if cell_id in geom_cache:
            return geom_cache[cell_id]
        cell = cell_map.get(cell_id)
        if cell is None:
            geom_cache[cell_id] = None
            return None
        geom_el = cell.find("mxGeometry")
        if geom_el is None:
            geom_cache[cell_id] = None
            return None
        x = float(geom_el.get("x", 0))
        y = float(geom_el.get("y", 0))
        w = float(geom_el.get("width", 0))
        h = float(geom_el.get("height", 0))
        parent_id = cell.get("parent")
        if parent_id and parent_id not in ("0", "1"):
            parent_geom = get_abs_geom(parent_id)
            if parent_geom is not None:
                x += parent_geom["x"]
                y += parent_geom["y"]
        geom_cache[cell_id] = {"x": x, "y": y, "w": w, "h": h}
        return geom_cache[cell_id]

    blocks: list[dict[str, object]] = []
    for cell in root.findall(".//mxCell"):
        if cell.get("vertex") != "1":
            continue
        cell_id = cell.get("id")
        if not cell_id:
            continue
        geom = get_abs_geom(cell_id)
        if geom is None or geom["w"] <= 0 or geom["h"] <= 0:
            continue
        style_text = cell.get("style", "")
        style = parse_style(style_text)
        blocks.append(
            {
                "id": cell_id,
                "value": cell.get("value", ""),
                "style": style,
                "is_text": "text" in style_text.split(";"),
                "geometry": geom,
            }
        )
    return blocks


def sample_block_color_hints(xml_text: str, reference_path: Path) -> list[dict[str, object]]:
    root = ET.fromstring(xml_text)
    page_width = float(root.get("pageWidth", 0) or 0)
    page_height = float(root.get("pageHeight", 0) or 0)

    with Image.open(reference_path) as image:
        ref = image.convert("RGB")
        img_w, img_h = ref.size

        if page_width > 0 and page_height > 0:
            scale_x = img_w / page_width
            scale_y = img_h / page_height
        else:
            scale_x = 1.0
            scale_y = 1.0

        hints: list[dict[str, object]] = []
        for block in absolute_vertex_geometries(xml_text):
            geom = block["geometry"]
            x0 = max(0, min(img_w - 1, int(round(float(geom["x"]) * scale_x))))
            y0 = max(0, min(img_h - 1, int(round(float(geom["y"]) * scale_y))))
            x1 = max(x0 + 1, min(img_w, int(round((float(geom["x"]) + float(geom["w"])) * scale_x))))
            y1 = max(y0 + 1, min(img_h, int(round((float(geom["y"]) + float(geom["h"])) * scale_y))))
            region = ref.crop((x0, y0, x1, y1))
            quantized = region.convert("P", palette=Image.ADAPTIVE, colors=4).convert("RGB")
            colors = quantized.getcolors(maxcolors=max(region.size[0] * region.size[1], 16)) or []
            colors.sort(key=lambda item: item[0], reverse=True)
            if colors:
                dominant_rgb = colors[0][1]
            else:
                dominant_rgb = region.getpixel((max(0, region.size[0] // 2), max(0, region.size[1] // 2)))
            hints.append(
                {
                    "id": block["id"],
                    "value": str(block["value"])[:80],
                    "geometry": {
                        "x": round(float(geom["x"]), 1),
                        "y": round(float(geom["y"]), 1),
                        "w": round(float(geom["w"]), 1),
                        "h": round(float(geom["h"]), 1),
                    },
                    "region_in_reference": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "reference_region_dominant_hex": rgb_to_hex(dominant_rgb),
                    "current_style": {
                        "fillColor": block["style"].get("fillColor", ""),
                        "strokeColor": block["style"].get("strokeColor", ""),
                        "fontColor": block["style"].get("fontColor", ""),
                        "gradientColor": block["style"].get("gradientColor", ""),
                    },
                }
            )
    return hints

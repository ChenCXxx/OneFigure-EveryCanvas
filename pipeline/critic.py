from __future__ import annotations

import json
from collections import Counter
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from config import Settings
from llm import LLMClient
from utils import (
    extract_json_object,
    format_prompt,
    load_prompt,
)


# ================================
# validation for layout agent
# ================================

_LAYOUT_DECORATION_SHAPES = {
    "shape=mxgraph.flowchart.annotation_1",
    "shape=curlyBracket",
}

SPAN_RATIO = 0.8
MAX_MARGIN_RATIO = 0.2
CENTER_TOL_RATIO = 0.05
AREA_RATIO_TOL = 0.2
IMAGE_RATIO_TOL = 0.15


def _is_layout_decoration(style: str) -> bool:
    return any(shape in style for shape in _LAYOUT_DECORATION_SHAPES)


def is_layout_decoration(style: str) -> bool:
    return _is_layout_decoration(style)


def rects_overlap(a: dict[str, float], b: dict[str, float]) -> bool:
    return not (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"]
                or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])


def rect_contains(outer: dict[str, float], inner: dict[str, float], tol: float = 1e-6) -> bool:
    return (outer["x"] <= inner["x"] + tol and outer["y"] <= inner["y"] + tol
            and outer["x"] + outer["w"] >= inner["x"] + inner["w"] - tol
            and outer["y"] + outer["h"] >= inner["y"] + inner["h"] - tol)


def overlap_pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


def is_parent_child_pair(left_id: str, right_id: str, block_map: dict[str, dict[str, Any]]) -> bool:
    left = block_map.get(left_id) or {}
    right = block_map.get(right_id) or {}
    return left.get("parent") == right_id or right.get("parent") == left_id


def collect_overlap_pairs(blocks: list[dict[str, Any]]) -> set[tuple[str, str]]:
    block_map = {b.get("id"): b for b in blocks if b.get("id")}
    rects = [(b["id"], b["geometry"]) for b in blocks
             if b.get("id") and b.get("geometry") and not is_layout_decoration(b.get("style", ""))]
    levels = {bid: sum(1 for oid, other in rects if oid != bid and rect_contains(other, geom))
              for bid, geom in rects}
    result: set[tuple[str, str]] = set()
    for i, (left_id, left) in enumerate(rects):
        for right_id, right in rects[i + 1:]:
            if is_parent_child_pair(left_id, right_id, block_map):
                continue
            if levels[left_id] == levels[right_id] and rects_overlap(left, right):
                result.add(overlap_pair_key(left_id, right_id))
    return result


def _parse_mxgraph_text(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    canvas = None
    graph_model = root if root.tag == "mxGraphModel" else root.find(".//mxGraphModel")
    if graph_model is not None:
        w, h = graph_model.get("pageWidth"), graph_model.get("pageHeight")
        if w is not None and h is not None:
            canvas = {"width": float(w), "height": float(h)}

    cell_map = {
        cell.get("id"): cell
        for cell in root.findall(".//mxCell")
        if cell.get("id")
    }
    cache: dict[str, dict[str, float] | None] = {}

    def absolute_geometry(cell_id: str):
        if cell_id in cache:
            return cache[cell_id]
        cell = cell_map.get(cell_id)
        geom = cell.find("mxGeometry") if cell is not None else None
        if geom is None:
            cache[cell_id] = None
            return None
        value = {"x": float(geom.get("x", 0)), "y": float(geom.get("y", 0)),
                 "w": float(geom.get("width", 0)), "h": float(geom.get("height", 0))}
        parent = cell.get("parent")
        if parent and parent not in ("0", "1"):
            parent_geometry = absolute_geometry(parent)
            if parent_geometry is not None:
                value["x"] += parent_geometry["x"]
                value["y"] += parent_geometry["y"]
        cache[cell_id] = value
        return value

    blocks = []
    for cell in root.findall(".//mxCell"):
        if cell.get("vertex") != "1" or "edgeLabel" in cell.get("style", "").split(";"):
            continue
        blocks.append({"id": cell.get("id"), "parent": cell.get("parent"),
                       "style": cell.get("style", ""),
                       "geometry": absolute_geometry(cell.get("id", ""))})
    edges = [{"id": cell.get("id"), "source": cell.get("source"),
              "target": cell.get("target"), "style": cell.get("style", ""),
              "geometry": cell.find("mxGeometry")}
             for cell in root.findall(".//mxCell") if cell.get("edge") == "1"]
    return {"canvas": canvas, "blocks": blocks, "edges": edges, "root": root}


def parse_mxgraph(xml_text: str) -> dict[str, Any]:
    """Legacy parse_mxgraph equivalent for the string-based pipeline."""
    return _parse_mxgraph_text(xml_text)


def validate_canvas(xml_text: str) -> list[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    graph_model = root if root.tag == "mxGraphModel" else root.find(".//mxGraphModel")
    if graph_model is None:
        return []
    try:
        canvas_w = float(graph_model.get("pageWidth", 0))
        canvas_h = float(graph_model.get("pageHeight", 0))
    except (TypeError, ValueError):
        return []
    if canvas_w <= 0 or canvas_h <= 0:
        return []

    reasons: list[str] = []
    output_vertex_cells = [c for c in root.findall(".//mxCell") if c.get("vertex") == "1"]
    if not output_vertex_cells:
        reasons.append('[R1] blocks are empty (mxCell vertex="1").')
    for cell in root.findall(".//mxCell"):
        if cell.get("vertex") != "1":
            continue
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        try:
            x = float(geom.get("x", 0))
            y = float(geom.get("y", 0))
            w = float(geom.get("width", 0))
            h = float(geom.get("height", 0))
        except (TypeError, ValueError):
            continue
        cell_id = cell.get("id", "")
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            reasons.append(f"[R2] block {cell_id} has invalid geometry (x,y,w,h).")
        elif x + w > canvas_w + 1e-6 or y + h > canvas_h + 1e-6:
            reasons.append(f"[R2] block {cell_id} exceeds canvas bounds.")
    return reasons


# Main validation function
def validate_xml(
    xml_path: str,
    input_xml_path: str | None = None,
) -> tuple[bool, list[str], str]:
    data = parse_mxgraph(xml_path)
    canvas = data["canvas"]
    blocks = data["blocks"]
    edges = data["edges"]
    allowed_overlap_pairs = set()

    if input_xml_path:
        input_data = parse_mxgraph(input_xml_path)
        allowed_overlap_pairs = collect_overlap_pairs(input_data["blocks"])

    reasons = []
    suggestions = []

    # 1. Basic format
    if not canvas or not isinstance(canvas.get("width"), (int, float)) or not isinstance(canvas.get("height"), (int, float)):
        reasons.append("[R1] canvas.width/height 缺失 (mxGraphModel@pageWidth/pageHeight)")
    if not blocks:
        reasons.append("[R1] blocks 為空 (mxCell vertex=\"1\")")
    for b in blocks:
        if not b.get("id"):
            reasons.append("[R1] block 缺少 id")
        g = b.get("geometry")
        if g is None or any(k not in g for k in ("x", "y", "w", "h")):
            reasons.append(f"[R1] block {b.get('id')} 缺少 geometry (x,y,w,h)")

    if not canvas:
        return False, reasons, "請補齊 pageWidth/pageHeight 或使用 --canvas"

    cw = canvas["width"]
    ch = canvas["height"]
    max_dim = max(cw, ch)

    rects = []
    block_map = {b.get("id"): b for b in blocks if b.get("id")}
    for b in blocks:
        g = b.get("geometry")
        if not g:
            continue
        style = b.get("style", "")
        if is_layout_decoration(style):
            continue
        rects.append((b.get("id"), g))

    level_map = {}
    for bid, r in rects:
        level_map[bid] = sum(
            1 for other_id, other_r in rects
            if other_id != bid and rect_contains(other_r, r)
        )

    # 2. Inside canvas
    for bid, r in rects:
        if r["x"] < 0 or r["y"] < 0 or r["w"] <= 0 or r["h"] <= 0:
            reasons.append(f"[R2] block {bid} 幾何值無效 (x,y,w,h)")
        if r["x"] + r["w"] > cw + 1e-6 or r["y"] + r["h"] > ch + 1e-6:
            reasons.append(f"[R2] block {bid} 超出畫布範圍")

    # 3. Overlap
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            left_id = rects[i][0]
            right_id = rects[j][0]
            if is_parent_child_pair(left_id, right_id, block_map):
                continue
            if level_map.get(left_id) != level_map.get(right_id):
                continue
            a = rects[i][1]
            b = rects[j][1]
            if rects_overlap(a, b):
                if overlap_pair_key(left_id, right_id) in allowed_overlap_pairs:
                    continue
                reasons.append(f"[R3] 同層 block {left_id} 與 {right_id} 重疊")

    # 5. Canvas usage
    xs = [r["x"] for _, r in rects] or [0]
    ys = [r["y"] for _, r in rects] or [0]
    x2s = [r["x"] + r["w"] for _, r in rects] or [0]
    y2s = [r["y"] + r["h"] for _, r in rects] or [0]
    span_w = max(x2s) - min(xs)
    span_h = max(y2s) - min(ys)
    if span_w / cw < SPAN_RATIO:
        reasons.append("[R5] blocks 水平跨度不足")
    if span_h / ch < SPAN_RATIO:
        reasons.append("[R5] blocks 垂直跨度不足")

    left_margin = min(xs)
    right_margin = cw - max(x2s)
    top_margin = min(ys)
    bottom_margin = ch - max(y2s)
    max_margin = max_dim * MAX_MARGIN_RATIO
    if left_margin > max_margin:
        reasons.append("[R5] 左側留白過大")
    if right_margin > max_margin:
        reasons.append("[R5] 右側留白過大")
    if top_margin > max_margin:
        reasons.append("[R5] 上側留白過大")
    if bottom_margin > max_margin:
        reasons.append("[R5] 下側留白過大")

    layout_cx = (min(xs) + max(x2s)) / 2
    layout_cy = (min(ys) + max(y2s)) / 2
    canvas_cx = cw / 2
    canvas_cy = ch / 2
    center_tol = max_dim * CENTER_TOL_RATIO
    if abs(layout_cx - canvas_cx) > center_tol:
        reasons.append("[R5] 佈局未水平置中")
    if abs(layout_cy - canvas_cy) > center_tol:
        reasons.append("[R5] 佈局未垂直置中")

    # 7. Area ratio (if input_xml_path)
    if input_xml_path:
        in_data = parse_mxgraph(input_xml_path)
        in_map = {b["id"]: b.get("geometry") for b in in_data["blocks"] if b.get("id")}
        in_style_map = {b["id"]: b.get("style", "") for b in in_data["blocks"] if b.get("id")}
        out_map = {bid: r for bid, r in rects}
        common = [bid for bid in out_map.keys() if bid in in_map and in_map[bid]]
        if common:
            in_areas = {bid: in_map[bid]["w"] * in_map[bid]["h"] for bid in common}
            out_areas = {bid: out_map[bid]["w"] * out_map[bid]["h"] for bid in common}
            in_sum = sum(in_areas.values())
            out_sum = sum(out_areas.values())
            if in_sum > 0 and out_sum > 0:
                for bid in common:
                    in_ratio = in_areas[bid] / in_sum
                    out_ratio = out_areas[bid] / out_sum
                    if abs(out_ratio - in_ratio) > AREA_RATIO_TOL:
                        reasons.append(f"[R7] block {bid} 面積比例變化過大")
                    if "shape=image" in in_style_map.get(bid, ""):
                        in_wh_ratio = in_map[bid]["w"] / max(in_map[bid]["h"], 1e-6)
                        out_wh_ratio = out_map[bid]["w"] / max(out_map[bid]["h"], 1e-6)
                        if abs(out_wh_ratio - in_wh_ratio) / max(in_wh_ratio, 1e-6) > IMAGE_RATIO_TOL:
                            reasons.append(f"[R7] image block {bid} 長寬比例變化過大")

    # 8. Relationship format (edges)
    block_ids = {b.get("id") for b in blocks if b.get("id")}
    if not edges:
        reasons.append("[R8] 沒有任何 edge（連線）")
    for e in edges:
        if e.get("source") and e.get("source") not in block_ids:
            reasons.append(f"[R8] edge {e.get('id')} source 不存在")
        if e.get("target") and e.get("target") not in block_ids:
            reasons.append(f"[R8] edge {e.get('id')} target 不存在")
    if input_xml_path:
        in_data = parse_mxgraph(input_xml_path)
        in_edge_pairs = Counter(
            (e.get("source"), e.get("target"))
            for e in in_data["edges"]
            if e.get("source") and e.get("target")
        )
        out_edge_pairs = Counter(
            (e.get("source"), e.get("target"))
            for e in edges
            if e.get("source") and e.get("target")
        )
        for (src, tgt), count in sorted((out_edge_pairs - in_edge_pairs).items()):
            reasons.append(f"[R8] 多出 relationship {src}->{tgt} ({count} 條)")
        for (src, tgt), count in sorted((in_edge_pairs - out_edge_pairs).items()):
            reasons.append(f"[R8] 缺少 relationship {src}->{tgt} ({count} 條)")

    # 9. Arrow points in canvas
    for e in edges:
        geom = e.get("geometry")
        if geom is None:
            continue
        for pt in geom.findall(".//mxPoint"):
            try:
                px = float(pt.get("x", 0))
                py = float(pt.get("y", 0))
            except ValueError:
                continue
            if px < 0 or py < 0 or px > cw or py > ch:
                reasons.append(f"[R9] edge {e.get('id')} 控制點超出畫布")

    if not reasons:
        suggestions.append("通過基本規則檢查。")

    suggestion = "；".join(suggestions) if suggestions else ""
    return not reasons, reasons, suggestion


# ================================
# Layout critics
# ================================
class LayoutCritic:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = LLMClient(settings)
        self.system_prompt = load_prompt("layout_critic_system.txt")

    def critique(
        self,
        *,
        rendered_image: Path,
        xml_text: str,
        reasons: list[str],
        decorations: list[dict[str, object]] | None = None,
    ) -> dict[str, Any]:
        prompt = format_prompt(
            "layout_critic_prompt.txt",
            xml_text=xml_text,
            reasons="\n".join(reasons),
            decorations=json.dumps(
                decorations or [], ensure_ascii=False, indent=2
            ),
        )
        raw = self.llm.generate(
            model=self.settings.critic_model,
            system_prompt=self.system_prompt,
            prompt=prompt,
            images=[rendered_image],
            component="layout_critic",
        )
        try:
            result = extract_json_object(raw)
        except (json.JSONDecodeError, ValueError):
            result = {"ok": False, "reasons": [], "suggestion": raw}
        if reasons:
            result["ok"] = False
        result.setdefault("ok", False)
        result.setdefault("reasons", [])
        result.setdefault("suggestion", "")
        return result


# ================================
# Parse critics
# ================================
class ParseCritic:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = LLMClient(settings)
        self.system_prompt = load_prompt("parse_critic_system.txt")

    def critique(
        self,
        *,
        target_image: Path,
        rendered_image: Path | None,
        parsed_xml: str,
        reasons: list[str],
        bbox_json: str = "",
    ) -> dict[str, Any]:
        prompt = format_prompt(
            "parse_critic_prompt.txt",
            parsed_xml=parsed_xml,
            reasons="\n".join(reasons),
            bbox_json=bbox_json,
        )
        raw = self._generate(prompt, target_image, rendered_image)
        try:
            result = extract_json_object(raw)
        except (json.JSONDecodeError, ValueError):
            result = {"ok": False, "reasons": [], "suggestion": raw}
        if reasons:
            result["ok"] = False
        result.setdefault("ok", False)
        result.setdefault("reasons", [])
        result.setdefault("suggestion", "")
        return result

    def _generate(
        self, prompt: str, target_image: Path, rendered_image: Path | None
    ) -> str:
        images = []
        if rendered_image:
            images.append(rendered_image)
        images.append(target_image)
        return self.llm.generate(
            model=self.settings.critic_model,
            system_prompt=self.system_prompt,
            prompt=prompt,
            images=images,
            component="parse_critic",
        )


# ================================
# Style critic
# ================================

class StyleCritic:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = LLMClient(settings)
        self.system_prompt = load_prompt("style_critic_system.txt")

    def critique(
        self,
        *,
        reference_image: Path,
        rendered_image: Path,
        styled_xml: str,
        block_color_hints: list[dict[str, object]],
    ) -> dict[str, Any]:
        prompt = format_prompt(
            "style_critic_prompt.txt",
            styled_xml=styled_xml,
            block_color_hints=json.dumps(
                block_color_hints, ensure_ascii=False, indent=2
            ),
        )
        raw = self.llm.generate(
            model=self.settings.critic_model,
            system_prompt=self.system_prompt,
            prompt=prompt,
            images=[reference_image, rendered_image],
            component="style_critic",
        )
        try:
            result = extract_json_object(raw)
        except (json.JSONDecodeError, ValueError):
            result = {"ok": False, "reasons": [], "suggestion": raw}
        result.setdefault("ok", False)
        result.setdefault("reasons", [])
        result.setdefault("suggestion", "")
        return result

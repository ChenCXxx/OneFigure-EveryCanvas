"""Background, occupancy-grid, and empty-region analysis."""
from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw

# ------------- Constants -------------------
COLOR_DISTANCE_THRESHOLD = 5
BACKGROUND_COLOR_AREA_THRESHOLD = 0.10
EXCLUDED_BACKGROUND_COLORS = {(221, 221, 221)}
GRID_ROWS = 20
GRID_COLS = 20
OCCUPIED_THRESHOLD = 0.01
EMPTY_COMPONENT_AREA_THRESHOLD = 0.10
QUALIFYING_OVERLAY_COLOR = (255, 0, 0, 70)
NON_QUALIFYING_OVERLAY_COLOR = (120, 255, 120, 70)
# -------------------------------------------

def color_distance(pixel: np.ndarray, color: np.ndarray) -> int:
    return int(np.abs(pixel.astype(np.int16) - color.astype(np.int16)).max())

def collect_background_candidates(arr: np.ndarray, min_area_ratio: float = BACKGROUND_COLOR_AREA_THRESHOLD) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    # flatten 2D to 1D array of pixels
    flat = arr.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    
    coverage, background_colors = [], []
    for color, count in zip(colors, counts):
        color_tuple = tuple(int(channel) for channel in color.tolist())
        # calculate the area ratio of this color
        area_ratio = count / flat.shape[0]
        
        is_background = area_ratio >= min_area_ratio and color_tuple not in EXCLUDED_BACKGROUND_COLORS
        coverage.append({"color": list(color_tuple), "pixel_count": int(count), "area_ratio": float(area_ratio), "is_background": bool(is_background)})
        if is_background:
            background_colors.append(np.array(color_tuple, dtype=np.uint8))
    coverage.sort(key=lambda item: item["pixel_count"], reverse=True)
    return coverage, background_colors

def build_background_mask(arr: np.ndarray, background_colors: list[np.ndarray], color_distance_threshold: int = COLOR_DISTANCE_THRESHOLD) -> np.ndarray:
    mask = np.zeros(arr.shape[:2], dtype=bool)
    for color in background_colors:
        distance = np.abs(arr.astype(np.int16) - color.astype(np.int16)).max(axis=2)
        mask |= distance <= color_distance_threshold
    return mask

def build_occupancy(
    foreground_mask: np.ndarray,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    occupied_threshold: float = OCCUPIED_THRESHOLD
) -> np.ndarray:
    occupancy = np.zeros((rows, cols), dtype=bool)
    height, width = foreground_mask.shape
    for row in range(rows):
        y1, y2 = round(row * height / rows), round((row + 1) * height / rows)
        for col in range(cols):
            x1, x2 = round(col * width / cols), round((col + 1) * width / cols)
            cell = foreground_mask[y1:y2, x1:x2]
            if cell.size: 
                occupancy[row, col] = cell.mean() >= occupied_threshold
    return occupancy

def build_background_label_grid(
    arr,
    background_colors,
    rows=GRID_ROWS,
    cols=GRID_COLS,
    color_distance_threshold=COLOR_DISTANCE_THRESHOLD,
):
    labels = [[None for _ in range(cols)] for _ in range(rows)]
    if not background_colors:
        return labels

    for r in range(rows):
        y1 = round(r * arr.shape[0] / rows)
        y2 = round((r + 1) * arr.shape[0] / rows)
        for c in range(cols):
            x1 = round(c * arr.shape[1] / cols)
            x2 = round((c + 1) * arr.shape[1] / cols)
            cell = arr[y1:y2, x1:x2]
            if cell.size == 0:
                continue

            best_color = None
            best_ratio = 0.0
            for bg_color in background_colors:
                distance = np.abs(cell.astype(np.int16) - bg_color.astype(np.int16)).max(axis=2)
                ratio = float((distance <= color_distance_threshold).mean())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_color = tuple(int(channel) for channel in bg_color.tolist())

            if best_color is not None and best_ratio >= 0.5:
                labels[r][c] = best_color
    return labels


def find_empty_regions(
    occupancy: np.ndarray,
    background_labels: list[list[tuple[int, int, int] | None]],
    area_threshold: float = EMPTY_COMPONENT_AREA_THRESHOLD
) -> list[dict[str, Any]]:
    rows, cols = occupancy.shape
    visited = np.zeros_like(occupancy, dtype=bool)
    regions = []
    for row in range(rows):
        for col in range(cols):
            if occupancy[row, col] or visited[row, col]: continue
            label = background_labels[row][col]
            visited[row, col] = True
            if label is None: continue
            queue, cells = deque([(row, col)]), []
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for next_row, next_col in ((current_row - 1, current_col), (current_row + 1, current_col), (current_row, current_col - 1), (current_row, current_col + 1)):
                    if 0 <= next_row < rows and 0 <= next_col < cols and not occupancy[next_row, next_col] and not visited[next_row, next_col] and background_labels[next_row][next_col] == label:
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            ratio = len(cells) / (rows * cols)
            regions.append(
                {
                    "cells": cells,
                    "background_color": list(label),
                    "area_ratio": float(ratio),
                    "qualifying": ratio >= area_threshold
                }
            )
    return regions


def grid_cell_to_pixels(
    row: int,
    col: int,
    image_width: int,
    image_height: int,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> tuple[int, int, int, int]:
    """Convert a grid cell coordinate into image pixel coordinates."""
    x1 = round(col * image_width / cols)
    x2 = round((col + 1) * image_width / cols)
    y1 = round(row * image_height / rows)
    y2 = round((row + 1) * image_height / rows)
    return x1, y1, x2, y2


# main function
def visualize_and_analyze_blank_area(
    image_path: Path,
    output_path: Path,
    empty_threshold: float = EMPTY_COMPONENT_AREA_THRESHOLD,
) -> dict[str, Any]:
    """Analyze blank areas, save an overlay, and return all metrics."""
    image_path = Path(image_path)
    output_path = Path(output_path)

    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB").copy()

    arr = np.asarray(image)
    color_coverage, background_colors = collect_background_candidates(arr)
    background_mask = build_background_mask(
        arr,
        background_colors,
    )
    foreground_mask = ~background_mask
    occupancy = build_occupancy(foreground_mask)
    background_labels = build_background_label_grid(
        arr,
        background_colors,
    )
    regions = find_empty_regions(
        occupancy,
        background_labels,
        empty_threshold,
    )

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(overlay)

    for region in regions:
        fill_color = (
            QUALIFYING_OVERLAY_COLOR
            if region["qualifying"]
            else NON_QUALIFYING_OVERLAY_COLOR
        )

        for row, col in region["cells"]:
            box = grid_cell_to_pixels(
                row,
                col,
                image.width,
                image.height,
            )
            draw.rectangle(box, fill=fill_color)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Image.alpha_composite(
        image.convert("RGBA"),
        overlay,
    ).convert("RGB").save(output_path)

    background_region_colors = [
        item["color"]
        for item in color_coverage
        if item["is_background"]
    ]
    primary_background_color = (
        background_region_colors[0]
        if background_region_colors
        else None
    )
    qualifying_ratio = sum(
        region["area_ratio"]
        for region in regions
        if region["qualifying"]
    )

    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "output_image_path": str(output_path),
        "background_color": primary_background_color,
        "background_colors": background_region_colors,
        "color_coverage": color_coverage,
        "foreground_ratio": float(foreground_mask.mean()),
        "occupancy": occupancy.astype(int).tolist(),
        "background_labels": [
            [list(label) if label is not None else None for label in row]
            for row in background_labels
        ],
        "background_regions": [
            {
                "color": region["background_color"],
                "area_ratio": region["area_ratio"],
            }
            for region in regions
            if region["qualifying"]
        ],
        "empty_area_ratio": qualifying_ratio,
    }

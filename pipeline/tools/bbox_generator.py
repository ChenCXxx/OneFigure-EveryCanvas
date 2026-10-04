"""Generate parse-stage bounding boxes from an input image with SAM3.

The generator writes only ``bbox.json``, the normalized object format consumed
by the parse prompt. The intermediate boxlib data remains in memory.

SAM3 and torch are imported lazily so importing this module does not require
the optional SAM3 runtime until automatic generation is actually requested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


MERGE_THRESHOLD = 0.9
MIN_AREA_RATIO = 0.6
SAM_PROMPT = "flowchart image,image,icon,square,rectangle,circle,triangle"
MIN_SCORE = 0.0


def run_sam3_inference(
    input_png: Path,
    sam_prompt: str = SAM_PROMPT,
    min_score: float = MIN_SCORE,
    merge_threshold: float = MERGE_THRESHOLD,
) -> list[dict[str, object]]:
    """Run local SAM3 and return merged pixel-coordinate detections."""
    # The local SAM3 checkout lives at <pipeline-root>/sam3/{sam3,assets}.
    # Add its repository directory so ``import sam3`` resolves to its package.
    sam3_repository = Path(__file__).resolve().parents[1] / "sam3"
    if (sam3_repository / "sam3" / "__init__.py").is_file():
        sys.path.insert(0, str(sam3_repository))

    import torch
    import sam3
    import sam3.model_builder as model_builder
    from sam3.model.decoder import TransformerDecoder
    from sam3.model.sam3_image_processor import Sam3Processor

    input_png = input_png.expanduser().resolve()
    if not input_png.is_file():
        raise FileNotFoundError(f"Input image not found: {input_png}")

    image = Image.open(input_png).convert("RGB")
    prompts = [prompt.strip() for prompt in sam_prompt.split(",") if prompt.strip()]
    if not prompts:
        raise ValueError("sam_prompt must contain at least one prompt")

    sam3_package_dir = (
        Path(sam3.__path__[0])
        if hasattr(sam3, "__path__")
        else Path(sam3.__file__).parent
    )
    bpe_name = "bpe_simple_vocab_16e6.txt.gz"
    bpe_path = sam3_package_dir / "assets" / bpe_name
    if not bpe_path.is_file():
        bpe_path = sam3_package_dir.parent / "assets" / bpe_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bbox] SAM3 device={device}", flush=True)

    # SAM3 currently precomputes positional encodings on a hard-coded CUDA
    # device. Disable only that optional precompute on CPU; runtime tensors
    # already follow the input device. This avoids modifying the SAM3 source.
    original_create_position_encoding = None
    original_get_coords = None
    if device == "cpu":
        original_create_position_encoding = model_builder._create_position_encoding
        original_get_coords = TransformerDecoder._get_coords

        def create_cpu_position_encoding(precompute_resolution=None):
            return original_create_position_encoding(precompute_resolution=None)

        def get_cpu_coords(height, width, coordinate_device):
            if coordinate_device == "cuda":
                coordinate_device = "cpu"
            return original_get_coords(height, width, coordinate_device)

        model_builder._create_position_encoding = create_cpu_position_encoding
        TransformerDecoder._get_coords = staticmethod(get_cpu_coords)

    try:
        model = model_builder.build_sam3_image_model(
            device=device,
            bpe_path=str(bpe_path) if bpe_path.exists() else None,
        )
    finally:
        if original_create_position_encoding is not None:
            model_builder._create_position_encoding = original_create_position_encoding
        if original_get_coords is not None:
            TransformerDecoder._get_coords = staticmethod(original_get_coords)
    processor = Sam3Processor(model, device=device)
    state = processor.set_image(image)

    detections: list[dict[str, object]] = []
    try:
        for prompt in prompts:
            result = processor.set_text_prompt(state=state, prompt=prompt)
            boxes = result["boxes"]
            scores = result["scores"]
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()

            count = 0
            for box, score in zip(boxes, scores):
                score = float(score)
                if score < min_score:
                    continue
                x1, y1, x2, y2 = map(int, box[:4])
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": score,
                        "prompt": prompt,
                    }
                )
                count += 1
            print(f"[bbox] prompt={prompt!r} detections={count}", flush=True)
    finally:
        del processor, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detections = merge_overlapping_boxes(detections, merge_threshold)
    shape_prompts = (
        "rectangle", "trapezoid", "square", "circle", "triangle",
        "polygon", "path", "ellipse", "line", "arrow",
    )
    icon_index = 1
    shape_index = 1
    for box in detections:
        prompt = str(box.get("prompt", "")).lower()
        is_shape = any(shape in prompt for shape in shape_prompts)
        box["is_shape"] = is_shape
        if is_shape:
            shape_type = next(
                (shape for shape in shape_prompts if shape in prompt),
                "shape",
            )
            box["label"] = f"<Shape_{shape_type}_{shape_index:02d}>"
            shape_index += 1
        else:
            box["label"] = f"<AF>{icon_index:02d}"
            icon_index += 1
    return detections


def calculate_overlap_ratio(
    box1: dict[str, object],
    box2: dict[str, object],
) -> float:
    left = max(float(box1["x1"]), float(box2["x1"]))
    top = max(float(box1["y1"]), float(box2["y1"]))
    right = min(float(box1["x2"]), float(box2["x2"]))
    bottom = min(float(box1["y2"]), float(box2["y2"]))
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area1 = (float(box1["x2"]) - float(box1["x1"])) * (
        float(box1["y2"]) - float(box1["y1"])
    )
    area2 = (float(box2["x2"]) - float(box2["x1"])) * (
        float(box2["y2"]) - float(box2["y1"])
    )
    smaller_area = min(area1, area2)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def calculate_area_ratio(
    box1: dict[str, object],
    box2: dict[str, object],
) -> float:
    area1 = (float(box1["x2"]) - float(box1["x1"])) * (
        float(box1["y2"]) - float(box1["y1"])
    )
    area2 = (float(box2["x2"]) - float(box2["x1"])) * (
        float(box2["y2"]) - float(box2["y1"])
    )
    if area1 <= 0 or area2 <= 0:
        return 0.0
    return min(area1, area2) / max(area1, area2)


def merge_two_boxes(
    box1: dict[str, object],
    box2: dict[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = {
        "x1": min(float(box1["x1"]), float(box2["x1"])),
        "y1": min(float(box1["y1"]), float(box2["y1"])),
        "x2": max(float(box1["x2"]), float(box2["x2"])),
        "y2": max(float(box1["y2"]), float(box2["y2"])),
        "score": max(float(box1.get("score", 0)), float(box2.get("score", 0))),
    }
    prompt1 = str(box1.get("prompt", ""))
    prompt2 = str(box2.get("prompt", ""))
    if prompt1 and prompt2:
        merged["prompt"] = (
            prompt1
            if float(box1.get("score", 0)) >= float(box2.get("score", 0))
            else prompt2
        )
    elif prompt1 or prompt2:
        merged["prompt"] = prompt1 or prompt2
    return merged


def merge_overlapping_boxes(
    boxes: list[dict[str, object]],
    overlap_threshold: float = MERGE_THRESHOLD,
) -> list[dict[str, object]]:
    if overlap_threshold <= 0 or len(boxes) <= 1:
        return [box.copy() for box in boxes]

    working = [box.copy() for box in boxes]
    merged = True
    while merged:
        merged = False
        for index, left in enumerate(working):
            for other_index in range(index + 1, len(working)):
                right = working[other_index]
                if (
                    calculate_overlap_ratio(left, right) >= overlap_threshold
                    and calculate_area_ratio(left, right) >= MIN_AREA_RATIO
                ):
                    combined = merge_two_boxes(left, right)
                    working = [
                        box
                        for box_index, box in enumerate(working)
                        if box_index not in {index, other_index}
                    ]
                    working.append(combined)
                    merged = True
                    break
            if merged:
                break

    result: list[dict[str, object]] = []
    for index, box in enumerate(working):
        output: dict[str, object] = {
            "id": index,
            "label": f"<AF>{index + 1:02d}",
            "x1": box["x1"],
            "y1": box["y1"],
            "x2": box["x2"],
            "y2": box["y2"],
            "score": box.get("score", 0),
        }
        if box.get("prompt"):
            output["prompt"] = box["prompt"]
        result.append(output)
    return result


def generate_boxlib(
    input_png: Path,
    output_dir: Path,
    sam_prompt: str = SAM_PROMPT,
    min_score: float = MIN_SCORE,
    merge_threshold: float = MERGE_THRESHOLD,
) -> dict[str, object]:
    with Image.open(input_png) as image:
        canvas_width, canvas_height = image.size

    boxes = run_sam3_inference(input_png, sam_prompt, min_score, merge_threshold)
    return {
        "image_size": {"width": canvas_width, "height": canvas_height},
        "prompts_used": [p.strip() for p in sam_prompt.split(",") if p.strip()],
        "boxes": boxes,
        "no_icon_mode": not boxes,
    }


def boxlib_to_bbox(
    boxlib_data: dict[str, object],
    bbox_path: Path,
) -> Path:
    image_size = boxlib_data.get("image_size", {})
    canvas_width = float(image_size.get("width", 0))
    canvas_height = float(image_size.get("height", 0))
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Invalid image_size in generated boxlib data")

    objects: list[dict[str, object]] = []
    for box in boxlib_data.get("boxes", []):
        x = float(box["x1"])
        y = float(box["y1"])
        width = float(box["x2"]) - x
        height = float(box["y2"]) - y
        if width <= 0 or height <= 0:
            continue
        obj: dict[str, object] = {
            "type": "rect" if box.get("is_shape", False) else "image",
            "x": round(x, 4),
            "y": round(y, 4),
            "width": round(width, 4),
            "height": round(height, 4),
        }
        for key in ("label", "prompt"):
            if box.get(key):
                obj[key] = box[key]
        objects.append(obj)

    result = {
        "canvas": {"width": canvas_width, "height": canvas_height},
        "objects": objects,
    }
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    bbox_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bbox] saved {bbox_path} objects={len(objects)}", flush=True)
    return bbox_path


def generate_bbox_json(
    input_png: Path,
    output_dir: Path,
    *,
    sam_prompt: str = SAM_PROMPT,
    min_score: float = MIN_SCORE,
    merge_threshold: float = MERGE_THRESHOLD,
) -> Path:
    """Generate and return ``bbox.json`` for ``input_png``."""
    output_dir = output_dir.expanduser().resolve()
    boxlib_data = generate_boxlib(
        input_png,
        output_dir,
        sam_prompt,
        min_score,
        merge_threshold,
    )
    return boxlib_to_bbox(boxlib_data, output_dir / "bbox.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bbox.json from an image with SAM3")
    parser.add_argument("--input_png", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--sam_prompt", default=SAM_PROMPT)
    parser.add_argument("--min_score", type=float, default=MIN_SCORE)
    parser.add_argument("--merge_threshold", type=float, default=MERGE_THRESHOLD)
    args = parser.parse_args()
    generate_bbox_json(
        args.input_png,
        args.output_dir,
        sam_prompt=args.sam_prompt,
        min_score=args.min_score,
        merge_threshold=args.merge_threshold,
    )


if __name__ == "__main__":
    main()

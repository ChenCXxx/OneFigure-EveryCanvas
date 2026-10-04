"""Image-analysis tools used by the Space benchmark."""
from .blank_area import visualize_and_analyze_blank_area
from .ocr import analyze_image_rotation
from .similarity import analyze_visual_similarity, calculate_lpips

__all__ = [
    "visualize_and_analyze_blank_area",
    "analyze_image_rotation",
    "analyze_visual_similarity",
    "calculate_lpips",
    "calculate_ssim",
]

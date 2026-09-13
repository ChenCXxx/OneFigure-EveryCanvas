"""SSIM and LPIPS checks between reference and candidate images."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

# ------------- Constants -------------------
MAX_SIDE = 512
LPIPS_THRESHOLD = 0.5
SSIM_THRESHOLD = 0.95  # SSIM is deprecated
# -------------------------------------------


def _aligned_pair(reference: Path, candidate: Path, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    """
    resize reference to candidate size
    if both images are too large, resize both to max_side
    """
    reference_image = Image.open(reference).convert("RGB")
    candidate_image = Image.open(candidate).convert("RGB")
    
    reference_image = reference_image.resize(
        candidate_image.size,
        Image.Resampling.LANCZOS,
    )
    
    scale = min(1.0, max_side / max(candidate_image.size))
    
    if scale < 1.0:  # image too large, resize both
        compare_size = (max(1, round(candidate_image.width * scale)), max(1, round(candidate_image.height * scale)))
        reference_image = reference_image.resize(compare_size, Image.Resampling.LANCZOS)
        candidate_image = candidate_image.resize(compare_size, Image.Resampling.LANCZOS)
    
    reference_array = np.asarray(reference_image)
    candidate_array = np.asarray(candidate_image)
    return reference_array, candidate_array

# ---------- SSIM ---------- (deprecated)
# def calculate_ssim(
#     reference: Path,
#     candidate: Path,
#     max_side: int = MAX_SIDE,
# ) -> float:
#     try:
#         from skimage.metrics import structural_similarity
#     except ImportError as exc:
#         raise RuntimeError("SSIM requires scikit-image") from exc
#     ref, cand = _aligned_pair(reference, candidate, max_side)
    
#     return float(structural_similarity(ref, cand, channel_axis=-1, data_range=255))

# ---------- LPIPS ----------
@lru_cache(maxsize=1)
def _lpips_model():
    """Create the LPIPS model once per process and reuse it."""
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS requires torch and lpips") from exc

    return lpips.LPIPS(net="alex").eval()


def calculate_lpips(
    reference: Path,
    candidate: Path,
    max_side: int = MAX_SIDE,
) -> float:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("LPIPS requires torch and lpips") from exc
    
    ref, cand = _aligned_pair(reference, candidate, max_side)
    def tensor(image: np.ndarray):
        return torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    
    model = _lpips_model()
    with torch.no_grad():
        return float(model(tensor(ref), tensor(cand)).item())


def analyze_visual_similarity(reference: Path, candidate: Path) -> dict[str, Any]:
    """
    Analyze the visual similarity between two images.
    Result structure:
    {
        "lpips": float,
        "lpips_threshold": float, (default = 0.5)
        "is_likely_direct_resize": bool,
    }
    """
    result: dict[str, Any] = {}
    # SSIM(deprecated)
    # try:
    #     result["ssim"] = calculate_ssim(reference, candidate)
    # except RuntimeError as exc:
    #     result["ssim_error"] = str(exc)
    
    # LPIPS
    try:
        result["lpips"] = calculate_lpips(reference, candidate)
    except RuntimeError as exc:
        result["lpips_error"] = str(exc)
    
    # The legacy benchmark uses LPIPS only for this heuristic.
    if "lpips" in result:
        result["lpips_threshold"] = LPIPS_THRESHOLD
        result["is_likely_direct_resize"] = (
            result["lpips"] < LPIPS_THRESHOLD
        )
    return result

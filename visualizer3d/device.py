from typing import Tuple

import open3d as o3d


def _cuda_available() -> bool:
    try:
        return bool(o3d.core.cuda.is_available())
    except Exception:
        return False


def resolve_device(requested: str) -> Tuple[str, bool, str]:
    if requested == "cpu":
        return "cpu", False, "Device: CPU"
    if requested == "cuda":
        if _cuda_available():
            return "cuda", True, "Device: CUDA"
        return "cpu", False, "CUDA not available, fallback to CPU"
    if _cuda_available():
        return "cuda", True, "Device: CUDA (auto)"
    return "cpu", False, "CUDA not available, using CPU"

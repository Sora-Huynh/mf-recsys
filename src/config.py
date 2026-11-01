"""Đọc cấu hình YAML và đặt seed.

Mục đích: gom mọi tham số về một chỗ để kết quả tái lập được. Không thuộc chương
nào của môn; đây là hạ tầng dùng chung.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Đọc file YAML và đổi các đường dẫn trong `paths` thành tuyệt đối.

    Args:
        path: Đường dẫn file YAML; None dùng `configs/default.yaml`.

    Returns:
        Dict cấu hình; `cfg["paths"]` chứa các `Path` tuyệt đối.

    Raises:
        ValueError: Nếu ba tỉ lệ chia train/val/test không có tổng bằng 1.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    total = sum(cfg["split"].values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Tổng tỉ lệ chia phải bằng 1, nhận được {total}")

    cfg["paths"] = {name: PROJECT_ROOT / rel for name, rel in cfg["paths"].items()}
    return cfg


def set_seed(seed: int) -> np.random.Generator:
    """Đặt seed cho `random`, `numpy` và trả về Generator riêng.

    Args:
        seed: Giá trị seed.

    Returns:
        `np.random.Generator`; code mới nên dùng đối tượng này thay vì trạng thái
        toàn cục. Lưu ý: hàm numba có RNG riêng, phải seed bên trong hàm numba.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)

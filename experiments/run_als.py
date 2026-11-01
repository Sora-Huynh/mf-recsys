"""Thí nghiệm ALS: đường hội tụ và số điều kiện theo λ.

Mục đích: kiểm tra hàm mục tiêu giảm đều và cho thấy λ cải thiện độ ổn định của hệ.
Liên quan Ch.2 và Ch.3.

Chạy: python -m experiments.run_als
"""

from __future__ import annotations

import argparse
from typing import Any

from src.config import load_config, set_seed


def run(cfg: dict[str, Any]) -> None:
    """Fit ALS, lưu nhật ký hội tụ, bảng κ theo λ và mô hình.

    Args:
        cfg: Cấu hình từ `load_config`.
    """
    # TODO: fit ALSRecommender; log.save; condition_vs_lambda trên Q đã học; lưu hình.
    raise NotImplementedError


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    set_seed(cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()

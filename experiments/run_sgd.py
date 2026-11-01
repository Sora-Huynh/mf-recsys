"""Thí nghiệm SGD: so sánh SGD, Momentum, Adam ở cùng k và λ.

Mục đích: chọn một optimizer đại diện để đưa vào so sánh tổng. Liên quan Ch.5 và Ch.6.

Chạy: python -m experiments.run_sgd
"""

from __future__ import annotations

import argparse
from typing import Any

from src.config import load_config, set_seed


def run(cfg: dict[str, Any]) -> None:
    """Chạy ba optimizer, lưu nhật ký hội tụ và mô hình tốt nhất theo val.

    Args:
        cfg: Cấu hình từ `load_config`.
    """
    # TODO: lặp ba optimizer với lr trong cfg["sgd"]["lr"]; lưu log; vẽ đường hội tụ.
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

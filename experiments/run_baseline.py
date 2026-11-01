"""Thí nghiệm baseline: RMSE/MAE của trung bình toàn cục và bias user/phim.

Mục đích: đặt mốc mà các mô hình phân rã phải vượt. Liên quan Ch.3.

Chạy: python -m experiments.run_baseline
"""

from __future__ import annotations

import argparse
from typing import Any

from src.config import load_config, set_seed


def run(cfg: dict[str, Any]) -> None:
    """Fit hai baseline, lưu bảng vào `results/tables/baseline.csv`.

    Args:
        cfg: Cấu hình từ `load_config`.
    """
    # TODO: load_processed, fit GlobalMean và BiasBaseline, tính RMSE/MAE trên val và test.
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

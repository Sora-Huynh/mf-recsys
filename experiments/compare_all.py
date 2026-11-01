"""So sánh tổng: bảng RMSE test và biểu đồ hội tụ.

Mục đích: tạo các hình/bảng chính của báo cáo. Chỉ so các mô hình đã chọn tham số
theo val: hai baseline, SVD, ALS và một optimizer SGD tốt nhất. Liên quan Ch.3–Ch.6.

Chạy: python -m experiments.compare_all
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src.config import load_config, set_seed


def evaluate_on_test(cfg: dict[str, Any]) -> pd.DataFrame:
    """Tính RMSE, MAE trên test của các mô hình đã lưu.

    Args:
        cfg: Cấu hình từ `load_config`.

    Returns:
        DataFrame cột `model`, `test_rmse`, `test_mae`, `fit_time_s`.
    """
    # TODO: đây là nơi DUY NHẤT dùng tập test.
    raise NotImplementedError


def run(cfg: dict[str, Any]) -> None:
    """Lưu bảng RMSE, biểu đồ cột và hai biểu đồ hội tụ (theo vòng lặp, theo thời gian).

    Args:
        cfg: Cấu hình từ `load_config`.
    """
    # TODO: evaluate_on_test, plot_bars, plot_convergence với x là epoch và elapsed_s.
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

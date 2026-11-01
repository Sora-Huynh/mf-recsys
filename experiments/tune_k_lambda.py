"""Quét k và λ trên tập val.

Mục đích: chọn tham số bằng val, không bằng test, để RMSE test cuối cùng không bị lạc
quan. Quét gọn: lưới (k, λ) cho ALS; chỉ k cho SVD. SGD dùng k tốt nhất của ALS và chỉ
quét reg của chính nó quanh giá trị mặc định. Liên quan Ch.3 và Ch.4.

Chạy: python -m experiments.tune_k_lambda
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src.config import load_config, set_seed


def sweep_als(cfg: dict[str, Any]) -> pd.DataFrame:
    """Quét lưới (k, λ) cho ALS.

    Args:
        cfg: Cấu hình từ `load_config`.

    Returns:
        DataFrame cột `k`, `reg`, `val_rmse`, `fit_time_s`.
    """
    # TODO: lặp cfg["tune"]["k_grid"] × cfg["tune"]["reg_grid"].
    raise NotImplementedError


def sweep_svd(cfg: dict[str, Any]) -> pd.DataFrame:
    """Quét k cho SVD.

    Args:
        cfg: Cấu hình từ `load_config`.

    Returns:
        DataFrame cột `k`, `val_rmse`, `fit_time_s`.
    """
    # TODO: lặp cfg["tune"]["k_grid"]; chỉ cần svds một lần với k lớn nhất rồi cắt bớt.
    raise NotImplementedError


def run(cfg: dict[str, Any]) -> None:
    """Chạy các lần quét, lưu bảng và hình ảnh hưởng của k, λ.

    Args:
        cfg: Cấu hình từ `load_config`.
    """
    # TODO: gọi sweep_als, sweep_svd; plot_sweep; in tham số tốt nhất để ghi vào config.
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

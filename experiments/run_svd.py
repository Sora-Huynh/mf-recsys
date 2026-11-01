"""Thí nghiệm SVD: RMSE, thời gian, và phổ giá trị kỳ dị.

Mục đích: phổ σ cho thấy dữ liệu xấp xỉ hạng thấp, là cơ sở của phân rã ma trận.
Liên quan Ch.4.

Chạy: python -m experiments.run_svd
"""

from __future__ import annotations

import argparse
from typing import Any

from src.config import load_config, set_seed


def run(cfg: dict[str, Any]) -> None:
    """Fit SVD, lưu bảng, hình phổ σ và mô hình.

    Args:
        cfg: Cấu hình từ `load_config`.
    """
    # TODO: fit BiasBaseline rồi TruncatedSVDRecommender; đo thời gian; lưu kết quả.
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

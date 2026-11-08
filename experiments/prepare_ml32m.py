"""Tiền xử lý MovieLens-32M bằng đúng pipeline của ML-1M.

Mục đích: `src.data.load.load_ratings` chỉ đọc `ratings.dat`, còn ML-32M là `ratings.csv`
với tên cột khác. File này chỉ thay bước đọc file; phần chia tập, ánh xạ ID và tạo CSR vẫn
dùng `src.data.preprocess`, nhờ đó không phải sửa code tiền xử lý chung của nhóm.

Liên quan Ch.2 (chuẩn bị dữ liệu cho ma trận thưa).

Chạy: python -m experiments.prepare_ml32m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data import preprocess as pipeline

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ml32m.yaml"


def load_ratings_csv(raw_dir: Path) -> pd.DataFrame:
    """Đọc `ratings.csv` của ML-32M về cùng dạng với `load_ratings`.

    Args:
        raw_dir: Thư mục chứa `ratings.csv`.

    Returns:
        DataFrame cột `user_id`, `movie_id`, `rating`, `timestamp`.
    """
    ratings = pd.read_csv(
        raw_dir / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    return ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    cfg = load_config(parser.parse_args().config)

    # `preprocess` gọi `load_ratings` theo tên trong module của nó, nên thay đúng tên đó.
    pipeline.load_ratings = load_ratings_csv
    data = pipeline.preprocess(cfg)

    total = len(data.train) + len(data.val) + len(data.test)
    print(f"Tổng số đánh giá: {total}")
    print(f"Train/val/test: {len(data.train)}/{len(data.val)}/{len(data.test)}")
    print(f"Kích thước ma trận: {data.shape}")


if __name__ == "__main__":
    main()

"""Ánh xạ ID sang chỉ số liên tục và chia train/val/test.

Mục đích: đưa dữ liệu về dạng mà các mô hình dùng chung.
- ID phim có khoảng trống (tới 3952 nhưng chỉ khoảng 3700 phim có đánh giá), dùng
  trực tiếp sẽ tạo cột rỗng, nên phải ánh xạ về 0..n-1.
- Chia ngẫu nhiên theo từng đánh giá với seed cố định: mọi mô hình so trên cùng tập.
- Phim/user không có trong train không thể học được vector ẩn, nên đánh giá của họ
  được chuyển về train. Đây không phải bài toán cold-start của đề tài.

Liên quan Ch.2 (chuẩn bị dữ liệu cho ma trận thưa).

Chạy: python -m src.data.preprocess
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config


@dataclass
class IdMaps:
    """Ánh xạ hai chiều giữa ID gốc và chỉ số liên tục.

    Attributes:
        user_ids: `user_ids[idx]` là UserID gốc của chỉ số `idx`.
        movie_ids: `movie_ids[idx]` là MovieID gốc của chỉ số `idx`.
        user_index: Dict UserID gốc -> chỉ số.
        movie_index: Dict MovieID gốc -> chỉ số.
    """

    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_index: dict[int, int]
    movie_index: dict[int, int]


@dataclass
class SplitData:
    """Kết quả tiền xử lý.

    Attributes:
        train: DataFrame cột `user_idx`, `item_idx`, `rating`.
        val: Như `train`.
        test: Như `train`.
        maps: Ánh xạ ID.
        shape: Kích thước ma trận (số user, số phim).
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    maps: IdMaps
    shape: tuple[int, int]


def split_ratings(
    ratings: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia ngẫu nhiên theo từng đánh giá.

    Args:
        ratings: DataFrame từ `load_ratings`.
        train_frac: Tỉ lệ train.
        val_frac: Tỉ lệ val; phần còn lại là test.
        seed: Seed của phép hoán vị.

    Returns:
        Bộ ba (train, val, test).
    """
    # TODO: hoán vị bằng np.random.default_rng(seed).permutation rồi cắt theo tỉ lệ.
    raise NotImplementedError


def move_cold_start_to_train(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chuyển đánh giá của user/phim không xuất hiện trong train về train.

    Args:
        train: Tập train theo ID gốc.
        val: Tập val theo ID gốc.
        test: Tập test theo ID gốc.

    Returns:
        Bộ ba mới, trong đó mọi user/phim ở val và test đều có trong train.
    """
    # TODO: lọc bằng isin; ghi lại số dòng đã chuyển để đưa vào báo cáo.
    raise NotImplementedError


def build_id_maps(train: pd.DataFrame) -> IdMaps:
    """Tạo ánh xạ ID từ tập train.

    Args:
        train: Tập train sau `move_cold_start_to_train`.

    Returns:
        `IdMaps`; chỉ số được sắp theo ID gốc tăng dần.
    """
    # TODO: dùng np.unique để lấy ID, dict comprehension để tạo ánh xạ ngược.
    raise NotImplementedError


def encode_ids(ratings: pd.DataFrame, maps: IdMaps) -> pd.DataFrame:
    """Đổi cột ID gốc sang chỉ số liên tục.

    Args:
        ratings: DataFrame có `user_id`, `movie_id`, `rating`.
        maps: Ánh xạ ID.

    Returns:
        DataFrame cột `user_idx`, `item_idx`, `rating`.
    """
    # TODO: vector hóa bằng Series.map, không lặp từng dòng.
    raise NotImplementedError


def preprocess(cfg: dict[str, Any]) -> SplitData:
    """Chạy toàn bộ tiền xử lý rồi lưu vào `data/processed/`.

    Args:
        cfg: Cấu hình từ `load_config`.

    Returns:
        `SplitData`.
    """
    # TODO: load -> split -> move_cold_start -> maps -> encode -> lưu parquet/npz + CSR.
    raise NotImplementedError


def load_processed(cfg: dict[str, Any]) -> SplitData:
    """Đọc dữ liệu đã tiền xử lý.

    Args:
        cfg: Cấu hình từ `load_config`.

    Returns:
        `SplitData`, giống hệt kết quả `preprocess`.
    """
    # TODO: đọc các file do preprocess ghi ra.
    raise NotImplementedError


if __name__ == "__main__":
    preprocess(load_config())

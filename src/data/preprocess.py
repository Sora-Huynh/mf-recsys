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
from src.data.load import load_ratings
from src.data.sparse import *


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

    if train_frac + val_frac > 1.0:
        raise ValueError(f"Tổng train_frac ({train_frac}) và val_frac ({val_frac}) không được vượt quá 1.0!")
    
    shuffled_ratings = ratings.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    row = len(ratings)
    train_row, val_row = int(row * train_frac), int(row * val_frac)

    train_df = shuffled_ratings.iloc[:train_row]
    val_df = shuffled_ratings.iloc[train_row:train_row + val_row]
    test_df = shuffled_ratings.iloc[train_row + val_row:]

    return train_df, val_df, test_df



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

    user_col = 'user_id'
    item_col = 'movie_id'

    val_mask =  val[user_col].isin(train[user_col]) & val[item_col].isin(train[item_col])
    test_mask = test[user_col].isin(train[user_col]) & test[item_col].isin(train[item_col])

    val_warm = val[val_mask]
    val_cold = val[~val_mask]

    test_warm = test[test_mask]
    test_cold = test[~test_mask]

    num_moved_val = len(val_cold)
    num_moved_test = len(test_cold)

    print(f"Số dòng được chuyển từ val về train là: {num_moved_val}")
    print(f"Số dòng được chuyển từ test về train là: {num_moved_test}")

    new_train = pd.concat([train, val_cold, test_cold], ignore_index=True)
    new_val = val_warm.reset_index(drop=True)
    new_test = test_warm.reset_index(drop=True)
    
    return new_train, new_val, new_test


def build_id_maps(train: pd.DataFrame) -> IdMaps:
    """Tạo ánh xạ ID từ tập train.

    Args:
        train: Tập train sau `move_cold_start_to_train`.

    Returns:
        `IdMaps`; chỉ số được sắp theo ID gốc tăng dần.
    """
    user_col = 'user_id'
    item_col = 'movie_id'

    unique_users = np.unique(train[user_col])
    unique_items = np.unique(train[item_col])

    user_index = {int(uid): idx for idx, uid in enumerate(unique_users)}
    movie_index = {int(iid): idx for idx, iid in enumerate(unique_items)}

    return IdMaps(
        user_ids=unique_users,
        movie_ids=unique_items,
        user_index=user_index,
        movie_index=movie_index
    )


def encode_ids(ratings: pd.DataFrame, maps: IdMaps) -> pd.DataFrame:
    """Đổi cột ID gốc sang chỉ số liên tục.

    Args:
        ratings: DataFrame có `user_id`, `movie_id`, `rating`.
        maps: Ánh xạ ID.

    Returns:
        DataFrame cột `user_idx`, `item_idx`, `rating`.
    """
    encoded_df = ratings.copy()

    encoded_df['user_idx'] = encoded_df['user_id'].map(maps.user_index)
    encoded_df['item_idx'] = encoded_df['movie_id'].map(maps.movie_index)

    if encoded_df[['user_idx', 'item_idx']].isna().any(axis=None):
        raise ValueError("Có user/phim không nằm trong ánh xạ ID")

    result_df = encoded_df[['user_idx', 'item_idx', 'rating']].astype(
        {'user_idx': 'int32', 'item_idx': 'int32'}
    )

    return result_df


def preprocess(cfg: dict[str, Any]) -> SplitData:
    """Chạy toàn bộ tiền xử lý rồi lưu vào `data/processed/`.

    Args:
        cfg: Cấu hình từ `load_config`.

    Returns:
        `SplitData`.
    """
    ratings = load_ratings(cfg["paths"]["raw_dir"])

    train_frac = cfg["split"]["train"]
    val_frac = cfg["split"]["val"]
    seed = cfg["seed"]
    
    train_df, val_df, test_df = split_ratings(ratings, train_frac, val_frac, seed)

    train_df, val_df, test_df = move_cold_start_to_train(train_df, val_df, test_df)

    maps = build_id_maps(train_df)

    train_encoded = encode_ids(train_df, maps)
    val_encoded = encode_ids(val_df, maps)
    test_encoded = encode_ids(test_df, maps)

    processed_dir = cfg["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_encoded.to_parquet(processed_dir / "train.parquet", index=False)
    val_encoded.to_parquet(processed_dir / "val.parquet", index=False)
    test_encoded.to_parquet(processed_dir / "test.parquet", index=False)
    np.savez(processed_dir / "id_maps.npz", user_ids=maps.user_ids, movie_ids=maps.movie_ids)

    shape = (len(maps.user_ids), len(maps.movie_ids))

    train_csr = build_csr(
        users= train_encoded['user_idx'].to_numpy(),
        items= train_encoded['item_idx'].to_numpy(),
        values= train_encoded['rating'].to_numpy(),
        shape=shape
    )

    save_csr(processed_dir / "train_csr.npz", train_csr)

    dense_bytes = dense_memory_bytes(shape)
    csr_bytes = csr_memory_bytes(train_csr)

    print(f"Dense memory: {dense_bytes} bytes")
    print(f"CSR memory: {csr_bytes} bytes")
    print(f"Ratio (CSR/Dense): {csr_bytes / dense_bytes:.4f}")

    return SplitData(
        train= train_encoded,
        val= val_encoded,
        test= test_encoded,
        maps= maps,
        shape= shape,
    )




def load_processed(cfg: dict[str, Any]) -> SplitData:
    """Đọc dữ liệu đã tiền xử lý.

    Args:
        cfg: Cấu hình từ `load_config`.

    Returns:
        `SplitData`, giống hệt kết quả `preprocess`.
    """
    processed_dir = cfg["paths"]["processed_dir"]

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    data = np.load(processed_dir / "id_maps.npz")
    user_ids = data["user_ids"]
    movie_ids = data["movie_ids"]

    maps = IdMaps(
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_index={int(uid): idx for idx, uid in enumerate(user_ids)},
        movie_index={int(mid): idx for idx, mid in enumerate(movie_ids)},
    )

    return SplitData(
        train=train_df,
        val=val_df,
        test=test_df,
        maps=maps,
        shape=(len(user_ids), len(movie_ids)),
    )


if __name__ == "__main__":
    preprocess(load_config())

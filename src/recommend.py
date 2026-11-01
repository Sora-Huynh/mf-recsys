"""Demo: nhập UserID, in top-10 phim gợi ý.

Mục đích: cho thấy mô hình đưa ra được gợi ý cụ thể, không chỉ một con số RMSE. Dùng
mô hình đã lưu, chỉ đổi giữa ID gốc và chỉ số liên tục. Liên quan Ch.4 và Ch.6 qua
mô hình đã huấn luyện.

Chạy: python -m src.recommend --user-id 42 --model als
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.data.preprocess import IdMaps
from src.models.base import Recommender


def recommend_for_user(
    model: Recommender,
    user_id: int,
    maps: IdMaps,
    movies: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Tạo bảng gợi ý cho một user.

    Args:
        model: Mô hình đã huấn luyện.
        user_id: UserID gốc.
        maps: Ánh xạ ID.
        movies: DataFrame từ `load_movies`.
        n: Số phim gợi ý.

    Returns:
        DataFrame cột `rank`, `title`, `genres`, `predicted_rating`.

    Raises:
        KeyError: Nếu UserID không tồn tại.
    """
    # TODO: maps.user_index -> model.recommend -> maps.movie_ids -> nối với movies.
    raise NotImplementedError


def main() -> None:
    """Đọc tham số dòng lệnh và in bảng gợi ý."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--model", choices=["svd", "als", "sgd"], default="als")
    parser.add_argument("--config", default=None)
    parser.parse_args()
    # TODO: load_config, Recommender.load, load_processed, recommend_for_user, in bảng.
    raise NotImplementedError


if __name__ == "__main__":
    main()

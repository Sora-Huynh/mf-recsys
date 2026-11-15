"""Demo: nhập UserID, in top-10 phim gợi ý.

Mục đích: cho thấy mô hình đưa ra được gợi ý cụ thể, không chỉ một con số RMSE. Dùng
mô hình đã lưu, chỉ đổi giữa ID gốc và chỉ số liên tục. Liên quan Ch.4 và Ch.6 qua
mô hình đã huấn luyện.

Chạy: python -m src.recommend --user-id 42 --model als
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.load import load_movies
from src.data.preprocess import IdMaps, load_processed
from src.models.base import Recommender


def recommend_for_user(
    model: Recommender,
    user_id: int,
    maps: IdMaps,
    movies: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Tạo bảng gợi ý cho một user.

    Phim user đã đánh giá trong train được loại sẵn bởi `Recommender.recommend`.

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
    if user_id not in maps.user_index:
        raise KeyError(f"UserID {user_id} không tồn tại trong dữ liệu.")
    user_idx = maps.user_index[user_id]

    item_indices, scores = model.recommend(user_idx, n=n)
    movie_ids = [maps.movie_ids[i] for i in item_indices]

    df_rec = pd.DataFrame({"movie_id": movie_ids, "predicted_rating": np.round(scores, 3)})
    merged = df_rec.merge(movies[["movie_id", "title", "genres"]], on="movie_id", how="left")
    merged.insert(0, "rank", range(1, len(merged) + 1))
    return merged[["rank", "title", "genres", "predicted_rating"]]


def main(argv: list[str] | None = None) -> None:
    """Đọc tham số dòng lệnh và in bảng gợi ý.

    Args:
        argv: Danh sách tham số; None thì lấy từ dòng lệnh.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True, help="UserID gốc của người dùng")
    parser.add_argument(
        "--model", choices=["baseline", "svd", "als", "sgd"], default="als", help="Tên mô hình"
    )
    parser.add_argument("--top-n", type=int, default=None, help="Số phim gợi ý (mặc định theo config)")
    parser.add_argument("--config", default=None, help="Đường dẫn file config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    top_n = args.top_n or cfg["recommend"]["top_n"]

    # Không tự huấn luyện thay thế: nếu thiếu mô hình thì báo rõ, tránh in nhãn sai
    model_path = Path(cfg["paths"]["models_dir"]) / f"{args.model}.pkl"
    if not model_path.exists():
        sys.exit(
            f"Không tìm thấy mô hình '{args.model}' tại {model_path}. "
            f"Hãy chạy: python -m experiments.run_{args.model}"
        )
    model = Recommender.load(model_path)

    maps = load_processed(cfg).maps
    movies = load_movies(Path(cfg["paths"]["raw_dir"]))

    try:
        recommendations = recommend_for_user(model, args.user_id, maps, movies, n=top_n)
    except KeyError as err:
        sys.exit(f"Lỗi: {err.args[0]}")

    print(f"\n=== TOP {top_n} GỢI Ý CHO USER {args.user_id} (mô hình: {args.model}) ===")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
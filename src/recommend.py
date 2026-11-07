"""Demo: nhập UserID, in top-10 phim gợi ý.

Mục đích: cho thấy mô hình đưa ra được gợi ý cụ thể, không chỉ một con số RMSE. Dùng
mô hình đã lưu, chỉ đổi giữa ID gốc và chỉ số liên tục. Liên quan Ch.4 và Ch.6 qua
mô hình đã huấn luyện.

Chạy: python -m src.recommend --user-id 42 --model als
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.load import load_movies
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
    # 1. Ánh xạ UserID gốc sang chỉ số user_index liên tục
    user_map = getattr(maps, "user_index", getattr(maps, "user2idx", None))
    if user_map is None or user_id not in user_map:
        raise KeyError(f"UserID {user_id} không tồn tại trong dữ liệu.")
    
    user_idx = user_map[user_id]

    # 2. Gọi hàm recommend của mô hình để lấy danh sách item_idx và dự đoán điểm
    rec_result = model.recommend(user_idx, n=n)
    if isinstance(rec_result, tuple):
        item_indices, scores = rec_result
    else:
        item_indices = rec_result
        scores = model.predict(np.full_like(item_indices, user_idx), item_indices)

    # 3. Ánh xạ item_index về MovieID gốc
    movie_map = getattr(maps, "movie_ids", getattr(maps, "idx2item", None))
    if movie_map is not None:
        movie_ids = [movie_map[i] for i in item_indices]
    else:
        movie_ids = item_indices

    # 4. Tạo DataFrame kết quả và nối thông tin tiêu đề, thể loại phim
    df_rec = pd.DataFrame({
        "movie_id": movie_ids,
        "predicted_rating": np.round(scores, 3)
    })

    id_col = "movie_id" if "movie_id" in movies.columns else ("MovieID" if "MovieID" in movies.columns else movies.columns[0])
    merged = df_rec.merge(movies, left_on="movie_id", right_on=id_col, how="left")

    merged["rank"] = range(1, len(merged) + 1)

    title_col = "title" if "title" in merged.columns else ("Title" if "Title" in merged.columns else "title")
    genres_col = "genres" if "genres" in merged.columns else ("Genres" if "Genres" in merged.columns else "genres")

    merged = merged.rename(columns={title_col: "title", genres_col: "genres"})
    
    return merged[["rank", "title", "genres", "predicted_rating"]]


def main() -> None:
    """Đọc tham số dòng lệnh và in bảng gợi ý."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True, help="UserID gốc của người dùng")
    parser.add_argument("--model", choices=["baseline", "svd", "als", "sgd"], default="als", help="Tên mô hình")
    parser.add_argument("--config", default=None, help="Đường dẫn file config")
    args = parser.parse_args()

    # 1. Load cấu hình
    cfg = load_config(args.config)

    # 2. Load ánh xạ ID và danh sách phim
    maps = IdMaps.load("data/processed/id_maps.pkl") if hasattr(IdMaps, "load") else IdMaps.load()
    movies = load_movies()

    # 3. Load mô hình đã huấn luyện
    model_path = f"results/models/{args.model}.pkl"
    if not os.path.exists(model_path):
        # Nếu chưa có file mô hình được lưu, khởi tạo và fit trực tiếp
        from src.data.load import load_processed_data
        from src.models.baseline import BiasBaseline
        
        train, val, _ = load_processed_data()
        model = BiasBaseline()
        model.fit(train, val)
    else:
        model = Recommender.load(model_path)

    # 4. Tạo và in bảng kết quả gợi ý
    recommendations = recommend_for_user(model, args.user_id, maps, movies, n=10)
    
    print(f"\n================ TOP 10 GỢI Ý CHO USER {args.user_id} (Model: {args.model.upper()}) ================")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
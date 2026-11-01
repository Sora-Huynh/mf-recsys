"""Đọc ratings.dat, movies.dat, users.dat thành DataFrame.

Mục đích: một điểm đọc dữ liệu duy nhất cho cả pipeline và notebook EDA. Hạ tầng dữ
liệu, không thuộc chương nào của môn.

Các file dùng dấu phân cách `::` và mã hóa latin-1 (tên phim có ký tự như "é").
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIES_COLUMNS = ["movie_id", "title", "genres"]
USERS_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]


def _read_dat(path: Path, columns: list[str], dtype: dict[str, str]) -> pd.DataFrame:
    """Đọc một file `.dat` phân cách bằng `::`.

    Args:
        path: Đường dẫn file.
        columns: Tên cột theo thứ tự trong file.
        dtype: Kiểu dữ liệu của từng cột.

    Returns:
        DataFrame có các cột `columns`.
    """
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=columns,
        dtype=dtype,
        encoding="latin-1",
    )


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    """Đọc `ratings.dat`.

    Args:
        raw_dir: Thư mục `ml-1m`.

    Returns:
        DataFrame cột `user_id`, `movie_id`, `rating`, `timestamp`.
    """
    dtype = {"user_id": "int32", "movie_id": "int32", "rating": "float32", "timestamp": "int64"}
    return _read_dat(raw_dir / "ratings.dat", RATINGS_COLUMNS, dtype)


def load_movies(raw_dir: Path) -> pd.DataFrame:
    """Đọc `movies.dat`.

    Args:
        raw_dir: Thư mục `ml-1m`.

    Returns:
        DataFrame cột `movie_id`, `title`, `genres` (các thể loại nối bằng `|`).
    """
    dtype = {"movie_id": "int32", "title": "string", "genres": "string"}
    return _read_dat(raw_dir / "movies.dat", MOVIES_COLUMNS, dtype)


def load_users(raw_dir: Path) -> pd.DataFrame:
    """Đọc `users.dat`.

    Args:
        raw_dir: Thư mục `ml-1m`.

    Returns:
        DataFrame cột `user_id`, `gender`, `age`, `occupation`, `zip_code`.
        `zip_code` là chuỗi vì có giá trị dạng `12345-6789`.
    """
    dtype = {
        "user_id": "int32",
        "gender": "string",
        "age": "int32",
        "occupation": "int32",
        "zip_code": "string",
    }
    return _read_dat(raw_dir / "users.dat", USERS_COLUMNS, dtype)

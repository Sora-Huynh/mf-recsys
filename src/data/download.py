"""Tải và giải nén MovieLens-1M vào `data/raw/`.

Mục đích: mỗi thành viên tự tải dữ liệu vì license ML-1M không cho phát hành lại,
nên dữ liệu thô không được commit. Hạ tầng dữ liệu, không thuộc chương nào của môn.

Chạy: python -m src.data.download
"""

from __future__ import annotations

import hashlib
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

from src.config import load_config

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_MD5 = "c4d9eecfca2ab87c1945afe126590906"  # theo ml-1m.zip.md5 của GroupLens
EXPECTED_FILES = ("ratings.dat", "movies.dat", "users.dat")


def is_downloaded(raw_dir: Path) -> bool:
    """Kiểm tra `raw_dir` đã có đủ ba file dữ liệu chưa.

    Args:
        raw_dir: Thư mục chứa dữ liệu, ví dụ `data/raw/ml-1m`.

    Returns:
        True nếu đủ `ratings.dat`, `movies.dat`, `users.dat`.
    """
    return all((raw_dir / name).is_file() for name in EXPECTED_FILES)


def md5sum(path: Path) -> str:
    """Tính MD5 của một file.

    Args:
        path: Đường dẫn file.

    Returns:
        Chuỗi hex.
    """
    with path.open("rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def fetch(url: str, destination: Path) -> None:
    """Tải một file qua HTTPS, có xác thực chứng chỉ.

    Dùng bộ CA của certifi vì kho chứng chỉ Windows có thể chưa nạp root CA của máy chủ
    GroupLens, làm Python báo CERTIFICATE_VERIFY_FAILED dù chứng chỉ hợp lệ.
    Ghi ra file tạm rồi đổi tên để lần tải đứt giữa chừng không để lại file hỏng.

    Args:
        url: Địa chỉ file.
        destination: Đường dẫn lưu.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    partial = destination.with_name(destination.name + ".part")
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        with partial.open("wb") as f:
            shutil.copyfileobj(response, f)
    partial.replace(destination)


def download_ml1m(raw_dir: Path, url: str = ML1M_URL) -> Path:
    """Tải file zip rồi giải nén; bỏ qua nếu dữ liệu đã có.

    Args:
        raw_dir: Thư mục đích, tên phải là `ml-1m` vì file zip chứa sẵn thư mục này.
        url: Địa chỉ file zip.

    Returns:
        `raw_dir`.

    Raises:
        ValueError: Nếu MD5 của file zip không khớp giá trị GroupLens công bố.
        FileNotFoundError: Nếu giải nén xong vẫn thiếu file.
    """
    if is_downloaded(raw_dir):
        return raw_dir

    parent = raw_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    zip_path = parent / "ml-1m.zip"
    if not zip_path.is_file():
        fetch(url, zip_path)

    digest = md5sum(zip_path)
    if digest != ML1M_MD5:
        raise ValueError(f"MD5 không khớp: {digest}. Xóa {zip_path} rồi tải lại.")

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(parent)

    if not is_downloaded(raw_dir):
        raise FileNotFoundError(f"Thiếu file dữ liệu trong {raw_dir}")
    return raw_dir


if __name__ == "__main__":
    print(download_ml1m(load_config()["paths"]["raw_dir"]))

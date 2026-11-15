"""Tạo ma trận CSR và so sánh bộ nhớ dense với CSR.

Mục đích: ma trận user × phim thưa khoảng 95,5%; lưu dense tốn khoảng 22 lần số giá
trị thật sự có. CSR còn lấy được nhanh một hàng, đúng thao tác ALS cần khi giải hệ
cho từng user.

Liên quan Ch.2 (định dạng CSR).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp


def build_csr(
    users: np.ndarray,
    items: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
) -> sp.csr_matrix:
    """Tạo ma trận CSR từ ba mảng tọa độ.

    Args:
        users: Chỉ số hàng.
        items: Chỉ số cột.
        values: Giá trị tại từng ô.
        shape: Kích thước ma trận.

    Returns:
        Ma trận CSR dtype float64.
    """
    return sp.coo_matrix((values, (users, items)), shape = shape, dtype = np.dtype("float64")).tocsr()


def dense_memory_bytes(shape: tuple[int, int], dtype: np.dtype = np.dtype("float64")) -> int:
    """Tính số byte nếu lưu dense.

    Args:
        shape: Kích thước ma trận.
        dtype: Kiểu phần tử.

    Returns:
        Số byte.
    """
    return shape[0] * shape[1] * dtype.itemsize



def csr_memory_bytes(matrix: sp.csr_matrix) -> int:
    """Tính số byte thực tế của ma trận CSR.

    Args:
        matrix: Ma trận CSR.

    Returns:
        Tổng byte của `data`, `indices`, `indptr`.
    """
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def save_csr(path: Path, matrix: sp.csr_matrix) -> None:
    """Lưu ma trận CSR ra file `.npz`.

    Args:
        path: Đường dẫn file.
        matrix: Ma trận CSR.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, matrix)


def load_csr(path: Path) -> sp.csr_matrix:
    """Đọc ma trận CSR từ file `.npz`.

    Args:
        path: Đường dẫn file.

    Returns:
        Ma trận CSR.
    """
    return sp.load_npz(path)

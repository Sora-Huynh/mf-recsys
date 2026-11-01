"""Thước đo sai số dự đoán: RMSE, MAE.

Mục đích: mọi mô hình dùng chung một định nghĩa sai số để so sánh công bằng.
Liên quan Ch.3: RMSE chính là căn của sai số bình phương trung bình mà bình phương
tối thiểu cực tiểu hóa.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_float_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Đổi hai dãy sang mảng float64 và kiểm tra hợp lệ.

    Args:
        y_true: Điểm thật.
        y_pred: Điểm dự đoán.

    Returns:
        Cặp mảng float64 cùng kích thước.

    Raises:
        ValueError: Nếu khác kích thước hoặc rỗng.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"Khác kích thước: {true.shape} và {pred.shape}")
    if true.size == 0:
        raise ValueError("Mảng rỗng")
    return true, pred


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Tính căn sai số bình phương trung bình.

    Args:
        y_true: Điểm thật.
        y_pred: Điểm dự đoán.

    Returns:
        Giá trị RMSE.
    """
    true, pred = _as_float_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Tính sai số tuyệt đối trung bình.

    Args:
        y_true: Điểm thật.
        y_pred: Điểm dự đoán.

    Returns:
        Giá trị MAE.
    """
    true, pred = _as_float_pair(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))

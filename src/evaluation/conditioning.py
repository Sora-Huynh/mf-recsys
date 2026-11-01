"""Số điều kiện của Q_uᵀQ_u + λI theo λ.

Mục đích: giải thích vai trò của λ dưới góc độ số. Với σ là giá trị kỳ dị của Q_u:
    κ₂(Q_uᵀQ_u + λI) = (σ_max² + λ) / (σ_min² + λ)
Vì vậy tăng λ làm hệ ổn định hơn; λ = 0 với user ít đánh giá cho hệ suy biến. Mất cỡ
log₁₀κ chữ số chính xác khi giải hệ, nên κ gắn trực tiếp với Ch.1.

Dùng `np.linalg.eigvalsh` vì ma trận đối xứng, nhanh và chính xác hơn `np.linalg.cond`.

Liên quan Ch.1 (sai số, số điều kiện), Ch.2 (hệ tuyến tính), Ch.3 (Ridge).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def regularized_gram(Q_u: np.ndarray, reg: float) -> np.ndarray:
    """Tạo ma trận Q_uᵀQ_u + λI.

    Args:
        Q_u: Các hàng của Q ứng với phim user đã đánh giá, kích thước (n_u, k).
        reg: Hệ số λ.

    Returns:
        Ma trận k×k.
    """
    # TODO: Q_u.T @ Q_u + reg * np.eye(k).
    raise NotImplementedError


def condition_number(A: np.ndarray) -> float:
    """Tính số điều kiện chuẩn 2 của ma trận đối xứng.

    Args:
        A: Ma trận đối xứng nửa xác định dương.

    Returns:
        κ₂(A); `np.inf` nếu trị riêng nhỏ nhất bằng 0 về mặt số.
    """
    # TODO: eigvalsh, lấy tỉ số trị riêng lớn nhất và nhỏ nhất.
    raise NotImplementedError


def condition_vs_lambda(
    Q: np.ndarray,
    train: sp.csr_matrix,
    reg_grid: list[float],
    n_users: int,
    seed: int,
) -> pd.DataFrame:
    """Tính thống kê κ theo λ trên một mẫu user.

    Args:
        Q: Ma trận phim đã học, kích thước (n_items, k).
        train: Ma trận điểm train CSR.
        reg_grid: Các giá trị λ cần quét.
        n_users: Số user lấy mẫu.
        seed: Seed lấy mẫu.

    Returns:
        DataFrame cột `reg`, `kappa_median`, `kappa_max`.
    """
    # TODO: dùng median vì κ của user ít đánh giá rất lớn, trung bình sẽ bị lệch.
    raise NotImplementedError

"""Mô hình cơ sở: trung bình toàn cục và trung bình + bias user/phim.

Mục đích: tạo mốc so sánh; mô hình phân rã phải thắng được các mốc này mới có ý
nghĩa. `BiasBaseline` còn là tầng dùng chung: SVD, ALS, SGD đều phân rã phần dư
r - (μ + b_u + b_i), nhờ đó ba phương pháp giải cùng một bài toán và so sánh được.

Công thức: r̂_ui = μ + b_u + b_i. Bias có shrinkage là bài toán Ridge thu gọn:
    b_i = Σ(r_ui - μ) / (n_i + reg),  b_u = Σ(r_ui - μ - b_i) / (n_u + reg)
vì phim ít đánh giá có trung bình nhiễu, cần kéo về 0.

Liên quan Ch.3 (Ridge).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.models.base import Recommender


class GlobalMean(Recommender):
    """Dự đoán mọi ô bằng trung bình toàn cục μ của tập train."""

    def __init__(self, rating_min: float = 1.0, rating_max: float = 5.0) -> None:
        """Khởi tạo mô hình.

        Args:
            rating_min: Cận dưới khi clip.
            rating_max: Cận trên khi clip.
        """
        super().__init__(rating_min, rating_max)
        self.mu: float = 0.0

    def _fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None) -> None:
        """Tính μ. Xem `Recommender._fit`."""
        if train.nnz > 0:
            self.mu = float(train.data.mean())
        else:
            self.mu = 0.0

    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Trả về mảng hằng μ. Xem `Recommender.predict_raw`."""
        return np.full(users.shape, self.mu, dtype=np.float64)


class BiasBaseline(Recommender):
    """Trung bình + bias user/phim có shrinkage."""

    def __init__(
        self,
        reg: float = 10.0,
        rating_min: float = 1.0,
        rating_max: float = 5.0,
    ) -> None:
        """Khởi tạo mô hình.

        Args:
            reg: Hệ số shrinkage.
            rating_min: Cận dưới khi clip.
            rating_max: Cận trên khi clip.
        """
        super().__init__(rating_min, rating_max)
        self.reg = reg
        self.mu: float = 0.0
        self.b_u: np.ndarray = np.array([])
        self.b_i: np.ndarray = np.array([])

    def _fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None) -> None:
        """Tính μ, b_u, b_i. Xem `Recommender._fit`."""
        coo = train.tocoo()
        u_idx = coo.row
        i_idx = coo.col
        ratings = coo.data

        n_users, n_items = train.shape

        if len(ratings) > 0:
            self.mu = float(ratings.mean())
        else:
            self.mu = 0.0

        r_minus_mu = ratings - self.mu

        # 1. Tính Item Bias: b_i = sum(r_ui - mu) / (n_i + reg)
        i_sums = np.bincount(i_idx, weights=r_minus_mu, minlength=n_items)
        i_counts = np.bincount(i_idx, minlength=n_items)
        self.b_i = i_sums / (i_counts + self.reg)

        # 2. Tính User Bias: b_u = sum(r_ui - mu - b_i) / (n_u + reg)
        r_minus_mu_bi = r_minus_mu - self.b_i[i_idx]
        u_sums = np.bincount(u_idx, weights=r_minus_mu_bi, minlength=n_users)
        u_counts = np.bincount(u_idx, minlength=n_users)
        self.b_u = u_sums / (u_counts + self.reg)

    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Tính μ + b_u + b_i; mô hình phân rã cộng thêm P Qᵀ vào giá trị này.

        Xem `Recommender.predict_raw`.
        """
        return self.mu + self.b_u[users] + self.b_i[items]

    def residual_matrix(self, ratings: sp.csr_matrix) -> sp.csr_matrix:
        """Tính ma trận phần dư r_ui - (μ + b_u + b_i) trên các ô đã có.

        Args:
            ratings: Ma trận điểm CSR.

        Returns:
            Ma trận CSR cùng cấu trúc thưa, giá trị là phần dư.
        """
        coo = ratings.tocoo()
        preds = self.predict_raw(coo.row, coo.col)
        res_data = coo.data - preds
        return sp.csr_matrix((res_data, (coo.row, coo.col)), shape=ratings.shape)
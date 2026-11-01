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
        self.rating_min = rating_min
        self.rating_max = rating_max

    def fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None = None) -> "GlobalMean":
        """Tính μ. Xem `Recommender.fit`."""
        # TODO: μ = train.data.mean().
        raise NotImplementedError

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Trả về mảng hằng μ. Xem `Recommender.predict`."""
        # TODO: np.full_like theo kích thước users.
        raise NotImplementedError


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
        self.reg = reg
        self.rating_min = rating_min
        self.rating_max = rating_max

    def fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None = None) -> "BiasBaseline":
        """Tính μ, b_u, b_i. Xem `Recommender.fit`."""
        # TODO: tính theo công thức đầu file bằng np.bincount, không lặp từng đánh giá.
        raise NotImplementedError

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Tính μ + b_u + b_i. Xem `Recommender.predict`."""
        # TODO: vector hóa bằng indexing rồi np.clip.
        raise NotImplementedError

    def raw_predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Tính μ + b_u + b_i, không clip.

        Args:
            users: Mảng chỉ số user.
            items: Mảng chỉ số phim.

        Returns:
            Giá trị baseline; các mô hình phân rã cộng phần này với P Qᵀ rồi mới clip.
        """
        # TODO: như predict nhưng không clip.
        raise NotImplementedError

    def residual_matrix(self, ratings: sp.csr_matrix) -> sp.csr_matrix:
        """Tính ma trận phần dư r_ui - (μ + b_u + b_i) trên các ô đã có.

        Args:
            ratings: Ma trận điểm CSR.

        Returns:
            Ma trận CSR cùng cấu trúc thưa, giá trị là phần dư.
        """
        # TODO: lấy tọa độ từ ratings.tocoo(), trừ raw_predict.
        raise NotImplementedError

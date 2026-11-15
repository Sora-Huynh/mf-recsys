"""Mô hình cơ sở: trung bình toàn cục và trung bình + bias user/phim.

Mục đích: tạo mốc so sánh; mô hình phân rã phải thắng được các mốc này mới có ý
nghĩa. `BiasBaseline` còn là tầng dùng chung: SVD, ALS, SGD đều phân rã phần dư
r - (μ + b_u + b_i), nhờ đó ba phương pháp giải cùng một bài toán và so sánh được.

Công thức: r̂_ui = μ + b_u + b_i. Bias có shrinkage (co về 0):
    b_i = Σ(r_ui - μ - b_u) / (n_i + reg),  b_u = Σ(r_ui - μ - b_i) / (n_u + reg)
vì phim ít đánh giá có trung bình nhiễu, cần kéo về 0.

Lưu ý: mặc định (`n_iters = 1`) bias chỉ được ước lượng MỘT lượt, b_i trước rồi b_u
(với b_u = 0 khi tính b_i). Đây là xấp xỉ của bài toán Ridge theo (b_u, b_i), không
phải nghiệm đồng thời. Tăng `n_iters` để luân phiên cập nhật hai vế, tiến dần tới
nghiệm chính xác (tọa độ luân phiên). Trên ML-1M, 5 lượt chỉ cải thiện RMSE cỡ 1e-3.

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
    """Trung bình + bias user/phim có shrinkage.

    Attributes:
        reg: Hệ số shrinkage, không âm.
        n_iters: Số lượt luân phiên cập nhật (b_i rồi b_u); mặc định 1.
        mu: Trung bình toàn cục của điểm train.
        b_u: Bias user, dài n_users; user không có đánh giá thì bằng 0.
        b_i: Bias phim, dài n_items; phim không có đánh giá thì bằng 0.
    """

    def __init__(
        self,
        reg: float = 10.0,
        rating_min: float = 1.0,
        rating_max: float = 5.0,
        n_iters: int = 1,
    ) -> None:
        """Khởi tạo mô hình.

        Args:
            reg: Hệ số shrinkage, không âm.
            rating_min: Cận dưới khi clip.
            rating_max: Cận trên khi clip.
            n_iters: Số lượt luân phiên cập nhật bias, ít nhất 1.

        Raises:
            ValueError: Nếu `reg` âm hoặc `n_iters` nhỏ hơn 1.
        """
        super().__init__(rating_min, rating_max)
        if reg < 0:
            raise ValueError(f"reg phải không âm, nhận {reg}")
        if n_iters < 1:
            raise ValueError(f"n_iters phải >= 1, nhận {n_iters}")
        self.reg = reg
        self.n_iters = n_iters
        self.mu: float = 0.0
        self.b_u: np.ndarray = np.array([])
        self.b_i: np.ndarray = np.array([])

    def _shrunk_mean(self, index: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
        """Trung bình co về 0 theo nhóm: Σ values / (n + reg).

        Nhóm không có đánh giá nào cho kết quả 0, kể cả khi reg = 0 (tránh 0/0 = NaN).

        Args:
            index: Chỉ số nhóm (user hoặc phim) của từng đánh giá.
            values: Phần dư tương ứng của từng đánh giá.
            size: Số nhóm.

        Returns:
            Mảng dài `size`.
        """
        sums = np.bincount(index, weights=values, minlength=size)
        counts = np.bincount(index, minlength=size)
        denom = counts + self.reg
        out = np.zeros(size)
        np.divide(sums, denom, out=out, where=denom > 0)
        return out

    def _fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None) -> None:
        """Tính μ, b_u, b_i. Xem `Recommender._fit`."""
        coo = train.tocoo()
        u_idx = coo.row
        i_idx = coo.col
        ratings = coo.data.astype(np.float64)
        n_users, n_items = train.shape

        self.mu = float(ratings.mean()) if len(ratings) > 0 else 0.0
        r_minus_mu = ratings - self.mu

        b_u = np.zeros(n_users)
        b_i = np.zeros(n_items)
        for _ in range(self.n_iters):
            b_i = self._shrunk_mean(i_idx, r_minus_mu - b_u[u_idx], n_items)
            b_u = self._shrunk_mean(u_idx, r_minus_mu - b_i[i_idx], n_users)
        self.b_u = b_u
        self.b_i = b_i

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
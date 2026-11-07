"""Điền thiếu rồi SVD cắt cụt.

Mục đích: định lý Eckart-Young nói SVD cắt cụt là xấp xỉ hạng k tốt nhất, nhưng chỉ
áp dụng cho ma trận đầy đủ, nên phải điền ô thiếu trước.

Cách điền: dùng baseline μ + b_u + b_i. Khi đó ma trận cần phân rã là phần dư, và ô
thiếu đúng bằng 0, tức ma trận vẫn thưa. Lý do chọn: giữ được CSR, không phải tạo ma
trận dense 6.040 × 3.706, và dùng được SVD lặp trên ma trận thưa.

Hạn chế cần nêu trong báo cáo: SVD coi các ô điền là dữ liệu thật, nên kết quả bị kéo
về baseline; ALS và SGD chỉ khớp trên ô có thật. Đây là điểm so sánh chính với ALS.

Liên quan Ch.4 (SVD).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds  # Bổ sung import quan trọng để không bị lỗi

from src.models.base import Recommender
from src.models.baseline import BiasBaseline


class TruncatedSVDRecommender(Recommender):
    """Mô hình R ≈ baseline + U_k Σ_k V_kᵀ."""

    def __init__(self, k: int, baseline: BiasBaseline) -> None:
        """Khởi tạo mô hình.

        Args:
            k: Số thành phần giữ lại.
            baseline: `BiasBaseline` đã fit.
        """
        super().__init__(baseline.rating_min, baseline.rating_max)
        self.k = k
        self.baseline = baseline
        self.u_: np.ndarray | None = None
        self.s_: np.ndarray | None = None
        self.vt_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    def _fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None) -> None:
        """Phân rã SVD phần dư. Xem `Recommender._fit`."""
        # 1. Luôn fit lại baseline, không dùng lệnh `if` để tránh dính dữ liệu cũ
        self.baseline.fit(train, val)

        # 2. Tính ma trận phần dư độc lập (an toàn tuyệt đối ngay cả khi bạn C chưa code)
        if hasattr(self.baseline, 'residual_matrix'):
            residual = self.baseline.residual_matrix(train)
        else:
            users, items = train.nonzero()
            base_preds = self.baseline.predict_raw(users, items)
            residual = train.copy()
            residual.data = residual.data - base_preds  # Thực tế - Baseline

        m, n = residual.shape
        min_dim = min(m, n)
        k_target = min(self.k, min_dim)

        # 3. Tiến hành phân rã ma trận
        if min_dim <= k_target + 1 or residual.nnz == m * n:
            dense_res = residual.toarray() if sp.issparse(residual) else np.asarray(residual)
            u, s, vt = np.linalg.svd(dense_res, full_matrices=False)
            u = u[:, :k_target]
            s = s[:k_target]
            vt = vt[:k_target, :]
        else:
            # Dùng trực tiếp hàm svds đã import
            u, s, vt = svds(residual, k=k_target)
            idx = np.argsort(s)[::-1]
            u = u[:, idx]
            s = s[idx]
            vt = vt[idx, :]

        self.u_ = u
        self.s_ = s
        self.vt_ = vt
        self.singular_values_ = s

    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Tính baseline + Σ_j u_j σ_j v_j. Xem `Recommender.predict_raw`."""
        users = np.asarray(users)
        items = np.asarray(items)
        base_pred = self.baseline.predict_raw(users, items)

        if self.u_ is None or self.s_ is None or self.vt_ is None or self.k == 0:
            return base_pred

        # Tối ưu: vt_.T[items] giúp cắt dòng ma trận nhanh hơn đáng kể
        svd_pred = np.sum(self.u_[users] * self.s_ * self.vt_.T[items], axis=1)
        return base_pred + svd_pred
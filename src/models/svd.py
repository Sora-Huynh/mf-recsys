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

    def _fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None) -> None:
        """Phân rã SVD phần dư. Xem `Recommender._fit`."""
        # TODO: scipy.sparse.linalg.svds(residual, k). Hàm trả σ tăng dần, cần đảo thứ tự.
        raise NotImplementedError

    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Tính baseline + Σ_j u_j σ_j v_j. Xem `Recommender.predict_raw`."""
        # TODO: einsum trên hàng của U và V, không dựng lại ma trận đầy đủ.
        raise NotImplementedError

"""ALS (Alternating Least Squares) cho phân rã ma trận.

Mục đích: bài toán min Σ (r_ui - p_uᵀq_i)² + λ(‖P‖² + ‖Q‖²) không lồi theo (P, Q)
nhưng là bình phương tối thiểu nếu giữ cố định một ma trận. Vì vậy ALS luân phiên cố
định Q giải P, rồi cố định P giải Q; mỗi bước có nghiệm đúng nên hàm mục tiêu không
tăng, và không cần chọn learning rate như SGD.

Mỗi user là một bài toán Ridge riêng:
    (Q_uᵀQ_u + λI) p_u = Q_uᵀ r_u
trong đó Q_u CHỈ gồm các hàng của phim user u đã đánh giá, không phải toàn bộ Q. Do
đó mỗi user có một ma trận k×k khác nhau. Ma trận này đối xứng xác định dương khi
λ > 0 nên dùng Cholesky: nhanh gấp đôi LU và ổn định, không cần chọn trụ.

Vì sao cần λ > 0: user ít đánh giá (n_u < k) làm Q_uᵀQ_u suy biến. ML-1M có user chỉ
20 đánh giá, còn khoảng 16 trong train, nên suy biến thật khi k ≥ 20.

Vòng lặp Python theo user/phim (vài nghìn lần) là chấp nhận được; không lặp theo từng
đánh giá.

Liên quan Ch.2 (Cholesky, CSR) và Ch.3 (Ridge).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.evaluation.convergence import ConvergenceLog
from src.models.base import Recommender
from src.models.baseline import BiasBaseline


def update_factors(
    fixed: np.ndarray,
    residual: sp.csr_matrix,
    reg: float,
    solver: str = "cholesky",
) -> np.ndarray:
    """Giải toàn bộ các hệ Ridge của một bên.

    Dùng chung cho hai bước: truyền `residual` CSR để cập nhật P từ Q, truyền chuyển vị
    (cũng ở dạng CSR) để cập nhật Q từ P.

    Args:
        fixed: Ma trận đang cố định, kích thước (m, k).
        residual: Ma trận phần dư CSR, mỗi hàng là một hệ cần giải.
        reg: Hệ số λ.
        solver: `cholesky` hoặc `gauss_seidel`.

    Returns:
        Ma trận nghiệm, kích thước (số hàng của `residual`, k).
    """
    # TODO: với mỗi hàng: lấy indices/data, tạo A = F_uᵀF_u + λI, b = F_uᵀr_u, gọi solver.
    raise NotImplementedError


def objective(
    P: np.ndarray,
    Q: np.ndarray,
    residual: sp.csr_matrix,
    reg: float,
) -> float:
    """Tính hàm mục tiêu Σ (r_ui - p_uᵀq_i)² + λ(‖P‖² + ‖Q‖²).

    Args:
        P: Ma trận user, kích thước (n_users, k).
        Q: Ma trận phim, kích thước (n_items, k).
        residual: Ma trận phần dư CSR.
        reg: Hệ số λ.

    Returns:
        Giá trị hàm mục tiêu; phải không tăng sau mỗi lần cập nhật, dùng để kiểm thử.
    """
    # TODO: vector hóa bằng tọa độ của residual.tocoo().
    raise NotImplementedError


class ALSRecommender(Recommender):
    """Mô hình R ≈ baseline + P Qᵀ, huấn luyện bằng ALS."""

    def __init__(
        self,
        k: int,
        reg: float,
        n_epochs: int,
        baseline: BiasBaseline,
        init_std: float = 0.1,
        solver: str = "cholesky",
        seed: int = 42,
    ) -> None:
        """Khởi tạo mô hình.

        Args:
            k: Số nhân tố ẩn.
            reg: Hệ số λ.
            n_epochs: Số vòng lặp; mỗi vòng cập nhật P rồi Q.
            baseline: `BiasBaseline` đã fit.
            init_std: Độ lệch chuẩn khởi tạo Q.
            solver: `cholesky` hoặc `gauss_seidel`.
            seed: Seed khởi tạo.
        """
        super().__init__(baseline.rating_min, baseline.rating_max)
        self.k = k
        self.reg = reg
        self.n_epochs = n_epochs
        self.baseline = baseline
        self.init_std = init_std
        self.solver = solver
        self.seed = seed
        self.log = ConvergenceLog()

    def _fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None) -> None:
        """Luân phiên cập nhật P, Q và ghi `self.log`. Xem `Recommender._fit`."""
        # TODO: khởi tạo Q; lặp update_factors hai chiều; ghi log, không tính giờ phần đánh giá.
        raise NotImplementedError

    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Tính baseline + p_uᵀq_i. Xem `Recommender.predict_raw`."""
        # TODO: baseline.predict_raw + einsum trên P[users], Q[items].
        raise NotImplementedError

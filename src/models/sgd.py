"""Phân rã ma trận bằng SGD / Momentum / Adam theo mini-batch.

Mục đích: ALS phải giải hàng nghìn hệ k×k mỗi vòng; SGD chỉ cần tính gradient trên
từng mini-batch nên mỗi bước rất rẻ, đổi lại phải chọn learning rate. Momentum và Adam
giảm dao động và giảm độ nhạy với learning rate.

Gradient theo một đánh giá, với e = r_ui - p_uᵀq_i:
    ∂L/∂p_u = -e q_i + λ p_u,    ∂L/∂q_i = -e p_u + λ q_i
Dùng gradient tay vì công thức ngắn và chạy nhanh; autodiff chỉ để đối chiếu (tùy chọn).

Vector hóa: trong một batch một user có thể lặp lại, nên cộng gradient bằng np.add.at
hoặc numba, không gán trực tiếp `grad[users] = ...` vì sẽ mất giá trị.

Lưu ý: λ ở đây áp dụng theo từng đánh giá, khác thang với λ tổng của ALS, nên không
được dùng chung một giá trị `reg` cho hai mô hình.

Liên quan Ch.5 (đạo hàm) và Ch.6 (GD, SGD, Momentum, Adam).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.evaluation.convergence import ConvergenceLog
from src.models.base import Recommender
from src.models.baseline import BiasBaseline


@dataclass
class OptimizerState:
    """Trạng thái tích lũy của optimizer cho một ma trận tham số.

    Attributes:
        velocity: Mô-men bậc một (Momentum, Adam).
        second_moment: Mô-men bậc hai (Adam).
        step: Số bước đã cập nhật, dùng hiệu chỉnh độ lệch của Adam.
    """

    velocity: np.ndarray
    second_moment: np.ndarray
    step: int = field(default=0)


def batch_gradients(
    P: np.ndarray,
    Q: np.ndarray,
    users: np.ndarray,
    items: np.ndarray,
    residuals: np.ndarray,
    reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tính gradient của P, Q trên một mini-batch.

    Args:
        P: Ma trận user, kích thước (n_users, k).
        Q: Ma trận phim, kích thước (n_items, k).
        users: Chỉ số user trong batch.
        items: Chỉ số phim trong batch.
        residuals: Phần dư sau baseline của từng đánh giá trong batch.
        reg: Hệ số λ theo từng đánh giá.

    Returns:
        Cặp gradient (grad_P, grad_Q), cùng kích thước với P, Q.
    """
    # TODO: tính e, gom gradient bằng np.add.at (hoặc numba nếu chậm).
    raise NotImplementedError


def apply_update(
    param: np.ndarray,
    grad: np.ndarray,
    state: OptimizerState,
    optimizer: str,
    lr: float,
    hyper: dict[str, float],
) -> None:
    """Cập nhật tại chỗ một ma trận tham số.

    Args:
        param: Ma trận P hoặc Q.
        grad: Gradient cùng kích thước.
        state: Trạng thái optimizer của ma trận này.
        optimizer: `sgd`, `momentum` hoặc `adam`.
        lr: Learning rate.
        hyper: Các hệ số `momentum_beta`, `adam_beta1`, `adam_beta2`, `adam_eps`.
    """
    # TODO: cài ba công thức cập nhật theo Ch.6.
    raise NotImplementedError


class SGDRecommender(Recommender):
    """Mô hình R ≈ baseline + P Qᵀ, huấn luyện bằng mini-batch."""

    def __init__(
        self,
        k: int,
        reg: float,
        lr: float,
        n_epochs: int,
        batch_size: int,
        baseline: BiasBaseline,
        optimizer: str = "adam",
        init_std: float = 0.1,
        hyper: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        """Khởi tạo mô hình.

        Args:
            k: Số nhân tố ẩn.
            reg: Hệ số λ theo từng đánh giá.
            lr: Learning rate.
            n_epochs: Số epoch.
            batch_size: Kích thước mini-batch.
            baseline: `BiasBaseline` đã fit.
            optimizer: `sgd`, `momentum` hoặc `adam`.
            init_std: Độ lệch chuẩn khởi tạo P, Q.
            hyper: Các hệ số của optimizer.
            seed: Seed khởi tạo và xáo dữ liệu.
        """
        super().__init__(baseline.rating_min, baseline.rating_max)
        self.k = k
        self.reg = reg
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.baseline = baseline
        self.optimizer = optimizer
        self.init_std = init_std
        self.hyper = hyper or {}
        self.seed = seed
        self.log = ConvergenceLog()

    def _fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None) -> None:
        """Lặp theo epoch và mini-batch, ghi `self.log`. Xem `Recommender._fit`."""
        # TODO: xáo dữ liệu mỗi epoch; batch_gradients + apply_update; không tính giờ phần đánh giá.
        raise NotImplementedError

    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Tính baseline + p_uᵀq_i. Xem `Recommender.predict_raw`."""
        # TODO: baseline.predict_raw + einsum trên P[users], Q[items].
        raise NotImplementedError

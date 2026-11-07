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
        # Lấy đặc trưng của các user và phim xuất hiện trong batch.
    p_batch = P[users]
    q_batch = Q[items]

    # Dự đoán phần dư: tích vô hướng của từng cặp user–phim.
    predictions = np.sum(p_batch * q_batch, axis=1)

    # Sai số giữa phần dư thật và phần dư dự đoán.
    errors = residuals - predictions

    # Gradient của từng lượt đánh giá.
    grad_p_batch = -errors[:, None] * q_batch + reg * p_batch
    grad_q_batch = -errors[:, None] * p_batch + reg * q_batch

    # Tạo nơi chứa gradient cho toàn bộ P và Q.
    grad_P = np.zeros_like(P)
    grad_Q = np.zeros_like(Q)

    # Cộng dồn khi user hoặc phim xuất hiện nhiều lần.
    np.add.at(grad_P, users, grad_p_batch)
    np.add.at(grad_Q, items, grad_q_batch)

    return grad_P, grad_Q


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
    if optimizer == "sgd":
        param -= lr * grad

    elif optimizer == "momentum":
        beta = hyper.get("momentum_beta", 0.9)

        state.velocity *= beta
        state.velocity += grad

        param -= lr * state.velocity

    elif optimizer == "adam":
        beta1 = hyper.get("adam_beta1", 0.9)
        beta2 = hyper.get("adam_beta2", 0.999)
        eps = hyper.get("adam_eps", 1e-8)

        state.step += 1

        # Trung bình trượt của gradient.
        state.velocity *= beta1
        state.velocity += (1 - beta1) * grad

        # Trung bình trượt của bình phương gradient.
        state.second_moment *= beta2
        state.second_moment += (1 - beta2) * grad ** 2

        # Hiệu chỉnh do trạng thái ban đầu bằng 0.
        m_hat = state.velocity / (1 - beta1 ** state.step)
        v_hat = state.second_moment / (1 - beta2 ** state.step)

        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    else:
        raise ValueError(f"Optimizer không hợp lệ: {optimizer}")

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
        from time import perf_counter

        rng = np.random.default_rng(self.seed)

        # 1. Khởi tạo đặc trưng của user và phim.
        n_users, n_items = train.shape

        self.P = rng.normal(
            0.0, self.init_std, size=(n_users, self.k)
        )
        self.Q = rng.normal(
            0.0, self.init_std, size=(n_items, self.k)
        )

        # 2. Mỗi ma trận có trạng thái optimizer riêng.
        state_P = OptimizerState(
            velocity=np.zeros_like(self.P),
            second_moment=np.zeros_like(self.P),
        )
        state_Q = OptimizerState(
            velocity=np.zeros_like(self.Q),
            second_moment=np.zeros_like(self.Q),
        )

        # 3. Lấy những lượt đánh giá có thật trong train.
        observed = train.tocoo()
        users = observed.row
        items = observed.col
        ratings = observed.data

        if len(ratings) == 0:
            raise ValueError("Tập train không có đánh giá.")
        if self.batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0.")

        # Baseline được học trước và giữ cố định.
        residuals = ratings - self.baseline.predict_raw(users, items)

        val_observed = None if val is None else val.tocoo()

        # 4. Hàm ghi sai số; không tính thời gian này vào huấn luyện.
        self.log = ConvergenceLog()

        def record_metrics(epoch, elapsed):
            train_predictions = self.predict(users, items)
            train_rmse = float(
                np.sqrt(np.mean((ratings - train_predictions) ** 2))
            )

            val_rmse = None
            if val_observed is not None and val_observed.nnz > 0:
                val_predictions = self.predict(
                    val_observed.row, val_observed.col
                )
                val_rmse = float(
                    np.sqrt(
                        np.mean(
                            (val_observed.data - val_predictions) ** 2
                        )
                    )
                )

            self.log.record(
                epoch=epoch,
                elapsed_s=elapsed,
                train_rmse=train_rmse,
                val_rmse=val_rmse,
            )

        elapsed = 0.0

        # Epoch 0: đo sai số trước khi học để có mốc so sánh.
        record_metrics(0, elapsed)

        # 5. Huấn luyện qua nhiều epoch.
        for epoch in range(1, self.n_epochs + 1):
            start = perf_counter()
            order = rng.permutation(len(ratings))

            for offset in range(0, len(order), self.batch_size):
                batch = order[offset:offset + self.batch_size]

                grad_P, grad_Q = batch_gradients(
                    self.P,
                    self.Q,
                    users[batch],
                    items[batch],
                    residuals[batch],
                    self.reg,
                )

                apply_update(
                    self.P, grad_P, state_P,
                    self.optimizer, self.lr, self.hyper,
                )
                apply_update(
                    self.Q, grad_Q, state_Q,
                    self.optimizer, self.lr, self.hyper,
                )

            elapsed += perf_counter() - start
            record_metrics(epoch, elapsed)

    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        # Điểm dự đoán từ baseline: trung bình + bias user/phim.
        baseline_scores = self.baseline.predict_raw(users, items)

        # Đặc trưng của các user và phim cần dự đoán.
        p = self.P[users]
        q = self.Q[items]

        # Độ phù hợp riêng của từng cặp user–phim.
        interaction = np.sum(p * q, axis=1)

        return baseline_scores + interaction
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

Vì sao cần λ > 0: user ít đánh giá (n_u < k) làm Q_uᵀQ_u suy biến. Sau khi chia tập ML-1M,
401 user có dưới 20 đánh giá trong train (ít nhất là 10), nên với k = 20 hệ suy biến thật.

Vòng lặp theo user/phim và theo đánh giá nằm trong hàm numba song song. Mỗi hàng được
giải độc lập nên kết quả không phụ thuộc số luồng.

Liên quan Ch.2 (Cholesky, CSR) và Ch.3 (Ridge).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from numba import njit, prange

from src.evaluation.convergence import ConvergenceLog
from src.evaluation.metrics import rmse
from src.models.base import Recommender
from src.models.baseline import BiasBaseline
from src.solvers.linear import _backward_transpose_core, _cholesky_core, _forward_core


@njit(parallel=True)
def _solve_rows(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    fixed: np.ndarray,
    reg: float,
) -> tuple[np.ndarray, int]:
    """Dựng và giải (F_uᵀF_u + λI)x = F_uᵀr_u cho từng hàng của ma trận CSR.

    Chỉ dựng tam giác dưới vì Cholesky chỉ đọc phần đó, giảm một nửa số phép tính.

    Args:
        indptr: Mảng `indptr` của CSR.
        indices: Mảng `indices` của CSR.
        data: Mảng `data` của CSR.
        fixed: Ma trận đang cố định, kích thước (m, k).
        reg: Hệ số λ.

    Returns:
        Cặp (ma trận nghiệm, chỉ số hàng thất bại đầu tiên hoặc -1 nếu không có).
    """
    n_rows = indptr.shape[0] - 1
    k = fixed.shape[1]
    solution = np.zeros((n_rows, k))
    # Hàm numba song song không ném lỗi được, nên ghi lại hàng thất bại để báo sau.
    failed = np.zeros(n_rows, dtype=np.bool_)
    for row in prange(n_rows):
        A = np.zeros((k, k))
        x = solution[row]
        for position in range(indptr[row], indptr[row + 1]):
            factor = fixed[indices[position]]
            value = data[position]
            for a in range(k):
                x[a] += value * factor[a]
                for c in range(a + 1):
                    A[a, c] += factor[a] * factor[c]
        for a in range(k):
            A[a, a] += reg
        if _cholesky_core(A):
            _forward_core(A, x)
            _backward_transpose_core(A, x)
        else:
            failed[row] = True
    bad = np.flatnonzero(failed)
    return solution, int(bad[0]) if bad.size > 0 else -1


@njit(parallel=True)
def _pair_dot(
    P: np.ndarray,
    Q: np.ndarray,
    users: np.ndarray,
    items: np.ndarray,
) -> np.ndarray:
    """Tính p_uᵀq_i cho từng cặp mà không dựng P[users] và Q[items].

    Hai mảng trung gian đó cần khoảng 8 GB với 25 triệu đánh giá và k = 20, nên không
    dùng `einsum` trên chúng được.

    Args:
        P: Ma trận user.
        Q: Ma trận phim.
        users: Chỉ số user, dài n.
        items: Chỉ số phim, dài n.

    Returns:
        Mảng dài n.
    """
    result = np.zeros(users.shape[0])
    for n in prange(users.shape[0]):
        p = P[users[n]]
        q = Q[items[n]]
        total = 0.0
        for a in range(p.shape[0]):
            total += p[a] * q[a]
        result[n] = total
    return result


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

    Raises:
        NotImplementedError: Nếu `solver` khác `cholesky`; Gauss-Seidel là phần tùy chọn
            đã bỏ trong ROADMAP.
        np.linalg.LinAlgError: Nếu có hệ không xác định dương, thường do λ = 0.
    """
    if solver != "cholesky":
        raise NotImplementedError(f"Chưa cài solver {solver!r}")
    solution, bad_row = _solve_rows(
        residual.indptr.astype(np.int64),
        residual.indices.astype(np.int64),
        residual.data.astype(np.float64),
        fixed,
        float(reg),
    )
    if bad_row >= 0:
        raise np.linalg.LinAlgError(f"Hệ của hàng {bad_row} không xác định dương (λ = {reg})")
    return solution


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
    coo = residual.tocoo()
    errors = coo.data - _pair_dot(P, Q, coo.row.astype(np.int64), coo.col.astype(np.int64))
    return float(errors @ errors + reg * (np.sum(P * P) + np.sum(Q * Q)))


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
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None

    def _fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None) -> None:
        """Luân phiên cập nhật P, Q và ghi `self.log`. Xem `Recommender._fit`."""
        n_users, n_items = train.shape
        rng = np.random.default_rng(self.seed)
        self.Q = rng.normal(0.0, self.init_std, size=(n_items, self.k))
        self.P = np.zeros((n_users, self.k))
        self.log = ConvergenceLog()

        residual = self.baseline.residual_matrix(train)
        residual_by_item = residual.T.tocsr()
        train_coo = train.tocoo()
        val_coo = val.tocoo() if val is not None else None

        # Biên dịch numba trước khi tính giờ: trên ML-1M, riêng bước này lâu gấp khoảng 40
        # lần một vòng ALS và sẽ làm sai phép so sánh hội tụ theo thời gian với SGD.
        update_factors(np.zeros((1, self.k)), sp.csr_matrix((1, 1)), 1.0, self.solver)

        elapsed = 0.0
        for epoch in range(1, self.n_epochs + 1):
            start = time.perf_counter()
            self.P = update_factors(self.Q, residual, self.reg, self.solver)
            self.Q = update_factors(self.P, residual_by_item, self.reg, self.solver)
            elapsed += time.perf_counter() - start

            train_rmse = rmse(train_coo.data, self.predict(train_coo.row, train_coo.col))
            val_rmse = None
            if val_coo is not None:
                val_rmse = rmse(val_coo.data, self.predict(val_coo.row, val_coo.col))
            self.log.record(
                epoch=epoch,
                elapsed_s=elapsed,
                train_rmse=train_rmse,
                val_rmse=val_rmse,
                objective=objective(self.P, self.Q, residual, self.reg),
            )

    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Tính baseline + p_uᵀq_i. Xem `Recommender.predict_raw`."""
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        return self.baseline.predict_raw(users, items) + _pair_dot(self.P, self.Q, users, items)

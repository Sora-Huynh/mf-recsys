"""Kiểm thử mô hình trên ma trận nhỏ tự tạo, biết trước đáp án.

Mục đích: dữ liệu thật không cho biết đáp án đúng; ma trận hạng thấp tự tạo thì có,
nên lỗi thuật toán lộ ra ngay mà không phải chờ chạy ML-1M.

Mỗi test đang ở trạng thái xfail. Khi cài xong, thay `raise NotImplementedError` bằng
phần kiểm tra và bỏ `pytestmark` khi file không còn test nào chưa làm.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="Chưa cài đặt")


@pytest.fixture
def low_rank_matrix() -> np.ndarray:
    """Ma trận 30 × 20 hạng đúng bằng 3, cố định theo seed.

    Returns:
        Ma trận dense P Qᵀ.
    """
    rng = np.random.default_rng(0)
    return rng.normal(size=(30, 3)) @ rng.normal(size=(20, 3)).T


@pytest.fixture
def observed(low_rank_matrix: np.ndarray) -> sp.csr_matrix:
    """Giữ ngẫu nhiên 70% ô của `low_rank_matrix` dưới dạng CSR.

    Args:
        low_rank_matrix: Ma trận đầy đủ.

    Returns:
        Ma trận CSR các ô quan sát được.
    """
    mask = np.random.default_rng(1).random(low_rank_matrix.shape) < 0.7
    return sp.csr_matrix(low_rank_matrix * mask)


def test_bias_baseline_recovers_known_biases() -> None:
    """Với r_ui = μ + b_u + b_i đầy đủ và reg = 0, RMSE phải gần 0."""
    # TODO: tạo ma trận cộng tính, fit BiasBaseline(reg=0.0).
    raise NotImplementedError


def test_cholesky_matches_numpy() -> None:
    """`cholesky_solve` phải khớp `np.linalg.solve` trên hệ SPD ngẫu nhiên."""
    # TODO: A = BᵀB + I; so sánh nghiệm với sai số nhỏ.
    raise NotImplementedError


def test_cholesky_rejects_non_spd() -> None:
    """`cholesky_factor` phải báo lỗi với ma trận không xác định dương."""
    # TODO: dùng np.diag([1.0, -1.0]), kiểm tra LinAlgError.
    raise NotImplementedError


def test_svd_recovers_full_low_rank(low_rank_matrix: np.ndarray) -> None:
    """Ma trận đầy đủ hạng 3 phải được khôi phục đúng với k = 3."""
    import numpy as np
    import scipy.sparse as sp
    from src.models.svd import TruncatedSVDRecommender

    # Tự tạo một baseline giả ngay bên trong hàm test để SVD gọi lệnh fit không bị lỗi,
    # đồng thời không đụng chạm đến class _ZeroBaseline bên ngoài của người khác.
    class DummyBaseline:
        rating_min = -100.0
        rating_max = 100.0
        def fit(self, train, val=None): 
            return self
        def predict_raw(self, users, items): 
            return np.zeros(len(users))
        def residual_matrix(self, ratings): 
            return ratings.copy()

    # 1. Chuyển ma trận hạng 3 thành dạng thưa (sparse CSR)
    train = sp.csr_matrix(low_rank_matrix)

    # 2. Khởi tạo mô hình SVD với k=3 và baseline giả[cite: 3]
    model = TruncatedSVDRecommender(k=3, baseline=DummyBaseline())
    model.fit(train)

    # 3. Lấy toạ độ các ô và tính điểm dự đoán
    users, items = np.nonzero(train.toarray())
    predicted = model.predict_raw(users, items)

    # 4. Kiểm tra sai số tái tạo ma trận (phải cực kỳ nhỏ)
    error = np.sqrt(np.mean((predicted - low_rank_matrix[users, items]) ** 2))
    assert error < 1e-7


def test_als_objective_never_increases(observed: sp.csr_matrix) -> None:
    """Hàm mục tiêu ALS không được tăng qua các vòng (tính chất của bước nghiệm đúng)."""
    # TODO: đọc cột objective trong model.log.
    raise NotImplementedError


def test_als_recovers_missing_entries(
    low_rank_matrix: np.ndarray,
    observed: sp.csr_matrix,
) -> None:
    """ALS với k = 3, λ nhỏ phải dự đoán đúng các ô bị ẩn."""
    # TODO: tính RMSE trên phần bù của observed.
    raise NotImplementedError


def test_sgd_gradient_matches_numeric() -> None:
    """Gradient tay phải khớp sai phân trung tâm (Ch.5)."""
    # TODO: so sánh batch_gradients với (L(x+h) - L(x-h)) / 2h trên vài phần tử.
    raise NotImplementedError


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_sgd_loss_decreases(observed: sp.csr_matrix, optimizer: str) -> None:
    """RMSE train cuối phải thấp hơn lúc đầu với cả ba optimizer."""
    # TODO: đọc cột train_rmse trong model.log.
    raise NotImplementedError

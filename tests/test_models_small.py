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
    from src.models.baseline import BiasBaseline

    # Bias có trung bình 0 nên μ, b_u, b_i được khôi phục đúng từng giá trị
    mu0 = 3.5
    bu = np.array([0.5, -0.25, -0.25, 0.0])
    bi = np.array([1.0, -0.5, -0.5])
    train = sp.csr_matrix(mu0 + bu[:, None] + bi[None, :])

    model = BiasBaseline(reg=0.0).fit(train)

    assert model.mu == pytest.approx(mu0)
    np.testing.assert_allclose(model.b_u, bu, atol=1e-12)
    np.testing.assert_allclose(model.b_i, bi, atol=1e-12)
    users, items = np.meshgrid(np.arange(4), np.arange(3), indexing="ij")
    pred = model.predict_raw(users.ravel(), items.ravel())
    assert np.sqrt(np.mean((pred - train.toarray().ravel()) ** 2)) < 1e-12


def test_bias_baseline_matches_loop_formula() -> None:
    """Với reg = 10 trên ma trận thưa, kết quả khớp công thức viết bằng vòng lặp."""
    from src.models.baseline import BiasBaseline

    rng = np.random.default_rng(0)
    dense = np.where(rng.random((20, 15)) < 0.5, rng.integers(1, 6, (20, 15)), 0).astype(float)
    reg = 10.0
    model = BiasBaseline(reg=reg).fit(sp.csr_matrix(dense))

    mu = dense[dense > 0].mean()
    b_i = np.zeros(15)
    for i in range(15):
        mask = dense[:, i] > 0
        b_i[i] = (dense[mask, i] - mu).sum() / (mask.sum() + reg)
    b_u = np.zeros(20)
    for u in range(20):
        mask = dense[u] > 0
        b_u[u] = (dense[u, mask] - mu - b_i[mask]).sum() / (mask.sum() + reg)

    assert model.mu == pytest.approx(mu)
    np.testing.assert_allclose(model.b_i, b_i)
    np.testing.assert_allclose(model.b_u, b_u)


def test_bias_baseline_zero_reg_has_no_nan() -> None:
    """Phim/user không có đánh giá và reg = 0 phải cho bias 0, không phải NaN."""
    from src.models.baseline import BiasBaseline

    train = sp.csr_matrix(np.array([[5.0, 0.0, 3.0], [4.0, 0.0, 2.0], [0.0, 0.0, 0.0]]))
    model = BiasBaseline(reg=0.0).fit(train)

    assert np.isfinite(model.b_i).all() and np.isfinite(model.b_u).all()
    assert model.b_i[1] == 0.0 and model.b_u[2] == 0.0


def test_bias_baseline_residual_matrix() -> None:
    """`residual_matrix` giữ cấu trúc thưa và cộng lại phần dự đoán thì ra điểm gốc."""
    from src.models.baseline import BiasBaseline

    rng = np.random.default_rng(2)
    dense = np.where(rng.random((12, 9)) < 0.5, rng.integers(1, 6, (12, 9)), 0).astype(float)
    train = sp.csr_matrix(dense)
    model = BiasBaseline(reg=10.0).fit(train)

    res = model.residual_matrix(train)
    assert sp.isspmatrix_csr(res) and res.nnz == train.nnz
    coo = train.tocoo()
    back = np.asarray(res[coo.row, coo.col]).ravel() + model.predict_raw(coo.row, coo.col)
    np.testing.assert_allclose(back, coo.data)


def test_global_mean_predicts_mu_and_clips() -> None:
    """GlobalMean dự đoán μ cho mọi ô và có clip về thang điểm."""
    from src.models.baseline import GlobalMean

    train = sp.csr_matrix(np.array([[5.0, 3.0], [4.0, 0.0]]))
    model = GlobalMean().fit(train)

    assert model.mu == pytest.approx(4.0)
    np.testing.assert_allclose(model.predict(np.array([0, 1]), np.array([0, 1])), 4.0)
    clipped = GlobalMean(rating_max=3.5).fit(train)
    assert clipped.predict(np.array([0]), np.array([0]))[0] == 3.5


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
    # TODO: so sánh tái tạo với low_rank_matrix.
    raise NotImplementedError


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
    from src.models.sgd import batch_gradients

    rng = np.random.default_rng(42)

    # Ma trận nhỏ: 3 user, 4 phim, mỗi đối tượng có 2 đặc trưng.
    P = rng.normal(size=(3, 2))
    Q = rng.normal(size=(4, 2))

    # Cố ý cho user và phim lặp để kiểm tra cộng dồn.
    users = np.array([0, 0, 1, 2])
    items = np.array([1, 2, 1, 3])
    residuals = np.array([0.5, -0.3, 1.2, 0.7])
    reg = 0.02

    # Loss phải khớp quy ước tổng theo batch của batch_gradients.
    def loss():
        p = P[users]
        q = Q[items]
        errors = residuals - np.sum(p * q, axis=1)

        squared_error = np.sum(errors ** 2)
        penalty = reg * (np.sum(p ** 2) + np.sum(q ** 2))

        return 0.5 * (squared_error + penalty)

    # Gradient tính bằng công thức.
    grad_P, grad_Q = batch_gradients(
        P, Q, users, items, residuals, reg
    )

    # Gradient tính bằng sai phân trung tâm.
    h = 1e-6

    for param, analytic_grad in [(P, grad_P), (Q, grad_Q)]:
        numeric_grad = np.zeros_like(param)

        for index in np.ndindex(param.shape):
            original = param[index]

            param[index] = original + h
            loss_plus = loss()

            param[index] = original - h
            loss_minus = loss()

            param[index] = original

            numeric_grad[index] = (loss_plus - loss_minus) / (2 * h)

        np.testing.assert_allclose(
            analytic_grad,
            numeric_grad,
            rtol=1e-5,
            atol=1e-7,
        )


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_sgd_loss_decreases(optimizer, monkeypatch):
    from src.models.sgd import SGDRecommender
    from src.evaluation.convergence import ConvergenceLog

    # Baseline đơn giản: luôn dự đoán 3 điểm.
    class ConstantBaseline:
        rating_min = 1.0
        rating_max = 5.0

        def predict_raw(self, users, items):
            return np.full(np.asarray(users).shape, 3.0)

    # Thay hàm ghi log chỉ trong test này.
    def temporary_record(
        self, epoch, elapsed_s, train_rmse,
        val_rmse=None, objective=None
    ):
        self.rows.append({
            "epoch": epoch,
            "elapsed_s": elapsed_s,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "objective": objective,
        })

    monkeypatch.setattr(
        ConvergenceLog, "record", temporary_record
    )

    # Dữ liệu nhỏ có quy luật, điểm đều nằm trong khoảng 1–5.
    user_factors = np.array([-1.0, -0.5, 0.5, 1.0])
    item_factors = np.array([-1.0, 0.5, 1.0])

    ratings = 3.0 + np.outer(user_factors, item_factors)
    train = sp.csr_matrix(ratings)

    learning_rates = {
        "sgd": 0.05,
        "momentum": 0.02,
        "adam": 0.02,
    }

    model = SGDRecommender(
        k=2,
        reg=0.001,
        lr=learning_rates[optimizer],
        n_epochs=100,
        batch_size=4,
        baseline=ConstantBaseline(),
        optimizer=optimizer,
        seed=42,
    )

    model.fit(train)

    initial_rmse = model.log.rows[0]["train_rmse"]
    final_rmse = model.log.rows[-1]["train_rmse"]

    print(
        f"\n{optimizer}: "
        f"RMSE ban đầu = {initial_rmse:.4f}, "
        f"RMSE cuối = {final_rmse:.4f}"
    )

    assert np.isfinite(final_rmse)
    assert final_rmse < initial_rmse

@pytest.mark.parametrize(
    "optimizer, expected",
    [
        ("sgd", 0.6),
        ("momentum", 0.42),
        ("adam", 0.8),
    ],
)
def test_optimizer_two_steps(optimizer, expected):
    from src.models.sgd import OptimizerState, apply_update

    param = np.array([[1.0]])
    grad = np.array([[2.0]])

    state = OptimizerState(
        velocity=np.zeros_like(param),
        second_moment=np.zeros_like(param),
    )

    # Cập nhật hai lần để kiểm tra cả trạng thái tích lũy.
    for _ in range(2):
        apply_update(
            param=param,
            grad=grad,
            state=state,
            optimizer=optimizer,
            lr=0.1,
            hyper={},
        )

    np.testing.assert_allclose(
        param,
        [[expected]],
        rtol=1e-6,
        atol=1e-8,
    )

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


class _ZeroBaseline:
    """Baseline luôn dự đoán 0, để test ALS không phụ thuộc `BiasBaseline`."""

    rating_min = -100.0
    rating_max = 100.0

    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Trả về mảng 0 cùng kích thước `users`."""
        return np.zeros(len(users))

    def residual_matrix(self, ratings: sp.csr_matrix) -> sp.csr_matrix:
        """Phần dư chính là điểm vì baseline bằng 0."""
        return ratings.copy()


class _MemoryLog:
    """Nhật ký tối giản, để test ALS không phụ thuộc `ConvergenceLog`."""

    def __init__(self) -> None:
        """Khởi tạo danh sách rỗng."""
        self.rows: list[dict[str, float | None]] = []

    def record(self, **row: float | None) -> None:
        """Lưu một dòng nhật ký."""
        self.rows.append(row)


def _fit_als(train: sp.csr_matrix, k: int, reg: float, n_epochs: int):
    """Fit ALS với baseline 0 và nhật ký tối giản.

    Args:
        train: Ma trận điểm.
        k: Số nhân tố ẩn.
        reg: Hệ số λ.
        n_epochs: Số vòng lặp.

    Returns:
        Mô hình ALS đã fit.
    """
    from src.models import als

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(als, "ConvergenceLog", _MemoryLog)
        model = als.ALSRecommender(
            k=k, reg=reg, n_epochs=n_epochs, baseline=_ZeroBaseline(), seed=0
        )
        return model.fit(train)


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
    from src.solvers.linear import cholesky_factor, cholesky_solve

    rng = np.random.default_rng(0)
    for k in (1, 2, 5, 20):
        B = rng.normal(size=(k + 3, k))
        A = B.T @ B + np.eye(k)
        b = rng.normal(size=k)
        np.testing.assert_allclose(cholesky_solve(A, b), np.linalg.solve(A, b), atol=1e-10)
        np.testing.assert_allclose(cholesky_factor(A), np.linalg.cholesky(A), atol=1e-10)

    # Ví dụ 6 của Ch.2, đáp án tính tay trong giáo trình.
    worked = np.array([[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]])
    expected = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 1.0, 2.0]])
    np.testing.assert_allclose(cholesky_factor(worked), expected)


def test_cholesky_rejects_non_spd() -> None:
    """`cholesky_factor` phải báo lỗi với ma trận không xác định dương."""
    from src.solvers.linear import cholesky_factor

    with pytest.raises(np.linalg.LinAlgError):
        cholesky_factor(np.diag([1.0, -1.0]))
    # Ma trận suy biến, cùng dạng Ví dụ 7 của Ch.3. Dùng số 1 để trụ thứ hai bằng đúng 0,
    # không phụ thuộc sai số làm tròn của căn bậc hai.
    with pytest.raises(np.linalg.LinAlgError):
        cholesky_factor(np.array([[1.0, 1.0], [1.0, 1.0]]))


def test_svd_recovers_full_low_rank(low_rank_matrix: np.ndarray) -> None:
    """Ma trận đầy đủ hạng 3 phải được khôi phục đúng với k = 3."""
    # TODO: so sánh tái tạo với low_rank_matrix.
    raise NotImplementedError


def test_als_objective_never_increases(observed: sp.csr_matrix) -> None:
    """Hàm mục tiêu ALS không được tăng qua các vòng (tính chất của bước nghiệm đúng)."""
    from src.models.als import objective

    model = _fit_als(observed, k=3, reg=1.0, n_epochs=30)
    values = [row["objective"] for row in model.log.rows]
    assert len(values) == 30
    assert all(after <= before * (1 + 1e-10) for before, after in zip(values, values[1:]))
    assert values[-1] < values[0]
    # Giá trị đã ghi phải đúng bằng công thức đầu file.
    expected = objective(model.P, model.Q, observed, 1.0)
    assert values[-1] == pytest.approx(expected)


def test_als_recovers_missing_entries(
    low_rank_matrix: np.ndarray,
    observed: sp.csr_matrix,
) -> None:
    """ALS với k = 3, λ nhỏ phải dự đoán đúng các ô bị ẩn."""
    model = _fit_als(observed, k=3, reg=1e-6, n_epochs=100)
    hidden = observed.toarray() == 0
    users, items = np.nonzero(hidden)
    predicted = model.predict(users, items)
    error = np.sqrt(np.mean((predicted - low_rank_matrix[users, items]) ** 2))
    assert error < 1e-2


def test_als_needs_positive_reg_when_user_has_few_ratings() -> None:
    """User có ít đánh giá hơn k: hệ suy biến khi λ = 0, giải được khi λ > 0 (Ch.3)."""
    from src.models.als import update_factors

    fixed = np.random.default_rng(0).normal(size=(5, 3))
    # Hàng rỗng cho A = 0 nên trụ bằng đúng 0. Với 1 đánh giá, trụ phụ thuộc làm tròn và
    # có thể là số dương rất nhỏ, nên không dùng để kiểm tra lỗi.
    with pytest.raises(np.linalg.LinAlgError):
        update_factors(fixed, sp.csr_matrix((1, 5)), reg=0.0)

    one_rating = sp.csr_matrix(np.array([[2.0, 0.0, 0.0, 0.0, 0.0]]))
    solution = update_factors(fixed, one_rating, reg=1.0)
    q = fixed[0]
    expected = np.linalg.solve(np.outer(q, q) + np.eye(3), 2.0 * q)
    np.testing.assert_allclose(solution[0], expected)


def test_als_results_do_not_depend_on_thread_count(observed: sp.csr_matrix) -> None:
    """Mỗi hệ giải độc lập nên số luồng không được làm đổi kết quả."""
    from numba import get_num_threads, set_num_threads

    original = get_num_threads()
    try:
        set_num_threads(1)
        single = _fit_als(observed, k=3, reg=1.0, n_epochs=5)
        set_num_threads(original)
        many = _fit_als(observed, k=3, reg=1.0, n_epochs=5)
    finally:
        set_num_threads(original)
    np.testing.assert_array_equal(single.P, many.P)
    np.testing.assert_array_equal(single.Q, many.Q)


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


def test_sgd_early_stopping_keeps_best_epoch(low_rank_matrix: np.ndarray) -> None:
    """Có val và patience: dừng khi val không giảm, và P, Q là của epoch val tốt nhất."""
    from src.models.sgd import SGDRecommender

    # Train là ô quan sát cộng nhiễu, val là các ô bị ẩn không nhiễu. Không điều chuẩn
    # và k lớn nên mô hình sẽ khớp cả nhiễu, val RMSE ngừng giảm rồi tăng lại.
    rng = np.random.default_rng(3)
    mask = rng.random(low_rank_matrix.shape) < 0.6
    noisy = low_rank_matrix + rng.normal(0.0, 0.5, low_rank_matrix.shape)
    train = sp.csr_matrix(np.where(mask, noisy, 0.0))
    val = sp.csr_matrix(np.where(~mask, low_rank_matrix, 0.0))

    patience = 3
    model = SGDRecommender(
        k=10, reg=0.0, lr=0.01, n_epochs=500, batch_size=64,
        baseline=_ZeroBaseline(), optimizer="sgd", init_std=0.1, seed=0,
        patience=patience,
    ).fit(train, val)

    log = model.log.to_frame()
    best = int(log.loc[log["val_rmse"].idxmin(), "epoch"])
    last = int(log["epoch"].iloc[-1])

    assert model.best_epoch_ == best
    assert last == best + patience < 500
    coo = val.tocoo()
    final = np.sqrt(np.mean((coo.data - model.predict(coo.row, coo.col)) ** 2))
    assert final == pytest.approx(log["val_rmse"].min())


def test_sgd_without_val_runs_all_epochs() -> None:
    """Không có val thì không dừng sớm, chạy đủ n_epochs."""
    from src.models.sgd import SGDRecommender

    train = sp.csr_matrix(np.arange(1.0, 13.0).reshape(4, 3) / 4.0)
    model = SGDRecommender(
        k=2, reg=0.0, lr=0.01, n_epochs=5, batch_size=4,
        baseline=_ZeroBaseline(), optimizer="sgd", seed=0, patience=1,
    ).fit(train)

    assert model.best_epoch_ is None
    assert int(model.log.to_frame()["epoch"].iloc[-1]) == 5

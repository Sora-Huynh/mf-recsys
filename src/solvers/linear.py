"""Bộ giải hệ tuyến tính k×k đối xứng xác định dương.

Mục đích: tự cài để thể hiện kiến thức Ch.2 và để ALS đổi được bộ giải.
- Cholesky (A = LLᵀ): chi phí k³/3, nghiệm chính xác, là lựa chọn mặc định.
- Gauss-Seidel (tùy chọn): lặp, mỗi vòng k²; hội tụ với ma trận đối xứng xác định
  dương, và có thể khởi tạo từ nghiệm vòng ALS trước.

Hàm được gọi hàng nghìn lần mỗi vòng ALS, nên phần tính được biên dịch bằng numba. Các
hàm `_..._core` là phần biên dịch: không cấp phát ma trận và không ném lỗi, nên ALS gọi
được chúng trong vòng lặp song song. Các hàm công khai chỉ kiểm tra đầu vào rồi gọi chúng.

Liên quan Ch.2 (Cholesky, Gauss-Seidel).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit
def _cholesky_core(A: np.ndarray) -> bool:
    """Ghi đè L lên tam giác dưới của A theo công thức Ch.2, tính theo từng cột.

    Args:
        A: Ma trận float64 (k, k); chỉ đọc tam giác dưới. Tam giác trên không được dùng.

    Returns:
        True nếu thành công; False nếu gặp trụ không dương hoặc NaN.
    """
    k = A.shape[0]
    for j in range(k):
        pivot = A[j, j]
        for m in range(j):
            pivot -= A[j, m] * A[j, m]
        # Viết `not pivot > 0` để bắt cả NaN, vì so sánh với NaN luôn sai.
        if not pivot > 0.0:
            return False
        l_jj = np.sqrt(pivot)
        A[j, j] = l_jj
        for i in range(j + 1, k):
            value = A[i, j]
            for m in range(j):
                value -= A[i, m] * A[j, m]
            A[i, j] = value / l_jj
    return True


@njit
def _forward_core(L: np.ndarray, x: np.ndarray) -> None:
    """Thế xuôi tại chỗ: thay x bằng nghiệm của Ly = x.

    Args:
        L: Ma trận có tam giác dưới là L.
        x: Vế phải float64, được ghi đè bằng nghiệm.
    """
    for i in range(x.shape[0]):
        value = x[i]
        for m in range(i):
            value -= L[i, m] * x[m]
        x[i] = value / L[i, i]


@njit
def _backward_transpose_core(L: np.ndarray, x: np.ndarray) -> None:
    """Thế ngược tại chỗ: thay x bằng nghiệm của Lᵀz = x.

    Đọc Lᵀ qua L nên không phải chuyển vị, tránh cấp phát trong vòng lặp ALS.

    Args:
        L: Ma trận có tam giác dưới là L.
        x: Vế phải float64, được ghi đè bằng nghiệm.
    """
    for i in range(x.shape[0] - 1, -1, -1):
        value = x[i]
        for m in range(i + 1, x.shape[0]):
            value -= L[m, i] * x[m]
        x[i] = value / L[i, i]


@njit
def _backward_core(U: np.ndarray, x: np.ndarray) -> None:
    """Thế ngược tại chỗ: thay x bằng nghiệm của Uz = x.

    Args:
        U: Ma trận có tam giác trên là U.
        x: Vế phải float64, được ghi đè bằng nghiệm.
    """
    for i in range(x.shape[0] - 1, -1, -1):
        value = x[i]
        for m in range(i + 1, x.shape[0]):
            value -= U[i, m] * x[m]
        x[i] = value / U[i, i]


def _as_square(A: np.ndarray) -> np.ndarray:
    """Sao chép A sang float64 và kiểm tra A vuông.

    Args:
        A: Ma trận đầu vào.

    Returns:
        Bản sao float64.

    Raises:
        ValueError: Nếu A không vuông.
    """
    matrix = np.array(A, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Cần ma trận vuông, nhận được {matrix.shape}")
    return matrix


def _right_hand_side(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sao chép b sang float64 và kiểm tra khớp kích thước với A.

    Args:
        A: Ma trận hệ số, kích thước (k, k).
        b: Vế phải.

    Returns:
        Bản sao float64 của b.

    Raises:
        ValueError: Nếu b không phải vector dài k.
    """
    x = np.array(b, dtype=np.float64)
    if x.shape != (A.shape[0],):
        raise ValueError(f"Vế phải có kích thước {x.shape}, cần ({A.shape[0]},)")
    return x


def cholesky_factor(A: np.ndarray) -> np.ndarray:
    """Phân rã Cholesky A = LLᵀ.

    Args:
        A: Ma trận đối xứng xác định dương, kích thước (k, k). Chỉ tam giác dưới được đọc.

    Returns:
        Ma trận tam giác dưới L.

    Raises:
        np.linalg.LinAlgError: Nếu A không xác định dương (trụ ≤ 0).
        ValueError: Nếu A không vuông.
    """
    L = _as_square(A)
    if not _cholesky_core(L):
        raise np.linalg.LinAlgError("Ma trận không xác định dương")
    return np.tril(L)


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Giải Ly = b với L tam giác dưới.

    Args:
        L: Ma trận tam giác dưới.
        b: Vế phải.

    Returns:
        Nghiệm y.

    Raises:
        ValueError: Nếu kích thước không khớp.
    """
    matrix = _as_square(L)
    y = _right_hand_side(matrix, b)
    _forward_core(matrix, y)
    return y


def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Giải Ux = y với U tam giác trên.

    Args:
        U: Ma trận tam giác trên.
        y: Vế phải.

    Returns:
        Nghiệm x.

    Raises:
        ValueError: Nếu kích thước không khớp.
    """
    matrix = _as_square(U)
    x = _right_hand_side(matrix, y)
    _backward_core(matrix, x)
    return x


def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Giải Ax = b bằng Cholesky.

    Args:
        A: Ma trận đối xứng xác định dương.
        b: Vế phải.

    Returns:
        Nghiệm x.

    Raises:
        np.linalg.LinAlgError: Nếu A không xác định dương.
        ValueError: Nếu kích thước không khớp.
    """
    L = cholesky_factor(A)
    return backward_substitution(L.T, forward_substitution(L, b))


def gauss_seidel_solve(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> tuple[np.ndarray, int]:
    """Giải Ax = b bằng Gauss-Seidel (tùy chọn).

    Args:
        A: Ma trận đối xứng xác định dương.
        b: Vế phải.
        x0: Nghiệm khởi tạo; None dùng vector 0.
        tol: Ngưỡng dừng theo chuẩn phần dư tương đối.
        max_iter: Số vòng tối đa.

    Returns:
        Cặp (nghiệm, số vòng đã chạy).
    """
    # TODO: chỉ làm nếu dư thời gian.
    raise NotImplementedError

"""Bộ giải hệ tuyến tính k×k đối xứng xác định dương.

Mục đích: tự cài để thể hiện kiến thức Ch.2 và để ALS đổi được bộ giải.
- Cholesky (A = LLᵀ): chi phí k³/3, nghiệm chính xác, là lựa chọn mặc định.
- Gauss-Seidel (tùy chọn): lặp, mỗi vòng k²; hội tụ với ma trận đối xứng xác định
  dương, và có thể khởi tạo từ nghiệm vòng ALS trước.

Hàm được gọi hàng nghìn lần mỗi vòng ALS, nên nên biên dịch bằng numba.

Liên quan Ch.2 (Cholesky, Gauss-Seidel).
"""

from __future__ import annotations

import numpy as np


def cholesky_factor(A: np.ndarray) -> np.ndarray:
    """Phân rã Cholesky A = LLᵀ.

    Args:
        A: Ma trận đối xứng xác định dương, kích thước (k, k).

    Returns:
        Ma trận tam giác dưới L.

    Raises:
        np.linalg.LinAlgError: Nếu A không xác định dương (trụ ≤ 0).
    """
    # TODO: cài theo cột; nếu trụ ≤ 0 thì báo lỗi, không lấy căn số âm.
    raise NotImplementedError


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Giải Ly = b với L tam giác dưới.

    Args:
        L: Ma trận tam giác dưới.
        b: Vế phải.

    Returns:
        Nghiệm y.
    """
    # TODO: thế xuôi.
    raise NotImplementedError


def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Giải Ux = y với U tam giác trên.

    Args:
        U: Ma trận tam giác trên.
        y: Vế phải.

    Returns:
        Nghiệm x.
    """
    # TODO: thế ngược.
    raise NotImplementedError


def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Giải Ax = b bằng Cholesky.

    Args:
        A: Ma trận đối xứng xác định dương.
        b: Vế phải.

    Returns:
        Nghiệm x.
    """
    # TODO: ghép cholesky_factor, forward_substitution, backward_substitution.
    raise NotImplementedError


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

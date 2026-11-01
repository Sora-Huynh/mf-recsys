"""Các hàm vẽ dùng chung.

Mục đích: mọi hình trong báo cáo có cùng phong cách. Quy ước: chữ tiếng Anh, hình rộng,
cỡ chữ lớn, biểu đồ cột ghi giá trị trên đầu cột. Không thuộc chương nào của môn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIGSIZE_WIDE = (14, 6)
FONT_SIZE = 16


def apply_style() -> None:
    """Đặt rcParams chung: kích thước hình, cỡ chữ, lưới."""
    # TODO: cập nhật plt.rcParams theo FIGSIZE_WIDE và FONT_SIZE.
    raise NotImplementedError


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Lưu hình PNG độ phân giải cao, tạo thư mục nếu chưa có.

    Args:
        fig: Hình cần lưu.
        path: Đường dẫn file.
    """
    # TODO: tight_layout, savefig dpi=200.
    raise NotImplementedError


def plot_bars(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    fmt: str = "{:.4f}",
) -> plt.Figure:
    """Vẽ biểu đồ cột có ghi giá trị trên đầu cột.

    Args:
        labels: Nhãn từng cột.
        values: Giá trị từng cột.
        title: Tiêu đề (tiếng Anh).
        ylabel: Nhãn trục y (tiếng Anh).
        fmt: Định dạng số ghi trên cột.

    Returns:
        Hình đã vẽ.
    """
    # TODO: ax.bar + ax.bar_label.
    raise NotImplementedError


def plot_convergence(
    logs: dict[str, pd.DataFrame],
    x: str = "epoch",
    y: str = "val_rmse",
    reference: dict[str, float] | None = None,
) -> plt.Figure:
    """Vẽ đường hội tụ của nhiều mô hình.

    Args:
        logs: Tên mô hình -> bảng từ `ConvergenceLog.to_frame`.
        x: Trục hoành, `epoch` hoặc `elapsed_s`.
        y: Cột cần vẽ.
        reference: Tên -> giá trị, vẽ đường ngang (baseline, SVD).

    Returns:
        Hình đã vẽ.
    """
    # TODO: mỗi mô hình một đường, có legend.
    raise NotImplementedError


def plot_sweep(
    table: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    logx: bool = False,
) -> plt.Figure:
    """Vẽ ảnh hưởng của tham số (k, λ).

    Args:
        table: Bảng kết quả quét.
        x: Cột tham số.
        y: Cột kết quả.
        hue: Cột tách thành nhiều đường.
        logx: True nếu trục x theo thang log.

    Returns:
        Hình đã vẽ.
    """
    # TODO: vẽ đường, đánh dấu điểm tốt nhất.
    raise NotImplementedError


def plot_condition(table: pd.DataFrame) -> plt.Figure:
    """Vẽ số điều kiện theo λ, hai trục log.

    Args:
        table: Bảng từ `condition_vs_lambda`.

    Returns:
        Hình đã vẽ.
    """
    # TODO: loglog kappa_median và kappa_max theo reg.
    raise NotImplementedError

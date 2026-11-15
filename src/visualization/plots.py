"""Các hàm vẽ dùng chung.

Mục đích: mọi hình trong báo cáo có cùng phong cách. Quy ước: chữ tiếng Anh, hình rộng,
cỡ chữ lớn, biểu đồ cột ghi giá trị trên đầu cột. Không thuộc chương nào của môn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

FIGSIZE_WIDE = (14, 6)
FONT_SIZE = 16
SERIES_COLORS = ("#2a78d6", "#eb6834")  # Gán theo thứ tự cố định, không lặp lại theo vòng.
REFERENCE_COLOR = "#52514e"
TEXT_COLOR = "#0b0b0b"
GRID_COLOR = "#e5e4e0"


def apply_style() -> None:
    """Đặt rcParams chung: kích thước hình, cỡ chữ, lưới."""
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE_WIDE,
            "font.size": FONT_SIZE,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": REFERENCE_COLOR,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Lưu hình PNG độ phân giải cao, tạo thư mục nếu chưa có.

    Args:
        fig: Hình cần lưu.
        path: Đường dẫn file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
    labels = {
        "epoch": "Epoch",
        "elapsed_s": "Training time (s)",
        "val_rmse": "Validation RMSE",
        "train_rmse": "Train RMSE",
        "objective": "Objective",
    }
    fig, ax = plt.subplots()
    for color, (name, table) in zip(SERIES_COLORS, logs.items()):
        ax.plot(table[x], table[y], color=color, marker="o", label=name)
    for name, value in (reference or {}).items():
        ax.axhline(value, color=REFERENCE_COLOR, linestyle="--", label=f"{name} ({value:.4f})")
    if x == "epoch":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(labels.get(x, x))
    ax.set_ylabel(labels.get(y, y))
    ax.legend()
    return fig


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
    fig, ax = plt.subplots()
    ax.loglog(table["reg"], table["kappa_max"], color=SERIES_COLORS[0], marker="o", label="Max over users")
    ax.loglog(table["reg"], table["kappa_median"], color=SERIES_COLORS[1], marker="o", label="Median over users")
    ax.set_xlabel("Regularization λ")
    ax.set_ylabel("Condition number κ₂")
    ax.legend()
    return fig

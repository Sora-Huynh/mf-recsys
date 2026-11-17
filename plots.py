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

    if len(labels) == 0 or len(values) == 0:
        raise ValueError("labels và values không được rỗng")

    if len(labels) != len(values):
        raise ValueError("labels và values phải có cùng độ dài")

    numeric_values = [float(value) for value in values]

    if not all(
        pd.notna(value) and abs(value) != float("inf")
        for value in numeric_values
    ):
        raise ValueError("values phải là các số hữu hạn")

    fig, ax = plt.subplots()

    bars = ax.bar(
        labels,
        numeric_values,
        color=SERIES_COLORS[0],
        width=0.65,
    )

    ax.bar_label(
        bars,
        labels=[fmt.format(value) for value in numeric_values],
        padding=4,
        fontsize=FONT_SIZE - 2,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y")

    ax.tick_params(
        axis="x",
        labelrotation=15 if len(labels) > 4 else 0,
    )

    lower = min(numeric_values)
    upper = max(numeric_values)

    span = max(
        upper - lower,
        abs(upper),
        abs(lower),
        1.0,
    )

    bottom = 0.0 if lower >= 0 else lower - 0.08 * span
    top = upper + 0.20 * span if upper > 0 else 0.0

    ax.set_ylim(
        bottom,
        top if top > bottom else bottom + 1.0,
    )

    return fig


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
    required = [x, y] + ([hue] if hue is not None else [])

    missing = [
        column for column in required
        if column not in table.columns
    ]

    if missing:
        raise ValueError(f"Thiếu cột trong bảng sweep: {missing}")

    if table.empty:
        raise ValueError("Bảng sweep không được rỗng")

    data = table[required].copy()

    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = pd.to_numeric(data[y], errors="coerce")

    data = data.dropna(subset=[x, y])

    if data.empty:
        raise ValueError("Bảng sweep không có cặp x/y hợp lệ")

    if not data[x].map(
        lambda value: abs(value) != float("inf")
    ).all():
        raise ValueError(f"Cột {x!r} chứa giá trị không hữu hạn")

    if not data[y].map(
        lambda value: abs(value) != float("inf")
    ).all():
        raise ValueError(f"Cột {y!r} chứa giá trị không hữu hạn")

    if logx and (data[x] <= 0).any():
        raise ValueError(
            "logx=True yêu cầu mọi giá trị trên trục x phải dương"
        )

    fig, ax = plt.subplots()

    if hue is None:
        groups = [(None, data)]
    else:
        groups = list(data.groupby(hue, sort=False, dropna=False))

    for index, (name, subset) in enumerate(groups):

        color = (
            SERIES_COLORS[index]
            if index < len(SERIES_COLORS)
            else REFERENCE_COLOR
        )

        subset = subset.sort_values(x)

        ax.plot(
            subset[x],
            subset[y],
            color=color,
            marker="o",
            label=str(name) if hue is not None else None,
        )

    best = data.loc[data[y].idxmin()]

    ax.scatter(
        [best[x]],
        [best[y]],
        color=REFERENCE_COLOR,
        marker="*",
        s=190,
        zorder=5,
        label=f"Best ({best[y]:.4f})",
    )

    axis_labels = {
        "k": "Latent factors (k)",
        "reg": "Regularization λ",
        "val_rmse": "Validation RMSE",
        "test_rmse": "Test RMSE",
        "fit_time_s": "Training time (s)",
    }

    ax.set_xlabel(axis_labels.get(x, x))
    ax.set_ylabel(axis_labels.get(y, y))

    if logx:
        ax.set_xscale("log")

    if x == "k":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if hue is not None or len(groups) == 1:
        ax.legend()

    return fig


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

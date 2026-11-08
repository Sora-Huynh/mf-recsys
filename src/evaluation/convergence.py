"""Ghi loss/RMSE theo vòng lặp và thời gian.

Mục đích: so sánh ALS và SGD cần cả hai trục. Một vòng ALS đắt hơn nhiều so với một
epoch SGD, nên chỉ nhìn theo vòng lặp sẽ gây hiểu nhầm; phải có trục thời gian.
Thời gian chỉ tính phần huấn luyện, không tính phần đánh giá RMSE.

SVD không lặp nên không có đường hội tụ; trên biểu đồ vẽ thành đường ngang.

Liên quan Ch.6 (đánh giá hội tụ).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ConvergenceLog:
    """Nhật ký hội tụ của một lần huấn luyện.

    Attributes:
        rows: Mỗi phần tử là một dict `epoch`, `elapsed_s`, `train_rmse`, `val_rmse`, `objective`.
    """

    rows: list[dict[str, float]] = field(default_factory=list)

    def record(
        self,
        epoch: int,
        elapsed_s: float,
        train_rmse: float,
        val_rmse: float | None = None,
        objective: float | None = None,
    ) -> None:
        """Thêm một dòng nhật ký.

        Args:
            epoch: Số thứ tự vòng lặp, bắt đầu từ 1.
            elapsed_s: Tổng thời gian huấn luyện tích lũy (giây).
            train_rmse: RMSE trên train.
            val_rmse: RMSE trên val.
            objective: Giá trị hàm mục tiêu.
        """
        self.rows.append(
            {
                "epoch": epoch,
                "elapsed_s": elapsed_s,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "objective": objective,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        """Đổi nhật ký thành DataFrame.

        Returns:
            DataFrame mỗi dòng một vòng lặp.
        """
        return pd.DataFrame(self.rows)

    def save(self, path: Path) -> None:
        """Lưu nhật ký ra CSV.

        Args:
            path: Đường dẫn file, thường trong `results/tables/`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)

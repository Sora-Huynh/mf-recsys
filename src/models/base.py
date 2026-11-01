"""Lớp trừu tượng cho mọi mô hình gợi ý.

Mục đích: cố định một giao diện chung để thí nghiệm và demo không phụ thuộc mô hình
cụ thể. File này là "hợp đồng" giữa các thành viên, cần chốt trước khi cài mô hình.
Không thuộc chương nào của môn.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import scipy.sparse as sp


class Recommender(ABC):
    """Giao diện chung của các mô hình dự đoán điểm.

    Quy ước: mọi chỉ số `user`, `item` là chỉ số liên tục, không phải ID gốc.
    """

    @abstractmethod
    def fit(
        self,
        train: sp.csr_matrix,
        val: sp.csr_matrix | None = None,
    ) -> "Recommender":
        """Huấn luyện mô hình.

        Args:
            train: Ma trận điểm train; ô không lưu nghĩa là chưa đánh giá.
            val: Ma trận điểm val, chỉ dùng để theo dõi hội tụ.

        Returns:
            Chính mô hình, để viết `model.fit(...).predict(...)`.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Dự đoán điểm cho các cặp (user, phim).

        Args:
            users: Mảng chỉ số user, dài n.
            items: Mảng chỉ số phim, dài n.

        Returns:
            Mảng dài n, đã clip về thang điểm [1, 5]. Phải vector hóa.
        """
        raise NotImplementedError

    def recommend(self, user: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Trả về n phim điểm dự đoán cao nhất mà user chưa xem trong train.

        Args:
            user: Chỉ số user.
            n: Số phim cần gợi ý.

        Returns:
            Cặp (chỉ số phim, điểm dự đoán), sắp giảm dần theo điểm.
        """
        # TODO: predict toàn bộ phim, loại phim đã có trong train, np.argpartition lấy top n.
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Lưu mô hình đã huấn luyện để demo không phải train lại.

        Args:
            path: Đường dẫn file, thường trong `results/models/`.
        """
        # TODO: pickle toàn bộ đối tượng.
        raise NotImplementedError

    @staticmethod
    def load(path: Path) -> "Recommender":
        """Đọc mô hình đã lưu.

        Args:
            path: Đường dẫn file.

        Returns:
            Mô hình đã huấn luyện.
        """
        # TODO: đọc pickle.
        raise NotImplementedError

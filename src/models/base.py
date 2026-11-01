"""Lớp trừu tượng cho mọi mô hình gợi ý.

Mục đích: cố định một giao diện chung để thí nghiệm và demo không phụ thuộc mô hình
cụ thể. File này là "hợp đồng" giữa các thành viên, đổi chữ ký phải báo cả nhóm.
Không thuộc chương nào của môn.

Mẫu template method: bên ngoài gọi `fit`, `predict`, `recommend`; mô hình con chỉ cài
`_fit` và `predict_raw`. Lý do: việc clip điểm và lưu ma trận train phải giống hệt nhau
ở mọi mô hình, nếu không RMSE không còn so sánh công bằng được.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
import scipy.sparse as sp


class Recommender(ABC):
    """Giao diện chung của các mô hình dự đoán điểm.

    Quy ước: mọi chỉ số `user`, `item` là chỉ số liên tục, không phải ID gốc.

    Attributes:
        rating_min: Cận dưới khi clip.
        rating_max: Cận trên khi clip.
        train_: Ma trận train CSR, được gán khi `fit`; None nếu chưa huấn luyện.
    """

    def __init__(self, rating_min: float = 1.0, rating_max: float = 5.0) -> None:
        """Khởi tạo phần dùng chung; lớp con phải gọi `super().__init__`.

        Args:
            rating_min: Cận dưới thang điểm.
            rating_max: Cận trên thang điểm.
        """
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.train_: sp.csr_matrix | None = None

    def fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None = None) -> Self:
        """Huấn luyện mô hình.

        `train_` được gán trước `_fit` để `predict` dùng được trong lúc huấn luyện, phục
        vụ ghi đường hội tụ.

        Args:
            train: Ma trận điểm train; ô không lưu nghĩa là chưa đánh giá.
            val: Ma trận điểm val, chỉ dùng để theo dõi hội tụ.

        Returns:
            Chính mô hình, để viết `model.fit(...).predict(...)`.
        """
        self.train_ = train.tocsr()
        self._fit(self.train_, val)
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Dự đoán điểm cho các cặp (user, phim), đã clip về thang điểm.

        Args:
            users: Mảng chỉ số user, dài n.
            items: Mảng chỉ số phim, dài n.

        Returns:
            Mảng dài n trong khoảng [rating_min, rating_max].

        Raises:
            RuntimeError: Nếu chưa `fit`.
            ValueError: Nếu `users` và `items` khác kích thước.
        """
        self._check_fitted()
        users = np.asarray(users)
        items = np.asarray(items)
        if users.shape != items.shape:
            raise ValueError(f"Khác kích thước: {users.shape} và {items.shape}")
        return np.clip(self.predict_raw(users, items), self.rating_min, self.rating_max)

    def recommend(self, user: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Trả về n phim điểm dự đoán cao nhất mà user chưa đánh giá trong train.

        Xếp hạng theo điểm chưa clip, vì sau clip nhiều phim cùng bằng 5 và thứ tự mất ý
        nghĩa. Dùng sắp xếp ổn định để kết quả tái lập được khi có điểm bằng nhau.

        Args:
            user: Chỉ số user.
            n: Số phim cần gợi ý.

        Returns:
            Cặp (chỉ số phim, điểm đã clip), giảm dần theo điểm chưa clip. Có thể ít hơn
            n phần tử nếu user đã đánh giá gần hết phim.

        Raises:
            RuntimeError: Nếu chưa `fit`.
            IndexError: Nếu `user` nằm ngoài khoảng hợp lệ.
        """
        self._check_fitted()
        n_users, n_items = self.train_.shape
        if not 0 <= user < n_users:
            raise IndexError(f"user {user} ngoài khoảng [0, {n_users})")

        items = np.arange(n_items)
        scores = self.predict_raw(np.full(n_items, user), items)
        seen = self.train_.indices[self.train_.indptr[user] : self.train_.indptr[user + 1]]
        candidates = np.delete(items, seen)
        top = candidates[np.argsort(-scores[candidates], kind="stable")[:n]]
        return top, np.clip(scores[top], self.rating_min, self.rating_max)

    def save(self, path: Path) -> None:
        """Lưu mô hình đã huấn luyện để demo không phải train lại.

        Args:
            path: Đường dẫn file, thường trong `results/models/`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Recommender":
        """Đọc mô hình đã lưu. Pickle có thể chạy code, chỉ mở file do nhóm tự tạo.

        Args:
            path: Đường dẫn file.

        Returns:
            Mô hình đã huấn luyện.

        Raises:
            TypeError: Nếu file không chứa `Recommender`.
        """
        with Path(path).open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, Recommender):
            raise TypeError(f"{path} không chứa Recommender")
        return model

    def _check_fitted(self) -> None:
        """Báo lỗi rõ ràng nếu mô hình chưa huấn luyện.

        Raises:
            RuntimeError: Nếu chưa `fit`.
        """
        if self.train_ is None:
            raise RuntimeError(f"{type(self).__name__} chưa được fit")

    @abstractmethod
    def _fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None) -> None:
        """Học tham số; lớp con cài.

        Args:
            train: Ma trận điểm train dạng CSR.
            val: Ma trận điểm val, có thể None.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Tính điểm chưa clip; lớp con cài, phải vector hóa.

        Args:
            users: Mảng chỉ số user, dài n.
            items: Mảng chỉ số phim, dài n.

        Returns:
            Mảng dài n.
        """
        raise NotImplementedError

"""Kiểm thử hợp đồng của `Recommender` bằng một mô hình giả biết trước điểm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from src.models.base import Recommender


class FixedScores(Recommender):
    """Mô hình giả: điểm của phim i là `scores[i]`, với mọi user."""

    def __init__(self, scores: list[float]) -> None:
        super().__init__()
        self.scores = np.asarray(scores, dtype=np.float64)

    def _fit(self, train: sp.csr_matrix, val: sp.csr_matrix | None) -> None:
        pass

    def predict_raw(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self.scores[items]


@pytest.fixture
def train() -> sp.csr_matrix:
    """User 0 đã xem phim 1 và 3; user 1 đã xem cả 5 phim."""
    return sp.csr_matrix(
        [
            [0.0, 4.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    )


@pytest.fixture
def model(train: sp.csr_matrix) -> FixedScores:
    """Điểm chưa clip có hai giá trị vượt 5 và một giá trị dưới 1."""
    return FixedScores([5.5, 9.0, 0.5, 3.0, 7.0]).fit(train)


def test_abstract_class_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Recommender()


def test_fit_returns_self_and_stores_csr(train: sp.csr_matrix) -> None:
    model = FixedScores([3.0] * 5)
    assert model.fit(train.tocoo()) is model
    assert sp.issparse(model.train_) and model.train_.format == "csr"


def test_predict_clips_to_rating_scale(model: FixedScores) -> None:
    result = model.predict(np.array([0, 0, 0]), np.array([0, 2, 3]))
    np.testing.assert_array_equal(result, [5.0, 1.0, 3.0])


def test_predict_rejects_mismatched_shapes(model: FixedScores) -> None:
    with pytest.raises(ValueError):
        model.predict(np.array([0, 1]), np.array([0]))


def test_unfitted_model_raises() -> None:
    model = FixedScores([3.0])
    with pytest.raises(RuntimeError):
        model.predict(np.array([0]), np.array([0]))
    with pytest.raises(RuntimeError):
        model.recommend(0)


def test_recommend_excludes_seen_and_ranks_by_raw_score(model: FixedScores) -> None:
    items, scores = model.recommend(0, n=2)
    # Phim 4 (7.0) phải đứng trước phim 0 (5.5) dù cả hai đều clip về 5.
    np.testing.assert_array_equal(items, [4, 0])
    np.testing.assert_array_equal(scores, [5.0, 5.0])


def test_recommend_returns_fewer_when_few_unseen(model: FixedScores) -> None:
    assert len(model.recommend(0, n=10)[0]) == 3
    assert len(model.recommend(1, n=10)[0]) == 0


def test_recommend_breaks_ties_by_item_index(train: sp.csr_matrix) -> None:
    items, _ = FixedScores([3.0] * 5).fit(train).recommend(0, n=3)
    np.testing.assert_array_equal(items, [0, 2, 4])


@pytest.mark.parametrize("user", [-1, 2])
def test_recommend_rejects_unknown_user(model: FixedScores, user: int) -> None:
    with pytest.raises(IndexError):
        model.recommend(user)


def test_save_and_load_roundtrip(model: FixedScores, tmp_path: Path) -> None:
    path = tmp_path / "models" / "fixed.pkl"
    model.save(path)
    loaded = Recommender.load(path)
    np.testing.assert_array_equal(loaded.recommend(0)[0], model.recommend(0)[0])

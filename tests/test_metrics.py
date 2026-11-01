"""Kiểm thử RMSE, MAE trên các giá trị tính tay được."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import mae, rmse


def test_perfect_prediction_gives_zero() -> None:
    y = [1.0, 3.0, 5.0]
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0


def test_known_values() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [2.0, 2.0, 3.0, 2.0]
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(5 / 4))
    assert mae(y_true, y_pred) == pytest.approx(3 / 4)


def test_rmse_is_at_least_mae() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.uniform(1, 5, size=100)
    y_pred = rng.uniform(1, 5, size=100)
    assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


@pytest.mark.parametrize("metric", [rmse, mae])
def test_shape_mismatch_raises(metric) -> None:
    with pytest.raises(ValueError):
        metric([1.0, 2.0], [1.0])


@pytest.mark.parametrize("metric", [rmse, mae])
def test_empty_raises(metric) -> None:
    with pytest.raises(ValueError):
        metric([], [])

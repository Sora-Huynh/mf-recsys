
"""Quét k và λ trên tập val.

Mục đích: chọn tham số bằng val, không bằng test, để RMSE test cuối cùng
không bị lạc quan.

Quét gọn:
- ALS: quét k rồi quét λ riêng, không làm lưới đầy đủ.
- SVD: chỉ quét k, phân rã một lần tại k lớn nhất.

Liên quan Ch.3 và Ch.4.

Chạy: python -m experiments.tune_k_lambda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.data.sparse import load_csr
from src.evaluation.metrics import rmse
from src.models.als import ALSRecommender
from src.models.baseline import BiasBaseline
from src.models.svd import TruncatedSVDRecommender
from src.visualization.plots import (
    apply_style,
    plot_sweep,
    save_figure,
)


def _load_data(cfg: dict[str, Any]):
    """Đọc train và val. Không sử dụng test."""

    directory = Path(cfg["paths"]["processed_dir"])

    train = load_csr(directory / "train_csr.npz").tocsr()
    val = pd.read_parquet(directory / "val.parquet")

    required = {"user_idx", "item_idx", "rating"}

    if not required.issubset(val.columns):
        raise ValueError("Validation thiếu cột bắt buộc")

    if val.empty:
        raise ValueError("Validation rỗng")

    users = val["user_idx"].to_numpy(dtype=np.int64)
    items = val["item_idx"].to_numpy(dtype=np.int64)
    ratings = val["rating"].to_numpy(dtype=np.float64)

    if not np.isfinite(ratings).all():
        raise ValueError("Validation chứa NaN hoặc Inf")

    if (
        users.min() < 0
        or users.max() >= train.shape[0]
        or items.min() < 0
        or items.max() >= train.shape[1]
    ):
        raise ValueError("Chỉ số validation nằm ngoài train")

    return train, users, items, ratings


def _make_baseline(cfg, train):
    """Huấn luyện baseline trên train."""

    baseline = BiasBaseline(
        reg=cfg["baseline"]["reg"],
        rating_min=cfg["ratings"]["min"],
        rating_max=cfg["ratings"]["max"],
    )

    baseline.fit(train)
    return baseline


def _positive_grid(values, name, integer=False):
    """Kiểm tra và loại bỏ tham số trùng."""

    result = []

    for value in values:
        number = float(value)

        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{name} phải chứa số dương hữu hạn")

        if integer:
            if not number.is_integer():
                raise ValueError(f"{name} phải chứa số nguyên")
            number = int(number)

        if number not in result:
            result.append(number)

    if not result:
        raise ValueError(f"{name} không được rỗng")

    return result


def sweep_als(cfg: dict[str, Any]) -> pd.DataFrame:
    """Quét k (giữ λ mặc định) rồi quét λ (giữ k mặc định) cho ALS.

    Args:
        cfg: Cấu hình từ `load_config`.

    Returns:
        DataFrame cột `k`, `reg`, `val_rmse`, `fit_time_s`.
    """

    train, users, items, ratings = _load_data(cfg)

    settings = cfg["als"]
    tune = cfg["tune"]

    k_grid = _positive_grid(
        tune["k_grid"], "k_grid", integer=True
    )
    reg_grid = _positive_grid(
        tune["reg_grid"], "reg_grid"
    )

    default_k = int(settings["k"])
    default_reg = float(settings["reg"])

    if default_k not in k_grid:
        raise ValueError("k_grid phải chứa als.k")

    if default_reg not in reg_grid:
        raise ValueError("reg_grid phải chứa als.reg")

    # Hai lát cắt, không phải lưới k × λ đầy đủ.
    candidates = list(dict.fromkeys(
        [(k, default_reg) for k in k_grid]
        + [(default_k, reg) for reg in reg_grid]
    ))

    baseline = _make_baseline(cfg, train)

    rows = []

    for k, reg in candidates:
        print(f"[ALS] k={k}, lambda={reg:g}")

        model = ALSRecommender(
            k=k,
            reg=reg,
            n_epochs=int(settings["n_epochs"]),
            baseline=baseline,
            init_std=float(settings["init_std"]),
            solver=settings["solver"],
            seed=int(cfg["seed"]),
        )

        start = time.perf_counter()
        model.fit(train)
        fit_time = time.perf_counter() - start

        predictions = model.predict(users, items)

        score = rmse(ratings, predictions)

        if not np.isfinite(score):
            raise RuntimeError(
                f"ALS có RMSE không hợp lệ: k={k}, reg={reg}"
            )

        rows.append({
            "k": k,
            "reg": reg,
            "val_rmse": score,
            "fit_time_s": fit_time,
        })

    return pd.DataFrame(rows).sort_values(
        ["k", "reg"], ignore_index=True
    )


def sweep_svd(cfg: dict[str, Any]) -> pd.DataFrame:
    """Quét k cho SVD.

    Args:
        cfg: Cấu hình từ `load_config`.

    Returns:
        DataFrame cột `k`, `val_rmse`, `fit_time_s`.
    """

    train, users, items, ratings = _load_data(cfg)

    k_grid = _positive_grid(
        cfg["tune"]["k_grid"],
        "k_grid",
        integer=True,
    )

    max_k = max(k_grid)

    if max_k >= min(train.shape):
        raise ValueError(
            "k lớn nhất phải nhỏ hơn chiều nhỏ nhất của ma trận"
        )

    set_seed(int(cfg["seed"]))
    baseline = _make_baseline(cfg, train)

    # Chỉ phân rã một lần ở rank lớn nhất.
    model = TruncatedSVDRecommender(
        k=max_k,
        baseline=baseline,
    )

    start = time.perf_counter()
    model.fit(train)
    fit_time = time.perf_counter() - start

    if (
        model.u_ is None
        or model.s_ is None
        or model.vt_ is None
    ):
        raise RuntimeError("SVD chưa được phân rã")

    # Đảm bảo thành phần được sắp theo singular value giảm dần.
    order = np.argsort(model.s_)[::-1]

    U = model.u_[:, order]
    S = model.s_[order]
    VT = model.vt_[order, :]

    base_predictions = model.baseline.predict_raw(
        users, items
    )

    rows = []

    for k in k_grid:
        # Cắt k thành phần đầu từ kết quả SVD chung.
        interaction = np.sum(
            U[users, :k]
            * S[:k]
            * VT[:k, items].T,
            axis=1,
        )

        predictions = np.clip(
            base_predictions + interaction,
            model.rating_min,
            model.rating_max,
        )

        score = rmse(ratings, predictions)

        if not np.isfinite(score):
            raise RuntimeError(f"SVD lỗi tại k={k}")

        print(f"[SVD] k={k}, val RMSE={score:.4f}")

        rows.append({
            "k": k,
            "val_rmse": score,
            # Thời gian phân rã chung, không phải thời gian
            # huấn luyện độc lập của từng k.
            "fit_time_s": fit_time,
        })

    return pd.DataFrame(rows).sort_values(
        "k", ignore_index=True
    )


def run(cfg: dict[str, Any]) -> None:
    """Chạy các lần quét, lưu bảng và hình ảnh hưởng của k, λ.

    Args:
        cfg: Cấu hình từ `load_config`.
    """

    paths = cfg["paths"]

    tables_dir = Path(paths["tables_dir"])
    figures_dir = Path(paths["figures_dir"])

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Quét ALS và SVD.
    als_results = sweep_als(cfg)
    svd_results = sweep_svd(cfg)

    als_results.to_csv(
        tables_dir / "als_k_lambda_sweep.csv",
        index=False,
    )

    svd_results.to_csv(
        tables_dir / "svd_k_sweep.csv",
        index=False,
    )

    default_k = int(cfg["als"]["k"])
    default_reg = float(cfg["als"]["reg"])

    # Lát cắt ALS theo k, giữ λ mặc định.
    als_k = als_results[
        np.isclose(als_results["reg"], default_reg)
    ].copy()

    # Lát cắt ALS theo λ, giữ k mặc định.
    als_reg = als_results[
        als_results["k"] == default_k
    ].copy()

    best_als_k = als_k.loc[
        als_k["val_rmse"].idxmin()
    ]

    best_als_reg = als_reg.loc[
        als_reg["val_rmse"].idxmin()
    ]

    best_svd = svd_results.loc[
        svd_results["val_rmse"].idxmin()
    ]

    # 2. Vẽ hình với style thống nhất.
    apply_style()

    # So sánh k giữa ALS và SVD.
    als_plot = als_k[["k", "val_rmse"]].copy()
    als_plot["model"] = "ALS"

    svd_plot = svd_results[["k", "val_rmse"]].copy()
    svd_plot["model"] = "SVD"

    k_plot = pd.concat(
        [als_plot, svd_plot],
        ignore_index=True,
    )

    save_figure(
        plot_sweep(
            k_plot,
            x="k",
            y="val_rmse",
            hue="model",
        ),
        figures_dir / "tune_k_val_rmse.png",
    )

    # Quét λ cho ALS.
    save_figure(
        plot_sweep(
            als_reg,
            x="reg",
            y="val_rmse",
            logx=True,
        ),
        figures_dir / "tune_lambda_val_rmse.png",
    )

    # 3. Thông báo kết quả lựa chọn trên validation.
    print("\n========== BEST VALIDATION PARAMETERS ==========")

    print(
        f"ALS best k: {int(best_als_k['k'])} "
        f"(lambda={default_reg:g}, "
        f"RMSE={best_als_k['val_rmse']:.4f})"
    )

    print(
        f"ALS best lambda: {best_als_reg['reg']:g} "
        f"(k={default_k}, "
        f"RMSE={best_als_reg['val_rmse']:.4f})"
    )

    print(
        f"SVD best k: {int(best_svd['k'])} "
        f"(RMSE={best_svd['val_rmse']:.4f})"
    )

    print(
        "\nCác lựa chọn ALS là kết quả của hai phép quét riêng, "
        "chưa chứng minh cặp k và lambda kết hợp là tối ưu."
    )

    print(
        "\nResults saved to:",
        tables_dir,
        figures_dir,
    )


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    set_seed(cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()

"""Thí nghiệm SGD: so sánh SGD, Momentum, Adam ở cùng k và λ.

Mục đích: chọn một optimizer đại diện để đưa vào so sánh tổng. Liên quan Ch.5 và Ch.6.

Chạy: python -m experiments.run_sgd
"""

from __future__ import annotations

import argparse
from typing import Any

from src.config import load_config, set_seed


def run(cfg: dict[str, Any]) -> None:
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from pathlib import Path

    from src.models.baseline import BiasBaseline
    from src.models.sgd import SGDRecommender
    from src.visualization.plots import (
        apply_style,
        plot_convergence,
        save_figure,
    )

    # 1. Chuẩn bị đường dẫn.
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])

    for directory in (tables_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 2. Chỉ đọc train và validation, không đọc test.
    train = sp.load_npz(processed_dir / "train_csr.npz").tocsr()
    val_df = pd.read_parquet(processed_dir / "val.parquet")

    val_users = val_df["user_idx"].to_numpy(dtype=np.int64)
    val_items = val_df["item_idx"].to_numpy(dtype=np.int64)
    val_ratings = val_df["rating"].to_numpy(dtype=float)

    if len(val_ratings) == 0:
        raise ValueError("Tập validation không có đánh giá.")

    val = sp.csr_matrix(
        (val_ratings, (val_users, val_items)),
        shape=train.shape,
    )

    # 3. Học baseline một lần, dùng chung cho ba optimizer.
    baseline = BiasBaseline(
        reg=cfg["baseline"]["reg"],
        rating_min=cfg["ratings"]["min"],
        rating_max=cfg["ratings"]["max"],
    )
    baseline.fit(train)

    settings = cfg["sgd"]
    logs = {}
    summary = []

    best_rmse = float("inf")
    best_optimizer = None

    # 4. Lần lượt chạy ba optimizer.
    for optimizer in ("sgd", "momentum", "adam"):
        print(f"\nDang huan luyen: {optimizer}")

        model = SGDRecommender(
            k=settings["k"],
            reg=settings["reg"],
            lr=settings["lr"][optimizer],
            n_epochs=settings["n_epochs"],
            batch_size=settings["batch_size"],
            baseline=baseline,
            optimizer=optimizer,
            init_std=settings["init_std"],
            hyper=settings,
            seed=cfg["seed"],
        )

        model.fit(train, val)

        # Đánh giá mô hình ở epoch cuối trên validation.
        predictions = model.predict(val_users, val_items)

        if not np.all(np.isfinite(predictions)):
            raise RuntimeError(
                f"{optimizer} sinh NaN/Inf. "
                "Kiem tra gradient va giam learning rate."
            )

        errors = val_ratings - predictions
        val_rmse = float(np.sqrt(np.mean(errors ** 2)))
        val_mae = float(np.mean(np.abs(errors)))

        # Lưu lịch sử học và mô hình của từng optimizer.
        history = model.log.to_frame()
        logs[optimizer] = history

        model.log.save(tables_dir / f"sgd_{optimizer}_history.csv")
        model.save(models_dir / f"sgd_{optimizer}.pkl")

        elapsed = float(history.iloc[-1]["elapsed_s"])

        summary.append({
            "optimizer": optimizer,
            "k": settings["k"],
            "reg": settings["reg"],
            "lr": settings["lr"][optimizer],
            "n_epochs": settings["n_epochs"],
            "batch_size": settings["batch_size"],
            "seed": cfg["seed"],
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "train_seconds": elapsed,
        })

        print(
            f"{optimizer}: RMSE={val_rmse:.4f}, "
            f"MAE={val_mae:.4f}, time={elapsed:.2f}s"
        )

        # Chọn optimizer có RMSE validation thấp nhất.
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_optimizer = optimizer
            model.save(models_dir / "sgd_best.pkl")

    # 5. Lưu bảng so sánh.
    table = pd.DataFrame(summary).sort_values("val_rmse")
    table.to_csv(tables_dir / "sgd_comparison.csv", index=False)

    print("\nBang so sanh:")
    print(table.to_string(index=False))
    print(f"\nOptimizer tot nhat theo validation: {best_optimizer}")

    # 6. Vẽ sai số theo epoch và thời gian.
    apply_style()

    for axis in ("epoch", "elapsed_s"):
        figure = plot_convergence(
            logs,
            x=axis,
            y="val_rmse",
        )
        save_figure(
            figure,
            figures_dir / f"sgd_convergence_{axis}.png",
        )

        import matplotlib.pyplot as plt
        plt.close(figure)


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    set_seed(cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()

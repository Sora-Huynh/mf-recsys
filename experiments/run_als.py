"""Thí nghiệm ALS: đường hội tụ và số điều kiện theo λ.

Mục đích: kiểm tra hàm mục tiêu giảm đều và cho thấy λ cải thiện độ ổn định của hệ.
Liên quan Ch.2 và Ch.3.

Chạy: python -m experiments.run_als
"""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Chỉ lưu hình ra file, không mở cửa sổ.

from src.config import load_config, set_seed
from src.data.preprocess import load_processed
from src.data.sparse import build_csr, load_csr
from src.evaluation.conditioning import condition_vs_lambda
from src.evaluation.metrics import rmse
from src.models.als import ALSRecommender
from src.models.baseline import BiasBaseline
from src.visualization.plots import apply_style, plot_condition, plot_convergence, save_figure


def run(cfg: dict[str, Any]) -> None:
    """Fit ALS, lưu nhật ký hội tụ, bảng κ theo λ và mô hình.

    Chỉ đo trên tập val: tập test để dành cho `experiments/compare_all.py`.

    Args:
        cfg: Cấu hình từ `load_config`.
    """
    paths = cfg["paths"]
    split = load_processed(cfg)
    train = load_csr(paths["processed_dir"] / "train_csr.npz")
    val = build_csr(
        split.val["user_idx"].to_numpy(),
        split.val["item_idx"].to_numpy(),
        split.val["rating"].to_numpy(),
        split.shape,
    )
    val_coo = val.tocoo()

    baseline = BiasBaseline(
        reg=cfg["baseline"]["reg"],
        rating_min=cfg["ratings"]["min"],
        rating_max=cfg["ratings"]["max"],
    ).fit(train)
    baseline_rmse = rmse(val_coo.data, baseline.predict(val_coo.row, val_coo.col))

    als_cfg = cfg["als"]
    model = ALSRecommender(
        k=als_cfg["k"],
        reg=als_cfg["reg"],
        n_epochs=als_cfg["n_epochs"],
        baseline=baseline,
        init_std=als_cfg["init_std"],
        solver=als_cfg["solver"],
        seed=cfg["seed"],
    ).fit(train, val)
    model.save(paths["models_dir"] / "als.pkl")

    log = model.log.to_frame()
    model.log.save(paths["tables_dir"] / "als_convergence.csv")
    conditioning = condition_vs_lambda(
        model.Q,
        train,
        cfg["conditioning"]["reg_grid"],
        cfg["conditioning"]["n_users"],
        cfg["seed"],
    )
    conditioning.to_csv(paths["tables_dir"] / "als_conditioning.csv", index=False)

    apply_style()
    reference = {"Bias baseline": baseline_rmse}
    for x, name in (("epoch", "als_val_rmse_by_epoch"), ("elapsed_s", "als_val_rmse_by_time")):
        figure = plot_convergence({"ALS": log}, x=x, y="val_rmse", reference=reference)
        save_figure(figure, paths["figures_dir"] / f"{name}.png")
    save_figure(plot_condition(conditioning), paths["figures_dir"] / "als_condition_vs_lambda.png")

    print(f"Baseline, RMSE val: {baseline_rmse:.4f}")
    print(log.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(conditioning.to_string(index=False, float_format=lambda value: f"{value:.4g}"))


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    set_seed(cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()

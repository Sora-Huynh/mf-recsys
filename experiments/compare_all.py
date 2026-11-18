"""So sánh tổng: bảng RMSE test và biểu đồ hội tụ.

Mục đích: tạo các hình/bảng chính của báo cáo. Chỉ so các mô hình đã chọn tham số
theo val: hai baseline, SVD, ALS và một optimizer SGD tốt nhất. Liên quan Ch.3–Ch.6.

Đây là nơi DUY NHẤT dùng tập test. Mô hình được đọc từ `results/models/` (do các
script `run_*` lưu), không huấn luyện lại, nên số liệu khớp với từng thí nghiệm riêng.

Thời gian huấn luyện:
- Mô hình lặp (ALS, SGD): lấy `elapsed_s` của vòng cuối trong nhật ký hội tụ, tức chỉ
  tính phần huấn luyện, không tính phần đánh giá. Cùng cách đo với biểu đồ hội tụ.
- Mô hình không lặp (baseline, SVD): fit lại một bản sao trên cùng tập train và đo giờ,
  vì các mô hình này không lưu thời gian huấn luyện. Chỉ mất dưới một giây.

Chạy: python -m experiments.compare_all
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Chỉ lưu hình ra file, không mở cửa sổ.

import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.data.preprocess import load_processed
from src.data.sparse import load_csr
from src.evaluation.metrics import mae, rmse
from src.models.base import Recommender
from src.models.baseline import GlobalMean
from src.visualization.plots import apply_style, plot_bars, plot_convergence, save_figure

# Tên hiển thị -> file trong `models_dir`. Thứ tự này là thứ tự cột trong bảng và hình.
MODEL_FILES = {
    "Bias baseline": "baseline.pkl",
    "SVD": "svd.pkl",
    "ALS": "als.pkl",
    "SGD": "sgd_best.pkl",
}


def _load_models(models_dir: Path) -> dict[str, Recommender]:
    """Đọc các mô hình đã lưu; mô hình nào thiếu file thì bỏ qua và báo.

    Args:
        models_dir: Thư mục chứa file `.pkl`.

    Returns:
        Dict tên hiển thị -> mô hình, theo thứ tự của `MODEL_FILES`.
    """
    models: dict[str, Recommender] = {}
    for name, filename in MODEL_FILES.items():
        path = models_dir / filename
        if not path.exists():
            print(f"[Bỏ qua] {name}: không có {path}. Chạy script run_* tương ứng trước.")
            continue
        model = Recommender.load(path)
        # SGD tốt nhất có thể là sgd, momentum hoặc adam; ghi rõ vào tên.
        optimizer = getattr(model, "optimizer", None)
        models[f"{name} ({optimizer})" if optimizer else name] = model
    return models


def _fit_time(model: Recommender) -> float:
    """Thời gian huấn luyện (giây), đo theo quy tắc ở đầu file.

    Args:
        model: Mô hình đã huấn luyện.

    Returns:
        Số giây.
    """
    log = getattr(model, "log", None)
    if log is not None and log.rows:
        return float(log.to_frame()["elapsed_s"].iloc[-1])
    fresh = copy.deepcopy(model)
    start = time.perf_counter()
    fresh.fit(model.train_)
    return time.perf_counter() - start


def evaluate_on_test(
    cfg: dict[str, Any], saved: dict[str, Recommender] | None = None
) -> pd.DataFrame:
    """Tính RMSE, MAE trên test của các mô hình đã lưu.

    Args:
        cfg: Cấu hình từ `load_config`.
        saved: Mô hình đã đọc sẵn; None thì đọc từ `models_dir`.

    Returns:
        DataFrame cột `model`, `test_rmse`, `test_mae`, `fit_time_s`.
    """
    paths = cfg["paths"]
    split = load_processed(cfg)
    train = load_csr(paths["processed_dir"] / "train_csr.npz")
    users = split.test["user_idx"].to_numpy()
    items = split.test["item_idx"].to_numpy()
    ratings = split.test["rating"].to_numpy(dtype=np.float64)

    # GlobalMean không được lưu vì demo không dùng; fit ở đây chỉ mất vài mili giây.
    models: dict[str, Recommender] = {
        "Global mean": GlobalMean(cfg["ratings"]["min"], cfg["ratings"]["max"]).fit(train)
    }
    models.update(saved if saved is not None else _load_models(paths["models_dir"]))

    rows = []
    for name, model in models.items():
        predictions = model.predict(users, items)
        rows.append(
            {
                "model": name,
                "test_rmse": rmse(ratings, predictions),
                "test_mae": mae(ratings, predictions),
                "fit_time_s": _fit_time(model),
            }
        )
    return pd.DataFrame(rows)


def _val_rmse(model: Recommender, val: pd.DataFrame) -> float:
    """RMSE trên val, dùng làm đường ngang trong biểu đồ hội tụ.

    Args:
        model: Mô hình đã huấn luyện.
        val: Bảng val từ `load_processed`.

    Returns:
        RMSE.
    """
    predictions = model.predict(val["user_idx"].to_numpy(), val["item_idx"].to_numpy())
    return rmse(val["rating"].to_numpy(dtype=np.float64), predictions)


def run(cfg: dict[str, Any]) -> None:
    """Lưu bảng RMSE, biểu đồ cột và hai biểu đồ hội tụ (theo vòng lặp, theo thời gian).

    Args:
        cfg: Cấu hình từ `load_config`.
    """
    paths = cfg["paths"]
    tables_dir, figures_dir = paths["tables_dir"], paths["figures_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    models = _load_models(paths["models_dir"])

    # 1. Bảng RMSE, MAE trên test.
    table = evaluate_on_test(cfg, models)
    table.to_csv(tables_dir / "compare_test.csv", index=False)
    print("RMSE, MAE trên tập test:")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    apply_style()

    # 2. Biểu đồ cột RMSE test.
    try:
        figure = plot_bars(
            table["model"].tolist(),
            table["test_rmse"].tolist(),
            title="Test RMSE by model",
            ylabel="Test RMSE",
        )
        save_figure(figure, figures_dir / "compare_test_rmse.png")
    except NotImplementedError:
        print("[Bỏ qua] Biểu đồ cột: `plot_bars` trong src/visualization/plots.py chưa cài.")

    # 3. Hội tụ: ALS và SGD là đường; baseline và SVD không lặp nên là đường ngang.
    split = load_processed(cfg)
    logs, reference = {}, {}
    for name, model in models.items():
        log = getattr(model, "log", None)
        if log is not None and log.rows:
            logs[name] = log.to_frame()
        else:
            reference[name] = _val_rmse(model, split.val)
    if logs:
        for x in ("epoch", "elapsed_s"):
            figure = plot_convergence(logs, x=x, y="val_rmse", reference=reference)
            save_figure(figure, figures_dir / f"compare_convergence_{x}.png")

    print(f"\nĐã lưu bảng vào {tables_dir / 'compare_test.csv'} và hình vào {figures_dir}")


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    set_seed(cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()

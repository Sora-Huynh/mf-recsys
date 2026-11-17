"""Thí nghiệm SVD: RMSE, thời gian, và phổ giá trị kỳ dị.

Mục đích: phổ σ cho thấy dữ liệu xấp xỉ hạng thấp, là cơ sở của phân rã ma trận.
Liên quan Ch.4.

Chạy: python -m experiments.run_svd
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config, set_seed
from src.data.sparse import load_csr
from src.evaluation.metrics import mae, rmse
from src.models.baseline import BiasBaseline
from src.models.svd import TruncatedSVDRecommender


def run(cfg: dict[str, Any]) -> None:
    """Fit SVD, lưu bảng, hình phổ σ và mô hình.

    Args:
        cfg: Cấu hình từ `load_config`.
    """
    # 1. Tải dữ liệu đúng theo cấu trúc file thực tế
    print("Đang tải dữ liệu...")
    train = load_csr(Path("data/processed/train_csr.npz"))
    
    # Tập val được lưu dưới dạng parquet thay vì CSR
    val_df = pd.read_parquet(Path("data/processed/val.parquet"))
    val_users = val_df.iloc[:, 0].values
    val_items = val_df.iloc[:, 1].values
    val_ratings = val_df.iloc[:, 2].values

    # 2. Khởi tạo mô hình
    print("Đang khởi tạo Baseline và TruncatedSVD...")
    baseline = BiasBaseline(reg=cfg["baseline"]["reg"])
    k = cfg["svd"]["k"]
    model = TruncatedSVDRecommender(k=k, baseline=baseline)

    # 3. Huấn luyện và đo thời gian thực thi
    print(f"Đang huấn luyện SVD với k={k}...")
    start_time = time.perf_counter()
    model.fit(train, val=None)  # SVD không cần validation lúc huấn luyện
    fit_time = time.perf_counter() - start_time
    print(f"Huấn luyện hoàn tất trong {fit_time:.2f} giây.")

    # 4. Dự đoán và đánh giá RMSE, MAE trên tập Validation
    preds = model.predict(val_users, val_items)

    val_rmse = rmse(val_ratings, preds)
    val_mae = mae(val_ratings, preds)
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")

    # 5. Lưu bảng kết quả
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame([{
        "model": "SVD",
        "k": k,
        "fit_time_s": fit_time,
        "val_rmse": val_rmse,
        "val_mae": val_mae
    }])
    csv_path = "results/tables/svd_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Đã lưu chỉ số đánh giá vào {csv_path}")

    # 6. Vẽ và lưu hình phổ giá trị kỳ dị (Scree plot cho Ch.4)
    if model.singular_values_ is not None:
        Path("results/figures").mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(model.singular_values_, marker='.', linestyle='-', color='b')
        plt.title('Phổ Giá trị kỳ dị (Scree Plot) - Truncated SVD')
        plt.xlabel('Thành phần k')
        plt.ylabel(r'Giá trị kỳ dị ($\sigma$)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = "results/figures/svd_spectrum.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Đã lưu biểu đồ phổ tại {plot_path}")

    # 7. Lưu mô hình vào models_dir để compare_all.py và src.recommend --model svd đọc được
    model_path = cfg["paths"]["models_dir"] / "svd.pkl"
    model.save(model_path)
    print(f"Đã lưu mô hình tại {model_path}")


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()
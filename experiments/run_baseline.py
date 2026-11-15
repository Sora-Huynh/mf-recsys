"""Thí nghiệm baseline: RMSE/MAE của trung bình toàn cục và bias user/phim.

Mục đích: đặt mốc mà các mô hình phân rã phải vượt. Liên quan Ch.3.

Chạy: python -m experiments.run_baseline
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config, set_seed
from src.data.preprocess import load_processed as load_processed_data
from src.data.sparse import build_csr
from src.models.baseline import BiasBaseline, GlobalMean


def evaluate_model(model, mat) -> tuple[float, float]:
    """Tính RMSE và MAE của mô hình trên ma trận CSR điểm số."""
    coo = mat.tocoo()
    preds = model.predict(coo.row, coo.col)
    actuals = coo.data
    
    rmse = float(pd.Series(actuals - preds).pow(2).mean() ** 0.5)
    mae = float(pd.Series(actuals - preds).abs().mean())
    return rmse, mae


def run(cfg: dict[str, Any]) -> None:
    """Fit hai baseline, lưu bảng vào `results/tables/baseline.csv`.

    Args:
        cfg: Cấu hình từ `load_config`.
    """
    # 1. Load dữ liệu đã qua tiền xử lý (dạng CSR matrix)
    data = load_processed_data(cfg)
    train = build_csr(
        users=data.train["user_idx"].to_numpy(),
        items=data.train["item_idx"].to_numpy(),
        values=data.train["rating"].to_numpy(),
        shape=data.shape,
    )
    val = build_csr(
        users=data.val["user_idx"].to_numpy(),
        items=data.val["item_idx"].to_numpy(),
        values=data.val["rating"].to_numpy(),
        shape=data.shape,
    )
    test = build_csr(
        users=data.test["user_idx"].to_numpy(),
        items=data.test["item_idx"].to_numpy(),
        values=data.test["rating"].to_numpy(),
        shape=data.shape,
    )
    baseline_cfg = cfg.get("baseline", {})
    reg = baseline_cfg.get("reg", 10.0)

    # 2. Khởi tạo 2 mô hình Baseline
    models = {
        "GlobalMean": GlobalMean(),
        "BiasBaseline": BiasBaseline(reg=reg),
    }

    results = []

    # 3. Huấn luyện và đánh giá từng mô hình
    for name, model in models.items():
        print(f"=== Huấn luyện mô hình: {name} ===")
        model.fit(train, val)

        # Tính chỉ số trên tập Val và Test
        val_rmse, val_mae = evaluate_model(model, val)
        test_rmse, test_mae = evaluate_model(model, test)

        print(f"[{name}] Val  RMSE: {val_rmse:.4f} | Val  MAE: {val_mae:.4f}")
        print(f"[{name}] Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}\n")

        # Chỉ lưu BiasBaseline cho demo: GlobalMean cho mọi phim cùng một điểm nên gợi ý vô nghĩa
        if name == "BiasBaseline":
            model.save(Path(cfg["paths"]["models_dir"]) / "baseline.pkl")
            print(f"Đã lưu mô hình vào '{Path(cfg['paths']['models_dir']) / 'baseline.pkl'}'")

        results.append(
            {
                "model": name,
                "val_rmse": round(val_rmse, 4),
                "val_mae": round(val_mae, 4),
                "test_rmse": round(test_rmse, 4),
                "test_mae": round(test_mae, 4),
            }
        )

    # 4. Xuất kết quả ra file results/tables/baseline.csv
    os.makedirs("results/tables", exist_ok=True)
    df_results = pd.DataFrame(results)
    out_path = "results/tables/baseline.csv"
    df_results.to_csv(out_path, index=False)
    print(f"Đã lưu bảng kết quả thành công vào '{out_path}'")


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    cfg = load_config(parser.parse_args().config)
    set_seed(cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()
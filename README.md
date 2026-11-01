# Hệ gợi ý dựa trên phân rã ma trận: So sánh SVD, ALS và SGD

Bài tập lớn môn AS2069 *Phương pháp số cho Khoa học dữ liệu*, ĐH Bách Khoa – ĐHQG-HCM.

## Đề tài

Dự đoán điểm còn thiếu trong ma trận user × phim của MovieLens-1M bằng phân rã R ≈ P Qᵀ,
và so sánh ba cách giải về độ chính xác, tốc độ hội tụ và tính chất số.

| Phương pháp | Ý tưởng | Chương |
|---|---|---|
| Baseline | Trung bình toàn cục; trung bình + bias user/phim | Ch.3 |
| SVD cắt cụt | Điền ô thiếu rồi lấy xấp xỉ hạng k | Ch.4 |
| ALS | Luân phiên giải hệ Ridge (QᵀQ + λI)p = Qᵀr bằng Cholesky | Ch.2–3 |
| SGD / Momentum / Adam | Giảm gradient theo mini-batch | Ch.5–6 |

Cả ba phương pháp phân rã cùng một ma trận phần dư r − (μ + b_u + b_i), nên kết quả
so sánh được với nhau.

**Dữ liệu:** MovieLens-1M, 1.000.209 đánh giá, 6.040 user, khoảng 3.700 phim, thưa
khoảng 95,5%.

**Bài báo nền:** Koren, Bell, Volinsky, *Matrix Factorization Techniques for
Recommender Systems*, IEEE Computer, 2009.

**Trích dẫn dữ liệu:** Harper, Konstan, *The MovieLens Datasets: History and Context*,
ACM TiiS 5(4), 2015. License không cho phát hành lại dữ liệu, nên `data/raw/` không được
commit.

## Cài đặt

Cần Python 3.11 trở lên. Chạy các lệnh tại thư mục `mf-recsys/`.

```bash
python -m venv .venv
```

Kích hoạt môi trường trên Windows:

```bash
.venv\Scripts\activate
```

Trên macOS/Linux:

```bash
source .venv/bin/activate
```

Cài thư viện và tải dữ liệu:

```bash
pip install -r requirements.txt
```

```bash
python -m src.data.download
```

## Cách chạy

> Hiện repo mới là khung: chỉ `config`, `download`, `load`, `metrics` đã hoàn chỉnh, các
> lệnh còn lại sẽ báo `NotImplementedError` cho tới khi được cài. Tiến độ xem ở
> [ROADMAP.md](ROADMAP.md).

Chạy từ thư mục gốc repo với `python -m` để `import src` hoạt động. Thứ tự thực hiện:

```bash
python -m src.data.preprocess
```

```bash
python -m experiments.run_baseline
```

```bash
python -m experiments.run_svd
```

```bash
python -m experiments.run_als
```

```bash
python -m experiments.run_sgd
```

```bash
python -m experiments.tune_k_lambda
```

```bash
python -m experiments.compare_all
```

Demo gợi ý:

```bash
python -m src.recommend --user-id 42 --model als
```

Kiểm thử:

```bash
python -m pytest
```

Mọi tham số nằm trong [configs/default.yaml](configs/default.yaml). Thử cấu hình khác:
chép file đó rồi thêm `--config configs/ten_file.yaml` vào lệnh.

## Cấu trúc repo

| Thư mục | Vai trò |
|---|---|
| `configs/` | Tham số thí nghiệm, một nguồn duy nhất |
| `data/` | Dữ liệu thô và đã xử lý, không commit |
| `src/data/` | Tải, đọc, tiền xử lý, CSR |
| `src/models/` | Baseline, SVD, ALS, SGD chung một giao diện |
| `src/solvers/` | Cholesky, Gauss-Seidel tự cài |
| `src/evaluation/` | Sai số, nhật ký hội tụ, số điều kiện |
| `src/visualization/` | Hàm vẽ chung |
| `experiments/` | Script tái tạo từng kết quả |
| `notebooks/` | EDA |
| `results/` | Hình, bảng, mô hình sinh ra, không commit |
| `reports/` | Báo cáo LaTeX và slide |
| `tests/` | Kiểm thử trên ví dụ biết trước đáp án |

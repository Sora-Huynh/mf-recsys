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

> Hiện repo mới là khung: chỉ `config`, `download`, `load`, `metrics`, `models/base` đã hoàn chỉnh, các
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

```text
mf-recsys/
├── README.md                     # mô tả đề tài, cài đặt, cách chạy
├── ROADMAP.md                    # lộ trình 2 tuần, tiêu chí "xong"
├── requirements.txt              # danh sách thư viện
├── .gitignore                    # bỏ qua dữ liệu, kết quả, CLAUDE.md
├── configs/
│   └── default.yaml              # toàn bộ tham số: seed, k, λ, lr, số epoch...
├── data/                         # không commit
│   ├── raw/                      # ML-1M gốc
│   └── processed/                # train/val/test, CSR đã lưu
├── doc/
│   └── PHAN_CONG.md              # phân công theo thành viên
├── src/
│   ├── __init__.py
│   ├── config.py                 # đọc YAML, đặt seed
│   ├── recommend.py              # demo: UserID -> top-10 phim
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py           # tải và giải nén ML-1M
│   │   ├── load.py               # đọc ratings/movies/users.dat
│   │   ├── preprocess.py         # ánh xạ ID, chia train/val/test
│   │   └── sparse.py             # ma trận CSR, so sánh bộ nhớ (Ch.2)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # giao diện chung của mô hình
│   │   ├── baseline.py           # global mean; mean + bias (Ch.3)
│   │   ├── svd.py                # điền thiếu + SVD cắt cụt (Ch.4)
│   │   ├── als.py                # ALS giải Ridge bằng Cholesky (Ch.2–3)
│   │   └── sgd.py                # SGD, Momentum, Adam (Ch.5–6)
│   ├── solvers/
│   │   ├── __init__.py
│   │   └── linear.py             # Cholesky tự cài (Ch.2)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # RMSE, MAE
│   │   ├── convergence.py        # nhật ký theo vòng lặp và thời gian
│   │   └── conditioning.py       # số điều kiện theo λ (Ch.1–3)
│   └── visualization/
│       ├── __init__.py
│       └── plots.py              # hàm vẽ chung
├── experiments/
│   ├── run_baseline.py           # mốc RMSE baseline
│   ├── run_svd.py                # RMSE và phổ giá trị kỳ dị
│   ├── run_als.py                # hội tụ ALS, số điều kiện theo λ
│   ├── run_sgd.py                # so sánh 3 optimizer
│   ├── tune_k_lambda.py          # quét k, λ trên tập val
│   └── compare_all.py            # bảng và hình tổng, nơi duy nhất dùng tập test
├── notebooks/
│   └── 01_eda.ipynb              # thống kê mô tả
├── results/                      # sinh tự động, không commit
│   ├── figures/
│   └── tables/
├── reports/
│   └── sections/                 # mỗi người viết một file báo cáo
└── tests/
    ├── test_metrics.py           # kiểm thử RMSE, MAE
    ├── test_base.py              # kiểm thử giao diện chung của mô hình
    └── test_models_small.py      # ma trận nhỏ biết trước đáp án
```

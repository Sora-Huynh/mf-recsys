# Tiến độ và nhận xét review

Cập nhật: 09/11/2026, trên nhánh `develop` sau khi merge PR #6 và #7 (commit `4903914`).

Tiêu chí "xong" của từng việc nằm trong [ROADMAP.md](../ROADMAP.md), phân công nằm trong [PHAN_CONG.md](PHAN_CONG.md).

## Tổng quan

| Phần | Người | Trạng thái | Ghi chú |
|---|---|---|---|
| Tiền xử lý, CSR | Duy | Xong (PR #1) | 800.203 / 100.002 / 100.004 dòng; CSR nhỏ hơn dense khoảng 18,6 lần |
| Baseline | Tiến Đạt | Xong (PR #3, #6) | Đúng công thức, có 5 test, có `n_iters` để tiến tới nghiệm Ridge chính xác |
| Demo gợi ý `recommend.py` | Tiến Đạt | Xong (PR #6) | Chạy được với `--model baseline`; tên file mô hình SVD, SGD chưa khớp (xem dưới) |
| SGD, Momentum, Adam | Hưng | Xong (PR #2) | `run_sgd.py` đã chạy hết được; còn overfit và vài chỗ dọn code |
| SVD | Gia Vang | Xong (PR #4) | Toán đúng; còn đường dẫn viết cứng và thiếu test |
| Cholesky, ALS | Anh Tuấn | Xong (PR #7) | Đúng, nhanh (numba), có 7 test và phần báo cáo |
| Convergence, conditioning | Trần Công Tuấn Anh | **Đã có** (Anh Tuấn cài trong PR #7) | Cần Trần Công Tuấn Anh rà lại và dùng tiếp, không viết lại |
| Plots | Trần Công Tuấn Anh | Một phần (PR #7) | Còn `plot_bars`, `plot_sweep` |
| `tune_k_lambda.py` | Trần Công Tuấn Anh | **Chưa có** | |
| EDA | Tuấn Anh | **Chưa có** | `01_eda.ipynb` vẫn là bản khung |
| `compare_all.py` | Duy | Chưa làm (tuần 2) | Đã đủ mô hình; còn chờ `plot_bars` |

`pytest` trên `develop`: 18 passed, 1 xfailed, 18 xpassed. Test duy nhất còn xfail là `test_svd_recovers_full_low_rank`; khi Gia Vang viết xong thì bỏ `pytestmark` ở đầu `tests/test_models_small.py`.

Các `NotImplementedError` còn lại có chủ ý: lớp trừu tượng trong `base.py`, Gauss-Seidel trong `linear.py`/`als.py` (phần tùy chọn đã bỏ).

## Kết quả hiện có (val RMSE trên ML-1M)

| Mô hình | Val RMSE | Thời gian huấn luyện | Ghi chú |
|---|---|---|---|
| GlobalMean | 1,1165 | | Test RMSE 1,1165 |
| BiasBaseline (reg = 10) | 0,9103 | dưới 0,1 giây | Test RMSE 0,9085 |
| SVD, k = 20 | 0,8840 | khoảng 0,5 giây | |
| **ALS**, k = 20, reg = 10, 20 vòng | **0,8654** | khoảng 1,2 giây | Hàm mục tiêu giảm đều, vòng 20 vẫn còn giảm nhẹ |
| SGD (epoch 30) | 0,8970 | | Tốt nhất 0,8717 ở epoch 3, sau đó overfit |
| Momentum (epoch 30) | 0,9096 | | Tốt nhất 0,8842 ở epoch 2, sau đó overfit |
| Adam (epoch 30) | 0,8858 | | Optimizer tốt nhất theo `run_sgd.py` |

Số SGD lấy từ `python -m experiments.run_sgd` trên `develop`, dùng `BiasBaseline` thật. Số "tốt nhất ở epoch …" lấy từ lần chạy thử trước. Cần tune `reg`/`lr` trên val thì mới so sánh công bằng với ALS.

SVD theo k:

| k | 0 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| Val RMSE | 0,9103 | 0,8991 | 0,8916 | 0,8867 | **0,8840** | 0,8843 | 0,8882 |

Sau k ≈ 20, RMSE tăng lại vì SVD coi ô thiếu là 0 như dữ liệu thật. Đây là điểm nên đưa vào báo cáo khi so với ALS.

BiasBaseline theo reg:

| reg | 0 | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| Val RMSE | 0,9103 | 0,9100 | **0,9098** | 0,9103 | 0,9133 | 0,9196 |

Số điều kiện κ₂ của hệ ALS theo λ (200 user ngẫu nhiên, Q đã học):

| λ | 0,01 | 0,1 | 1 | 10 | 100 |
|---|---|---|---|---|---|
| κ trung vị | 13,3 | 13,1 | 10,9 | 4,1 | 1,5 |
| κ lớn nhất | 2476 | 248,5 | 39,3 | 6,2 | 2,1 |

## Kết quả kiểm tra tính đúng

### BiasBaseline (Tiến Đạt)

| Phép kiểm tra | Kết quả |
|---|---|
| Ma trận đầy đủ r = μ + b_u + b_i, reg = 0 | Khôi phục đúng μ, b_u, b_i, sai số khoảng 1e-15 |
| reg = 10, so với công thức tự tính bằng vòng lặp | b_i và b_u khớp hoàn toàn |
| `residual_matrix` | Giữ cấu trúc thưa, giá trị đúng bằng r − (μ + b_u + b_i), trả về CSR |
| User/phim không có đánh giá, kể cả reg = 0 | Bias bằng 0, không ra NaN |
| `reg < 0` hoặc `n_iters < 1` | Báo `ValueError` |
| Lưu rồi nạp lại mô hình | Dự đoán giữ nguyên |
| `n_iters` tăng dần, so với nghiệm Ridge đồng thời giải bằng `spsolve` | Hàm mục tiêu giảm dần về tối ưu; 30 lượt còn lệch tương đối 4e-6. Val RMSE: 1 lượt 0,91027, 5 lượt 0,90984, nghiệm chính xác 0,90977 |
| Demo user 42; UserID không tồn tại; thiếu file mô hình | Ra top 10 đúng; báo lỗi rõ ràng trong hai trường hợp sau |

Docstring ghi "5 lượt cải thiện RMSE cỡ 1e-3", số đo được là khoảng 4e-4. Không ảnh hưởng gì, có thể sửa câu chữ khi tiện.

### SVD (Gia Vang)

| Phép kiểm tra | Kết quả |
|---|---|
| Ma trận hạng 3 đầy đủ, baseline = 0, k = 3, 5, 19 | Khôi phục đúng, sai số khoảng 3e-15 |
| Ma trận thưa 200×150 (đi qua `svds`), so với SVD đầy đủ của numpy | σ khớp và giảm dần; ma trận tái tạo khớp xấp xỉ hạng k tốt nhất (Eckart–Young), sai số khoảng 1e-14 |
| Dùng `BiasBaseline` thật với reg = 0 | Phân rã đúng ma trận phần dư |

### Cholesky, ALS (Anh Tuấn)

| Phép kiểm tra | Kết quả |
|---|---|
| `cholesky_solve`, `cholesky_factor` so với numpy, k = 1, 2, 5, 20; ví dụ tính tay của Ch.2 | Khớp (test của Anh Tuấn) |
| Ma trận không xác định dương, ma trận suy biến | Báo `LinAlgError` |
| `update_factors` trên 300 user ML-1M, so với `np.linalg.solve` | Sai số tương đối lớn nhất 2e-15 |
| Hàm mục tiêu qua các vòng (ma trận nhỏ và ML-1M) | Không bao giờ tăng |
| Ma trận hạng 3 ẩn 30% ô | ALS dự đoán đúng các ô bị ẩn |
| Số luồng numba khác nhau | Kết quả giống hệt |

### SGD (Hưng)

Gradient khớp với sai phân hữu hạn trung tâm, và loss giảm với cả 3 optimizer.

## Việc cần sửa theo từng người

### Hưng (SGD)

1. Bỏ monkeypatch `ConvergenceLog.record` trong `tests/test_models_small.py`; `convergence.py` đã có.
2. Bỏ giá trị mặc định cứng trong `hyper.get(...)` của `apply_update`; đọc thẳng từ config.
3. SGD và Momentum overfit sau 2–3 epoch. Tune `reg`/`lr` trên val, hoặc báo cáo epoch có val RMSE tốt nhất.
4. `run_sgd.py` lưu `sgd_best.pkl`, `sgd_{optimizer}.pkl`, trong khi `recommend.py --model sgd` tìm `sgd.pkl`. Thống nhất với Tiến Đạt.
5. Việc nhỏ:
   - Dùng lại `load_processed`/`load_csr` thay vì tự đọc file.
   - Đưa import ra đầu file và thêm lại docstring.
   - Gradient đang là tổng theo batch, nên `lr` phụ thuộc `batch_size`; cần ghi rõ trong báo cáo.

### Gia Vang (SVD)

1. Viết `test_svd_recovers_full_low_rank`. Dùng baseline luôn trả về 0 và ma trận hạng 3 đầy đủ; với k = 3, sai số phải khoảng 1e-15. Đây là test cuối cùng còn xfail.
2. `run_svd.py` đang viết cứng `data/processed/...` và lấy cột theo vị trí `iloc[:, 0]`. Dùng `cfg["paths"]`, `load_processed(cfg)` và lấy cột theo tên.
3. Lưu mô hình vào `cfg["paths"]["models_dir"] / "svd.pkl"` (hiện là `results/svd_model.pkl`) để `recommend.py --model svd` chạy được.
4. Bỏ đoạn dự phòng khi thiếu `residual_matrix` trong `svd.py`.
5. Vẽ phổ σ khoảng 100–200 giá trị thay vì chỉ 20.

### Anh Tuấn (ALS)

1. Mô tả PR #7 còn nhắc `configs/ml32m.yaml`, `prepare_ml32m.py` dù commit cuối đã bỏ. Không cần sửa code.
2. Phần báo cáo đang là `.tex`, trong khi Tiến Đạt viết `.md`. Cả nhóm cần chốt một định dạng (xem mục Quy trình).

### Trần Công Tuấn Anh (đánh giá, biểu đồ)

1. `convergence.py`, `conditioning.py` và một phần `plots.py` đã được Anh Tuấn cài trong PR #7. Rà lại, giữ phong cách hình (màu, cỡ chữ) cho thống nhất, không viết lại từ đầu.
2. Cài `plot_bars` (cần cho `compare_all.py`), `plot_sweep` và `tune_k_lambda.py`.

### Tuấn Anh (EDA)

Chạy `01_eda.ipynb` trên dữ liệu đã tiền xử lý.

### Tiến Đạt (baseline, demo)

Không còn việc bắt buộc. Thống nhất tên file mô hình với Hưng và Gia Vang để demo chạy được với mọi `--model`.

## Quy trình

- Người mở PR không tự merge; trưởng nhóm duyệt rồi merge. Repo private miễn phí không có branch protection, nên GitHub không tự chặn được.
- Trước khi cài file của người khác, báo người đó trước. PR #7 cài luôn `convergence.py`, `conditioning.py`, `plots.py` của Trần Công Tuấn Anh: lần này giữ lại vì đúng và giúp `run_sgd.py` chạy được, nhưng dễ làm trùng việc.
- Chốt định dạng phần báo cáo trong `reports/sections/` (`.md` hay `.tex`) trước khi mọi người viết tiếp.
- Sửa các điểm trên bằng **nhánh mới tách từ `develop`** rồi mở PR mới. Không dùng lại nhánh đã merge.
- Đặt tên commit rõ nội dung; tránh "Add files via upload".

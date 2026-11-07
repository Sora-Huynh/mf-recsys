# Tiến độ và nhận xét review

Cập nhật: 07/11/2026, trên nhánh `develop` (sau khi merge PR #4, commit `907016d`).

Tiêu chí "xong" của từng việc nằm trong [ROADMAP.md](../ROADMAP.md), phân công nằm trong [PHAN_CONG.md](PHAN_CONG.md).

## Tổng quan

| Phần | Người | Trạng thái | Ghi chú |
|---|---|---|---|
| Tiền xử lý, CSR | Duy | Xong, đã merge (PR #1) | 800.203 / 100.002 / 100.004 dòng; CSR nhỏ hơn dense khoảng 18,6 lần |
| Baseline | Tiến Đạt | Xong, đã merge (PR #3) | Chạy đúng; còn thiếu test |
| Demo gợi ý `recommend.py` | Tiến Đạt | Đã merge, **còn lỗi** | Hàm `recommend_for_user` đúng; `main()` chạy bị lỗi |
| SGD, Momentum, Adam | Hưng | Xong, đã merge (PR #2) | Test đầy đủ; `run_sgd.py` còn chờ `convergence.py`, `plots.py` |
| SVD | Gia Vang | Xong, đã merge (PR #4) | Toán đúng; còn đường dẫn viết cứng và thiếu test |
| Cholesky, ALS | Anh Tuấn | **Chưa có** | `linear.py`, `als.py`, `run_als.py` vẫn là khung |
| EDA | Tuấn Anh | **Chưa có** | `01_eda.ipynb` vẫn là bản khung, chưa chạy |
| Convergence, plots, conditioning, tune | Trần Công Tuấn Anh | **Chưa có** | Cả 4 file vẫn là khung |
| `compare_all.py` | Duy | Chưa làm (tuần 2) | Cần ALS, `convergence.py`, `plots.py` |

`pytest` trên `develop`: 18 passed, 6 xfailed, 7 xpassed. Các test xfail còn lại là baseline, SVD, Cholesky và ALS.

## Kết quả hiện có (val RMSE trên ML-1M)

| Mô hình | Val RMSE | Ghi chú |
|---|---|---|
| GlobalMean | 1,1165 | Test RMSE 1,1165 |
| BiasBaseline (reg = 10) | 0,9103 | Test RMSE 0,9085 |
| SVD, k = 20 | 0,8840 | Huấn luyện khoảng 0,5–0,7 giây |
| SGD | 0,8717 | Tốt nhất ở epoch 3, sau đó overfit |
| Momentum | 0,8842 | Tốt nhất ở epoch 2, sau đó overfit |
| Adam | 0,8934 | Ở epoch 8 vẫn còn giảm |

Số của SGD được đo bằng bias baseline tạm, cho cùng val RMSE 0,9103 với `BiasBaseline` thật. Cần chạy lại bằng `run_sgd.py` khi có `convergence.py`.

SVD theo k:

| k | 0 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| Val RMSE | 0,9103 | 0,8991 | 0,8916 | 0,8867 | **0,8840** | 0,8843 | 0,8882 |

Sau k ≈ 20, RMSE tăng lại vì SVD coi ô thiếu là 0 như dữ liệu thật. Đây là điểm nên đưa vào báo cáo khi so với ALS.

BiasBaseline theo reg:

| reg | 0 | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| Val RMSE | 0,9103 | 0,9100 | **0,9098** | 0,9103 | 0,9133 | 0,9196 |

Giá trị reg = 10 trong config gần như tốt nhất. Shrinkage chỉ giúp rất ít vì trên ML-1M phim nào cũng có khá nhiều đánh giá.

## Kết quả kiểm tra tính đúng

### BiasBaseline (Tiến Đạt)

| Phép kiểm tra | Kết quả |
|---|---|
| Ma trận đầy đủ r = μ + b_u + b_i, reg = 0 | Khôi phục đúng μ, b_u, b_i, sai số khoảng 1e-15 |
| reg = 10, so với công thức tự tính bằng vòng lặp | b_i và b_u khớp hoàn toàn |
| `residual_matrix` | Giữ cấu trúc thưa, giá trị đúng bằng r − (μ + b_u + b_i), trả về CSR |
| User/phim không có đánh giá trong train | Bias bằng 0, dự đoán bằng μ |
| `GlobalMean` | μ khớp, có clip |
| `recommend_for_user` cho user 42 | Ra top 10 hợp lý: Seven Samurai, Usual Suspects, Schindler's List, Casablanca… |

Hai điểm nên ghi vào báo cáo (không phải lỗi):

1. **Bias chỉ được ước lượng trong một lượt:** tính b_i trước, rồi mới tính b_u. Đây không phải nghiệm Ridge đồng thời. Trên ma trận thưa 50%, dù không có nhiễu và reg = 0, RMSE vẫn còn khoảng 0,058. Docstring gọi là "Ridge thu gọn" thì nên nói rõ là xấp xỉ. Muốn nghiệm chính xác thì phải lặp xen kẽ b_i và b_u.
2. **Nếu reg = 0 mà có phim không có đánh giá nào trong train**, mẫu số n_i + reg bằng 0 nên kết quả ra NaN. Hiện chưa xảy ra vì tiền xử lý đã chuyển mọi user/phim về train, và config dùng reg = 10.

### SVD (Gia Vang)

| Phép kiểm tra | Kết quả |
|---|---|
| Ma trận hạng 3 đầy đủ, baseline = 0, k = 3, 5, 19 | Khôi phục đúng, sai số khoảng 3e-15 |
| Ma trận thưa 200×150 (đi qua `svds`), so với SVD đầy đủ của numpy | σ khớp và giảm dần; ma trận tái tạo khớp xấp xỉ hạng k tốt nhất (Eckart–Young), sai số khoảng 1e-14 |
| Dùng `BiasBaseline` thật với reg = 0 | Phân rã đúng ma trận phần dư |

### SGD (Hưng)

Gradient khớp với sai phân hữu hạn trung tâm, và loss giảm với cả 3 optimizer. Kiểm tra bằng test của Hưng và chạy thử trên ML-1M.

## Việc cần sửa theo từng người

### Hưng (SGD)

1. Khi `convergence.py` xong, bỏ monkeypatch `ConvergenceLog.record` trong `tests/test_models_small.py`.
2. Bỏ giá trị mặc định cứng trong `hyper.get(...)` của `apply_update`; đọc thẳng từ config.
3. SGD và Momentum overfit sau 2–3 epoch, trong khi `run_sgd.py` đánh giá ở epoch cuối. Cần tune `reg`/`lr` trên val, hoặc báo cáo epoch có val RMSE tốt nhất.
4. Việc nhỏ:
   - Dùng lại `load_processed`/`load_csr` thay vì tự đọc file.
   - Đưa import ra đầu file và thêm lại docstring.
   - Gradient đang là tổng theo batch, nên `lr` phụ thuộc `batch_size`; cần ghi rõ trong báo cáo.

### Tiến Đạt (baseline, demo)

1. `python -m src.recommend --user-id 42 --model baseline` bị lỗi vì `main()` gọi những thứ không tồn tại:
   - `IdMaps.load(...)`: không có hàm này. Dùng `load_processed(cfg).maps`.
   - `load_processed_data` từ `src.data.load`: không có. Dùng `load_processed` trong `src.data.preprocess` và `load_csr`.
   - `load_movies()`: thiếu tham số `raw_dir`. Truyền vào `cfg["paths"]["raw_dir"]`.
2. Viết `test_bias_baseline_recovers_known_biases`. Dùng ma trận đầy đủ r = μ + b_u + b_i và reg = 0; RMSE phải khoảng 1e-15.
3. Trong docstring và báo cáo, nói rõ bias được ước lượng một lượt, là xấp xỉ chứ không phải nghiệm Ridge đồng thời.
4. Bỏ các nhánh `getattr(..., "user2idx")`, `"idx2item"`, `"MovieID"`... trong `recommend_for_user`. Tên trường đã cố định là `user_index`, `movie_ids`, nên không cần đoán.

### Gia Vang (SVD)

1. Viết `test_svd_recovers_full_low_rank`. Dùng baseline luôn trả về 0 và ma trận hạng 3 đầy đủ; với k = 3, sai số phải khoảng 1e-15.
2. `run_svd.py` đang viết cứng `data/processed/...` và lấy cột theo vị trí `iloc[:, 0]`. Nên dùng `cfg["paths"]`, `load_processed(cfg)` và lấy cột theo tên `user_idx`, `item_idx`, `rating`.
3. Mô hình được lưu vào `results/svd_model.pkl`, còn `recommend.py` tìm ở `results/models/svd.pkl`. Cần thống nhất một chỗ.
4. Bỏ đoạn dự phòng khi thiếu `residual_matrix` trong `svd.py` (dòng 50–56). `BiasBaseline` đã có hàm này.
5. Hình phổ chỉ có k = 20 giá trị σ. Để chứng minh dữ liệu xấp xỉ hạng thấp, nên vẽ khoảng 100–200 giá trị.

### Anh Tuấn, Tuấn Anh, Trần Công Tuấn Anh

Chưa nộp code. Ưu tiên:

- **Trần Công Tuấn Anh**: `convergence.py` và `plots.py` trước, vì `run_sgd.py` và `compare_all.py` đều cần hai file này.
- **Anh Tuấn**: `linear.py` (Cholesky), sau đó `als.py`. `BiasBaseline` đã có, nên dùng thẳng `residual_matrix`.
- **Tuấn Anh**: chạy `01_eda.ipynb` trên dữ liệu đã tiền xử lý.

## Quy trình

- PR #3 được người mở PR tự merge khi chưa có ai review. Từ nay, mọi PR vào `develop` cần trưởng nhóm duyệt. Repo private miễn phí không có branch protection, nên GitHub không tự chặn được: người mở PR không tự merge, mà để trưởng nhóm merge.
- Sửa các điểm trên bằng **nhánh mới tách từ `develop`** rồi mở PR mới. Không dùng lại nhánh đã merge.
- Đặt tên commit rõ nội dung; tránh "Add files via upload".

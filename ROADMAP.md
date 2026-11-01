# Roadmap 2 tuần

Nguyên tắc: chốt giao diện trước, cài song song sau. Trọng tâm là so sánh **gọn**: mỗi câu
hỏi một thí nghiệm, không quét tràn lan. Giữ đủ phạm vi bắt buộc, bỏ toàn bộ phần tùy chọn.

## Mốc bắt buộc

| Ngày | Mốc |
|---|---|
| 1 | Repo trên GitHub, cả nhóm chạy được, chốt `base.py` |
| 3 | Có dữ liệu sạch, CSR, baseline |
| 7 | SVD, ALS, SGD đều thắng baseline và có đường hội tụ |
| 10 | Chốt tham số, xong bảng và hình tổng; từ đây không đổi kết quả |
| 11 | Mỗi người nộp phần báo cáo và slide của mình |
| 13 | Xong báo cáo, slide, kiểm tra tái lập |
| 14 | Tập thuyết trình, nộp |

## Phân công

Mỗi người sở hữu trọn một nhóm file, nhờ đó ít xung đột git.

| Người | Vai trò | File sở hữu |
|---|---|---|
| A | Điều phối + tiền xử lý | `src/config.py`, `configs/`, `src/models/base.py`, `src/data/*`, `experiments/compare_all.py`, `tests/test_base.py`, `tests/test_models_small.py`, `README.md` |
| B | EDA + ghép báo cáo | `notebooks/01_eda.ipynb`, `reports/main.tex` |
| C | Baseline + demo + ghép slide | `src/models/baseline.py`, `experiments/run_baseline.py`, `src/recommend.py`, `tests/test_metrics.py`, slide |
| D | SVD | `src/models/svd.py`, `experiments/run_svd.py` |
| E | ALS | `src/models/als.py`, `src/solvers/linear.py`, `experiments/run_als.py` |
| F | SGD | `src/models/sgd.py`, `experiments/run_sgd.py` |
| G | Đánh giá + biểu đồ | `src/evaluation/convergence.py`, `src/evaluation/conditioning.py`, `src/visualization/plots.py`, `experiments/tune_k_lambda.py` |

Hai file dùng chung, quy ước để không đụng nhau:
- `configs/default.yaml`: mỗi người chỉ sửa khối của mình.
- `tests/test_models_small.py`: đã tách sẵn từng hàm; D, E, F, C chỉ điền hàm của mình.
- Báo cáo: mỗi người viết một file trong `reports/sections/` và 1–2 slide cho phần đó, nộp trước hết ngày 11; B ghép báo cáo, C ghép slide.

## Giai đoạn 1 (ngày 1–3): Nền móng

| Bước | Ngày | Việc | File cần điền | Đầu ra | Tiêu chí "xong" | Phụ trách |
|---|---|---|---|---|---|---|
| 1.1 | 1 | Khởi tạo git, môi trường, tải dữ liệu | `.gitignore`, `requirements.txt` | Repo chung, mọi người chạy được | Cả 7 người chạy `python -m pytest` và `python -m src.data.download` thành công | A |
| 1.2 | 1 | Chốt hợp đồng mô hình | `src/models/base.py`, `tests/test_base.py` | Giao diện `fit/predict/recommend/save/load`; mô hình con chỉ cài `_fit` và `predict_raw` | Test hợp đồng qua; cả nhóm đồng ý; từ đây chữ ký không đổi nếu không báo trước | A |
| 1.3 | 1–2 | Tiền xử lý | `src/data/preprocess.py` | Train/val/test, ánh xạ ID trong `data/processed/` | Chạy hai lần cho kết quả giống hệt; mọi user/phim ở val, test đều có trong train; tổng đúng 1.000.209 dòng | A |
| 1.4 | 2 | CSR | `src/data/sparse.py` | Ma trận train CSR, bảng bộ nhớ dense vs CSR | `nnz` khớp số dòng train; lưu rồi đọc lại không đổi | A |
| 1.5 | 2–3 | EDA | `notebooks/01_eda.ipynb` | Hình phân phối điểm, đuôi dài, độ thưa | Mỗi hình có một nhận xét dẫn tới lựa chọn ở bước sau | B |
| 1.6 | 1–3 | Hạ tầng vẽ và hội tụ | `src/visualization/plots.py`, `src/evaluation/convergence.py` | Hàm vẽ, `ConvergenceLog` | Hình mẫu đúng quy ước: chữ Anh, rộng, chữ lớn, ghi số trên cột | G |
| 1.7 | 2–3 | Baseline | `src/models/baseline.py`, `experiments/run_baseline.py` | `results/tables/baseline.csv` | Test nhỏ qua; RMSE test cỡ 1,12 (global) và 0,91 (bias) | C |

Trong lúc chờ dữ liệu, D, E, F viết và test trên ma trận nhỏ ngay từ ngày 1.

## Giai đoạn 2 (ngày 1–7): Ba mô hình

D, E, F làm song song.

| Bước | Ngày | Việc | File cần điền | Đầu ra | Tiêu chí "xong" | Phụ trách |
|---|---|---|---|---|---|---|
| 2.1 | 1–5 | SVD | `src/models/svd.py`, `experiments/run_svd.py` | RMSE, thời gian, hình phổ σ | Test khôi phục ma trận hạng 3 qua; RMSE test thấp hơn bias baseline | D |
| 2.2 | 1–3 | Cholesky | `src/solvers/linear.py` | Bộ giải tự cài | Khớp `np.linalg.solve`; báo lỗi với ma trận không SPD | E |
| 2.3 | 3–5 | ALS | `src/models/als.py` | Mô hình ALS | Hàm mục tiêu không tăng; test ô ẩn qua | E |
| 2.4 | 6–7 | Chạy ALS trên ML-1M | `experiments/run_als.py` | Nhật ký hội tụ, mô hình lưu | RMSE test cỡ 0,85–0,87 | E |
| 2.5 | 1–3 | Gradient SGD | `src/models/sgd.py` (`batch_gradients`) | Gradient vector hóa | Khớp sai phân trung tâm | F |
| 2.6 | 4–7 | Ba optimizer | `src/models/sgd.py`, `experiments/run_sgd.py` | Đường hội tụ SGD, Momentum, Adam | Loss giảm với cả ba; một epoch không lặp Python theo từng đánh giá | F |
| 2.7 | 4–7 | Số điều kiện | `src/evaluation/conditioning.py` | Bảng và hình κ theo λ | κ giảm khi λ tăng; khớp công thức (σ²max + λ)/(σ²min + λ) | G |

## Giai đoạn 3 (ngày 8–14): So sánh và báo cáo

| Bước | Ngày | Việc | File cần điền | Đầu ra | Tiêu chí "xong" | Phụ trách |
|---|---|---|---|---|---|---|
| 3.1 | 8–9 | Chọn tham số trên val | `experiments/tune_k_lambda.py` | Bảng và hình ảnh hưởng của k, λ | Tham số tốt nhất đã ghi vào `configs/default.yaml`; không chạm tập test | G |
| 3.2 | 10 | So sánh tổng | `experiments/compare_all.py` | Bảng RMSE, biểu đồ cột, hội tụ theo vòng lặp và thời gian | Một lệnh sinh lại đủ hình bảng từ mô hình đã lưu | A |
| 3.3 | 8–10 | Demo | `src/recommend.py` | `python -m src.recommend --user-id N` in top-10 | Tên phim đúng, không có phim user đã xem; báo lỗi rõ ràng với ID sai | C |
| 3.4 | 11–12 | Kiểm tra tái lập | Không có file mới | Xác nhận từ máy khác | Một người khác clone, chạy lại, RMSE khớp tới 3 chữ số | D |
| 3.5 | 10–13 | Báo cáo | `reports/main.tex`, `reports/sections/*` | PDF | Mỗi phương pháp có: lý do, chương liên quan, kết quả; chép hình cuối vào `reports/figures/` | B + tất cả |
| 3.6 | 11–14 | Slide, tập thuyết trình | `reports/` | Slide | Đã tập đủ giờ, mỗi người giải thích được phần của mình | C + tất cả |

## So sánh nào được làm

Để giữ phạm vi gọn, chỉ làm các so sánh sau:

| Câu hỏi | Đối tượng so sánh | Thí nghiệm |
|---|---|---|
| Mô hình nào chính xác nhất? | Baseline, SVD, ALS, SGD tốt nhất | Bảng RMSE test |
| Mô hình nào hội tụ nhanh hơn? | ALS và SGD tốt nhất, SVD là đường ngang | Theo vòng lặp và theo thời gian |
| k ảnh hưởng thế nào? | ALS (giữ λ mặc định), SVD | Quét 3 giá trị k |
| λ ảnh hưởng thế nào? | ALS (giữ k mặc định) | Quét 4 giá trị λ, kèm số điều kiện |
| Optimizer nào tốt? | SGD, Momentum, Adam ở cùng k, λ | Chỉ trong `run_sgd.py` |

Quét k và λ riêng, không làm lưới đầy đủ: ALS chỉ cần 6 lần chạy thay vì 12.

Bỏ toàn bộ phần tùy chọn: Cholesky vs Gauss-Seidel, gradient tay vs autodiff, float32 vs float64.

## Rủi ro

| Rủi ro | Phương án |
|---|---|
| Bước 1.3 trễ làm cả nhóm chờ | D, E, F viết và test trên ma trận nhỏ trước, không cần ML-1M |
| ALS quá chậm khi k lớn | Biên dịch solver bằng numba; giảm `k_grid` |
| SGD phân kỳ | Giảm `lr`; thử Adam trước vì ít nhạy với `lr` |
| Ngày 7 chưa xong một mô hình | Vẫn chốt so sánh ngày 10 với mô hình đã chạy được; A báo ngay cho cả nhóm vào ngày 7 |
| `results/` không được commit | Hình dùng trong báo cáo chép sang `reports/figures/` |

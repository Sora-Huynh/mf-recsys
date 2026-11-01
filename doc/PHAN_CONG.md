# Phân công công việc

BTL AS2069: Hệ gợi ý dựa trên phân rã ma trận, so sánh SVD, ALS và SGD. Thực hiện trong 2 tuần.

Tiêu chí "xong" của từng việc nằm trong [ROADMAP.md](../ROADMAP.md).

## Bảng phân công

| Thành viên | Vai trò | Tuần 1 | Tuần 2 |
|---|---|---|---|
| **Huỳnh Hữu Duy** (trưởng nhóm) | Điều phối + tiền xử lý | Đẩy repo lên GitHub, chốt `base.py`<br>`preprocess.py`, `sparse.py` | Review, merge code<br>`compare_all.py`<br>Viết phần mở đầu, tiền xử lý, kết luận |
| **Tuấn Anh** | EDA + ghép báo cáo | `01_eda.ipynb` | Viết phần EDA<br>Ghép báo cáo |
| **Tiến Đạt** | Baseline + demo + ghép slide | `baseline.py`, `run_baseline.py` | `recommend.py`<br>Viết phần baseline, demo<br>Ghép slide |
| **Nguyễn Vũ Gia Vang** | SVD | `svd.py`, `run_svd.py` | Kiểm tra tái lập<br>Viết phần SVD |
| **Hồ Nguyễn Anh Tuấn** | ALS | `linear.py` (Cholesky)<br>`als.py`, `run_als.py` | Sửa lỗi, chạy lại nếu cần<br>Viết phần ALS, Cholesky |
| **Nguyễn Minh Hưng** | SGD | `sgd.py`, `run_sgd.py` | Sửa lỗi, chạy lại nếu cần<br>Viết phần SGD, optimizer |
| **Trần Công Tuấn Anh** | Đánh giá + biểu đồ | `plots.py`, `convergence.py`<br>`conditioning.py` | `tune_k_lambda.py`<br>Viết phần số điều kiện, ảnh hưởng k và λ |

## Việc chung của cả nhóm

- Mỗi người làm 1–2 slide cho phần của mình.
- Mỗi người viết phần báo cáo vào một file riêng trong `reports/sections/`.
- Cả nhóm tập thuyết trình trước khi nộp; ai cũng phải giải thích được phần mình làm.

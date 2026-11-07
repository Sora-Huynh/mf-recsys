# 3. Mô hình Baseline Predictor và Kịch bản Gợi ý (Demo)

## 3.1. Phương pháp Baseline Predictor
Mô hình Baseline ước lượng điểm số đánh giá $\hat{r}_{u,i}$ dựa trên mức trung bình toàn cục $\mu$, độ lệch của người dùng $b_u$ (User Bias) và độ lệch của bộ phim $b_i$ (Item Bias):

$$\hat{r}_{u,i} = \mu + b_u + b_i$$

Hàm mục tiêu được tối ưu hóa với thuật toán Cập nhật Luân phiên (Alternating Optimization) kết hợp chuẩn hóa $L_2$:
- **Item Bias:** 
  $$b_i = \frac{\sum_{u \in R(i)} (r_{u,i} - \mu - b_u)}{|R(i)| + \lambda_i}$$
- **User Bias:** 
  $$b_u = \frac{\sum_{i \in R(u)} (r_{u,i} - \mu - b_i)}{|R(u)| + \lambda_u}$$

## 3.2. Kết quả Thử nghiệm
Mô hình Baseline đóng vai trò là mốc chuẩn (benchmark) so sánh cho các phương pháp Phân rã ma trận (SVD, ALS, SGD). 
- **GlobalMean:** Validation RMSE ≈ 1.1171 | MAE ≈ 0.9334
- **BiasBaseline:** Validation RMSE ≈ 0.9120 | MAE ≈ 0.7226

## 3.3. Quy trình Demo Gợi ý (Recommend CLI)
Kịch bản gợi ý thực hiện qua các bước:
1. Xác định danh sách các phim người dùng đã xem trong lịch sử tương tác.
2. Dự đoán điểm cho tất cả các phim chưa xem bằng phương thức `predict_raw()`.
3. Sắp xếp điểm dự đoán giảm dần để chọn ra Top-N phim có điểm số cao nhất.
4. Ánh xạ `item_index` liên tục về `MovieID` gốc và ghép nối với dữ liệu phim để hiển thị tiêu đề, thể loại.
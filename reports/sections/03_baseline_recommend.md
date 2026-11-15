# 3. Mô hình Baseline và kịch bản gợi ý (demo)

## 3.1. Mô hình Baseline

Mô hình ước lượng điểm đánh giá của người dùng $u$ cho phim $i$ bằng trung bình toàn cục $\mu$ (trung bình điểm trên tập train), độ lệch của người dùng $b_u$ và độ lệch của phim $b_i$:

$$\hat{r}_{ui} = \mu + b_u + b_i$$

Hai nhóm bias được ước lượng từ bài toán bình phương tối thiểu có hiệu chỉnh $L_2$ (Ridge):

$$\min_{b_u,\, b_i} \sum_{(u,i)\in\Omega} \bigl(r_{ui} - \mu - b_u - b_i\bigr)^2 + \lambda\Bigl(\sum_u b_u^2 + \sum_i b_i^2\Bigr)$$

trong đó $\Omega$ là tập các ô đã có điểm trong tập train. Khi cố định một vế thì vế còn lại có nghiệm đóng:

$$b_i = \frac{\sum_{u \in R(i)} (r_{ui} - \mu - b_u)}{|R(i)| + \lambda}, \qquad
b_u = \frac{\sum_{i \in R(u)} (r_{ui} - \mu - b_i)}{|R(u)| + \lambda}$$

với $R(i)$ là tập người dùng đã đánh giá phim $i$ và $R(u)$ là tập phim người dùng $u$ đã đánh giá. Hệ số $\lambda$ (trong cấu hình là `reg = 10`) dùng chung cho cả hai nhóm bias. Mẫu số $|R|+\lambda$ kéo bias về 0 khi có ít đánh giá: phim chỉ có vài lượt đánh giá có trung bình nhiễu nên không nên tin hoàn toàn. Khi $|R| \gg \lambda$, bias xấp xỉ trung bình phần dư thông thường.

**Cách cài đặt.** Mặc định, bias chỉ được tính một lượt: lấy $b_u = 0$ để tính $b_i$, sau đó tính $b_u$ từ $b_i$ vừa có. Đây là xấp xỉ của bài toán Ridge, không phải nghiệm đồng thời của $(b_u, b_i)$. Lặp xen kẽ hai công thức trên (tham số `n_iters > 1`) là phương pháp tọa độ luân phiên và sẽ tiến dần tới nghiệm Ridge; một lượt được giữ làm mặc định để baseline đơn giản và nhanh. Với user hoặc phim không có đánh giá nào trong train, bias bằng 0 và dự đoán rơi về $\mu$; bước tiền xử lý đã đảm bảo mọi user và phim ở val, test đều có trong train.

**Vai trò chung.** Baseline còn là tầng dùng chung của cả nhóm: SVD, ALS và SGD đều phân rã ma trận phần dư $r_{ui} - (\mu + b_u + b_i)$ thay vì ma trận điểm gốc, nhờ đó ba phương pháp giải cùng một bài toán và so sánh được với nhau.

## 3.2. Kết quả thử nghiệm

Baseline là mốc mà các phương pháp phân rã ma trận (SVD, ALS, SGD) phải vượt qua.

| Mô hình | Val RMSE | Val MAE | Test RMSE | Test MAE |
|---|---|---|---|---|
| GlobalMean | 1,1165 | 0,9340 | 1,1165 | 0,9332 |
| BiasBaseline ($\lambda = 10$) | 0,9103 | 0,7200 | 0,9085 | 0,7183 |

Thêm bias user/phim giảm RMSE khoảng 18,5% so với dự đoán bằng trung bình toàn cục. Tập test chỉ dùng để báo cáo, không dùng để chọn tham số.

**Ảnh hưởng của hệ số $\lambda$** (Val RMSE của BiasBaseline):

| $\lambda$ | 0 | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| Val RMSE | 0,9103 | 0,9100 | **0,9098** | 0,9103 | 0,9133 | 0,9196 |

$\lambda = 5$ cho kết quả tốt nhất nhưng chỉ hơn $\lambda = 10$ khoảng 0,0005, nên giữ $\lambda = 10$ theo cấu hình. Hiệu chỉnh $L_2$ giúp rất ít trên ML-1M vì phim nào cũng có khá nhiều đánh giá, nên trung bình của từng phim vốn đã ít nhiễu. Khi $\lambda$ quá lớn (từ 25), bias bị kéo về 0 quá mạnh nên mô hình khớp dữ liệu kém đi.

## 3.3. Kịch bản gợi ý (demo)

Chạy bằng dòng lệnh, ví dụ cho người dùng có UserID 42:

```bash
python -m src.recommend --user-id 42 --model baseline
```

Quy trình gồm các bước:

1. Ánh xạ `UserID` gốc sang chỉ số liên tục; báo lỗi rõ ràng nếu UserID không tồn tại.
2. Dự đoán điểm cho mọi phim bằng `predict_raw()`, rồi loại các phim người dùng đã đánh giá trong tập train.
3. Sắp xếp giảm dần theo điểm **chưa clip** để chọn Top-N phim. Sau khi clip về thang 1–5, nhiều phim cùng bằng 5 nên thứ tự mất ý nghĩa; sắp xếp ổn định giúp kết quả tái lập được khi có điểm bằng nhau.
4. Ánh xạ chỉ số phim về `MovieID` gốc và ghép với dữ liệu phim để hiển thị tiêu đề, thể loại.

Kết quả cho người dùng 42:

| Hạng | Phim | Thể loại | Điểm dự đoán |
|---|---|---|---|
| 1 | Seven Samurai (1954) | Action, Drama | 4,716 |
| 2 | The Usual Suspects (1995) | Crime, Thriller | 4,674 |
| 3 | The Wrong Trousers (1993) | Animation, Comedy | 4,672 |
| 4 | Schindler's List (1993) | Drama, War | 4,667 |
| 5 | A Close Shave (1995) | Animation, Comedy, Thriller | 4,660 |
| 6 | Sunset Blvd. (1950) | Film-Noir | 4,653 |
| 7 | Rear Window (1954) | Mystery, Thriller | 4,636 |
| 8 | Paths of Glory (1957) | Drama, War | 4,605 |
| 9 | Casablanca (1942) | Drama, Romance, War | 4,590 |
| 10 | The Third Man (1949) | Mystery, Thriller | 4,586 |

**Hạn chế của baseline.** Vì $\hat{r}_{ui} = \mu + b_u + b_i$, với cùng một người dùng thì $\mu + b_u$ là hằng số, nên thứ tự các phim chỉ phụ thuộc vào $b_i$. Danh sách gợi ý của mọi người dùng vì thế có cùng thứ tự các phim được đánh giá cao nhất, chỉ khác nhau ở việc loại phim mỗi người đã xem. Điều này đúng với kết quả ở trên, toàn các phim kinh điển có điểm trung bình cao. Baseline vì vậy dự đoán điểm khá tốt nhưng không cá nhân hóa việc xếp hạng. Đây chính là lý do cần các mô hình phân rã ma trận, vốn học thêm tương tác $p_u^\top q_i$ giữa người dùng và phim.
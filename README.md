# 📊 E-Commerce Data Analytics & Predictive Machine Learning Framework

> **Đồ án môn học:** Máy Học Thống Kê (Statistical Machine Learning)
>
> **Trường:** Đại học Công nghệ TP. HCM (HUTECH) — Khoa Đào tạo Chất lượng cao / Ngành Khoa học Dữ liệu
>
> **Giảng viên hướng dẫn:** Th.S Nguyễn Quang Phúc
>
> **Năm học:** 2026

## 👨‍💻 Thành viên thực hiện (Team Members)

| 

| **STT** | **Họ và Tên** | **Mã số Sinh viên** | **Vai trò** | 
| 1 | **Nguyễn Đăng Khoa** | `2386400026` | Thành viên | 
| 2 | **Võ Thị Ngọc Hiền** | `2386400019` | Thành viên | 
| 3 | **Phan Xuân Dương** | `2386400966` | Thành viên | 
| 4 | **Nguyễn Đức Vinh** | `23864.....` | Thành viên | 

## 📌 Tổng quan dự án (Project Overview)

Trong bối cảnh Thương mại Điện tử (E-Commerce) phát triển mạnh mẽ trên thị trường đa quốc gia (Hoa Kỳ, Đức, Pháp, Anh, v.v.), việc thấu hiểu hành vi khách hàng, quản trị doanh thu và giữ chân người dùng đóng vai trò chiến lược sống còn.

Dự án này xây dựng một hệ thống phân tích toàn diện và áp dụng các mô hình **Máy Học Thống Kê (Statistical Machine Learning)** tiên tiến từ cơ bản đến nâng cao:

1. **Phân tích Hồi quy (Linear & Logistic Regression):** Dự đoán tổng giá trị hóa đơn giao dịch & xác suất trả hàng.

2. **Phân cụm Không giám sát (K-Means, DBSCAN, PCA):** Phân đoạn nhóm khách hàng chiến lược và phát hiện điểm dị biệt.

3. **Phân lớp Dự báo Churn (K-NN, SVM, Decision Tree, Random Forest, MLP):** Phát hiện sớm rủi ro khách hàng rời bỏ dịch vụ kết hợp xử lý mất cân bằng lớp (SMOTE) & phân tích RFM.

4. **Dự báo Chuỗi thời gian (ARIMA, SARIMA, Holt-Winters):** Dự báo xu hướng doanh thu thương mại điện tử theo tuần giai đoạn 2020–2026.

## 📁 Cấu trúc Tập dữ liệu (Datasets & Schema)

Hệ thống sử dụng bộ dữ liệu tổng hợp giao dịch E-Commerce bao gồm 4 tập dữ liệu quan trọng có liên kết logic chặt chẽ:

* **`customers.csv` (8,000 khách hàng):** Định danh (`customer_id`), quốc gia (`country`), tuổi, giới tính, cấp thành viên (`membership_tier`), tổng chi tiêu (`total_spend_usd`), số ngày từ lần mua cuối (`days_since_last_purchase`), và biến mục tiêu rời bỏ (`churned`).

* **`orders.csv` (25,000 giao dịch):** Chi tiết đơn hàng (`order_id`, `order_date`), giá trị đơn (`total_amount_usd`), đơn giá, số lượng, giảm giá, chi phí vận chuyển, phương thức thanh toán, thiết bị, và trạng thái hoàn trả (`returned`).

* **`monthly_revenue.csv`:** Tập chuỗi thời gian tổng hợp doanh thu theo tháng/quý.

* **`products_summary.csv`:** Hiệu suất kinh doanh và đánh giá theo dòng sản phẩm.

## ⚙️ Quy trình Tiền xử lý & Feature Engineering

1. **Xử lý khuyết thiếu & trùng lặp:** Loại bỏ trùng lặp khóa chính; điền giá trị thiếu (Missing Values) bằng `Median` và gán nhãn `Unknown`.

2. **Xử lý Ngoại lai (Outlier Clipping):** Áp dụng kỹ thuật `IQR Clipping` giới hạn khoảng $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$.

3. **Biến đổi Đặc trưng (Feature Engineering):** Tính số ngày gắn kết (`customer_tenure_days`), gán nhãn thứ bậc (`membership_tier_ordinal`), biến đổi `One-Hot Encoding` biến danh định.

4. **Chuẩn hóa & Giảm chiều:** Dùng `StandardScaler` đưa về thang đo chuẩn $N(0,1)$ và dùng `PCA` nén không gian đặc trưng.

## 🚀 Nội dung & Kết quả Nghiên cứu Chi tiết

### 1️⃣ Mô hình Hồi quy (Regression Models)

* **Hồi quy Tuyến tính (Linear Regression - Dự đoán `total_amount_usd`):**

  * Mô hình đa biến tinh chỉnh bằng thuật toán **Backward Elimination** giảm từ 41 biến xuống **5 biến độc lập cốt lõi**, đạt $R^2 = 0.8016$, $RMSE = 66.44$.

  * Kiểm soát đa cộng tuyến thông qua chỉ số $VIF < 5$ và kiểm định độc lập phần dư Durbin-Watson $\approx 2.0$.


* **Hồi quy Logistic (Logistic Regression - Dự đoán `returned`):**

  * Do mất cân bằng lớp nghiêm trọng (chỉ \~8.3% đơn bị hoàn trả), mô hình gốc dự đoán 100% lớp 0 ($Recall = 0$).

  * **Giải pháp SMOTE:** Tái cân bằng dữ liệu giúp $Recall$ tăng lên **14%**, $ROC-AUC$ tăng lên **0.5122**.

| **Trạng thái** | **Lớp 0 (Không trả)** | **Lớp 1 (Có trả)** | **Tổng mẫu Train** | 
| **Trước SMOTE** | \~16,090 | \~1,410 | \~17,500 | 
| **Sau SMOTE** | \~16,090 | \~16,090 | \~32,180 | 

### 2️⃣ Mô hình Phân cụm & Giảm chiều (Clustering & Dimensionality Reduction)

* **PCA (Principal Component Analysis):** Chọn $n\_components = 10$ giải thích $>80\%$ tổng phương sai dữ liệu.


* **K-Means vs DBSCAN Comparison:**

  * **K-Means (**$K=2$**):** Đạt **Silhouette Score = 0.2512** vượt trội. Chia thành 2 cụm rõ rệt: *Cụm 0 (Khách hàng giá trị cao - 21.9%)* và *Cụm 1 (Khách hàng phổ thông - 78.1%)*.

  * **DBSCAN (**$\epsilon=2.018, min\_samples=5$**):** Tách được 4 cụm mật độ và lọc $5.72\%$ điểm nhiễu (outliers).

### 3️⃣ Phân lớp Cơ bản & Dự báo Rời bỏ (Churn Classification: K-NN vs SVM)

Nhằm phát hiện khách hàng có nguy cơ ngưng tương tác (`churned = 1`):

* **K-NN (**$K=3$**):** Độ chính xác tổng thể $Accuracy = 90.0\%$, nhưng bị thiên lệch lớp đa số ($Recall\_lớp\_1 = 0.08$).

* **SVM (`class_weight='balanced'`):** Cho khả năng bắt Churn tốt hơn gấp 7 lần với $Recall\_lớp\_1 = 0.59$ và $ROC-AUC = 0.687$.

### 4️⃣ Các Mô hình Học máy Nâng cao (Advanced Machine Learning & EDA)

* **Phân tích Khám phá EDA & RFM:**

  * Tập dữ liệu tập trung lớn nhất tại Hoa Kỳ (2,509 KH), Vương quốc Anh (800 KH), Ấn Độ (711 KH).

  * Khách hàng hạng **Free** có tỷ lệ Churn cao nhất (9.5%), trong khi hạng **Silver/Platinum** có độ gắn kết cao hơn đáng kể.


* **Đánh giá So sánh Mô hình Học máy Churn Prediction:** Thử nghiệm trên 1,600 mẫu kiểm thử tĩnh kết hợp với SMOTE:

| **Model** | **Accuracy** | **Precision (Lớp 1)** | **Recall (Lớp 1)** | **F1-Score** | **ROC-AUC** | 
| **Logistic Regression** | **0.8778** | 0.3135 | **0.3059** | **0.3087** | **0.75** | 
| **Random Forest** | 0.9105 | **0.4087** | 0.0315 | 0.0583 | 0.73 | 
| **Decision Tree** | 0.8936 | 0.3606 | 0.2430 | 0.2891 | 0.62 | 
| **MLP (Deep Neural Net)** | 0.9106 | 0.3750 | 0.0105 | 0.0201 | 0.62 | 

> 🌟 **Kết luận:** **Logistic Regression** đạt chỉ số $ROC-AUC = 0.75$ và $Recall = 30.59\%$ tốt nhất trong việc nhận diện khách hàng rời đi, trong khi **Random Forest** tối ưu cho bài toán cần $Precision$ cao.

### 5️⃣ Dự báo Chuỗi thời gian (Time-Series Revenue Forecasting)

Tổng hợp 326 tuần doanh thu thương mại điện tử từ 2020 đến 2026. Chuỗi gốc đạt tính dừng ADF ($p-value = 2.24 \times 10^{-30} < 0.05 \implies d=0$).

* **Kết quả Đánh giá Mô hình trên Tập Test 12 Tuần:**

  * **Holt-Winters Additive (Expanding Window):** Xếp hạng 1 toàn diện với $MAPE = 14.27\%$ (Độ chính xác đạt $85.73\%$), $MAE = 1663.50$.

  * **SARIMA(1,1,1)x(1,0,0)12:** Bắt nhịp mùa vụ Quý tốt với $MAPE = 14.72\%$.

  * **ARIMA(1,0,1):** Cho xu hướng phẳng dài hạn với $MAPE = 15.53\%$.

## 🛠️ Công nghệ & Thư viện Sử dụng (Tech Stack)

* **Language:** Python 3.10+

* **Data Processing & Analysis:** `pandas`, `numpy`, `scipy`

* **Machine Learning & Analytics:** `scikit-learn`, `imbalanced-learn` (SMOTE)

* **Time-Series Forecasting:** `statsmodels` (ARIMA, SARIMA, Holt-Winters)

* **Data Visualization:** `matplotlib`, `seaborn`

## 💡 Đề xuất Ứng dụng Quản trị Kinh doanh (Business Action Plan)

1. **Khách hàng giá trị cao (High-Value Cluster - K-Means):** Triển khai đặc quyền Chăm sóc VIP, Loyalty Reward cá nhân hóa và chính sách miễn phí hoàn trả.

2. **Khách hàng Free Tier & Churn Risk:** Tối ưu hóa chuỗi Email Marketing cá nhân hóa, tung mã kích thích chuyển đổi sang hạng trả phí (Silver/Gold).

3. **Quản trị Chuỗi cung ứng:** Áp dụng mô hình **Holt-Winters (Expanding Window)** dự báo nhu cầu hàng hóa theo tuần nhằm tối ưu kho bãi và giảm chi phí vận hành.

## 📜 Tài liệu Tham khảo (References)

* \[1\] P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," *J. Comput. Appl. Math.*, 1987.

* \[2\] N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," *J. Artif. Intell. Res.*, 2002.

* \[3\] G. E. P. Box, G. M. Jenkins, and G. C. Reinsel, *Time Series Analysis: Forecasting and Control*, 5th ed., Wiley, 2015.

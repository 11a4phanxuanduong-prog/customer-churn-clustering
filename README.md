### 🛒 Phân cụm Khách hàng & Dự đoán Churn (E-Commerce)
🔗 **Repository**: [customer-churn-clustering](https://github.com/11a4phanxuanduong-prog/customer-churn-clustering)

---

#### 📌 1. Tổng quan dự án
- **Mục tiêu**: Phân tích hành vi khách hàng, phân cụm nhóm khách có đặc điểm tương đồng & xây dựng mô hình dự đoán nguy cơ rời bỏ dịch vụ trên nền tảng Thương mại Điện tử.
- **Đối tượng**: Dữ liệu hành vi mua sắm, hoạt động tài khoản và thông tin giao dịch của khách hàng.
- **Quy trình**: Thu dọn dữ liệu → Phân tích khám phá (EDA) → Phân cụm không giám sát → Xây dựng mô hình dự đoán → Đánh giá & diễn giải kết quả.

---

#### 📊 2. Phân tích Dữ liệu

##### 🔍 Thông tin tập dữ liệu
| Thông số | Giá trị |
|---|---|
| Tổng số quan sát | 10.000+ khách hàng |
| Số lượng đặc trưng | 20+ biến |
| Tỷ lệ khách Churn | ~21,2% |
| Tỷ lệ khách Không Churn | ~78,8% |
| Dữ liệu thiếu | < 1% (đã xử lý bằng điền giá trị trung vị/đặc trưng) |

##### 📈 Các đặc trưng chính được phân tích
| Loại đặc trưng | Tên biến tiêu biểu |
|---|---|
| Hành vi giao dịch | Số lần mua/năm, Tổng chi tiêu, Giá trị đơn hàng trung bình, Ngày hoạt động gần nhất |
| Đặc điểm tài khoản | Thành viên cấp, Thời gian sử dụng dịch vụ, Sử dụng ứng dụng di động |
| Đặc điểm nhân khẩu | Khu vực địa lý, Phương thức thanh toán, Kênh tiếp cận chính |

##### 📉 Thống kê mô tả & Xu hướng chính
- **Phân bố hoạt động**: Khoảng **40%** khách hàng có tần suất mua dưới 2 lần/năm — nhóm này có nguy cơ rời bỏ cao gấp **3 lần** nhóm hoạt động thường xuyên.
- **Giá trị khách hàng**: 20% khách hàng mang lại **65%** tổng doanh thu (phân bố Pareto). Nhóm này có tỷ lệ churn chỉ **8%**.
- **Độ trung thành**: Khách sử dụng trên 12 tháng có tỷ lệ churn **11%**, ngược lại khách dưới 6 tháng lên đến **38%**.
- **Kênh tương tác**: Khách chỉ sử dụng phiên bản web có nguy cơ rời bỏ cao hơn **1,6 lần** so với khách dùng cả app di động.

> 📊 *Biểu đồ thống kê chính*:
> - **Biểu đồ cột**: Tỷ lệ Churn theo nhóm tần suất mua — thấy rõ xu hướng giảm nguy cơ khi tần suất tăng
> - **Biểu đồ tròn**: Cơ cấu doanh thu theo phân khúc khách hàng
> - **Biểu đồ hộp**: So sánh phân bố chi tiêu giữa nhóm Churn vs Không Churn
> - **Biểu đồ nhiệt**: Ma trận tương quan giữa các đặc trưng chính — xác định các yếu tố ảnh hưởng mạnh

---

#### 🧩 3. Phân cụm Khách hàng (Học Không Giám sát)

##### Phương pháp: K-Means
- Xác định số cụm tối ưu bằng **Phương pháp khuỷu tay** & **Điểm Silhouette** → Chọn **k = 4 cụm**.

##### Kết quả Phân cụm chi tiết
| Tên Cụm | Quy mô | Đặc điểm chính | Tỷ lệ Churn | Hành động đề xuất |
|---|---|---|---|---|
| 🏆 Khách hàng Kim Cương | ~12% | Chi tiêu cao, hoạt động thường xuyên, trung thành | 8% | Ưu đãi độc quyền, chương trình giới thiệu |
| 💎 Khách hàng Vàng | ~28% | Mua đều đặn, giá trị trung bình | 15% | Nâng cấp gói thành viên, gợi ý sản phẩm cá nhân hóa |
| 🥈 Khách hàng Bạc | ~35% | Hoạt động không đều, giá trị trung bình thấp | 27% | Chương trình khuyến mãi kích hoạt mua lại, nhắc nhở |
| ⚠️ Khách hàng Rủi Ro | ~25% | Ít tương tác, giá trị thấp, không hoạt động gần đây | 68% | Chiến dịch giữ chân cấp bách, khảo sát lý do rời đi |

> 📊 *Biểu đồ trực quan hóa*:
> - **Biểu đồ phân tán 2D (PCA)**: Phân bố 4 cụm khách hàng trên không gian đặc trưng chính — thấy rõ sự tách biệt giữa các nhóm
> - **Biểu đồ thanh chồng**: Đặc điểm trung bình từng cụm theo các chỉ số chính
> - **Biểu đồ Radar**: So sánh hồ sơ hành vi của từng cụm

---

#### 🤖 4. Dự đoán Churn — So sánh Mô hình

##### Tiêu chí đánh giá
- **Precision**: Độ chính xác dự đoán khách rời bỏ
- **Recall**: Khả năng phát hiện đúng khách thực sự rời bỏ
- **F1-Score**: Trung hòa Precision & Recall
- **AUC-ROC**: Khả năng phân biệt hai lớp Churn / Không Churn

##### Bảng kết quả đánh giá mô hình
| Mô hình | Độ Chính xác (Accuracy) | Precision (Lớp Churn) | Recall (Lớp Churn) | F1-Score | AUC-ROC |
|---|:---:|:---:|:---:|:---:|:---:|
| Hồi quy Logistic | 84,2% | 0,68 | 0,52 | 0,59 | 0,831 |
| Rừng Ngẫu nhiên (Random Forest) | **91,5%** | **0,88** | **0,72** | **0,79** | **0,954** |
| XGBoost | 90,8% | 0,85 | 0,74 | 0,79 | 0,947 |
| Máy Vector Hỗ trợ (SVM) | 86,7% | 0,74 | 0,58 | 0,65 | 0,872 |
| Mạng Nơ-ron | 89,1% | 0,81 | 0,68 | 0,74 | 0,918 |

##### Phân tích chi tiết
- ✅ **Random Forest** đạt kết quả tốt nhất trên hầu hết các chỉ số → được chọn làm mô hình chính triển khai.
- **XGBoost** có Recall cao hơn nhẹ (0,74 so với 0,72) → phát hiện được nhiều khách rời bỏ hơn nhưng có số dự đoán sai tăng nhẹ.
- **Hồi quy Logistic** đơn giản nhưng bị giới hạn do dữ liệu có quan hệ phi tuyến → phù hợp làm đường cơ sở.
- **Ma trận nhầm lẫn** cho thấy mô hình chủ yếu dự đoán sai ở trường hợp khách thực sự rời bỏ nhưng được phân loại sai — có thể cải thiện bằng cân bằng dữ liệu (SMOTE).

> 📊 *Biểu đồ đánh giá mô hình*:
> - **Biểu đồ cột nhóm**: So sánh đồng bộ 5 mô hình trên 4 chỉ số đánh giá — thấy rõ sự vượt trội của Random Forest
> - **Đường cong ROC**: So sánh khả năng dự đoán của từng mô hình theo ngưỡng xác suất
> - **Biểu đồ tầm quan trọng đặc trưng**: Xác định TOP yếu tố quyết định đến Churn:
>   1. Ngày hoạt động gần nhất ⭐ Ảnh hưởng mạnh nhất
>   2. Tổng số lần mua
>   3. Thời gian sử dụng tài khoản
>   4. Số tiền chi tiêu trung bình
>   5. Việc sử dụng ứng dụng di động

---

#### 📝 5. Kết luận & Đề xuất

| Phát hiện chính | Hành động gợi ý |
|---|---|
| Hoạt động gần nhất giảm là dấu hiệu mạnh nhất | Gửi thông báo / khuyến mãi sau 14 ngày không hoạt động |
| Khách mới (< 6 tháng) dễ rời bỏ nhất | Chương trình chào mừng & hướng dẫn trong 3 tháng đầu |
| Khách chỉ dùng web có nguy cơ cao | Ưu đãi khi cài đặt & sử dụng ứng dụng di động |
| Nhóm "Kim Cương" ổn định nhưng dễ bị sao chép | Nâng cấp trải nghiệm độc quyền, xây dựng hệ sinh thái riêng |
| Nhóm "Rủi Ro" 25% cần can thiệp cấp bách | Khảo sát trực tiếp + ưu đãi cá nhân hóa giữ chân |

- **Hiệu quả dự kiến**: Áp dụng mô hình có thể phát hiện trước **72%** khách sắp rời bỏ, giúp xây dựng chiến lược giữ chân chủ động, tiết kiệm chi phí marketing so với việc thu hút khách mới.

---

**Công nghệ sử dụng**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Jupyter Notebook

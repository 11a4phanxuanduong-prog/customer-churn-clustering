# Phân cụm Khách hàng & Dự đoán Churn (E-Commerce)

## Bài toán
Phân nhóm hành vi khách hàng và dự đoán khả năng rời bỏ (churn) 
trên bộ dữ liệu 8.000 khách hàng E-Commerce.

## Vai trò cá nhân
Phụ trách toàn bộ phần Phân cụm (K-Means, DBSCAN) và Phân loại 
(K-NN, SVM) trong đồ án nhóm môn Máy học Thống kê.

## Phương pháp
- Tiền xử lý: IQR clipping, StandardScaler, OneHotEncoder, PCA (10 
  thành phần cho mô hình, 2 thành phần để trực quan hóa)
- Phân cụm: K-Means (chọn K bằng Elbow + Silhouette), DBSCAN 
  (tối ưu epsilon bằng K-distance plot)
- Phân loại churn: K-NN (chọn K theo F1-score), SVM 
  (class_weight='balanced' do dữ liệu lệch lớp ~8.9% churn)

## Kết quả

### Phân cụm
| Chỉ số | K-Means | DBSCAN |
|---|---|---|
| Số cụm | 2 | 4 |
| Silhouette Score | 0.251 | 0.131 |
| Calinski-Harabasz | 1679.9 | 319.2 |
| Tỉ lệ điểm nhiễu | 0% | 10.27% |

→ K-Means cho kết quả phân cụm tốt hơn, chia rõ 2 nhóm: 
Khách hàng giá trị cao (21.9%) và Khách hàng phổ thông (78.1%)

### Phân loại churn
| Chỉ số | K-NN (K=3) | SVM |
|---|---|---|
| Accuracy | 0.900 | 0.660 |
| Recall (churn) | 0.080 | 0.590 |
| ROC-AUC | 0.570 | 0.687 |

→ SVM phát hiện churn tốt hơn đáng kể (Recall cao hơn) dù 
Accuracy thấp hơn — phù hợp hơn cho dữ liệu mất cân bằng lớp

[Xem biểu đồ ROC](results/roc_curve_knn_svm.png)

## Công nghệ
Python, Scikit-learn, Pandas, Matplotlib, Seaborn

## Cách chạy
\`\`\`bash
pip install -r requirements.txt
jupyter notebook notebooks/
\`\`\`

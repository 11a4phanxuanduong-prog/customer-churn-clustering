# 🛒 E-Commerce Customer Analytics: Segmentation, Churn Prediction & Revenue Forecasting

**Course project:** Statistical Machine Learning · HUTECH University, Data Science · 2026
**Supervisor:** MSc. Nguyễn Quang Phúc
**Team:** Nguyễn Đăng Khoa, Võ Thị Ngọc Hiền, **Phan Xuân Dương**, Nguyễn Đức Vinh

📄 [Full report (Vietnamese, PDF)](%C4%90%E1%BB%92%20%C3%81N%20M%C3%94N%20M%C3%81Y%20H%E1%BB%8CC%20TH%E1%BB%90NG%20K%C3%8A.pdf) · 📓 [Main notebook](code_k_mean-dbscan-knn-svm%20%281%29.ipynb)

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Data](#-data)
- [Preprocessing & Feature Engineering](#-preprocessing--feature-engineering)
- [Results](#-results)
- [Business Recommendations](#-business-recommendations)
- [Tech Stack](#-tech-stack)

---

## 📖 Overview
An end-to-end statistical machine-learning study on a multi-country e-commerce dataset. It answers four business questions:

| # | Business question | Techniques |
|---|---|---|
| 1 | What drives order value and returns? | Linear Regression (backward elimination), Logistic Regression + **SMOTE** |
| 2 | Which customer segments exist? | **PCA**, **K-Means**, **DBSCAN** |
| 3 | Who is about to churn? | K-NN, SVM, Decision Tree, Random Forest, MLP, Logistic Regression, **RFM** analysis |
| 4 | How will weekly revenue evolve? | **ARIMA**, **SARIMA**, **Holt-Winters** |

## 🗂 Data
| File | Content |
|---|---|
| `customers.csv` | **8,000 customers**: country, age, gender, membership tier, total spend, recency, churn label |
| `orders.csv` | **25,000 orders**: amount, unit price, quantity, discount, shipping, payment, device, returned |
| `monthly_revenue.csv` | Aggregated monthly / quarterly revenue |
| `products_summary.csv` | Product-line performance and ratings |

## ⚙️ Preprocessing & Feature Engineering
1. Removed duplicate keys and imputed missing values (median / `Unknown`).
2. **IQR clipping** of outliers to [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
3. Built features: `customer_tenure_days`, ordinal membership tier, one-hot encoding.
4. `StandardScaler` → **PCA (10 components, > 80% variance)**.

## 📊 Results

### 1️⃣ Regression
- **Linear Regression** (order value): backward elimination cut 41 → **5 predictors**, with **R² = 0.80** and RMSE = 66.4. VIF < 5 and Durbin-Watson ≈ 2.0.
- **Logistic Regression** (returns, only 8.3% positive): the baseline predicted only class 0 (recall = 0). **SMOTE** raised recall to 14%.

### 2️⃣ Customer Segmentation
| Method | Result |
|---|---|
| **K-Means (K = 2)** | Silhouette **0.251**: *High-value customers (21.9%)* vs *Mainstream customers (78.1%)* |
| **DBSCAN** (ε = 2.018, min_samples = 5) | 4 density clusters + **5.72% outliers** detected |

### 3️⃣ Churn Prediction (1,600 test samples, SMOTE)
| Model | Accuracy | Precision (churn) | Recall (churn) | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.878 | 0.314 | **0.306** | **0.309** | **0.75** |
| Random Forest | 0.911 | **0.409** | 0.032 | 0.058 | 0.73 |
| Decision Tree | 0.894 | 0.361 | 0.243 | 0.289 | 0.62 |
| MLP | 0.911 | 0.375 | 0.011 | 0.020 | 0.62 |
| SVM (balanced) | — | — | 0.59 | — | 0.687 |

💡 Accuracy is misleading on imbalanced churn data: Random Forest and MLP reach ~91% accuracy but miss almost every churner. **Logistic Regression gives the best churn detection (ROC-AUC 0.75)**, and a class-balanced SVM catches **7× more churners than K-NN**.

**EDA / RFM:** the largest markets are the US (2,509), the UK (800) and India (711). **Free-tier** members churn the most (9.5%).

### 4️⃣ Weekly Revenue Forecasting (326 weeks, 2020–2026, 12-week test)
| Model | MAPE |
|---|---|
| **Holt-Winters Additive (expanding window)** | **14.27%** (85.7% accuracy) |
| SARIMA(1,1,1)×(1,0,0)₁₂ | 14.72% |
| ARIMA(1,0,1) | 15.53% |

## 💼 Business Recommendations
1. **High-value cluster:** VIP care, personalized loyalty rewards, free returns.
2. **Free-tier and at-risk customers:** personalized email journeys and upgrade incentives (Silver/Gold).
3. **Operations:** use Holt-Winters weekly forecasts for inventory and warehouse planning.

## 🛠 Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![statsmodels](https://img.shields.io/badge/statsmodels-4051B5?style=for-the-badge) ![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-6C757D?style=for-the-badge) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

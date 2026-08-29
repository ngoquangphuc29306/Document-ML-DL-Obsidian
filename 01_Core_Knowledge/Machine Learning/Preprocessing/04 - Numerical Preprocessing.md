---
title: "Numerical Preprocessing"
tags: [preprocessing, numerical-features, scaling]
---

# Numerical Preprocessing

## 1. Khi nào scaling quan trọng?

Đặc biệt với:
- KNN / K-Means;
- SVM;
- PCA;
- Linear/Logistic Regression;
- Neural Network.
Tree models thường không bắt buộc scale.
## 2. Standardization

$$
x'=\frac{x-\mu}{\sigma}
$$

Sau transform, train feature có mean gần 0 và std gần 1.

> [!important]
> StandardScaler **không yêu cầu dữ liệu phải Gaussian**.

### Dùng khi
- model nhạy scale;
- feature có đơn vị rất khác nhau;
- không cần range cố định.

### Hạn chế
Mean/std vẫn bị ảnh hưởng bởi outlier.

## 3. Min-Max Scaling

$$
x'=\frac{x-x_{min}}{x_{max}-x_{min}}
$$
Thường đưa train data về \([0,1]\).

Hạn chế:
- rất nhạy min/max cực trị;
- dữ liệu mới ngoài range train có thể transform ra ngoài \([0,1]\).
## 4. RobustScaler
$$x'=\frac{x-median(x)}{Q_3-Q_1}$$
Ít nhạy outlier hơn.
## 5. Log Transform
$$x'=\log(1+x)$$
Hữu ích cho right-skewed positive data như revenue, views, transaction amount.
## 6. Power Transform
- Box-Cox: dữ liệu dương.
- Yeo-Johnson: dùng được với zero/âm.
## 7. Chọn nhanh

| Trường hợp        | Gợi ý                    |
| ----------------- | ------------------------ |
| SVM/KNN/PCA       | StandardScaler           |
| Nhiều outlier     | RobustScaler             |
| Cần range cố định | MinMaxScaler             |
| Right-skewed      | Log / Power transform    |
| Tree-only         | Thường không cần scaling |

> [!warning]
> Chỉ `fit` scaler trên train.

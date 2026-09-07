---
title: "Tổng quan Data Preprocessing"
tags: [machine-learning, preprocessing, fundamentals]
---

# Tổng quan Data Preprocessing

## 1. Khái niệm

Data preprocessing biến dữ liệu thô thành dữ liệu:
- đúng kiểu;
- sạch và nhất quán;
- phù hợp với model;
- không làm rò rỉ thông tin từ validation/test.

Ví dụ:

| age | income | city | target |
|---:|---:|---|---:|
| 22 | 8,000,000 | Hanoi | 1 |
| NaN | 12,000,000 | Hue | 0 |
| 150 | -1 | Hanoi | 1 |

Cần đặt câu hỏi:
- `age = NaN` xử lý thế nào?
- `age = 150` là lỗi hay outlier thật?
- `income = -1` có phải sentinel cho missing?
- `city` là nominal hay ordinal?

## 2. Preprocessing phụ thuộc model

### Distance-based
KNN, K-Means, SVM rất nhạy với scale.

### Linear models
Linear/Logistic Regression thường cần scaling, encoding; skewness đôi khi cần transform.

### Tree-based
Decision Tree, Random Forest, Gradient Boosting, XGBoost thường **không bắt buộc scaling** vì split theo threshold từng feature.

> [!NOTE]
> Tree model không cần scaling không có nghĩa là không cần xử lý schema, missing, category hay leakage.

## 3. Fit và Transform

- `fit`: học tham số từ train.
- `transform`: áp dụng tham số đã học.
- `fit_transform`: fit rồi transform cùng một tập.

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
```

## 4. Quy trình

```text
Hiểu dữ liệu
→ Chọn split phù hợp
→ Split
→ Fit preprocessing trên train
→ Transform train/valid/test
→ Train model
→ Evaluate
```

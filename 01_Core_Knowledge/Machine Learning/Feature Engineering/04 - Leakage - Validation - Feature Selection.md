---
title: "Leakage, Validation và Feature Selection"
tags: [feature-engineering, leakage, validation, feature-selection]
---

# Leakage, Validation và Feature Selection

# 1. Data leakage

Data leakage xảy ra khi model dùng thông tin không tồn tại tại thời điểm dự đoán.

Hậu quả:

- Validation rất cao
- Production giảm mạnh
- Đánh giá không đáng tin

## Target leakage

Ví dụ dự đoán trả nợ nhưng dùng:

```text
loan_repaid_date
final_status
```

Hai feature này chỉ có sau kết quả.

## Train-test contamination

Sai:

```python
scaler.fit(X_all)
```

Đúng:

```python
scaler.fit(X_train)
```

Mọi transformer phải fit trên train:

- Imputer
- Scaler
- PCA
- Feature selector
- Target encoder

## Group leakage

Cùng user, bệnh nhân, tài liệu hoặc thiết bị xuất hiện ở cả train và test.

Giải pháp:

- GroupKFold
- Split theo entity
- Loại duplicate và near-duplicate

# 2. Validation phải mô phỏng production

| Tình huống | Cách split |
|---|---|
| Dự báo tương lai | Time split |
| Dự đoán user mới | Group split theo user |
| Mẫu độc lập | Random split |
| Dữ liệu theo thiết bị | Group split theo device |

# 3. Feature selection

## Filter methods

- Variance threshold
- Correlation
- Mutual information
- Chi-square
- ANOVA

## Wrapper methods

- Recursive Feature Elimination
- Sequential Feature Selection

## Embedded methods

- L1 regularization
- Tree-based selection
- Boosting

## Permutation importance

1. Tính metric ban đầu
2. Trộn một feature
3. Tính lại metric
4. Mức giảm là importance

## SHAP

Dùng để giải thích đóng góp feature ở mức global và local.

> [!warning]
> Feature importance và SHAP không chứng minh quan hệ nhân quả.

# 4. Ablation study

So sánh từng nhóm:

```text
Baseline
+ Calendar features
+ Lag features
+ Weather features
+ Customer history
```

Đây là cách thực tế để biết nhóm feature nào tạo giá trị.

# 5. Feature availability table

| Feature | Có lúc inference? | Delay | Rủi ro |
|---|---:|---:|---|
| age | Có | 0 | thấp |
| spend_30d | Có | 1 giờ | trung bình |
| final_status | Không | sau kết quả | leakage |

---
title: "Data Leakage, Pipeline và ColumnTransformer"
tags: [preprocessing, data-leakage, pipeline, sklearn]
---

# Data Leakage, Pipeline và ColumnTransformer

## 1. Leakage là gì?

Model được tiếp cận thông tin không tồn tại lúc prediction hoặc validation/test ảnh hưởng train.

Hậu quả:
- validation quá tốt;
- production giảm mạnh.

## 2. Leakage do preprocessing

Sai:

```python
scaler.fit(X_all)
```

Đúng:

```python
scaler.fit(X_train)
```

Tương tự với imputer, PCA, feature selector, target encoder, TF-IDF.

## 3. Target Leakage

Ví dụ dự đoán trả nợ nhưng dùng `loan_repaid_date` nếu cột này chỉ có sau outcome.

## 4. Time Leakage

Dự đoán \(y_t\) nhưng dùng $y_{t+1}$, rolling window chứa tương lai hoặc weather thực tế tương lai.

## 5. Group Leakage

Cùng user/patient/device/document xuất hiện cả train và test có thể làm đánh giá lạc quan.

## 6. Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", model),
])
```

## 7. ColumnTransformer

Dùng xử lý khác nhau cho numerical/categorical columns.

## 8. Vì sao Pipeline tốt cho CV?

Mỗi fold:
```text
Fold Train → fit preprocessing
Fold Valid → transform only
```

## 9. Production

Nên lưu `preprocessing + model` cùng nhau để inference dùng đúng transformation.

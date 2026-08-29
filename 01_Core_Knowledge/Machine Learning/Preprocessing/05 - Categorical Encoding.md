---
title: "Categorical Encoding"
tags: [preprocessing, categorical-encoding]
---

# Categorical Encoding

## 1. Nominal vs Ordinal
### Nominal
- Không có thứ tự
- Ví dụ: `Hanoi`, `Hue`, `Da Nang`.
- Các giá trị chỉ là nhãn phân loại, không so sánh lớn/nhỏ.
### Ordinal
- Có thứ tự thật 
- Ví dụ: `low < medium < high`.
- Thứ tự mang ý nghĩa trong dữ liệu.
## 2. Label Encoding
Trong scikit-learn, `LabelEncoder` chủ yếu dành cho target \(y\).

```text
spam → 1
not_spam → 0
```

> [!warning]
> Không nên dùng LabelEncoder cho **nominal feature** vì sẽ tạo thứ tự giả (ví dụ: `Red=0, Blue=1, Green=2` khiến model hiểu sai là có quan hệ thứ tự).
## 3. Ordinal Encoding

```text
low = 0
medium = 1
high = 2
```

chỉ khi thứ tự thật sự tồn tại.
- Ví dụ: mức độ hài lòng, cấp độ rủi ro
## 4. One-Hot Encoding
Biến mỗi category thành một vector nhị phân.
Ví dụ:
```
Color = {Red, Green, Blue}

Red   → [1, 0, 0]
Green → [0, 1, 0]
Blue  → [0, 0, 1]
```
Ưu điểm:
- Không tạo thứ tự giả.
- Tốt với nominal có cardinality thấp/trung bình.
Nhược điểm:
- Số chiều tăng mạnh khi cardinality cao.
- Sinh ra ma trận thưa (sparse matrix).
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X_categorical)
```
## 5. Frequency Encoding
$$FE(c)=\frac{count(c)}{N}$$
- Một cột duy nhất.
- Hữu ích khi cardinality cao.
- Nhưng có thể gây rò rỉ thông tin nếu target liên quan đến tần suất.
## 6. Target Encoding
Mã hóa bằng kỳ vọng của target theo category:
$$TE(c)=E[y|c]$$
Một dạng smoothing:
$$TE(c)=\frac{n_c\bar y_c+\alpha\mu}{n_c+\alpha}$$
- Fit trên train
- Dùng out-of-fold để tránh leakage
- Thường dùng cho high-cardinality categorical.
## 7. Rare Categories

- Các category hiếm có thể gộp thành `"Other"`.
- Giúp giảm nhiễu và số chiều.
- Tránh overfitting vào những giá trị xuất hiện quá ít.

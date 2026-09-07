---
title: "Outliers và Distribution Transform"
tags: [preprocessing, outliers, skewness]
---

# Outliers và Distribution Transform

## 1. Outlier không đồng nghĩa dữ liệu sai

Có thể là:
- lỗi nhập;
- sensor lỗi;
- giao dịch lớn hợp lệ;
- fraud;
- sự kiện hiếm.

## 2. IQR Rule

$IQR=Q_3-Q_1$
$L=Q_1-1.5IQR,\qquad U=Q_3+1.5IQR$
Đây là heuristic phát hiện, không phải luật bắt buộc xóa.
## 3. Z-score
$$z=\frac{x-\mu}{\sigma}$$

`|z| > 3` đôi khi dùng nếu dữ liệu gần normal, nhưng không phù hợp với phân phối lệch mạnh.

## 4. Cách xử lý

- sửa nếu chắc chắn lỗi;
- giữ nếu hợp lệ;
- clipping/winsorization;
- log/power transform;
- RobustScaler;
- outlier indicator.

## 5. Clipping

$$x'=\min(\max(x,L),U)$$

Chỉ dùng khi có lý do nghiệp vụ hoặc validation chứng minh hữu ích.

## 6. Theo model

- Linear/KNN: thường nhạy hơn.
- Tree: ít nhạy scale hơn nhưng outlier vẫn có thể ảnh hưởng split.
- AdaBoost: có thể nhạy với noisy samples.
- XGBoost: có regularization nhưng không miễn nhiễm outlier.

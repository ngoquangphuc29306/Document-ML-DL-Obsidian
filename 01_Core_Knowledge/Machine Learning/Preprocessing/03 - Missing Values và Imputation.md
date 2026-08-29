---
title: "Missing Values và Imputation"
tags: [preprocessing, missing-values, imputation]
---

# Missing Values và Imputation

## 1. Missing có thể mang thông tin

`last_purchase_date = NaN` có thể nghĩa là **chưa từng mua hàng**.

Nguyên nhân:
- không thu thập được;
- user không cung cấp;
- sensor lỗi;
- feature không áp dụng;
- sự kiện chưa xảy ra.

## 2. Có nên xóa?

Không có luật “missing > 50% thì xóa”. Cần xem:
- giá trị nghiệp vụ;
- pattern missing;
- khả năng thu thập ở production;
- feature thay thế.

## 3. Numerical Imputation

### Mean
$x_{miss}\leftarrow \bar{x}$
Nhanh nhưng nhạy outlier và làm giảm variance.
### Median
$x_{miss}\leftarrow median(x)$
Ổn định hơn khi dữ liệu lệch hoặc có outlier.
### Constant
`0`, `-1`, `999` chỉ khi có ý nghĩa rõ và không trùng giá trị thật.
## 4. Missing Indicator

```text
income_missing = 1
```

Sau đó vẫn có thể impute `income`.

## 5. Categorical Missing

- mode;
- `"Unknown"` / `"Missing"`.

Nếu missing có ý nghĩa riêng, category riêng thường rõ ràng hơn mode.

## 6. Model-based Imputation

- `KNNImputer`
- `IterativeImputer`
Có thể tận dụng các feature khác nhưng tốn chi phí hơn và dễ leakage nếu không nằm trong pipeline.
## 7. Chống Leakage

```python
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_valid = imputer.transform(X_valid)
```

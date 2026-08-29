---
title: "Data Inspection và Cleaning"
tags: [preprocessing, data-cleaning, data-quality]
---

# Data Inspection và Cleaning

## 1. Hiểu schema

Với mỗi cột cần biết:
- kiểu dữ liệu;
- ý nghĩa;
- đơn vị;
- miền hợp lệ;
- missing có nghĩa gì;
- có phải ID không;
- có sẵn lúc inference không.

## 2. Sai kiểu dữ liệu

Ví dụ `"1000000"` lưu dưới dạng string có thể làm sort, aggregation và model sai.

## 3. Duplicate

- **Exact duplicate:** toàn bộ dòng giống nhau.
- **Entity duplicate:** cùng thực thể xuất hiện ngoài mong muốn.

> [!warning]
> Không xóa dòng chỉ vì thấy “trùng”. Trong transaction data, một user có nhiều dòng thường là hợp lệ.

## 4. Giá trị bất hợp lý

Ví dụ:

```text
age = -3
temperature = 999
income = -1
```

Cần xác định là lỗi nhập, sentinel, sai đơn vị hay trường hợp hiếm hợp lệ.

## 5. Chuẩn hóa category và đơn vị

```text
Hanoi / Ha Noi / hanoi / HN
```

có thể là cùng category.

Tương tự `1.72 m = 172 cm`.

## 6. ID Columns

`customer_id`, `order_id`, `uuid` thường không nên đưa trực tiếp vào model. Nên biến thành feature có nghĩa như:
- số order;
- recency;
- tổng chi tiêu.

## 7. EDA tối thiểu

Numerical:
- min/max;
- mean/median;
- quantile;
- missing rate;
- unique count.

Categorical:
- cardinality;
- top categories;
- rare categories;
- invalid values.

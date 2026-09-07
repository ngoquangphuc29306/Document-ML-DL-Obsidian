---
title: "Nền tảng Feature Engineering"
tags: [machine-learning, feature-engineering]
---
# Phần 1 — Nền tảng Feature Engineering

## 1. Feature engineering là gì?

Feature engineering là quá trình chọn, biến đổi, kết hợp hoặc tạo mới các biến đầu vào để mô hình học được quy luật tốt hơn.

$$X = \text{features}, \qquad y = \text{target}$$

**Ví dụ:**
$$\text{Average Order Value} = \frac{\text{Total Revenue}}{\text{Order Count}}$$

$$\text{Days Since Last Order} = \text{Prediction Time} - \text{Last Order Time}$$

---

## 2. Vì sao quan trọng?

Mô hình chỉ học từ thông tin có trong $X$. Nếu feature không biểu diễn đúng tín hiệu, đổi sang mô hình mạnh hơn chưa chắc giải quyết được.

$$\text{Performance} \approx \text{Data Quality} + \text{Feature Quality} + \text{Model} + \text{Validation}$$

---

## 3. Preprocessing khác Feature Engineering

### Preprocessing
*Làm dữ liệu hợp lệ:*
* Xử lý missing
* Sửa kiểu dữ liệu
* Chuẩn hóa
* Encode category
* Xóa duplicate

### Feature engineering
*Làm dữ liệu hữu ích hơn:*
* Ngày sinh $\rightarrow$ tuổi
* Timestamp $\rightarrow$ giờ, thứ, tháng
* Giao dịch $\rightarrow$ tổng chi tiêu 30 ngày
* Chuỗi thời gian $\rightarrow$ lag, rolling mean

> [!NOTE] Tóm tắt
> * **Preprocessing:** Làm dữ liệu dùng được.
> * **Feature engineering:** Làm dữ liệu giàu thông tin hơn.

---

## 4. Feature engineering theo model

### Linear models
**Thường cần:**
* Scaling
* Log transform
* Interaction
* Polynomial features
* Encoding

### Tree-based models
Có thể tự học threshold và interaction, nhưng vẫn cần feature nghiệp vụ:
* Recency
* Frequency
* Monetary value
* Lag
* Rolling statistics
* Ratios

### Neural networks
Có thể tự học representation, nhưng vẫn phụ thuộc vào:
* Tokenization
* Windowing
* Normalization
* Input design

---

## 5. Câu hỏi trước khi tạo feature

* Feature biểu diễn tín hiệu nào?
* Có liên quan hợp lý đến target không?
* Có tồn tại tại prediction time không?
* Có dùng thông tin tương lai không?
* Có tính được trong production không?
* Có ổn định theo thời gian không?
* Có trùng lặp với feature khác không?

> [!WARNING] Warning
> Nhiều feature hơn không đồng nghĩa mô hình tốt hơn. Feature dư thừa có thể làm tăng overfitting, độ trễ và rủi ro leakage.
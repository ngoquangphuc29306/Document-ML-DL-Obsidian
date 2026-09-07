---
title: "Generalization"
tags: [machine-learning, foundations, generalization]
status: complete
---
---

# Generalization

## 1. Khái niệm

Generalization (Khả năng khái quát hóa) là khả năng model hoạt động tốt trên **dữ liệu mới chưa từng thấy** (unseen data).
Mục tiêu chính của ML không phải là đạt train score cao nhất, mà là học được pattern đủ ổn định để dự đoán chính xác trên unseen data.

---
## 2. Training Error
$$E_{\text{train}}$$

Là lỗi của mô hình trên tập dữ liệu được dùng để huấn luyện (fit model).

---

## 3. Generalization Error
$$E_{\text{gen}} = \mathbb{E}_{(x,y) \sim P}[L(y, f(x))]$$

Trong thực tế ta không biết phân phối $P$ chính xác, nên phải ước lượng $E_{\text{gen}}$ thông qua tập **Validation / Test / Cross-Validation (CV)**.

---
## 4. Overfitting
Thường thấy khi:
$$E_{\text{train}} \ll E_{\text{validation}}$$
Mô hình học cả tín hiệu quan trọng (signal) lẫn nhiễu (noise) trong tập train.

---
## 5. Underfitting

Cả Train error và Validation error đều ở mức cao.

**Nguyên nhân có thể do:**
* Model quá đơn giản
* Thiếu feature quan trọng
* Regularization quá mạnh
* Quá trình optimization chưa đủ (ví dụ: số epoch/iteration quá ít)

---
## 6. Generalization Gap
Trực giác:
$$\text{Gap} = E_{\text{validation}} - E_{\text{train}}$$

Gap quá lớn thường gợi ý hiện tượng **Overfitting** (High Variance).

---

## 7. Yếu tố ảnh hưởng đến Generalization

* Model complexity (Độ phức tạp mô hình)
* Amount of data (Lượng dữ liệu huấn luyện)
* Noise (Nhiễu trong dữ liệu)
* Regularization (Kỹ thuật điều tiết)
* Feature quality (Chất lượng đặc trưng)
* Validation design (Thiết kế chiến lược kiểm thử)
* Distribution shift (Sự dịch chuyển phân phối dữ liệu)

---

## 8. Distribution Shift

$$P_{\text{production}}(X,Y) \neq P_{\text{train}}(X,Y)$$

Model có thể đạt kết quả tốt trên tập test, nhưng sẽ giảm hiệu năng rõ rệt trên môi trường production nếu phân phối dữ liệu thay đổi.

---

## 9. Cách cải thiện Generalization

* Thêm dữ liệu có tính đại diện cao
* Áp dụng Regularization
* Giảm độ phức tạp mô hình khi bị overfit
* Đầu tư vào Feature Engineering
* Sử dụng Cross-Validation chuẩn xác
* Dùng Early Stopping khi huấn luyện

---

> [!TIP] Ghi nhớ cốt lõi
> **Generalization** = Học quy luật thực sự, không phải ghi nhớ training data.
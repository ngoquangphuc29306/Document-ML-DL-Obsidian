---
tags:
  - ml/core-concepts
  - theory
  - evaluation
created: 2026-07-21
---

# Bias-Variance Tradeoff

**Bias-Variance Tradeoff** là một khung lý thuyết cơ bản trong [[Machine Learning Overview|Machine Learning]], giải thích nguồn gốc của sai số (error) trong các mô hình dự đoán. Việc hiểu rõ Tradeoff này giúp ta biết cách điều chỉnh độ phức tạp của mô hình để đạt hiệu suất tối ưu.

---

## 1. Total Error Decomposition

Giả sử dữ liệu thực tế được tạo ra từ một hàm số ẩn:
$$y = f(x) + \epsilon$$

Trong đó:
* $f(x)$: Hàm số thực tế mà mô hình $\hat{f}(x)$ cố gắng xấp xỉ.
* $\epsilon$: Nhiễu ngẫu nhiên (irreducible error), có trung bình bằng 0 và độ biến động $\sigma^2$. Nhiễu này không thể loại bỏ dù mô hình có hoàn hảo đến đâu.

Khi đo lường sai số dự đoán trung bình (Mean Squared Error - MSE) trên dữ liệu mới, ta có thể phân rã toán học thành **3 thành phần**:

$$\text{Expected Test MSE} = \text{Bias}^2(\hat{f}(x)) + \text{Variance}(\hat{f}(x)) + \sigma^2$$

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

---

## 2. Định nghĩa chi tiết Bias và Variance
![Pasted image 20260723103253](../../../05_Assets/Images/Pasted%20image%2020260723103253.png)

### 2.1. Bias (Độ chệch)
* **Khái niệm:** Bias là độ sai lệch giữa **giá trị trung bình mà mô hình dự đoán** so với **giá trị thực tế**.
* **Bản chất:** ML method không thể nắm bắt được mối quan hệ thực sự của data.
* **Bias cao (High Bias):** Mô hình quá đơn giản, không học được cấu trúc/quy luật phức tạp của dữ liệu.
* **Hậu quả:** Gây ra hiện tượng [[Overfitting vs Underfitting#Underfitting Học vẹt chưa tới Quá đơn giản|Underfitting]].

### 2.2. Variance (Độ biến động)
* **Khái niệm:** đo mức độ dự đoán của mô hình thay đổi khi ta huấn luyền cũng một phương phpas trên các training set khác nhau được lấy từ cùng một phân phối dữ liệu
* **Bản chất:** Đại diện cho độ nhạy của mô hình đối với sự thay đổi nhỏ trong dữ liệu Train.
* **Variance cao (High Variance):** Mô hình "học thuộc lòng" cả các điểm nhiễu (noise) của tập Train.
* **Hậu quả:** Gây ra hiện tượng [[Overfitting vs Underfitting#Overfitting Học vẹt quá đà Quá phức tạp|Overfitting]] (khi đổi sang tập Test, kết quả bị dao động mạnh và biến đổi tiêu cực).
![Pasted image 20260721095112](../../../05_Assets/Images/Pasted%20image%2020260721095112.png)
---